"""Bounded metadata parsing and cross-file dimension contracts.

This validates metadata and raw payload length, not the numeric contents of arrays.
"""

from __future__ import annotations

import ast
import json
import os
import re
import stat
from pathlib import Path
from typing import BinaryIO

MAX_CONTRACT = 1_048_576
MAX_HEADER = 10_000
MAX_ARRAYS = 1024
MAX_RANK = 32
MAX_INTEGER = (1 << 63) - 1
DEFAULT_ARRAY_BYTES = 1 << 30
DEFAULT_TOTAL_BYTES = 4 << 30
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_DTYPE = re.compile(r"([<>|])([biufc])([0-9]{1,2})\Z")
_SIZES = {"b": {1}, "i": {1, 2, 4, 8}, "u": {1, 2, 4, 8}, "f": {2, 4, 8}, "c": {8, 16}}


class ContractError(ValueError):
    """A stable reason code, intentionally excluding raw input or host paths."""


def _integer(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value <= MAX_INTEGER


def _dtype(value: object) -> int:
    if not isinstance(value, str):
        raise ContractError("unsupported_dtype")
    if re.fullmatch(r"[<>=|]?O[0-9]*", value):
        raise ContractError("object_dtype_forbidden")
    match = _DTYPE.fullmatch(value)
    if not match:
        raise ContractError("unsupported_dtype")
    endian, kind, size_text = match.groups()
    size = int(size_text)
    if size not in _SIZES[kind] or (endian == "|" and size != 1):
        raise ContractError("unsupported_dtype")
    return size


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate_contract_key")
        result[key] = value
    return result


def _path_parts(value: object) -> list[str]:
    if not isinstance(value, str) or len(value) > 512 or not value.endswith(".npy"):
        raise ContractError("invalid_array_path")
    parts = value.split("/")
    if len(parts) > 16 or any(not _COMPONENT.fullmatch(part) for part in parts):
        raise ContractError("invalid_array_path")
    return parts


def validate_contract(contract: object) -> dict:
    """Validate the entire configuration before opening any array."""
    if not isinstance(contract, dict) or set(contract) != {"version", "arrays", "limits"}:
        raise ContractError("invalid_contract_fields")
    if type(contract["version"]) is not int or contract["version"] != 1:
        raise ContractError("unsupported_contract_version")
    limits = contract["limits"]
    if not isinstance(limits, dict) or set(limits) != {"array_bytes", "total_bytes"}:
        raise ContractError("invalid_limits")
    if not all(_integer(v, minimum=1) for v in limits.values()):
        raise ContractError("invalid_limits")
    arrays = contract["arrays"]
    if not isinstance(arrays, list) or not 1 <= len(arrays) <= MAX_ARRAYS:
        raise ContractError("invalid_array_count")
    paths: set[str] = set()
    for rule in arrays:
        if not isinstance(rule, dict) or set(rule) != {"path", "dtype", "shape", "order"}:
            raise ContractError("invalid_array_rule")
        parts = _path_parts(rule["path"])
        key = "/".join(parts).casefold()
        if key in paths:
            raise ContractError("duplicate_array_path")
        paths.add(key)
        _dtype(rule["dtype"])
        if type(rule["order"]) is not str or rule["order"] not in ("C", "F", "either"):
            raise ContractError("invalid_order")
        shape = rule["shape"]
        if not isinstance(shape, list) or len(shape) > MAX_RANK:
            raise ContractError("invalid_contract_shape")
        for dimension in shape:
            if dimension is None or _integer(dimension):
                continue
            if isinstance(dimension, str) and _SYMBOL.fullmatch(dimension):
                continue
            raise ContractError("invalid_contract_shape")
    return contract


def load_contract(path: str | Path) -> dict:
    try:
        with open(path, "rb") as stream:
            content = stream.read(MAX_CONTRACT + 1)
        if len(content) > MAX_CONTRACT:
            raise ContractError("contract_too_large")
        contract = json.loads(content.decode("utf-8"), object_pairs_hook=_pairs)
    except ContractError:
        raise
    except (OSError, ValueError, RecursionError) as exc:
        raise ContractError("contract_unreadable_or_invalid_json") from exc
    return validate_contract(contract)


def _read(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ContractError("truncated_npy_header")
    return data


def _header(stream: BinaryIO) -> tuple[dict, int, list[int]]:
    if _read(stream, 6) != b"\x93NUMPY":
        raise ContractError("invalid_npy_magic")
    version = tuple(_read(stream, 2))
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ContractError("unsupported_npy_version")
    size = int.from_bytes(_read(stream, 2 if version == (1, 0) else 4), "little")
    if not 1 <= size <= MAX_HEADER:
        raise ContractError("header_size_limit")
    raw = _read(stream, size)
    if not raw.endswith(b"\n"):
        raise ContractError("header_missing_newline")
    try:
        tree = ast.parse(
            raw.decode("utf-8" if version == (3, 0) else "latin1").strip(), mode="eval"
        )
        if sum(1 for _ in ast.walk(tree)) > 2048:
            raise ContractError("header_complexity_limit")
        if not isinstance(tree.body, ast.Dict):
            raise ContractError("invalid_header_dictionary")
        keys = tree.body.keys
        if len(keys) != 3 or any(
            not isinstance(key, ast.Constant) or type(key.value) is not str for key in keys
        ):
            raise ContractError("invalid_header_keys")
        if {key.value for key in keys} != {"descr", "fortran_order", "shape"}:
            raise ContractError("invalid_header_keys")
        metadata = ast.literal_eval(tree)
    except (SyntaxError, ValueError, UnicodeError, RecursionError, MemoryError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("invalid_header_literal") from exc
    if type(metadata["fortran_order"]) is not bool or type(metadata["shape"]) is not tuple:
        raise ContractError("invalid_header_types")
    shape = metadata["shape"]
    if len(shape) > MAX_RANK or any(not _integer(dim) for dim in shape):
        raise ContractError("invalid_header_shape")
    itemsize = _dtype(metadata["descr"])
    # Strides must be representable even for empty arrays: a zero axis cannot
    # erase an otherwise unrepresentable product of the nonzero axes.
    nonzero_elements = 1
    for dimension in shape:
        nonzero_elements *= max(1, dimension)
        if nonzero_elements > MAX_INTEGER // itemsize:
            raise ContractError("declared_size_overflow")
    elements = 0 if 0 in shape else nonzero_elements
    offset = (10 if version == (1, 0) else 12) + size
    if offset % 16:
        raise ContractError("header_alignment")
    return metadata, elements * itemsize, list(version)


def _regular(info: os.stat_result) -> bool:
    # All Windows reparse points, including junctions/placeholders, are refused.
    return stat.S_ISREG(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400


def _inspect(path: Path, max_bytes: int) -> dict:
    before = path.lstat()
    if not _regular(before):
        raise ContractError("not_regular_file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        if not _regular(opened) or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ContractError("file_changed")
        metadata, data_bytes, version = _header(stream)
        if data_bytes > max_bytes:
            raise ContractError("array_byte_budget")
        expected_size = stream.tell() + data_bytes
        if opened.st_size != expected_size:
            raise ContractError("payload_length_mismatch")
        after = os.fstat(stream.fileno())
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ContractError("file_changed")
    return {
        "dtype": metadata["descr"],
        "shape": list(metadata["shape"]),
        "order": "F" if metadata["fortran_order"] else "C",
        "data_bytes": data_bytes,
        "npy_version": version,
    }


def inspect_npy(path: str | Path, *, max_bytes: int = DEFAULT_ARRAY_BYTES) -> dict:
    """Inspect one stable regular file. No payload bytes are read or deserialized."""
    if not _integer(max_bytes, minimum=1):
        raise ContractError("invalid_byte_budget")
    try:
        return _inspect(Path(path), max_bytes)
    except OSError as exc:
        raise ContractError("array_unreadable") from exc


def _in_root(root: Path, relative: str) -> Path:
    cursor = root
    for part in _path_parts(relative):
        cursor = cursor / part
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ContractError("linked_path_forbidden")
    return cursor


def verify(contract: dict, root: str | Path) -> dict:
    """Return a versioned receipt; every rule is checked, failed runs never pass."""
    validate_contract(contract)
    root = Path(root)
    try:
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ContractError("invalid_root")
    except OSError as exc:
        raise ContractError("invalid_root") from exc
    bindings: dict[str, int] = {}
    results = []
    total = 0
    for index, rule in enumerate(contract["arrays"], 1):
        result = {"array": index, "path": rule["path"], "ok": False, "issues": []}
        try:
            metadata = inspect_npy(
                _in_root(root, rule["path"]), max_bytes=contract["limits"]["array_bytes"]
            )
            result["observed"] = metadata
            total += metadata["data_bytes"]
            issues = result["issues"]
            if metadata["dtype"] != rule["dtype"]:
                issues.append("dtype_mismatch")
            if rule["order"] != "either" and metadata["order"] != rule["order"]:
                issues.append("order_mismatch")
            if len(rule["shape"]) != len(metadata["shape"]):
                issues.append("rank_mismatch")
            else:
                for dimension, observed in zip(rule["shape"], metadata["shape"], strict=True):
                    if dimension is None:
                        continue
                    if isinstance(dimension, str):
                        if dimension in bindings and bindings[dimension] != observed:
                            issues.append("shared_dimension_mismatch")
                        else:
                            bindings[dimension] = observed
                    elif dimension != observed:
                        issues.append("dimension_mismatch")
            result["issues"] = sorted(set(issues))
            result["ok"] = not result["issues"]
        except ContractError as exc:
            result["issues"] = [str(exc)]
        except OSError:
            result["issues"] = ["array_unreadable"]
        results.append(result)
    global_issues = ["total_byte_budget"] if total > contract["limits"]["total_bytes"] else []
    return {
        "report_version": 1,
        "ok": not global_issues and all(row["ok"] for row in results),
        "arrays": results,
        "bindings": bindings,
        "observed_data_bytes": total,
        "issues": global_issues,
        "payload_values_checked": False,
    }
