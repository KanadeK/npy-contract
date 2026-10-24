import contextlib
import copy
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from npy_contract import ContractError, inspect_npy, load_contract, verify
from npy_contract.cli import main
from npy_contract.core import MAX_HEADER, _header, validate_contract


def npy(*, shape=(2, 3), dtype="<f4", order=False, version=(1, 0), payload=None, header=None):
    prefix = b"\x93NUMPY" + bytes(version)
    base = 10 if version == (1, 0) else 12
    text = header or repr({"descr": dtype, "fortran_order": order, "shape": shape})
    raw = text.encode("utf8" if version == (3, 0) else "latin1")
    raw += b" " * ((-(base + len(raw) + 1)) % 64) + b"\n"
    size = struct.pack("<H" if version == (1, 0) else "<I", len(raw))
    if payload is None:
        count = 1
        for dim in shape:
            count *= dim
        payload = bytes(count * int(dtype[2:]))
    return prefix + size + raw + payload


def contract():
    return {
        "version": 1,
        "limits": {"array_bytes": 1024, "total_bytes": 2048},
        "arrays": [
            {"path": "features.npy", "dtype": "<f4", "shape": ["rows", 3], "order": "C"},
            {"path": "labels.npy", "dtype": "|u1", "shape": ["rows"], "order": "either"},
        ],
    }


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.feature = self.root / "features.npy"
        self.label = self.root / "labels.npy"
        self.feature.write_bytes(npy())
        self.label.write_bytes(npy(shape=(2,), dtype="|u1"))

    def single(self, content):
        self.feature.write_bytes(content)
        return inspect_npy(self.feature)

    def assert_error(self, code, content):
        with self.assertRaisesRegex(ContractError, f"^{code}$"):
            self.single(content)

    def test_complete_batch_shared_dimension(self):
        report = verify(contract(), self.root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["bindings"], {"rows": 2})
        self.assertEqual(report["observed_data_bytes"], 26)
        self.assertFalse(report["payload_values_checked"])

    def test_all_versions_and_endianness(self):
        for version in [(1, 0), (2, 0), (3, 0)]:
            for dtype in ["<f4", ">f8", "|i1", "<c8"]:
                with self.subTest(version=version, dtype=dtype):
                    data = npy(version=version, dtype=dtype)
                    result = self.single(data)
                    self.assertEqual(result["dtype"], dtype)
                    self.assertEqual(result["npy_version"], list(version))

    def test_zero_dimensions_and_scalar(self):
        self.assertEqual(self.single(npy(shape=()))["data_bytes"], 4)
        self.assertEqual(self.single(npy(shape=(0, 10)))["data_bytes"], 0)
        self.assertEqual(self.single(npy(shape=(0, 1_000_000), payload=b""))["data_bytes"], 0)

    def test_empty_array_still_requires_representable_nonzero_axes(self):
        self.assert_error("declared_size_overflow", npy(shape=(0, (1 << 63) - 1), payload=b""))
        self.assert_error("declared_size_overflow", npy(shape=(1 << 40, 0, 1 << 40), payload=b""))

    def test_long_json_integer_fails_as_structured_input_error(self):
        path = self.root / "large-integer.json"
        path.write_text('{"version":' + "9" * 5000 + "}", encoding="utf8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([str(path), "--root", str(self.root)])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])

    def test_shared_dimension_conflict(self):
        self.label.write_bytes(npy(shape=(3,), dtype="|u1"))
        report = verify(contract(), self.root)
        self.assertFalse(report["ok"])
        self.assertIn("shared_dimension_mismatch", report["arrays"][1]["issues"])

    def test_repeated_symbol_inside_one_shape(self):
        data = contract()
        data["arrays"][0]["shape"] = ["square", "square"]
        self.assertIn("shared_dimension_mismatch", verify(data, self.root)["arrays"][0]["issues"])

    def test_wildcard_shape(self):
        data = contract()
        data["arrays"][0]["shape"] = [None, 3]
        self.assertTrue(verify(data, self.root)["ok"])

    def test_dtype_order_rank_and_dimension_mismatches(self):
        for key, value, code in [
            ("dtype", ">f4", "dtype_mismatch"),
            ("order", "F", "order_mismatch"),
            ("shape", [2], "rank_mismatch"),
            ("shape", [2, 4], "dimension_mismatch"),
        ]:
            with self.subTest(key=key):
                data = contract()
                data["arrays"][0][key] = value
                self.assertIn(code, verify(data, self.root)["arrays"][0]["issues"])

    def test_header_order_flag_is_used_even_singleton(self):
        self.assertEqual(self.single(npy(shape=(1,), order=True))["order"], "F")

    def test_truncated_and_trailing_payload(self):
        data = npy()
        for content in [data[:-1], data + b"another array or trailing byte"]:
            self.assert_error("payload_length_mismatch", content)

    def test_object_payload_is_never_deserialized(self):
        self.assert_error("object_dtype_forbidden", npy(dtype="|O", payload=b"not even pickle"))

    def test_structured_strings_native_and_extended_dtype_fail_closed(self):
        for dtype in ["<f16", "=i4", "<U4", "|f4", "f4", "<V8", [("secret", "|O")]]:
            with self.subTest(dtype=dtype):
                self.assert_error("unsupported_dtype", npy(dtype=dtype, payload=b""))

    def test_missing_unknown_and_duplicate_header_keys(self):
        for header in [
            "{'descr':'<f4','shape':(2,)}",
            "{'descr':'<f4','shape':(2,),'fortran_order':False,'extra':0}",
            "{'descr':'<f4','descr':'<f4','shape':(2,)}",
        ]:
            with self.subTest(header=header):
                self.assert_error("invalid_header_keys", npy(header=header))

    def test_non_literal_header_cannot_execute(self):
        self.assert_error(
            "invalid_header_literal",
            npy(
                header="{'descr':__import__('not_a_real_module'),'shape':(2,),'fortran_order':False}"
            ),
        )

    def test_invalid_shape_and_bool(self):
        for shape in [(True,), (-1,), (1.5,), [2], (1,) * 33]:
            with self.subTest(shape=shape):
                header = repr({"descr": "<f4", "shape": shape, "fortran_order": False})
                with self.assertRaises(ContractError):
                    self.single(npy(header=header))

    def test_invalid_order_type(self):
        self.assert_error("invalid_header_types", npy(order=0))

    def test_claimed_four_gigabyte_header_rejected_before_read(self):
        class Tracked(io.BytesIO):
            maximum_read = 0

            def read(self, count=-1):
                self.maximum_read = max(self.maximum_read, count)
                return super().read(count)

        stream = Tracked(b"\x93NUMPY\x02\x00" + b"\xff" * 4)
        with self.assertRaisesRegex(ContractError, "header_size_limit"):
            _header(stream)
        self.assertLessEqual(stream.maximum_read, 6)

    def test_header_budget_and_complexity(self):
        self.assert_error("header_size_limit", npy(header=" " * MAX_HEADER + "{}"))
        header = "{'descr':'<f4','shape':(" + "0," * 2100 + "),'fortran_order':False}"
        self.assert_error("header_complexity_limit", npy(header=header))

    def test_huge_declared_array_without_allocation(self):
        self.assert_error("declared_size_overflow", npy(shape=((1 << 63) - 1, 2), payload=b""))

    def test_array_budget(self):
        data = contract()
        data["limits"]["array_bytes"] = 23
        result = verify(data, self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(result["arrays"][0]["issues"], ["array_byte_budget"])

    def test_aggregate_budget(self):
        data = contract()
        data["limits"]["total_bytes"] = 25
        self.assertEqual(verify(data, self.root)["issues"], ["total_byte_budget"])

    def test_missing_array_is_failure_and_others_checked(self):
        self.feature.unlink()
        report = verify(contract(), self.root)
        self.assertFalse(report["ok"])
        self.assertTrue(report["arrays"][1]["ok"])

    def test_in_root_directory_and_final_symlinks_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "features.npy").write_bytes(npy())
        link = self.root / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not permit symlink creation for this user")
        data = contract()
        data["arrays"][0]["path"] = "link/features.npy"
        self.assertEqual(verify(data, self.root)["arrays"][0]["issues"], ["linked_path_forbidden"])
        self.feature.unlink()
        self.feature.symlink_to(outside / "features.npy")
        self.assertEqual(
            verify(contract(), self.root)["arrays"][0]["issues"], ["linked_path_forbidden"]
        )

    def test_contract_invalid_fields_types_and_unknown_options(self):
        original = contract()
        bad = []
        for key, value in [("version", True), ("arrays", []), ("limits", {"array_bytes": 0})]:
            item = copy.deepcopy(original)
            item[key] = value
            bad.append(item)
        item = copy.deepcopy(original)
        item["ignored"] = True
        bad.append(item)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate_contract(value)

    def test_traversal_absolute_and_duplicate_paths_refused(self):
        for path in ["../features.npy", "/features.npy", "a\\features.npy", "a//b.npy", "D:/x.npy"]:
            data = contract()
            data["arrays"][0]["path"] = path
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ContractError, "invalid_array_path"),
            ):
                validate_contract(data)
        data = contract()
        data["arrays"][1]["path"] = "FEATURES.npy"
        with self.assertRaisesRegex(ContractError, "duplicate_array_path"):
            validate_contract(data)

    def test_invalid_contract_shapes(self):
        for shape in [[True], [-1], [""], ["a.b"], [1.5], "rows", [None] * 33]:
            data = contract()
            data["arrays"][0]["shape"] = shape
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ContractError, "invalid_contract_shape"),
            ):
                validate_contract(data)

    def test_contract_file_duplicates_and_size(self):
        path = self.root / "contract.json"
        path.write_text('{"version":1,"version":1}', encoding="utf8")
        with self.assertRaisesRegex(ContractError, "duplicate_contract_key"):
            load_contract(path)
        path.write_bytes(b" " * (1_048_576 + 1))
        with self.assertRaisesRegex(ContractError, "contract_too_large"):
            load_contract(path)

    def test_non_regular_and_invalid_root(self):
        with self.assertRaisesRegex(ContractError, "not_regular_file"):
            inspect_npy(self.root)
        with self.assertRaisesRegex(ContractError, "invalid_root"):
            verify(contract(), self.feature)

    def test_invalid_magic_version_truncation_alignment_and_newline(self):
        self.assert_error("invalid_npy_magic", b"NOTNPYabcdef")
        self.assert_error("unsupported_npy_version", b"\x93NUMPY\x09\x00")
        self.assert_error("truncated_npy_header", b"\x93NUM")
        content = npy()
        header_end = 10 + int.from_bytes(content[8:10], "little")
        self.assert_error(
            "header_missing_newline", content[: header_end - 1] + b" " + content[header_end:]
        )
        raw = b"{'descr':'<f4','shape':(0,),'fortran_order':False}\n"
        self.assert_error(
            "header_alignment", b"\x93NUMPY\x01\x00" + struct.pack("<H", len(raw)) + raw
        )

    def test_invalid_budget_api(self):
        for budget in [True, -1, 0, 1.5]:
            with self.assertRaisesRegex(ContractError, "invalid_byte_budget"):
                inspect_npy(self.feature, max_bytes=budget)

    def test_reports_omit_payload_values_and_absolute_paths(self):
        marker = b"PRIVATE_PAYLOAD_MARKER___"
        self.feature.write_bytes(npy(payload=marker))
        output = json.dumps(verify(contract(), self.root))
        self.assertNotIn(marker.decode(), output)
        self.assertNotIn(str(self.root), output)

    def test_same_length_changed_values_are_outside_scope(self):
        self.feature.write_bytes(npy(payload=b"z" * 24))
        self.assertTrue(verify(contract(), self.root)["ok"])

    def test_cli_exit_codes_and_input_stays_unchanged(self):
        path = self.root / "contract.json"
        path.write_text(json.dumps(contract()), encoding="utf8")
        before = self.feature.read_bytes()
        for action, expected in [
            (lambda: None, 0),
            (lambda: self.label.unlink(), 1),
            (lambda: path.write_text("{bad", encoding="utf8"), 2),
        ]:
            action()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main([str(path), "--root", str(self.root)])
            self.assertEqual(code, expected)
            self.assertEqual(json.loads(output.getvalue())["ok"], expected == 0)
        self.assertEqual(self.feature.read_bytes(), before)

    def test_real_cli_subprocess(self):
        path = self.root / "contract.json"
        path.write_text(json.dumps(contract()), encoding="utf8")
        result = subprocess.run(
            [sys.executable, "-m", "npy_contract", str(path), "--root", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["ok"])

    @unittest.skipUnless(os.name != "nt", "POSIX FIFO test")
    def test_fifo_refused_without_blocking(self):
        fifo = self.root / "fifo.npy"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ContractError, "not_regular_file"):
            inspect_npy(fifo)

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_windows_junction_directory_refused(self):
        target = self.root / "target"
        target.mkdir()
        (target / "features.npy").write_bytes(npy())
        link = self.root / "junction"
        result = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = contract()
        data["arrays"][0]["path"] = "junction/features.npy"
        self.assertEqual(verify(data, self.root)["arrays"][0]["issues"], ["linked_path_forbidden"])


if __name__ == "__main__":
    unittest.main()
