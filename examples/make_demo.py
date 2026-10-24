"""Create two tiny synthetic delivery batches, using only the standard library."""

import json
import struct
import sys
from pathlib import Path


def write_array(path, shape, dtype, values):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode("ascii")
    header += b" " * ((-(10 + len(header) + 1)) % 64) + b"\n"
    data = b"".join(struct.pack("<f" if dtype == "<f4" else "B", value) for value in values)
    with path.open("xb") as stream:
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data)


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    contract = {
        "version": 1,
        "limits": {"array_bytes": 1_048_576, "total_bytes": 2_097_152},
        "arrays": [
            {"path": "features.npy", "dtype": "<f4", "shape": ["samples", 2], "order": "C"},
            {"path": "labels.npy", "dtype": "|u1", "shape": ["samples"], "order": "either"},
        ],
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf8")
    for name, labels in [("good", [0, 1, 0]), ("mismatch", [0, 1])]:
        folder = output / name
        folder.mkdir()
        write_array(folder / "features.npy", (3, 2), "<f4", [1, 2, 3, 4, 5, 6])
        write_array(folder / "labels.npy", (len(labels),), "|u1", labels)
    print(
        "Created good and mismatched synthetic batches; existing directories are never overwritten."
    )


if __name__ == "__main__":
    main()
