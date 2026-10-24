"""Independent format oracle: real NumPy writes, including versions and layouts.

Development/CI requires NumPy; the installed runtime and smoke workflow do not.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from npy_contract import ContractError, inspect_npy


class NumPyOracle(unittest.TestCase):
    def test_numpy_written_numeric_arrays_all_supported_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.npy"
            for dtype in [
                "?",
                "i1",
                "u1",
                "<i2",
                ">i4",
                "<i8",
                "<u2",
                ">u4",
                "<u8",
                "<f2",
                ">f4",
                "<f8",
                "<c8",
                ">c16",
            ]:
                for shape in [(), (0,), (2, 3), (2, 1, 3)]:
                    for order in ["C", "F"]:
                        for version in [(1, 0), (2, 0), (3, 0)]:
                            with self.subTest(
                                dtype=dtype, shape=shape, order=order, version=version
                            ):
                                array = np.zeros(shape, dtype=dtype, order=order)
                                with path.open("wb") as stream:
                                    np.lib.format.write_array(
                                        stream, array, version=version, allow_pickle=False
                                    )
                                result = inspect_npy(path)
                                self.assertEqual(result["dtype"], array.dtype.str)
                                self.assertEqual(result["shape"], list(array.shape))
                                self.assertEqual(result["data_bytes"], array.nbytes)
                                self.assertEqual(result["npy_version"], list(version))

    def test_numpy_object_and_structured_arrays_never_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.npy"
            for array in [
                np.array(["synthetic"], dtype=object),
                np.zeros(2, dtype=[("number", "i4"), ("object", object)]),
            ]:
                np.save(path, array, allow_pickle=True)
                with self.assertRaises(ContractError):
                    inspect_npy(path)
