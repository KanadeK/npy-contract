"""Read-only dense NPY delivery contracts; no NumPy runtime dependency."""

from .core import ContractError, inspect_npy, load_contract, verify

__version__ = "0.1.0"
__all__ = ["ContractError", "inspect_npy", "load_contract", "verify"]
