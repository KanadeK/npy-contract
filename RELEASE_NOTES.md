# v0.1.0

Initial usable release of dense NPY delivery contracts.

- Read-only bounded NPY 1.0/2.0/3.0 metadata and raw payload length checks.
- Exact dtype/layout, literal/wildcard/shared dimensions and declared byte budgets.
- Value-free JSON receipts, fail-closed unsupported formats and stable error codes.
- Standard-library runtime, synthetic examples, real NumPy format-oracle tests,
  clean offline wheel installation and Linux/Windows CI.

Limits: numeric scalar descriptors only; no NPZ/object/structured/string arrays,
numeric content validation, payload hashes or filesystem snapshot guarantee.
Read the README for full format, resource and privacy boundaries.
