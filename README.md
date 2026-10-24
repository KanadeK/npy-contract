# npy-contract

Check a batch of dense NPY files against a delivery contract **before loading the
arrays**. Catch a feature/label row mismatch, wrong endian dtype, changed layout,
truncation, object dtype or excessive declared memory in a local command or CI.

[中文说明](README.zh-CN.md) · [MIT](LICENSE) · Python 3.11+ · no runtime packages

```text
features.npy  shape [samples, 2]  dtype <f4
labels.npy    shape [samples]     dtype |u1
                         └──── shared dimension must agree
```

NumPy already has safe loading defaults and memory mapping. This tool supplies a
small, declarative **cross-file delivery gate** without installing NumPy, loading
the numeric data or executing pickle. It checks metadata and raw payload length;
it does not check values, hashes, numeric ranges or model correctness.

## Install and try

Clone the repository, then use a virtual environment:

```sh
git clone https://github.com/KanadeK/npy-contract.git
cd npy-contract
python -m venv .venv
# POSIX:
. .venv/bin/activate
# Windows PowerShell, instead of activation, prefix commands with .venv/Scripts/python.exe
python -m pip install .
python examples/make_demo.py demo
python -m npy_contract demo/contract.json --root demo/good
python -m npy_contract demo/contract.json --root demo/mismatch
```

The good batch exits **0** and binds `samples` to 3. The mismatch batch exits **1**:
its two labels cannot match three feature rows. Demo data are synthetic and created
by the included script; it refuses to overwrite an existing destination directory.
The installed `npy-contract` command is equivalent to `python -m npy_contract`.

For an offline installation, download the wheel from GitHub Releases and run
`python -m pip install --no-index path/to/npy_contract-0.1.0-py3-none-any.whl`.
No package-registry publication is implied.

## Write a contract

```json
{
  "version": 1,
  "limits": {"array_bytes": 1073741824, "total_bytes": 4294967296},
  "arrays": [
    {"path": "features.npy", "dtype": "<f4", "shape": ["samples", 16], "order": "C"},
    {"path": "labels.npy", "dtype": "|u1", "shape": ["samples"], "order": "either"}
  ]
}
```

- Integer dimensions must match exactly; `null` accepts any dimension; a name
  such as `samples` binds on its first observation and must agree everywhere,
  including repeated use in one array. Names match `[A-Za-z][A-Za-z0-9_]{0,63}`.
- `dtype` is an **exact on-disk descriptor**, including endianness. Supported:
  `b1`, `i1/i2/i4/i8`, `u1/u2/u4/u8`, `f2/f4/f8`, `c8/c16` preceded by `<` or `>`;
  `|` is allowed only for one-byte types. NumPy boolean is normally `|b1`.
- `order` is `C`, `F` or `either`, compared with the NPY header's order flag.
  NumPy can write a singleton array flagged `C` even if allocated with order `F`.
- Paths are explicit, relative `.npy` paths with `/` separators. Each component
  starts with an ASCII letter, digit or underscore and then allows letters,
  digits, underscores, dots or hyphens. Absolute paths, traversal, backslashes,
  empty components, case-insensitive duplicates and linked paths are refused.
- Every field shown above is required. Unknown fields, duplicate JSON keys,
  booleans used as integers, unsupported dtypes and invalid limits fail closed.
  Extra files in the directory are ignored; only the declared files are opened.

Reports contain expected relative filenames, observed dtype/shape/order/size,
dimension bindings and stable reason codes. They exclude array values, absolute
input paths and raw exceptions. Filenames and shapes can themselves be private;
review receipts before sharing. `observed_data_bytes` sums successfully inspected
files, not files whose header or per-file budget already failed. Any failed rule
still makes the entire receipt fail.

Exit codes: **0** all rules and total budget pass; **1** an array or total budget
fails; **2** invalid/unreadable contract, invalid root or command usage. JSON is
written to stdout for checks; argparse help/usage uses its normal text interface.
Save it with shell redirection if needed. Inputs are never rewritten.

## Resource and trust boundaries

NPY versions 1.0, 2.0 and 3.0 are supported for the scalar numeric types above.
Object/pickle, structured, subarray, string, datetime, extended-float descriptors,
NPZ archives and concatenated NPY streams are unsupported and fail rather than
silently skipping. A same-length corrupted or malicious numeric payload can pass.
This is not a security certification or a replacement for content validation.

Headers are capped at 10,000 bytes **before reading the advertised length**;
literal syntax trees at 2,048 nodes, rank at 32, dimensions and declared byte counts
at 2^63−1. The nonzero-axis product times item size must also fit that limit,
including for empty arrays, so stride sizes remain representable. Contracts are
capped at 1 MiB and 1,024 arrays. Paths have at most
16 components, each at most 128 ASCII characters, and 512 characters in total.
The user declares positive per-file and total data-byte budgets. Only bounded
headers are read with unbuffered IO; payload lengths are compared against file
size. There is no payload allocation or memory-map operation.

Use stable regular local files. In-root symlinks and Windows reparse points are
refused. Filesystem changes during a check, hostile concurrent replacement and
remote/filesystem IO stalls are outside the guarantee. Keep inputs stable until
the downstream program opens them. Header/schema validation does not authenticate
the producer. The runtime makes no network requests and invokes no subprocesses.

## Python API

```python
from npy_contract import load_contract, verify, inspect_npy

receipt = verify(load_contract("contract.json"), "delivery")
assert receipt["ok"], receipt
metadata = inspect_npy("delivery/features.npy", max_bytes=1 << 30)
```

## Verify, contribute and clean up

```sh
python -m pip install -e . "numpy>=2.1,<3" ruff==0.17.2 build==1.6.1
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m build
python scripts/smoke_install.py
```

NumPy is a development-only oracle: tests write real files across three NPY
versions, supported dtypes, shapes and storage layouts. Synthetic regressions
exercise invalid headers, huge sizes, shared dimensions, path boundaries, error
codes and read-only behavior. The wheel smoke test installs in a new virtual
environment without network or NumPy and runs both documented batch outcomes,
then uninstalls. Linux/Windows CI runs Python 3.11 and 3.13, without write access.

Uninstall with `python -m pip uninstall npy-contract`. Remove only the demo folder
you created, then deactivate and remove the optional `.venv` directory. There is
no service, background process, configuration in the home folder or persistent
cache. See [CONTRIBUTING](CONTRIBUTING.md), [SECURITY](SECURITY.md) and
[third-party notices](THIRD_PARTY.md).

Related tools: [NumPy format IO](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
for loading/mapping arrays; [SimpleNpyIO](https://github.com/NaokiHori/SimpleNpyIO)
and [cnpy](https://github.com/rogersce/cnpy) for C/C++ consumers. Use those when you
need array access; use this gate when explicit metadata delivery contracts fit.
