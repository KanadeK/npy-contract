# Third-party components

Project source and synthetic fixtures were written for this project. No competitor
implementation or third-party fixture is vendored. Source is under the MIT license
in LICENSE; dependencies retain their own copyright and terms.

There are no third-party Python runtime package dependencies. Python and its
standard library are separately supplied by the user under the Python Software
Foundation license and bundled component notices. These components are not
redistributed in the wheel.

Development/build tools are installed separately, not included in the wheel:
setuptools (MIT), build (MIT), Ruff (MIT). CI uses actions/checkout and
actions/setup-python (MIT), pinned to verified Git commit IDs. Their transitive
dependencies retain upstream licenses; no upstream code has been relicensed here.

Related projects linked in the README are comparisons, not bundled dependencies.

NumPy is used only as an independent test oracle, installed separately; it is not
imported by the runtime, bundled, or needed for the clean wheel smoke test. NumPy
source is BSD-3-Clause; its separately distributed wheels include additional
component notices (including BLAS and supporting libraries). Preserve the notices
shipped with that distribution. No NumPy implementation or fixture is copied here.
