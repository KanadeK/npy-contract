# Contributing

Open an issue with a reproducible scenario and a small synthetic fixture. Explain
the expected behavior and why existing tools do not fit. Never upload production
data, credentials, or private archives/traces/databases.

Use Python 3.11+ in a virtual environment. Install with `python -m pip install -e .`
and install development checks with
`python -m pip install -r requirements-dev.txt` (includes a NumPy test oracle).
Run `python -m unittest discover -s tests -v`, `ruff check .`,
`ruff format --check .`, `python -m build`, and `python scripts/smoke_install.py`.
Tests must exercise real behavior and include regressions for fixes. Keep runtime
dependencies minimal and document any changed limits or report fields.

The maintainer reviews all external suggestions before integration. CI only tests,
builds and checks; it never commits or merges. Third-party copyright and license
notices must remain intact. Do not add co-author trailers or unrelated attribution
to maintainer-authored commits; preserve the authorship of any actual third-party
work. Do not submit copied implementations without provenance and compatible terms.
