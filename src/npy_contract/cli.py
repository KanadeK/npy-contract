"""CLI writes receipts to stdout and never modifies inputs or creates files."""

import argparse
import json

from . import __version__
from .core import ContractError, load_contract, verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check dense NPY delivery metadata without loading arrays"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("contract", help="Explicit version-1 JSON contract")
    parser.add_argument("--root", required=True, help="Directory containing the contracted arrays")
    args = parser.parse_args(argv)
    try:
        report = verify(load_contract(args.contract), args.root)
    except ContractError as exc:
        print(json.dumps({"report_version": 1, "ok": False, "error": str(exc)}))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["ok"] else 1
