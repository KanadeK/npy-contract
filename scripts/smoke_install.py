"""Build first; install the actual wheel without network or runtime packages."""

import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("npy_contract-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Build exactly one wheel first")
    with tempfile.TemporaryDirectory(prefix="npy-contract-install-") as directory:
        clean = Path(directory)
        venv.create(clean / "venv", with_pip=True)
        python = clean / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-index", str(wheels[0])],
            cwd=clean,
            check=True,
        )
        subprocess.run(
            [str(python), str(root / "examples/make_demo.py"), str(clean / "demo")],
            cwd=clean,
            check=True,
        )
        for name, expected in [("good", 0), ("mismatch", 1)]:
            result = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-m",
                    "npy_contract",
                    str(clean / "demo/contract.json"),
                    "--root",
                    str(clean / "demo" / name),
                ],
                cwd=clean,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != expected or json.loads(result.stdout)["ok"] != (expected == 0):
                raise RuntimeError("Fresh installed CLI failed the documented batch workflow")
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.util; assert importlib.util.find_spec('numpy') is None",
            ],
            cwd=clean,
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "uninstall", "-y", "npy-contract"], cwd=clean, check=True
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.util; assert importlib.util.find_spec('npy_contract') is None",
            ],
            cwd=clean,
            check=True,
        )
    print("Fresh offline wheel install, good/mismatch workflows, no NumPy, and uninstall passed")


if __name__ == "__main__":
    main()
