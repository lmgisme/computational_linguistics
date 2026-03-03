"""
setup_environment.py
--------------------
Run this script once to verify and install all required dependencies for
Phase 1 of the Turkic substrate detection project.

Usage:
    python setup_environment.py

This script:
  1. Checks Python version (3.9+ required).
  2. Attempts to import each required package.
  3. Installs any missing packages via pip.
  4. Runs a minimal smoke test of LingPy's IPA tokenizer and pandas.
  5. Prints a go/no-go status table.
"""

import sys
import subprocess
import importlib
from typing import NamedTuple

REQUIRED_PACKAGES = [
    # (import_name, pip_install_name, minimum_version_str_or_None)
    ("pandas",   "pandas",  "1.4"),
    ("requests", "requests", None),
    ("lingpy",   "lingpy",   None),
]


class PkgStatus(NamedTuple):
    name:      str
    available: bool
    version:   str
    note:      str


def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print(f"ERROR: Python 3.9+ required. You have {major}.{minor}.")
        sys.exit(1)
    print(f"Python {major}.{minor}.{sys.version_info[2]} — OK")


def try_import(import_name: str) -> tuple[bool, str]:
    """Try to import a package; return (success, version_string)."""
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        return True, ver
    except ImportError:
        return False, ""


def install_package(pip_name: str) -> bool:
    """Install a package via pip. Returns True on success."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name],
        capture_output=True, text=True
    )
    return result.returncode == 0


def run_smoke_tests() -> list[str]:
    """
    Run minimal imports and operations that exercise the core pipeline.
    Returns a list of failure messages (empty list = all passed).
    """
    failures = []

    # Test pandas
    try:
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert len(df) == 2, "pandas DataFrame construction failed"
    except Exception as e:
        failures.append(f"pandas smoke test: {e}")

    # Test requests
    try:
        import requests
        _ = requests.Session()
    except Exception as e:
        failures.append(f"requests smoke test: {e}")

    # Test LingPy IPA tokenizer
    try:
        from lingpy.sequence.sound_classes import ipa2tokens
        tokens = ipa2tokens("tʃaɣ")
        assert isinstance(tokens, list) and len(tokens) > 0, \
            f"ipa2tokens returned unexpected result: {tokens}"
    except ImportError:
        failures.append("LingPy import failed — tokenization will use fallback regex.")
    except Exception as e:
        failures.append(f"LingPy smoke test: {e}")

    return failures


def main():
    print("=" * 55)
    print("  Turkic Substrate Detection — Environment Setup")
    print("=" * 55)

    # 1. Python version check
    check_python_version()
    print()

    # 2. Check / install packages
    statuses: list[PkgStatus] = []
    for import_name, pip_name, _ in REQUIRED_PACKAGES:
        found, ver = try_import(import_name)
        if found:
            statuses.append(PkgStatus(pip_name, True, ver, "already installed"))
        else:
            print(f"  Installing {pip_name}...")
            success = install_package(pip_name)
            if success:
                _, ver = try_import(import_name)
                statuses.append(PkgStatus(pip_name, True, ver, "just installed"))
            else:
                statuses.append(PkgStatus(pip_name, False, "", "INSTALL FAILED"))

    # 3. Print package table
    print()
    print(f"  {'PACKAGE':<12} {'STATUS':<16} {'VERSION':<12} NOTES")
    print("  " + "-" * 52)
    for s in statuses:
        status_str = "OK" if s.available else "MISSING"
        print(f"  {s.name:<12} {status_str:<16} {s.version:<12} {s.note}")

    # 4. Smoke tests
    print()
    print("  Running smoke tests...")
    failures = run_smoke_tests()
    if not failures:
        print("  All smoke tests passed.")
    else:
        print("  Smoke test failures:")
        for f in failures:
            print(f"    - {f}")

    # 5. Go/no-go
    print()
    critical_ok = all(s.available for s in statuses if s.name in ("pandas", "requests"))
    if critical_ok:
        print("  GO: Critical dependencies satisfied. You can run main.py.")
    else:
        print("  NO-GO: Critical dependencies missing. Resolve above errors before running.")
    print("=" * 55)


if __name__ == "__main__":
    main()
