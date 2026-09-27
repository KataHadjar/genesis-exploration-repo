"""Report the local Python and Genesis environment."""

from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Genesis importable: {importlib.util.find_spec('genesis') is not None}")
    print(f"PyTorch importable: {importlib.util.find_spec('torch') is not None}")
    print(f"Executable: {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
