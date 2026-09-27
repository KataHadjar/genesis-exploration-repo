"""Report the local Python and Genesis environment."""

from __future__ import annotations

import importlib.util
import platform
import sys

from genesis_exploration.runtime import get_runtime_info


def main() -> int:
    info = get_runtime_info()
    print(f"Python: {info.python}")
    print(f"Platform: {info.platform}")
    print(f"Machine: {info.machine}")
    print(f"Genesis importable: {importlib.util.find_spec('genesis') is not None}")
    print(f"PyTorch importable: {importlib.util.find_spec('torch') is not None}")
    print(f"Executable: {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
