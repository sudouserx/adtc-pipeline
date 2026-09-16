#!/usr/bin/env python3
"""Clone, pin, and build llama.cpp tools into TOOLS_DIR."""

from __future__ import annotations

from common import setup_llama_cpp


def main() -> int:
    binaries = setup_llama_cpp()
    for name, path in binaries.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
