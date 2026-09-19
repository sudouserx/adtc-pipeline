#!/usr/bin/env python3
"""Clone, pin, and build llama.cpp into the shared KUZA_WORK_DIR/tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import config

if str(config.PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(config.PIPELINE_DIR))

import config as pipeline_config  # noqa: E402

pipeline_config.WORK_DIR = Path(os.environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline"))
pipeline_config.TOOLS_DIR = config.TOOLS_DIR
pipeline_config.LLAMA_CPP_COMMIT = config.LLAMA_CPP_COMMIT

from common import setup_llama_cpp  # noqa: E402


def main() -> int:
    binaries = setup_llama_cpp()
    for name, path in binaries.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
