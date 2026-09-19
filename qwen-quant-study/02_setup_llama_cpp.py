#!/usr/bin/env python3
"""Clone, pin, and build llama.cpp into the shared KUZA_WORK_DIR/tools."""

from __future__ import annotations

import study_config
from pipeline_env import bootstrap_pipeline

bootstrap_pipeline(study_config)

from common import setup_llama_cpp  # noqa: E402


def main() -> int:
    binaries = setup_llama_cpp()
    for name, path in binaries.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
