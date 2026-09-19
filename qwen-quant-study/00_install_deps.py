#!/usr/bin/env python3
"""Install pinned CUDA/Python packages from the main pipeline config."""

from __future__ import annotations

import sys
from pathlib import Path

import config

PIPELINE = config.PIPELINE_DIR
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import config as pipeline_config  # noqa: E402

pipeline_config.WORK_DIR = Path(
    __import__("os").environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline")
)
pipeline_config.TOOLS_DIR = config.TOOLS_DIR

from common import (  # noqa: E402
    package_version,
    public_version,
    pytorch_index_url,
    require_llama_cpp_build_tools,
    run,
)


def main() -> int:
    require_llama_cpp_build_tools()
    pinned = pipeline_config.PINNED_PACKAGES
    mismatches = {
        name: (package_version(name), version)
        for name, version in pinned.items()
        if public_version(package_version(name)) != version
    }
    if not mismatches:
        print("Pinned packages already match config.py")
        return 0
    wanted = ", ".join(f"{name}=={want}" for name, (_, want) in mismatches.items())
    print(f"Installing mismatched packages: {wanted}")
    cuda_packages = [
        f"{name}=={pinned[name]}" for name in pipeline_config.TORCH_CUDA_PACKAGES
    ]
    remaining = [
        f"{name}=={version}"
        for name, version in pinned.items()
        if name not in pipeline_config.TORCH_CUDA_PACKAGES
    ]
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--index-url",
            pytorch_index_url(),
            *cuda_packages,
        ]
    )
    if remaining:
        run([sys.executable, "-m", "pip", "install", "--upgrade", *remaining])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
