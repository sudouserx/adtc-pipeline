#!/usr/bin/env python3
"""Install pinned CUDA/Python packages from config.py."""

from __future__ import annotations

import sys

import config
from common import (
    package_version,
    public_version,
    pytorch_index_url,
    require_llama_cpp_build_tools,
    run,
)


def main() -> int:
    require_llama_cpp_build_tools()
    mismatches = {
        name: (package_version(name), version)
        for name, version in config.PINNED_PACKAGES.items()
        if public_version(package_version(name)) != version
    }
    if not mismatches:
        print("Pinned packages already match config.py")
        return 0
    wanted = ", ".join(f"{name}=={want}" for name, (_, want) in mismatches.items())
    print(f"Installing mismatched packages: {wanted}")
    cuda_packages = [
        f"{name}=={config.PINNED_PACKAGES[name]}" for name in config.TORCH_CUDA_PACKAGES
    ]
    remaining = [
        f"{name}=={version}"
        for name, version in config.PINNED_PACKAGES.items()
        if name not in config.TORCH_CUDA_PACKAGES
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
