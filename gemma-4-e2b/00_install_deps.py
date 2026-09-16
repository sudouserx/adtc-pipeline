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
    # #region agent log
    import json
    import time
    from pathlib import Path as _Path

    _jinja = package_version("Jinja2")
    if _jinja == "missing":
        _jinja = package_version("jinja2")
    _rec = {
        "sessionId": "846a88",
        "runId": "pre-fix",
        "hypothesisId": "A",
        "location": "00_install_deps.py:main",
        "message": "pinned vs jinja2",
        "data": {
            "jinja2_pinned": "jinja2" in config.PINNED_PACKAGES
            or "Jinja2" in config.PINNED_PACKAGES,
            "jinja2_installed": _jinja,
            "mismatch_count": sum(
                1
                for name, version in config.PINNED_PACKAGES.items()
                if public_version(package_version(name)) != version
            ),
        },
        "timestamp": int(time.time() * 1000),
    }
    _line = json.dumps(_rec) + "\n"
    for _path in (
        _Path("/home/ebrahim/Desktop/adtc pipeline/.cursor/debug-846a88.log"),
        _Path(__file__).resolve().parents[1] / ".cursor" / "debug-846a88.log",
    ):
        try:
            _path.parent.mkdir(parents=True, exist_ok=True)
            with _path.open("a", encoding="utf-8") as _handle:
                _handle.write(_line)
        except Exception:
            pass
    print("DEBUG_LOG", _line, flush=True)
    # #endregion
    mismatches = {
        name: (package_version(name), version)
        for name, version in config.PINNED_PACKAGES.items()
        if public_version(package_version(name)) != version
    }
    if not mismatches:
        print("Pinned packages already match config.py")
        # #region agent log
        _after = package_version("Jinja2")
        if _after == "missing":
            _after = package_version("jinja2")
        _rec2 = {
            "sessionId": "846a88",
            "runId": "post-fix",
            "hypothesisId": "A",
            "location": "00_install_deps.py:main",
            "message": "jinja2 already matched",
            "data": {
                "jinja2_installed": _after,
                "wanted": config.PINNED_PACKAGES.get("jinja2"),
            },
            "timestamp": int(time.time() * 1000),
        }
        print("DEBUG_LOG", json.dumps(_rec2) + "\n", flush=True)
        # #endregion
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
    # #region agent log
    _after = package_version("Jinja2")
    if _after == "missing":
        _after = package_version("jinja2")
    _rec2 = {
        "sessionId": "846a88",
        "runId": "post-fix",
        "hypothesisId": "A",
        "location": "00_install_deps.py:main",
        "message": "jinja2 after install",
        "data": {"jinja2_installed": _after, "wanted": config.PINNED_PACKAGES.get("jinja2")},
        "timestamp": int(time.time() * 1000),
    }
    _line2 = json.dumps(_rec2) + "\n"
    print("DEBUG_LOG", _line2, flush=True)
    for _path in (
        _Path("/home/ebrahim/Desktop/adtc pipeline/.cursor/debug-846a88.log"),
        _Path(__file__).resolve().parents[1] / ".cursor" / "debug-846a88.log",
    ):
        try:
            _path.parent.mkdir(parents=True, exist_ok=True)
            with _path.open("a", encoding="utf-8") as _handle:
                _handle.write(_line2)
        except Exception:
            pass
    # #endregion
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
