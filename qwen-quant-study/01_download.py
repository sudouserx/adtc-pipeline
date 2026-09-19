#!/usr/bin/env python3
"""Selective download of kuza-qwen-3.5-4b artifacts from Hugging Face."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import study_config as config
import paths
from study_common import hf_token, require_file, sha256_file, write_json


def fetch_hf_manifest() -> dict:
    url = f"https://huggingface.co/{config.HF_REPO}/raw/main/upload_manifest.json"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    token = hf_token()
    dest = config.ARTIFACTS_DIR
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"downloading {len(config.DOWNLOAD_INCLUDE)} paths to {dest}")
    snapshot_download(
        repo_id=config.HF_REPO,
        repo_type="model",
        local_dir=str(dest),
        allow_patterns=config.DOWNLOAD_INCLUDE,
        token=token,
    )

    manifest = fetch_hf_manifest()
    expected = {
        entry["path"]: entry["sha256"]
        for entry in manifest.get("files", [])
        if entry["path"] in config.DOWNLOAD_INCLUDE
    }
    verified: list[dict] = []
    missing: list[str] = []
    bad_hash: list[str] = []
    for rel, want_sha in sorted(expected.items()):
        path = dest / rel
        if not path.is_file():
            missing.append(rel)
            continue
        got = sha256_file(path)
        ok = got == want_sha
        if not ok:
            bad_hash.append(rel)
        verified.append(
            {
                "path": rel,
                "size": path.stat().st_size,
                "sha256": got,
                "expected_sha256": want_sha,
                "ok": ok,
            }
        )
        print(f"{'ok' if ok else 'BAD'}  {path.stat().st_size:>12}  {rel}")

    require_file(paths.reference_path(), "reference/kuza-bf16.gguf missing after download")
    require_file(paths.imatrix_path(), "imatrix/kuza.imatrix missing after download")
    require_file(paths.eval_path(), "imatrix/eval.txt missing after download")
    require_file(config.HIDDEN_PROMPTS, "hidden prompt set missing in repo")

    if missing:
        raise RuntimeError(f"Download incomplete; missing: {missing}")
    if bad_hash:
        raise RuntimeError(f"sha256 mismatch: {bad_hash}")

    write_json(
        dest / "download_manifest.json",
        {
            "repo_id": config.HF_REPO,
            "dest": str(dest),
            "files": verified,
            "total_bytes": sum(item["size"] for item in verified),
        },
    )
    print(f"verified {len(verified)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
