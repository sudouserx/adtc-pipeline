#!/usr/bin/env python3
"""Upload the finished Gemma 4 E2B run directory to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
import model
from common import adapter_dir, hf_token, require_file, run_dir, sha256_file, write_json

IGNORE_NAMES = {"__pycache__"}
IGNORE_SUFFIXES = {".tmp"}


def ignored(path: Path) -> bool:
    if path.suffix in IGNORE_SUFFIXES or path.name.endswith(".tmp"):
        return True
    return any(part in IGNORE_NAMES for part in path.parts)


def iter_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not ignored(path.relative_to(root))
    )


def model_card(repo_id: str) -> str:
    mix = model.MIX
    quant_lines = "\n".join(
        f"- `{name}/{spec['filename']}` (`{spec['base_type']}`)"
        for name, spec in model.QUANT_CANDIDATES.items()
    )
    return f"""---
license: gemma
base_model: {model.BASE_MODEL}
tags:
  - agriculture
  - east-africa
  - gemma4
  - gguf
  - lora
  - peft
language:
  - en
  - sw
---

# Kuza Gemma 4 E2B

Full training-run archive for Kuza (East Africa agricultural assistant),
fine-tuned from [`{model.BASE_MODEL}`](https://huggingface.co/{model.BASE_MODEL}).
Weights, logs, checkpoints, GGUFs, and provenance are stored with the same
layout as `$KUZA_WORK_DIR/{config.RUN_ID}/`.

This derivative is subject to the [Gemma license](https://ai.google.dev/gemma/terms).

## Training mix

- 100% English train from `{model.DATASETS["english"]}`
- {mix["swahili_of_english"]:.0%} Swahili from `{model.DATASETS["swahili"]}`
- {mix["general_of_english"]:.0%} `{model.DATASETS["general"]}`
- {mix["adversarial_of_english"]:.0%} adversarial from `{model.DATASETS["adversarial"]}`
- all multiturn from `{model.DATASETS["multiturn"]}`

LoRA: RsLoRA r={model.LORA["r"]}, alpha={model.LORA["lora_alpha"]}, QAT `{model.LORA["qat_scheme"]}`.
Sequence length {config.MAX_SEQ_LENGTH}, {config.EPOCHS:g} epochs, LR `{config.LEARNING_RATE}`.
Thinking is off. GGUFs are text-only (PLE kept; vision/audio dropped).

## Files

- `adapter/` — PEFT adapter, tokenizer, SFT metrics and manifests
- `training/` — Trainer checkpoints including `checkpoint-best`
- `merged_bf16/` — text-only merged Hugging Face BF16 weights
- `reference/` — text-only BF16 GGUF (`kuza-bf16.gguf`) and smoke log
- `imatrix/` — calibration corpus, eval corpus, imatrix, logs
- `quants/` — quantized GGUF candidates:
{quant_lines}
- `screen/` — GPU KLD diagnostic, hidden-set scores, and `results.json` winner
- `provenance/` — copied adapter metrics, recipes, screen JSON
- `upload_manifest.json` — path, size, and sha256 for every uploaded file

## Download

```bash
huggingface-cli download {repo_id} --local-dir ./{config.RUN_ID}
```

`screen/results.json` ranks by hidden-set accuracy, then GGUF size, then GPU
TPS. Do not treat GPU TPS as an ADTC laptop measurement.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=config.UPLOAD_REPO)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and bytes; write card/manifest locally; do not upload",
    )
    args = parser.parse_args()

    token = hf_token()
    root = run_dir()
    require_file(adapter_dir() / "adapter_config.json", "Run 02_sft.py first.")
    if not root.is_dir():
        raise RuntimeError(f"Run directory is missing: {root}")

    (root / "README.md").write_text(model_card(args.repo), encoding="utf-8")

    files = [path for path in iter_files(root) if path.name != "upload_manifest.json"]
    entries = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        relative = path.relative_to(root).as_posix()
        entry: dict[str, str | int] = {"path": relative, "size": size}
        if not args.dry_run:
            entry["sha256"] = sha256_file(path)
        entries.append(entry)
        print(f"{size:>12}  {relative}", flush=True)

    write_json(
        root / "upload_manifest.json",
        {
            "run_id": config.RUN_ID,
            "repo_id": args.repo,
            "run_dir": str(root),
            "file_count": len(entries),
            "total_bytes": total,
            "hashed": not args.dry_run,
            "files": entries,
        },
    )
    print(f"files: {len(entries)}")
    print(f"bytes: {total}")
    print(f"manifest: {root / 'upload_manifest.json'}")
    if args.dry_run:
        print(f"dry-run: skip upload to {args.repo}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    url = api.create_repo(
        args.repo, repo_type="model", exist_ok=True, token=token
    )
    print(f"repo: {url}")
    api.upload_large_folder(
        repo_id=args.repo,
        folder_path=root,
        repo_type="model",
        ignore_patterns=["**/*.tmp", "**/__pycache__/**"],
    )
    print(f"uploaded: {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
