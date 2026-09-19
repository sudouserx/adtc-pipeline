#!/usr/bin/env python3
"""Stage 08 — Upload the finished Gemma 4 E2B run directory to Hugging Face.

FINAL version (2026-09-20). Pairs with config.py; works against the
UNMODIFIED repo common.py / model.py.

Changes vs the original 08_upload.py:
  - Model card is driven by config (mix fractions, dataset registry,
    effective train length/LR) instead of model.py's hard-coded values.
  - DPO-aware: when adapter/ holds the promoted preference-aligned policy
    (02b_dpo.py), the card documents the DPO stage, and adapter-sft/ /
    adapter-dpo/ are listed as provenance artifacts.
  - Explicit note that the preference dataset (kuzaai/kuza_dpo_preference)
    is maintained and uploaded MANUALLY by the repo owner — this stage only
    archives the model run and never pushes datasets.

Usage:
  python 08_upload.py [--repo kuzaai/kuza-gemma-4-e2b] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
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


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def quant_listing(root: Path) -> str:
    """Enumerate candidates from 05_quants.py recipes; fall back to model.py."""
    quants = root / "quants"
    if quants.is_dir():
        recipes = sorted(quants.glob("*/recipe.json"))
        if recipes:
            return "\n".join(
                f"- `{p.parent.name}/` (`{read_json(p).get('base_type', '?')}`, from {read_json(p).get('source', 'reference')})"
                for p in recipes
            )
    from model import QUANT_CANDIDATES

    return "\n".join(
        f"- `{name}/{spec['filename']}` (`{spec['base_type']}`)"
        for name, spec in QUANT_CANDIDATES.items()
    )


def model_card(repo_id: str, root: Path) -> str:
    mix = config.MIX
    datasets = config.DATASETS
    from model import BASE_MODEL, LORA

    quant_lines = quant_listing(root)

    dpo_manifest = read_json(adapter_dir() / "dpo_manifest.json")
    sft_manifest = read_json(adapter_dir() / "sft_manifest.json")

    mix_lines = [
        f"- 100% English train from `{datasets['english']}`",
        f"- {mix['swahili_of_english']:.0%} Swahili from `{datasets['swahili']}`",
        f"- {mix['general_of_english']:.0%} `{datasets['general']}`",
        f"- {mix['adversarial_of_english']:.0%} adversarial from `{datasets['adversarial']}`",
        f"- {mix.get('multiturn_of_english', 0.03):.0%} multiturn from `{datasets['multiturn']}`",
    ]
    if mix.get("code_switch_of_english"):
        mix_lines.append(
            f"- {mix['code_switch_of_english']:.0%} code-switching (Sw-En) from `{datasets.get('code_switch', 'kuzaai/kuza_sft_code_switch')}`"
        )
    if mix.get("grounding_of_english"):
        mix_lines.append(
            f"- {mix['grounding_of_english']:.0%} passage-grounded QA from `{datasets.get('grounding', 'kuzaai/kuza_sft_grounding')}`"
        )
    mix_block = "\n".join(mix_lines)

    dpo_block = ""
    files_block = "- `adapter/` — PEFT policy adapter (SFT), tokenizer, metrics and manifests"
    if dpo_manifest is not None:
        dpo_hparams = dpo_manifest.get("hyperparameters", {})
        dpo_ds = dpo_manifest.get("dataset", {})
        origin = dpo_ds.get("repo") or dpo_ds.get("path") or config.DPO_DATASET
        dpo_block = f"""
## Preference alignment (DPO)

`adapter/` holds the SFT+DPO policy (promoted by `02b_dpo.py`). Preference
pairs: {dpo_manifest.get('train_pairs', '?')} train / {dpo_manifest.get('eval_pairs', '?')} validation from `{origin}`.
DPO: beta `{dpo_hparams.get('beta', config.DPO_BETA)}`, RPO alpha `{dpo_hparams.get('rpo_alpha', config.DPO_RPO_ALPHA)}`,
LR `{dpo_hparams.get('learning_rate', config.DPO_LR)}`, {dpo_hparams.get('num_train_epochs', config.DPO_EPOCHS):g} epoch(s),
loss `{dpo_hparams.get('loss_type', 'sigmoid')}`. The pristine SFT adapter is preserved in `adapter-sft/`;
the pre-promotion DPO output is preserved in `adapter-dpo/`.

The preference dataset [`{config.DPO_DATASET}`](https://huggingface.co/datasets/{config.DPO_DATASET})
is maintained and uploaded separately by the repo owner; it is never pushed by
the training pipeline.
"""
        files_block = "\n".join(
            [
                "- `adapter/` — PEFT **policy** adapter (SFT+DPO), tokenizer, metrics and manifests",
                "- `adapter-sft/` — frozen pristine SFT adapter (pre-alignment snapshot)",
                "- `adapter-dpo/` — DPO output as trained (pre-promotion copy)",
            ]
        )
    else:
        dpo_block = """
## Preference alignment (DPO)

Not applied for this run (`KUZA_SKIP_DPO=1` or the preference dataset was
unavailable); `adapter/` is the pure SFT policy.
"""

    provenance = ""
    if sft_manifest is not None:
        lora = sft_manifest.get("lora", {})
        provenance = (
            f"LoRA: RsLoRA r={lora.get('r', LORA['r'])}, alpha={lora.get('alpha', LORA['lora_alpha'])}, "
            f"QAT `{lora.get('qat_scheme', LORA['qat_scheme'])}`."
        )

    qat_files_line = (
        "- `reference_qat/` — QAT lattice-exact Q4_0 export (`kuza-qat-q4_0-lattice.gguf`)\n"
        if (root / "reference_qat").is_dir()
        else ""
    )

    return f"""---
license: gemma
base_model: {BASE_MODEL}
tags:
  - agriculture
  - east-africa
  - gemma4
  - gguf
  - lora
  - peft
  - dpo
language:
  - en
  - sw
---

# Kuza Gemma 4 E2B

Full training-run archive for Kuza (East Africa agricultural assistant),
fine-tuned from [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}).
Weights, logs, checkpoints, GGUFs, and provenance are stored with the same
layout as `$KUZA_WORK_DIR/{config.RUN_ID}/`.

This derivative is subject to the [Gemma license](https://ai.google.dev/gemma/terms).

## Training mix

{mix_block}

{provenance}
Sequence length {config.TRAIN_MAX_SEQ_LENGTH}, {config.EPOCHS:g} epochs, LR `{config.LEARNING_RATE}`.
Thinking is off. GGUFs are text-only (PLE kept; vision/audio dropped).
{dpo_block}
## Files

{files_block}
- `training/` — Trainer checkpoints including `checkpoint-best`
- `merged_bf16/` — text-only merged Hugging Face BF16 weights
- `reference/` — text-only BF16 GGUF (`kuza-bf16.gguf`) and smoke log
{qat_files_line}- `imatrix/` — calibration corpus, eval corpus, imatrix, logs
- `quants/` — quantized GGUF candidates:
{quant_lines}
- `screen/` — GPU KLD diagnostic, hidden-set scores, and `results.json` winner
- `provenance/` — copied adapter metrics, recipes, screen JSON
- `upload_manifest.json` — path, size, and sha256 for every uploaded file

## Download

```bash
huggingface-cli download {repo_id} --local-dir ./{config.RUN_ID}
```

`screen/results.json` applies the quality gates (safety, language, same-top-p,
KLD ceiling) first, then ranks gate-passers within KLD noise by hidden-set
accuracy, CPU generation t/s, GGUF size, and peak RSS. CPU numbers are the
deployment signal for the edge target; GPU t/s and KLD are diagnostics.
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

    (root / "README.md").write_text(model_card(args.repo, root), encoding="utf-8")

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
