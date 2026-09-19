#!/usr/bin/env python3
"""Upload all qwen-quant-study artifacts and results to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
import quant_specs
from study_common import hf_token, read_json, sha256_file, write_json

IGNORE_DIR_NAMES = {".hf-cache", "__pycache__", ".git"}
IGNORE_SUFFIXES = {".tmp"}


def ignored(relative: Path) -> bool:
    if relative.suffix in IGNORE_SUFFIXES or relative.name.endswith(".tmp"):
        return True
    return any(part in IGNORE_DIR_NAMES for part in relative.parts)


def iter_upload_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not ignored(path.relative_to(root))
    )


def upload_roots() -> list[Path]:
    roots = [
        config.ARTIFACTS_DIR,
        config.QUANTS_DIR,
        config.SCREEN_DIR,
        config.RESULTS_DIR,
    ]
    return [path for path in roots if path.is_dir()]


def model_card(repo_id: str, summary: dict | None) -> str:
    past = "\n".join(
        f"- `{name}` — `{spec['base_type']}` (from [{config.HF_REPO}](https://huggingface.co/{config.HF_REPO}))"
        for name, spec in quant_specs.PAST_QUANT_CANDIDATES.items()
    )
    new = "\n".join(
        f"- `{name}` — `{spec['base_type']}` — {spec.get('notes', '')}"
        for name, spec in quant_specs.NEW_QUANT_CANDIDATES.items()
    )
    verdict = (summary or {}).get("diagnosis", "pending")
    explanation = (summary or {}).get("explanation", "Run the study pipeline to populate.")
    winner = (summary or {}).get("winner", "pending")
    bf16 = (summary or {}).get("bf16_hidden_mean", "pending")

    return f"""---
license: apache-2.0
base_model: {config.HF_REPO}
tags:
  - agriculture
  - east-africa
  - qwen
  - gguf
  - quantization
  - evaluation
language:
  - en
  - sw
---

# Kuza Qwen 3.5-4B Quant Study

Quantization vs finetuning diagnosis for the Kuza East Africa agricultural
assistant. Source weights and past quants from
[`{config.HF_REPO}`](https://huggingface.co/{config.HF_REPO}).

## Diagnosis

- **Verdict:** `{verdict}`
- **BF16 hidden_mean:** {bf16}
- **Screen winner:** `{winner}`
- {explanation}

## Layout

- `artifacts/` — downloaded reference GGUF, imatrix, tokenizer, past quants
- `quants/` — newly quantized GGUF candidates from this study
- `screen/` — KLD logs, hidden-set reports, bench logs, `results.json`
- `results/` — baseline eval, `report/analysis.md`, per-model JSON
- `upload_manifest.json` — path, size, sha256 for every uploaded file

## Past-run quants (re-evaluated)

{past}

## New quant candidates

{new}

## Download

```bash
huggingface-cli download {repo_id} --local-dir ./qwen-quant-study
```

Primary metric: hidden-set rubric score on 36 EN/SW agriculture prompts
(`data/hidden_prompts.jsonl` in the source repo). Ranking: hidden_mean desc,
then GGUF size asc, then GPU generation TPS desc.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=config.UPLOAD_REPO)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and write manifest/card locally; do not upload",
    )
    args = parser.parse_args()

    token = hf_token()
    root = config.WORK_DIR
    root.mkdir(parents=True, exist_ok=True)

    present = upload_roots()
    if not present:
        raise RuntimeError(
            f"No study output under {root}. Run the pipeline before uploading."
        )

    summary_path = config.REPORT_DIR / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    (root / "README.md").write_text(model_card(args.repo, summary), encoding="utf-8")

    files = [
        path
        for path in iter_upload_files(root)
        if path.name != "upload_manifest.json" and path != root / "README.md"
    ]
    files.append(root / "README.md")

    entries: list[dict[str, str | int]] = []
    total = 0
    for path in sorted(set(files)):
        if not path.is_file():
            continue
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
            "study_id": "qwen-quant-study",
            "source_repo": config.HF_REPO,
            "repo_id": args.repo,
            "work_dir": str(root),
            "upload_roots": [str(path.relative_to(root)) for path in present],
            "file_count": len(entries),
            "total_bytes": total,
            "hashed": not args.dry_run,
            "diagnosis": (summary or {}).get("diagnosis"),
            "winner": (summary or {}).get("winner"),
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
        ignore_patterns=[
            "**/.hf-cache/**",
            "**/__pycache__/**",
            "**/*.tmp",
            "**/.git/**",
        ],
    )
    print(f"uploaded: {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
