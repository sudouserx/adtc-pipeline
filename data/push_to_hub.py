#!/usr/bin/env python3
"""Push the four Kuza SFT JSONL files to their Hugging Face dataset repos.

Each local file is normalized to a common chat schema and uploaded as that
repo's ``train`` split, then a MIT dataset card is written over README.md.

The four Hub repos must already exist. This script does not create them and
does not upload HuggingFaceH4/no_robots. Auth uses the cached Hugging Face
login (``huggingface-cli login``); no ``HF_TOKEN`` env var is required.

    python data/push_to_hub.py
    python data/push_to_hub.py --dry-run
    python data/push_to_hub.py --cards-only
    python data/push_to_hub.py --only english
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "final"

FEATURES_COLUMNS = (
    "messages",
    "instruction",
    "response",
    "language",
    "source",
    "source_id",
)

SISTER_REPOS = (
    "kuzaai/kuza_sft_english",
    "kuzaai/kuza_sft_swahili",
    "kuzaai/kuza_sft_adversarial",
    "kuzaai/kuza_sft_multiturn",
)

ISO_LANGUAGE = {"english": "en", "swahili": "sw"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    filename: str
    env_var: str
    default_repo: str
    default_language: str
    pretty_name: str
    tags: tuple[str, ...]
    mix_note: str
    origin: str
    extra_card: str


SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="english",
        filename="kuza_sft_english.jsonl",
        env_var="KUZA_EN_DATASET",
        default_repo="kuzaai/kuza_sft_english",
        default_language="english",
        pretty_name="Kuza SFT English",
        tags=("agriculture", "east-africa", "sft", "datasets", "conversational"),
        mix_note=(
            "Kuza supervised fine-tuning uses **100%** of this English train "
            "split (then mixes the other Kuza datasets against that count)."
        ),
        origin=(
            "This English SFT split is the filtered agricultural Q&A used to "
            "train Kuza.\n\n"
            "It starts from the Digital Green FarmerChat large corpus "
            "([`DigiGreen/farmerchat-queries-large`](https://huggingface.co/datasets/DigiGreen/farmerchat-queries-large)): "
            "1,446,866 farmer-initiated Q&A pairs from India, Kenya, Nigeria, "
            "Ethiopia and other regions (Kenya 858,674; Ethiopia 38,114). The "
            "examples are grounded in real farmer questions rather than being "
            "generated entirely synthetically.\n\n"
            "Preparation kept useful agricultural information per example "
            "rather than the full corpus. High-priority and high-frequency "
            "questions were separated from the long tail, using semantic "
            "representations for sampling diversity. 50K priority examples "
            "and 20K long-tail examples were then cleaned with "
            "Llama-3.3-70B-Versatile (filler removed, denser answers). That "
            "produced an intermediate ~56K corpus "
            "([`kuzaai/agri_sft_prod_56k`](https://huggingface.co/datasets/kuzaai/agri_sft_prod_56k)). "
            "Malformed records, exact and near duplicates, and answers that "
            "were too close semantically were then dropped, leaving ~25.6K "
            "examples "
            "([`kuzaai/agri_sft_prod_dedup_25k`](https://huggingface.co/datasets/kuzaai/agri_sft_prod_dedup_25k)).\n\n"
            "This Hub dataset is that filtered English corpus after additional "
            "local cleanup (near-duplicates and hidden-eval overlap removed). "
            "The 56K intermediate set remains public so dropped rows can be "
            "inspected; it was not used to train the submitted model.\n\n"
            "An alternative path — extracting facts from about 200 extension "
            "guidelines and reverse-generating farmer conversations — was "
            "tried and discarded because the material was hard to gather at "
            "scale and less representative of real farmer questions."
        ),
        extra_card="",
    ),
    DatasetSpec(
        key="swahili",
        filename="kuza_sft_swahili.jsonl",
        env_var="KUZA_SW_DATASET",
        default_repo="kuzaai/kuza_sft_swahili",
        default_language="swahili",
        pretty_name="Kuza SFT Swahili",
        tags=("agriculture", "east-africa", "sft", "datasets", "swahili", "conversational"),
        mix_note=(
            "Kuza supervised fine-tuning samples **35%** of the English train "
            "count from this Swahili split."
        ),
        origin=(
            "This is a **new** Kenyan Swahili SFT set. It is not "
            "[`kuzaai/agri_sft_25k_swahili`](https://huggingface.co/datasets/kuzaai/agri_sft_25k_swahili) "
            "and should not be treated as a relabel of that older upload.\n\n"
            "It was produced by translating the filtered English Kuza SFT "
            "examples with GPT-OSS-120B (Groq), then applying numeral, "
            "glossary, and length QC. Rows that failed QC were dropped, so "
            "this split is smaller than English. Residual translation "
            "artifacts remain. Large teacher models were used only during "
            "dataset preparation."
        ),
        extra_card="",
    ),
    DatasetSpec(
        key="adversarial",
        filename="kuza_sft_adversarial.jsonl",
        env_var="KUZA_ADV_DATASET",
        default_repo="kuzaai/kuza_sft_adversarial",
        default_language="english",
        pretty_name="Kuza SFT Adversarial",
        tags=("agriculture", "east-africa", "sft", "datasets", "safety", "conversational"),
        mix_note=(
            "Kuza supervised fine-tuning samples **5%** of the English train "
            "count from this split."
        ),
        origin=(
            "This is a **new** hand-authored safety set. It is **not** derived "
            "from FarmerChat and is **not** one of the previously published "
            "Kuza agricultural Q&A corpora "
            "([`kuzaai/agri_sft_prod_56k`](https://huggingface.co/datasets/kuzaai/agri_sft_prod_56k), "
            "[`kuzaai/agri_sft_prod_dedup_25k`](https://huggingface.co/datasets/kuzaai/agri_sft_prod_dedup_25k), "
            "[`kuzaai/agri_sft_25k_swahili`](https://huggingface.co/datasets/kuzaai/agri_sft_25k_swahili)).\n\n"
            "Prompts try to elicit unlabeled pesticide doses, jailbreaks, "
            "leftover-chemical recipes, and illegal agrochemical shopping. "
            "Assistant turns refuse, decline invented rates, and redirect to "
            "a locally registered product label or an extension officer. Do "
            "not treat adversarial *prompts* as permitted behavior; only the "
            "refusal responses are the training target."
        ),
        extra_card=(
            "## Safety\n\n"
            "User turns are intentionally harmful-looking. They are included "
            "so a farm assistant learns to **refuse** unsafe agrochemical "
            "requests, not to teach those behaviors. Do not treat prompts as "
            "advice. Responses are refusals, not mixing instructions.\n"
        ),
    ),
    DatasetSpec(
        key="multiturn",
        filename="kuza_sft_multiturn.jsonl",
        env_var="KUZA_MT_DATASET",
        default_repo="kuzaai/kuza_sft_multiturn",
        default_language="english",
        pretty_name="Kuza SFT Multi-turn",
        tags=("agriculture", "east-africa", "sft", "datasets", "swahili", "conversational"),
        mix_note=(
            "Kuza supervised fine-tuning uses **all** rows in this split "
            "(no further downsampling)."
        ),
        origin=(
            "Hand-authored four-turn English and Swahili dialogs. The first "
            "assistant turn asks a clarifying question (crop, location, "
            "symptom); the second gives concrete farm advice and avoids "
            "invented pesticide or veterinary doses."
        ),
        extra_card="",
    ),
)


def content_from_turn(turn: dict[str, Any]) -> str:
    value = turn.get("content", turn.get("value", ""))
    if isinstance(value, list):
        value = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in value
        )
    return str(value).strip()


def clean_messages(turns: Any) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    if not isinstance(turns, list):
        return cleaned
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("from", ""))).lower()
        if role in {"human"}:
            role = "user"
        if role in {"model", "gpt"}:
            role = "assistant"
        content = content_from_turn(turn)
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def last_pair(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    user = ""
    assistant = ""
    for turn in messages:
        if turn["role"] == "user":
            user = turn["content"]
        elif turn["role"] == "assistant" and user:
            assistant = turn["content"]
    if user and assistant:
        return user, assistant
    return None


def extract_raw_messages(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, list):
        return clean_messages(raw)
    if not isinstance(raw, dict):
        return []
    messages = clean_messages(raw.get("messages") or raw.get("conversations"))
    if messages:
        return messages
    instruction = str(raw.get("instruction") or raw.get("user") or "").strip()
    response = str(raw.get("response") or raw.get("assistant") or "").strip()
    if instruction and response:
        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    return []


def normalize_row(
    raw: Any, index: int, spec: DatasetSpec
) -> dict[str, Any] | None:
    messages = extract_raw_messages(raw)
    pair = last_pair(messages)
    if not pair:
        return None
    instruction, response = pair
    record = raw if isinstance(raw, dict) else {}
    language = str(record.get("language") or spec.default_language).strip()
    if language.lower() in {"en", "eng"}:
        language = "english"
    elif language.lower() in {"sw", "swa", "kiswahili"}:
        language = "swahili"
    source = str(record.get("source") or spec.key).strip() or spec.key
    source_id = str(
        record.get("source_id") or record.get("id") or f"{spec.key}-{index}"
    ).strip()
    return {
        "messages": messages,
        "instruction": instruction,
        "response": response,
        "language": language,
        "source": source,
        "source_id": source_id,
    }


def load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}: line {line_number} is not valid JSON: {exc}"
                ) from exc
    return rows


def load_normalized(path: Path, spec: DatasetSpec) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Missing dataset file: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty dataset file: {path}")
    raw_rows = load_jsonl(path)
    if not raw_rows:
        raise SystemExit(f"No JSONL rows in {path}")
    normalized: list[dict[str, Any]] = []
    bad: list[int] = []
    for index, raw in enumerate(raw_rows):
        row = normalize_row(raw, index, spec)
        if row is None:
            bad.append(index)
            continue
        normalized.append(row)
    if bad:
        preview = ", ".join(str(i) for i in bad[:12])
        extra = "" if len(bad) <= 12 else f" (+{len(bad) - 12} more)"
        raise SystemExit(
            f"{path}: {len(bad)} row(s) have no user+assistant pair "
            f"(indices {preview}{extra})."
        )
    return normalized


def dataset_features() -> Any:
    from datasets import Features, Value

    # List-of-dict syntax = sequence of structs. Sequence({...}) would mean a
    # dict of lists and explode `messages` on encode.
    return Features(
        {
            "messages": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ],
            "instruction": Value("string"),
            "response": Value("string"),
            "language": Value("string"),
            "source": Value("string"),
            "source_id": Value("string"),
        }
    )


def build_hf_dataset(rows: list[dict[str, Any]]) -> Any:
    from datasets import Dataset, DatasetDict

    dataset = Dataset.from_list(rows, features=dataset_features())
    return DatasetDict({"train": dataset})


def size_category(count: int) -> str:
    if count < 1_000:
        return "n<1K"
    if count < 10_000:
        return "1K<n<10K"
    if count < 100_000:
        return "10K<n<100K"
    return "100K<n<1M"


def iso_languages(rows: list[dict[str, Any]], spec: DatasetSpec) -> list[str]:
    seen: list[str] = []
    for row in rows:
        code = ISO_LANGUAGE.get(row["language"], row["language"])
        if code not in seen:
            seen.append(code)
    if not seen:
        seen.append(ISO_LANGUAGE.get(spec.default_language, "en"))
    return seen


def yaml_list(values: list[str] | tuple[str, ...], indent: int = 0) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {value}" for value in values)


def language_table(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["language"] for row in rows)
    lines = ["| language | rows |", "|---|---|"]
    for language, count in sorted(counts.items()):
        lines.append(f"| {language} | {count:,} |")
    return "\n".join(lines)


def fields_table() -> str:
    return """| field | type | description |
|---|---|---|
| `messages` | list of `{role, content}` | Full dialog. Roles are `user` and `assistant`. English and Swahili rows are typically 2 turns; multi-turn rows are 4 turns. |
| `instruction` | string | Last user turn (same text SFT uses as the prompt side of the pair). |
| `response` | string | Last assistant turn. |
| `language` | string | `english` or `swahili`. |
| `source` | string | Origin bucket (`english`, `swahili`, `adversarial`, or `multiturn`). |
| `source_id` | string | Stable row id from prep, or `{source}-{index}` when the local file had none. |"""


def sister_links(current_repo: str) -> str:
    lines = []
    for repo in SISTER_REPOS:
        mark = " (this dataset)" if repo == current_repo else ""
        lines.append(f"- [`{repo}`](https://huggingface.co/datasets/{repo}){mark}")
    return "\n".join(lines)


def dataset_card(
    spec: DatasetSpec, repo: str, rows: list[dict[str, Any]]
) -> str:
    languages = iso_languages(rows, spec)
    n = len(rows)
    extra = spec.extra_card
    if extra and not extra.endswith("\n"):
        extra += "\n"
    extra_block = f"\n{extra}" if extra else ""
    return f"""---
license: mit
pretty_name: {spec.pretty_name}
language:
{yaml_list(languages)}
task_categories:
- text-generation
tags:
{yaml_list(spec.tags)}
size_categories:
- {size_category(n)}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# {spec.pretty_name}

Supervised fine-tuning data for **Kuza**, an offline agricultural assistant
for smallholder farmers and agricultural extension workers in East Africa
(English and Swahili). This repository is one of four Kuza SFT datasets.

## Dataset description

{spec.origin}

{spec.mix_note}

- **Rows:** {n:,}
- **Split:** `train` only
- **License:** MIT
{extra_block}
## Languages

{language_table(rows)}

## Dataset structure

All four Kuza SFT repos share this schema:

{fields_table()}

### Splits

| split | rows |
|---|---|
| train | {n:,} |

## How to load

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train")
print(ds[0]["messages"])
```

## Intended use

Fine-tune a chat model as an East African farm assistant: direct, specific
answers in the user's language, with concrete steps rather than generic
advice, and without inventing pesticide or veterinary doses.

This split is meant to be mixed with the sister Kuza datasets below, not used
as a complete training mixture by itself.

## Limitations

- Advice is **not** a substitute for a local product label, veterinarian, or
  extension officer. Rates, products, and legal rules vary by country.
- Domain is English and Swahili agriculture in East Africa. It is not a
  general instruction corpus.

## License

MIT License. You may use, copy, modify, merge, publish, distribute, and
sublicense this dataset.

## Sister datasets

{sister_links(repo)}
"""


def require_repo(repo: str) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        HfApi().repo_info(repo, repo_type="dataset")
    except RepositoryNotFoundError as exc:
        raise SystemExit(
            f"Dataset repo {repo} does not exist. Create it on the Hub first "
            f"(MIT license), then re-run."
        ) from exc


def push_card(
    spec: DatasetSpec,
    repo: str,
    rows: list[dict[str, Any]],
) -> None:
    from huggingface_hub import DatasetCard

    require_repo(repo)
    DatasetCard(dataset_card(spec, repo, rows)).push_to_hub(
        repo, repo_type="dataset"
    )
    print(f"Wrote dataset card for {repo}")


def push_dataset(
    spec: DatasetSpec,
    repo: str,
    rows: list[dict[str, Any]],
    *,
    private: bool,
) -> None:
    require_repo(repo)
    print(f"Pushing {len(rows):,} rows to {repo} (split=train)")
    build_hf_dataset(rows).push_to_hub(repo, private=private)
    push_card(spec, repo, rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Kuza SFT JSONL files as four Hub train splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with kuza_sft_*.jsonl (default: data/final)",
    )
    for spec in SPECS:
        parser.add_argument(
            f"--{spec.key}-repo",
            default=os.environ.get(spec.env_var, spec.default_repo),
            help=f"Hub dataset id (env {spec.env_var})",
        )
    parser.add_argument(
        "--only",
        choices=[spec.key for spec in SPECS],
        help="Upload a single dataset (retry). Still requires that local file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalize and print cards; do not call the Hub.",
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Rewrite Hub README cards from local JSONL; do not re-upload parquet.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Keep Hub repos private (default: public, matching the existing MIT repos).",
    )
    return parser.parse_args(argv)


def selected_specs(only: str | None) -> tuple[DatasetSpec, ...]:
    if only is None:
        return SPECS
    return tuple(spec for spec in SPECS if spec.key == only)


def preview_row(row: dict[str, Any]) -> dict[str, Any]:
    preview = {key: row[key] for key in FEATURES_COLUMNS}
    preview["messages"] = [
        {"role": turn["role"], "content": _clip(turn["content"])}
        for turn in row["messages"]
    ]
    preview["instruction"] = _clip(row["instruction"])
    preview["response"] = _clip(row["response"])
    return preview


def _clip(text: str, limit: int = 160) -> str:
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir
    specs = selected_specs(args.only)

    if not args.only:
        missing = [
            spec.filename
            for spec in SPECS
            if not (data_dir / spec.filename).is_file()
            or (data_dir / spec.filename).stat().st_size == 0
        ]
        if missing:
            raise SystemExit(
                "All four local files are required unless --only is set. "
                "Missing or empty: " + ", ".join(missing)
            )

    loaded: list[tuple[DatasetSpec, str, list[dict[str, Any]]]] = []
    for spec in specs:
        path = data_dir / spec.filename
        repo = getattr(args, f"{spec.key}_repo")
        rows = load_normalized(path, spec)
        langs = ", ".join(
            f"{language}={count}"
            for language, count in sorted(
                Counter(row["language"] for row in rows).items()
            )
        )
        print(f"{spec.key}: {len(rows):,} rows from {path} -> {repo} ({langs})")
        loaded.append((spec, repo, rows))

    if args.dry_run:
        for spec, repo, rows in loaded:
            print(f"\n=== {repo} schema sample ===")
            print(json.dumps(preview_row(rows[0]), ensure_ascii=False, indent=2))
            print(f"\n=== {repo} dataset card ===\n")
            print(dataset_card(spec, repo, rows).rstrip())
        print("\nDry run only. Nothing was uploaded.")
        return 0

    for spec, repo, rows in loaded:
        if args.cards_only:
            print(f"Updating dataset card for {repo} ({len(rows):,} rows)")
            push_card(spec, repo, rows)
        else:
            push_dataset(spec, repo, rows, private=args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
