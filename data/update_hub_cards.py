#!/usr/bin/env python3
"""Patch Hub dataset-card YAML only. Does not re-upload parquet.

Removes invalid task_categories value ``conversational`` (Hub warning).
Keeps ``text-generation`` and moves conversational onto tags.

    python data/update_hub_cards.py
"""

from __future__ import annotations

import sys

REPOS = (
    "kuzaai/kuza_sft_english",
    "kuzaai/kuza_sft_swahili",
    "kuzaai/kuza_sft_adversarial",
    "kuzaai/kuza_sft_multiturn",
)


def patch(card: object) -> bool:
    data = card.data
    cats = data.task_categories or []
    if isinstance(cats, str):
        cats = [cats]
    cats = [c for c in cats if c != "conversational"]
    if "text-generation" not in cats:
        cats.insert(0, "text-generation")
    tags = data.tags or []
    if isinstance(tags, str):
        tags = [tags]
    tags = list(tags)
    if "conversational" not in tags:
        tags.append("conversational")
    changed = data.task_categories != cats or data.tags != tags
    data.task_categories = cats
    data.tags = tags
    return changed


def main() -> int:
    from huggingface_hub import DatasetCard

    for repo in REPOS:
        card = DatasetCard.load(repo)
        if not patch(card):
            print(f"{repo}: already clean")
            continue
        card.push_to_hub(repo, repo_type="dataset")
        print(
            f"{repo}: task_categories={card.data.task_categories} "
            f"tags={card.data.tags}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
