#!/usr/bin/env python3
"""Copy Gate 2 provenance artifacts out of a finished run directory."""

from __future__ import annotations

import shutil
from pathlib import Path

import config
from common import adapter_dir, quants_dir, run_dir, screen_dir, write_json

ADAPTER_FILES = (
    "adapter_config.json",
    "sft_manifest.json",
    "train_metrics.json",
    "trainer_state.json",
    "data_quality.json",
    "kuza_system_prompt.json",
)


def write_provenance(dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    adapter = adapter_dir()
    for name in ADAPTER_FILES:
        src = adapter / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for src in adapter.glob("adapter_model*.safetensors"):
        shutil.copy2(src, dest / src.name)
    screen = screen_dir() / "results.json"
    if screen.is_file():
        shutil.copy2(screen, dest / "screen_results.json")
    recipes: list[str] = []
    if quants_dir().is_dir():
        for recipe in quants_dir().glob("*/recipe.json"):
            shutil.copy2(recipe, dest / f"{recipe.parent.name}_recipe.json")
            recipes.append(recipe.parent.name)
    write_json(
        dest / "provenance_index.json",
        {
            "run_id": config.RUN_ID,
            "run_dir": str(run_dir()),
            "quant_recipes": recipes,
        },
    )
    return recipes


def main() -> int:
    run_dest = run_dir() / "provenance"
    experiments_dest = (
        Path(__file__).resolve().parents[1] / "experiments" / "provenance" / config.RUN_ID
    )
    write_provenance(run_dest)
    write_provenance(experiments_dest)
    print(f"provenance: {run_dest}")
    print(f"provenance: {experiments_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
