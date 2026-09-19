"""Artifact path helpers for the quant study."""

from __future__ import annotations

from pathlib import Path

import study_config as config


def artifacts() -> Path:
    return config.ARTIFACTS_DIR


def reference_path() -> Path:
    return artifacts() / "reference" / "kuza-bf16.gguf"


def imatrix_path() -> Path:
    return artifacts() / "imatrix" / "kuza.imatrix"


def eval_path() -> Path:
    return artifacts() / "imatrix" / "eval.txt"


def adapter_dir() -> Path:
    return artifacts() / "adapter"


def past_quant_path(name: str, filename: str) -> Path:
    return artifacts() / "quants" / name / filename


def new_quant_path(name: str, filename: str) -> Path:
    return config.QUANTS_DIR / name / filename


def candidate_dir(name: str) -> Path:
    return config.QUANTS_DIR / name
