"""Shared helpers: re-export pipeline utilities and study-specific wrappers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import config

# Import pipeline modules read-only.
if str(config.PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(config.PIPELINE_DIR))

import model as pipeline_model  # noqa: E402
from common import (  # noqa: E402
    generation_tps,
    gguf_inventory,
    help_has,
    installed_packages,
    llama_cpp_binaries,
    parse_bench_tps,
    parse_mean_kld,
    require_file,
    run,
    setup_llama_cpp,
    sha256_file,
    smoke_load,
    tool_help,
    write_json,
    read_json,
    require_llama_cpp_build_tools,
    package_version,
    public_version,
    pytorch_index_url,
)

# Patch pipeline config so llama.cpp tools resolve under shared TOOLS_DIR.
import config as pipeline_config  # noqa: E402

pipeline_config.WORK_DIR = Path(os.environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline"))
pipeline_config.TOOLS_DIR = config.TOOLS_DIR
pipeline_config.MAX_SEQ_LENGTH = config.MAX_SEQ_LENGTH
pipeline_config.SEED = config.SEED
pipeline_config.FLASH_ATTN = config.FLASH_ATTN
pipeline_config.CACHE_TYPE_K = config.CACHE_TYPE_K
pipeline_config.CACHE_TYPE_V = config.CACHE_TYPE_V


def configure_hf_cache() -> None:
    config.HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(config.HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(config.HF_CACHE_DIR / "hub"))


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set in the environment")
    configure_hf_cache()
    return token


def load_tokenizer():
    return pipeline_model.load_tokenizer(config.ARTIFACTS_DIR / "adapter")


def runtime_flags() -> list[str]:
    return [
        "-fa",
        str(config.FLASH_ATTN),
        "-ctk",
        str(config.CACHE_TYPE_K),
        "-ctv",
        str(config.CACHE_TYPE_V),
    ]


def breakdown_hidden(report: dict) -> dict[str, object]:
    rows = report.get("rows") or []
    by_language: dict[str, list[float]] = {}
    by_category: dict[str, list[float]] = {}
    for row in rows:
        score = float(row.get("score") or 0)
        lang = str(row.get("language") or "unknown")
        cat = str(row.get("category") or "unknown")
        by_language.setdefault(lang, []).append(score)
        by_category.setdefault(cat, []).append(score)

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "by_language": {key: mean(vals) for key, vals in sorted(by_language.items())},
        "by_category": {key: mean(vals) for key, vals in sorted(by_category.items())},
    }
