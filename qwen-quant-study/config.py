"""Paths and knobs for the Qwen quant diagnosis study."""

from __future__ import annotations

import os
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STUDY_ROOT.parent
PIPELINE_DIR = REPO_ROOT / "qwen-3.5-4b"

_kuza_work = os.environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline")
WORK_DIR = Path(os.environ.get("STUDY_WORK_DIR", f"{_kuza_work}/qwen-quant-study"))
TOOLS_DIR = Path(os.environ.get("KUZA_WORK_DIR", _kuza_work)) / "tools"
HF_CACHE_DIR = WORK_DIR / ".hf-cache"

HF_REPO = os.environ.get("STUDY_HF_REPO", "kuzaai/kuza-qwen-3.5-4b")
UPLOAD_REPO = os.environ.get(
    "STUDY_UPLOAD_REPO", "kuzaai/kuza-qwen-3.5-4b-quant-study"
)
SOURCE_RUN_ID = "kuza-qwen-3.5-4b"

ARTIFACTS_DIR = WORK_DIR / "artifacts"
RESULTS_DIR = WORK_DIR / "results"
QUANTS_DIR = WORK_DIR / "quants"
SCREEN_DIR = WORK_DIR / "screen"
REPORT_DIR = RESULTS_DIR / "report"

HIDDEN_PROMPTS = REPO_ROOT / "data" / "hidden_prompts.jsonl"
EVAL_HIDDEN = REPO_ROOT / "experiments" / "eval_hidden.py"

SEED = 42
MAX_SEQ_LENGTH = 1024
LLAMA_CPP_COMMIT = "aac810230f9ef0cf73a47c56e46e87d0988be348"

FLASH_ATTN = "on"
CACHE_TYPE_K = "q8_0"
CACHE_TYPE_V = "q8_0"

# Hidden mean above this suggests finetuning is acceptable; gap vs quants => quant issue.
HIDDEN_GOOD_THRESHOLD = 0.75

DOWNLOAD_INCLUDE = [
    "reference/kuza-bf16.gguf",
    "reference/reference_manifest.json",
    "reference/kuza_system_prompt.json",
    "imatrix/kuza.imatrix",
    "imatrix/eval.txt",
    "imatrix/calibration.txt",
    "imatrix/corpus_manifest.json",
    "adapter/tokenizer.json",
    "adapter/tokenizer_config.json",
    "adapter/chat_template.jinja",
    "adapter/heldout.jsonl",
    "adapter/kuza_system_prompt.json",
    "quants/q4_k_m_imatrix/kuza-qwen-q4_k_m.gguf",
    "quants/q4_k_m_imatrix/recipe.json",
    "quants/q4_k_xl_ssm/kuza-qwen-q4_k_xl.gguf",
    "quants/q4_k_xl_ssm/recipe.json",
    "quants/q4_k_s_ssm/kuza-qwen-q4_k_s.gguf",
    "quants/q4_k_s_ssm/recipe.json",
]
