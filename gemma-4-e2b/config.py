"""config.py — the single authoritative config for kuza-pipeline.

Consolidates all three prior versions (2026-09-20):
  - original repo config.py (uploaded same day)      -> base, every name kept
  - kuza-pipeline-review/code/config.py              -> training/screening fixes
  - kuza-dpo-upgrade/config.py                       -> DPO add-on block

Replaces BOTH earlier configs. It is drop-in for:
  - the unmodified repo scripts (02_sft ... 08_upload): every original name
    is still defined, values only change where a fix was justified;
  - the reviewed stage scripts (02_sft / 02b_dpo / 03b_qat_export / 04_imatrix /
    05_quants / 06_screen): all split/new knobs are defined;
  - 02b_dpo.py: all flat DPO_* knobs are defined — including
    DPO_VALIDATION_SPLIT, which 02b reads directly and which was MISSING
    from the earlier dpo-upgrade config (latent AttributeError when loading
    from the Hub — fixed here).

Tags: [ORIG] unchanged from the repo config. [FIX] value changed, reason
given. [NEW] knob that did not exist before.

DPO dataset: you upload it manually to kuzaai/kuza_dpo_preference. Nothing
here or in the pipeline pushes it to the Hub — stages only consume it.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Machine / provider [ORIG]
# ---------------------------------------------------------------------------
WORK_DIR = Path(os.environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline"))
TOOLS_DIR = WORK_DIR / "tools"
HF_CACHE_DIR = WORK_DIR / ".hf-cache"
_LOCAL_DATA = os.environ.get("KUZA_LOCAL_DATA", "").strip()
LOCAL_DATA_DIR = Path(_LOCAL_DATA) if _LOCAL_DATA else None

RUN_ID = "kuza-gemma-4-e2b"
UPLOAD_REPO = os.environ.get("KUZA_UPLOAD_REPO", "kuzaai/kuza-gemma-4-e2b")

SEED = 42

# ---------------------------------------------------------------------------
# Sequence-length knobs [NEW — split of the old single MAX_SEQ_LENGTH]
# ---------------------------------------------------------------------------
# Original scripts use MAX_SEQ_LENGTH for everything (train, imatrix ctx,
# perplexity ctx). 2048 gives multiturn/grounded data headroom; eval stages
# running at 2048 instead of 1024 stay self-consistent because reference and
# candidate always share the same context.
# Reviewed scripts use the split knobs below and ignore this one.
MAX_SEQ_LENGTH = 2048                          # [FIX] was 1024
TRAIN_MAX_SEQ_LENGTH = int(os.environ.get("KUZA_TRAIN_SEQ", "2048"))   # [NEW]
CALIBRATION_CTX = 1024                         # [NEW] llama-imatrix chunk ctx
EVAL_CTX = 1024                                # [NEW] perplexity / KLD ctx
SMOKE_CTX = 1024                               # [NEW] smoke-generation ctx

# ---------------------------------------------------------------------------
# SFT hyperparameters
# ---------------------------------------------------------------------------
EPOCHS = float(os.environ.get("KUZA_EPOCHS", "2.0"))                   # [ORIG]
TRAIN_BATCH_SIZE = 8                                           # [ORIG]
GRAD_ACCUMULATION = 1                                          # [ORIG]
# [FIX] 2e-5 -> 1e-4. 2e-5 is a full-fine-tune LR; the archived LoRA run's
# eval loss was still falling at its final step. Reproduce the old behaviour
# with KUZA_LR=2e-5.
LEARNING_RATE = float(os.environ.get("KUZA_LR", "1e-4"))
LR_SCHEDULER = "cosine_with_warmup"                            # [NEW]
WARMUP_RATIO = 0.03                                            # [NEW]
WEIGHT_DECAY = 0.01                                            # [NEW]
# [NEW] eval cadence + early stop (original: 4 eval points per whole run —
# too coarse for best-checkpoint selection to mean anything).
EVALS_PER_EPOCH = 4
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_THRESHOLD = 1e-4
SAVE_TOTAL_LIMIT = 6

MAX_INVALID_FRACTION = 0.01                                    # [ORIG]
MAX_TRUNCATED_FRACTION = 0.10                                  # [ORIG]

# ---------------------------------------------------------------------------
# Training mix & dataset registry [NEW]
# ---------------------------------------------------------------------------
# Single source of truth for source names. model.py re-exports these as
# model.DATASETS and model.MIX at import time so every stage reads the same
# registry. Fractions are shares OF THE ENGLISH TRAIN POOL.
DATASETS = {
    "english": os.environ.get("KUZA_EN_DATASET", "kuzaai/kuza_sft_english"),
    "swahili": os.environ.get("KUZA_SW_DATASET", "kuzaai/kuza_sft_swahili"),
    # Optional — Hub repos are placeholders until uploaded; use KUZA_LOCAL_DATA
    # or override KUZA_*_DATASET when a repo becomes available.
    "swahili_native": os.environ.get("KUZA_SW_NATIVE_DATASET", "kuzaai/kuza_sft_swahili_native"),
    "code_switch": os.environ.get("KUZA_CS_DATASET", "kuzaai/kuza_sft_code_switch"),
    "grounding": os.environ.get("KUZA_GROUND_DATASET", "kuzaai/kuza_sft_grounding"),
    "adversarial": os.environ.get("KUZA_ADV_DATASET", "kuzaai/kuza_sft_adversarial"),
    "multiturn": os.environ.get("KUZA_MT_DATASET", "kuzaai/kuza_sft_multiturn"),
    "general": os.environ.get("KUZA_GENERAL_DATASET", "HuggingFaceH4/no_robots"),
    # Manually uploaded by you — consumed only, never pushed, by the pipeline.
    "preference": os.environ.get("KUZA_DPO_DATASET", "kuzaai/kuza_dpo_preference"),
}

MIX = {
    "swahili_of_english": 0.50,          # [FIX] was 0.35 — first-class language share
    "general_of_english": 0.08,
    "adversarial_of_english": 0.05,      # [FIX] archived run had ~117 rows (0.34%)
    "multiturn_of_english": 0.03,
    "swahili_native_of_english": 0.0,    # future; enable when Hub repo exists
    "code_switch_of_english": 0.10,      # [NEW] Sw-En mixed queries (deployment-real)
    "grounding_of_english": 0.06,        # [NEW] passage-grounded QA (anti-hallucination)
    "agri_eval_fraction": 0.05,
    "adversarial_eval_fraction": 0.10,
    "general_max_instruction_words": 200,
    "general_max_response_words": 250,
    "eval_group_max": 500,
}

# SFT dataset roles — only SFT_REQUIRED_SOURCES are resolved from Hub by default.
SFT_REQUIRED_SOURCES = (
    "english",
    "swahili",
    "general",
    "adversarial",
    "multiturn",
)
SFT_OPTIONAL_SOURCES = (
    "swahili_native",
    "code_switch",
    "grounding",
)

# Sources that must have a train/eval split BEFORE mixing (leakage fix). [NEW]
SPLIT_SOURCES = ("english", "swahili", "swahili_native", "code_switch",
                 "grounding", "adversarial", "multiturn", "general")

# ---------------------------------------------------------------------------
# Calibration / eval corpora (imatrix stage)
# ---------------------------------------------------------------------------
CALIBRATION_PER_LANGUAGE = 800                                 # [FIX] was 400
CALIBRATION_GENERIC = 200                                      # [ORIG]
EVAL_PER_LANGUAGE = 200                                        # [FIX] was 100
# [NEW] TOKEN-based floor. The archived corpus collapsed to ~10K tokens via a
# broken row-count estimate; this floor makes that impossible.
CALIBRATION_MIN_TOKENS = int(os.environ.get("KUZA_CALIB_TOKENS", "512000"))
# [NEW] max fraction of calibration chunks allowed to contain the system
# prompt (archived corpus repeated a ~150-token prompt in 100% of rows,
# biasing the importance matrix toward one fixed string).
CALIBRATION_SYSTEM_PROMPT_MAX_FRACTION = 0.25

# ---------------------------------------------------------------------------
# Runtime flags for llama-cli / llama-bench [ORIG] — not baked into the GGUF
# ---------------------------------------------------------------------------
FLASH_ATTN = "on"
CACHE_TYPE_K = "q8_0"
CACHE_TYPE_V = "q8_0"

# Threads for imatrix/perplexity stages on the training box. [NEW]
TOOL_THREADS = min(8, os.cpu_count() or 4)

# ---------------------------------------------------------------------------
# CPU-representative edge bench + screening gates [NEW] (06_screen.py)
# Deployment target is a CPU-only 8-16GB laptop; GPU t/s are diagnostics only.
# ---------------------------------------------------------------------------
EDGE_BENCH = {
    "ngl": 0,                       # CPU-only arm
    "threads": int(os.environ.get("KUZA_EDGE_THREADS", "0")),  # 0 = auto (physical)
    "prompt_tokens": 512,
    "gen_tokens": 128,
    "repeats": 3,
}

# A candidate failing any gate cannot win regardless of hidden-score ranking.
SCREEN_GATES = {
    "min_same_top_p": 92.0,         # percent, vs BF16 logits
    "max_mean_kld": 0.060,          # absolute ceiling; Q8_0 defines the floor
    "max_safety_failures": 0,
    "max_language_failures": 0,
}
# [FIX] archived run screened over ~16K tokens — cannot resolve the 0.005-0.02
# KLD gaps that separate 4-bit candidates. 128 chunks ~ 128K tokens.
SCREEN_KLD_CHUNKS = int(os.environ.get("KUZA_KLD_CHUNKS", "128"))

# ---------------------------------------------------------------------------
# DPO preference alignment (02b_dpo.py — flat vars are the source of truth;
# the DPO dict below is rebuilt from them for 02b_dpo.py compatibility)
# ---------------------------------------------------------------------------
# Dataset is uploaded manually to the Hub as kuzaai/kuza_dpo_preference
# (train/validation). $KUZA_LOCAL_DATA/preference.jsonl wins when present.
# KUZA_SKIP_DPO=1 skips 02b; KUZA_REQUIRE_DPO=1 turns a missing dataset into
# a hard error instead of a graceful skip.
DPO_DATASET = os.environ.get(
    "KUZA_DPO_DATASET",
    os.environ.get("KUZA_PREF_DATASET", "kuzaai/kuza_dpo_preference"),
)
DPO_VALIDATION_SPLIT = "validation"     # [FIX] 02b reads this directly — was missing
DPO_BETA = 0.1
DPO_RPO_ALPHA = 1.0                     # RPO (pid-wise) term, adds stability at small beta
DPO_LR = 5e-6
DPO_EPOCHS = 1.0
DPO_TRAIN_BATCH_SIZE = 4
DPO_GRAD_ACCUMULATION = 2               # effective batch 8
DPO_MAX_PROMPT_LENGTH = 640
DPO_MAX_COMPLETION_LENGTH = 384
DPO_MAX_LENGTH = MAX_SEQ_LENGTH
DPO_WARMUP_RATIO = 0.10

# 02b_dpo.py compatibility view (reads config.DPO[...]). Values are the same
# source of truth as the flat vars above — do not edit independently.
DPO = {
    "enabled": os.environ.get("KUZA_DPO_ENABLED", "1") == "1",
    "dataset": DPO_DATASET,
    "beta": DPO_BETA,
    "learning_rate": DPO_LR,
    "epochs": DPO_EPOCHS,
    "batch_size": DPO_TRAIN_BATCH_SIZE,
    "grad_accumulation": DPO_GRAD_ACCUMULATION,
    "max_prompt_length": DPO_MAX_PROMPT_LENGTH,
    "max_completion_length": DPO_MAX_COMPLETION_LENGTH,
    "loss": "sigmoid",                  # DPO; set "kto_pair" for scalar-labelled pairs
    "r": 16,                            # smaller adapter for the refinement stage
    "lora_alpha": 32,
}

# ---------------------------------------------------------------------------
# Smoke generation [FIX] 24 tokens was too short to reveal degeneration
# ---------------------------------------------------------------------------
SMOKE_N_TOKENS = 128
SMOKE_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Build / packaging [ORIG]
# ---------------------------------------------------------------------------
CUDA_ARCH: str | None = None

LLAMA_CPP_COMMIT = "aac810230f9ef0cf73a47c56e46e87d0988be348"

PINNED_PACKAGES = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "xformers": "0.0.35",
    "bitsandbytes": "0.50.1",
    "torchao": "0.16.0",
    "fbgemm-gpu-genai": "1.5.0",
    "unsloth": "2026.8.19",
    "unsloth-zoo": "2026.8.13",
    "transformers": "5.5.0",
    "peft": "0.19.1",
    "trl": "0.24.0",
    "datasets": "4.3.0",
    "accelerate": "1.13.0",
    "huggingface-hub": "1.11.0",
    "safetensors": "0.7.0",
    "sentencepiece": "0.2.1",
    "tokenizers": "0.22.2",
    "protobuf": "5.29.5",
    "jinja2": "3.1.6",
}
TORCH_CUDA_PACKAGES = ("torch", "torchvision", "xformers")
