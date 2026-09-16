"""Provider paths and run knobs for kuza-pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Machine / provider. On RunPod: KUZA_WORK_DIR=/workspace/kuza-pipeline
WORK_DIR = Path(os.environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline"))
TOOLS_DIR = WORK_DIR / "tools"
HF_CACHE_DIR = WORK_DIR / ".hf-cache"
_LOCAL_DATA = os.environ.get("KUZA_LOCAL_DATA", "").strip()
LOCAL_DATA_DIR = Path(_LOCAL_DATA) if _LOCAL_DATA else None

RUN_ID = "kuza-gemma-4-e2b"

SEED = 42
MAX_SEQ_LENGTH = 1024
EPOCHS = 2.0
TRAIN_BATCH_SIZE = 8
GRAD_ACCUMULATION = 1
LEARNING_RATE = 2e-5

MAX_INVALID_FRACTION = 0.01
MAX_TRUNCATED_FRACTION = 0.10
CALIBRATION_PER_LANGUAGE = 400
CALIBRATION_GENERIC = 200
EVAL_PER_LANGUAGE = 100

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
}
TORCH_CUDA_PACKAGES = ("torch", "torchvision", "xformers")
