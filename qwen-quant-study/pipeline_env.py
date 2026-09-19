"""Load qwen-3.5-4b pipeline modules without clashing with study_config."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_pipeline_config(study_config: ModuleType) -> ModuleType:
    path = Path(study_config.PIPELINE_DIR) / "config.py"
    spec = importlib.util.spec_from_file_location("kuza_qwen_pipeline_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_pipeline(study_config: ModuleType) -> ModuleType:
    pipeline_dir = Path(study_config.PIPELINE_DIR)
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))

    pipeline_config = load_pipeline_config(study_config)
    pipeline_config.WORK_DIR = Path(
        os.environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline")
    )
    pipeline_config.TOOLS_DIR = Path(study_config.TOOLS_DIR)
    for name in (
        "MAX_SEQ_LENGTH",
        "SEED",
        "FLASH_ATTN",
        "CACHE_TYPE_K",
        "CACHE_TYPE_V",
        "LLAMA_CPP_COMMIT",
    ):
        if hasattr(study_config, name):
            setattr(pipeline_config, name, getattr(study_config, name))

    # common.py and model.py do `import config`; give them the pipeline module.
    sys.modules["config"] = pipeline_config
    return pipeline_config
