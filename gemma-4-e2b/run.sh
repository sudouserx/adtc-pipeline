#!/usr/bin/env bash
# =============================================================================
# run.sh — one-shot orchestration for the Kuza Gemma 4 E2B pipeline
#
# Pipeline order:
#   preflight -> deps -> llama.cpp -> SFT -> DPO -> merge -> QAT export
#   -> imatrix -> quants -> screen -> provenance -> upload
#
# Knobs (all optional; see config.py for the full list):
#   HF_TOKEN             required — Hugging Face write token
#   KUZA_WORK_DIR        run root          (default /workspace/kuza-pipeline)
#   KUZA_LOCAL_DATA      dir of local {source}.jsonl overrides; put
#                        preference.jsonl here to use a local preference set
#   KUZA_UPLOAD_REPO     target HF repo    (default kuzaai/kuza-gemma-4-e2b)
#   KUZA_LR / KUZA_TRAIN_SEQ / KUZA_EPOCHS        SFT overrides
#   KUZA_DPO_DATASET     preference repo   (default kuzaai/kuza_dpo_preference
#                        — uploaded MANUALLY by the owner; never pushed here)
#   KUZA_SKIP_DPO=1      skip preference alignment (adapter/ stays pure SFT)
#   KUZA_DPO_ENABLED=0   same as KUZA_SKIP_DPO (config.DPO["enabled"])
#   KUZA_FORCE_DPO=1     retrain DPO even if adapter-dpo/ exists
#   KUZA_REQUIRE_DPO=1   missing preference dataset -> hard error
#   KUZA_SKIP_UPLOAD=1   stop after provenance, no HF upload
#   KUZA_SKIP_QAT_EXPORT=1  skip QAT lattice export (05_quants skips q4_0_qat_export)
#   KUZA_DRY_RUN=1       08_upload.py lists+hashes only, uploads nothing
#   KUZA_CALIB_TOKENS / KUZA_KLD_CHUNKS / KUZA_EDGE_THREADS   screening knobs
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

import config
import model

if not os.environ.get("HF_TOKEN"):
    sys.exit("HF_TOKEN must be set")
if shutil.which("nvidia-smi") is None:
    sys.exit("nvidia-smi not found; a CUDA GPU is required")
missing = [name for name in ("cmake", "g++", "make", "nvcc") if shutil.which(name) is None]
if missing:
    sys.exit(
        "llama.cpp build tools missing: "
        + ", ".join(missing)
        + ". Install them, then resume from 01_setup_llama_cpp.py: "
        "apt-get update && apt-get install -y cmake g++ make git"
    )

for key in config.SFT_REQUIRED_SOURCES:
    print(f"preflight: required {key} <- {model.DATASETS[key]}")
for key in config.SFT_OPTIONAL_SOURCES:
    print(
        f"preflight: optional {key} <- {model.DATASETS[key]} "
        "(skipped unless local JSONL or Hub repo exists)"
    )

if config.LOCAL_DATA_DIR is not None:
    print(f"preflight: KUZA_LOCAL_DATA={config.LOCAL_DATA_DIR}")
    for key in (*config.SFT_REQUIRED_SOURCES, *config.SFT_OPTIONAL_SOURCES):
        path = config.LOCAL_DATA_DIR / f"{key}.jsonl"
        if path.is_file() and path.stat().st_size > 0:
            print(f"preflight: local {key} <- {path}")
        elif key in config.SFT_REQUIRED_SOURCES:
            print(f"preflight: local {key} missing; will use Hub")
        else:
            print(f"preflight: local {key} missing; optional source will be skipped")
    pref = config.LOCAL_DATA_DIR / "preference.jsonl"
    if pref.is_file() and pref.stat().st_size > 0:
        print(f"preflight: local preference pairs <- {pref}")
else:
    leftover = Path(config.__file__).resolve().parents[1] / "data" / "cleaned"
    found = sorted(
        path.name
        for path in leftover.glob("*.jsonl")
        if path.is_file() and path.stat().st_size > 0
    )
    if found:
        print(
            f"preflight: warning: {leftover} has {', '.join(found)}; "
            "ignored unless KUZA_LOCAL_DATA is set"
        )

dpo_disabled = (
    os.environ.get("KUZA_SKIP_DPO") == "1"
    or not config.DPO.get("enabled", True)
)
if dpo_disabled:
    reason = "KUZA_SKIP_DPO=1" if os.environ.get("KUZA_SKIP_DPO") == "1" else "KUZA_DPO_ENABLED=0"
    print(f"preflight: DPO disabled ({reason})")
else:
    pref_local = None
    if config.LOCAL_DATA_DIR is not None:
        candidate = config.LOCAL_DATA_DIR / "preference.jsonl"
        if candidate.is_file() and candidate.stat().st_size > 0:
            pref_local = candidate
    if pref_local is not None:
        print(f"preflight: preference pairs <- {pref_local}")
    else:
        print(
            "preflight: preference pairs <- "
            f"{config.DPO_DATASET} (Hub, uploaded manually by the repo owner)"
        )
        print(
            "preflight: WARNING: no local preference.jsonl; 02b_dpo will load "
            "from Hub after deps install or skip if unavailable"
        )
        if os.environ.get("KUZA_REQUIRE_DPO") == "1":
            sys.exit(
                "KUZA_REQUIRE_DPO=1 but preference.jsonl is missing from "
                "KUZA_LOCAL_DATA; set it before starting or unset KUZA_REQUIRE_DPO"
            )

print("preflight: ok")
PY

python 00_install_deps.py
python 01_setup_llama_cpp.py

python 02_sft.py

python 02b_dpo.py

python 03b_qat_export.py

python 04_imatrix.py
python 05_quants.py
python 06_screen.py
python 07_provenance.py

python 08_upload.py
