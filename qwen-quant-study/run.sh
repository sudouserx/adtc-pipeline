#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python - <<'PY'
import os
import shutil
import sys

if not os.environ.get("HF_TOKEN"):
    sys.exit("HF_TOKEN must be set")
if shutil.which("nvidia-smi") is None:
    sys.exit("nvidia-smi not found; a CUDA GPU is required")
missing = [name for name in ("cmake", "g++", "make", "nvcc") if shutil.which(name) is None]
if missing:
    sys.exit(
        "llama.cpp build tools missing: "
        + ", ".join(missing)
        + ". Install them, then resume from 02_setup_llama_cpp.py"
    )
print("preflight: ok")
PY

python 00_install_deps.py
python 02_setup_llama_cpp.py
python 01_download.py

if [ "${STUDY_SKIP_BASELINE:-0}" != "1" ]; then
  python 03_baseline_eval.py
fi

if [ "${STUDY_SKIP_QUANT:-0}" != "1" ]; then
  python 04_quantize.py
  python 05_screen.py
  python 06_report.py
else
  echo "quant/screen/report: skipped (STUDY_SKIP_QUANT=1)"
fi

if [ "${STUDY_SKIP_UPLOAD:-0}" = "1" ]; then
  echo "upload: skipped (STUDY_SKIP_UPLOAD=1)"
else
  python 07_upload.py
fi

echo "done: see ${STUDY_WORK_DIR:-$KUZA_WORK_DIR/qwen-quant-study}/results/report/"
