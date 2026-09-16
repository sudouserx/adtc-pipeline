#!/usr/bin/env bash
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

for key, repo in model.DATASETS.items():
    print(f"preflight: {key} <- {repo}")

if config.LOCAL_DATA_DIR is not None:
    print(f"preflight: KUZA_LOCAL_DATA={config.LOCAL_DATA_DIR}")
    for key in model.DATASETS:
        path = config.LOCAL_DATA_DIR / f"{key}.jsonl"
        if path.is_file() and path.stat().st_size > 0:
            print(f"preflight: local {key} <- {path}")
        else:
            print(f"preflight: local {key} missing; will use Hub")
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

print("preflight: ok")
PY

python 00_install_deps.py
python 01_setup_llama_cpp.py
python 02_sft.py
python 03_reference.py
python 04_imatrix.py
python 05_quants.py
python 06_screen.py  # hidden-set rank; KLD diagnostic; GPU TPS is not ADTC
python 07_provenance.py
if [ "${KUZA_SKIP_UPLOAD:-}" = "1" ]; then
  echo "upload: skipped (KUZA_SKIP_UPLOAD=1)"
else
  python 08_upload.py
fi
