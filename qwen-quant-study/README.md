# Qwen 3.5-4B Quantization vs Finetuning Diagnosis

Self-contained study to determine whether poor model quality comes from **finetuning** or **quantization**. Downloads reusable artifacts from [`kuzaai/kuza-qwen-3.5-4b`](https://huggingface.co/kuzaai/kuza-qwen-3.5-4b), evaluates the BF16 reference, re-screens past quants, and quantizes an expanded candidate matrix (including Q3_K_M and IQ3_XS).

Does **not** modify [`qwen-3.5-4b/`](../qwen-3.5-4b/) pipeline code.

## Requirements

- CUDA GPU (`nvidia-smi`)
- `HF_TOKEN` with read access to `kuzaai/kuza-qwen-3.5-4b` and write access for upload
- Build tools: cmake, g++, make, nvcc, git
- ~25–30 GB disk under `$STUDY_WORK_DIR`

## Quick start

```bash
export HF_TOKEN=hf_...
export KUZA_WORK_DIR=/workspace/kuza-pipeline
cd qwen-quant-study && bash run.sh
```

Optional:

```bash
export STUDY_WORK_DIR=/path/to/study-work   # default: $KUZA_WORK_DIR/qwen-quant-study
export STUDY_SKIP_QUANT=1                     # baseline eval only
export STUDY_SKIP_BASELINE=0                  # default; set 1 to skip BF16 gate
export STUDY_SKIP_UPLOAD=1                    # skip Hub upload (upload runs by default)
export STUDY_UPLOAD_REPO=kuzaai/kuza-qwen-3.5-4b-quant-study
```

## Stages

| Script | Purpose |
|--------|---------|
| `00_install_deps.py` | Install pinned Python packages |
| `02_setup_llama_cpp.py` | Build llama.cpp (shared `$KUZA_WORK_DIR/tools`) |
| `01_download.py` | Selective HF download + sha256 verify |
| `03_baseline_eval.py` | BF16 + past quants on hidden set |
| `04_quantize.py` | New quant candidates from reference GGUF |
| `05_screen.py` | KLD + hidden + bench for all models |
| `06_report.py` | `analysis.md` + `summary.json` verdict |
| `07_upload.py` | Upload all artifacts and results to Hugging Face |

## Results

Written under `$STUDY_WORK_DIR/results/`:

- `baseline/` — BF16 gate and past-quant comparison
- `screen/` — per-model KLD, bench, hidden JSON
- `report/analysis.md` — human-readable verdict
- `report/summary.json` — machine-readable leaderboard

## Diagnosis rule

1. Evaluate `reference/kuza-bf16.gguf` on the 36-prompt hidden set.
2. If BF16 `hidden_mean` ≥ 0.75 but quants score much lower → **quantization issue**.
3. If BF16 is also low → **finetuning or template issue**.

## Quant candidates

**From HF (past run):** `q4_k_m_imatrix`, `q4_k_xl_ssm`, `q4_k_s_ssm`

**New local quants:** `q4_k_m_plain`, `q5_k_m_imatrix_ssm`, `q3_k_m_imatrix_ssm`, `iq3_xs_imatrix_ssm`, `iq4_xs_imatrix_ssm`, `q6_k_imatrix`

## Upload

After the pipeline completes, `07_upload.py` uploads everything under
`$STUDY_WORK_DIR` to [`kuzaai/kuza-qwen-3.5-4b-quant-study`](https://huggingface.co/kuzaai/kuza-qwen-3.5-4b-quant-study)
(creates the repo if missing). Override with `STUDY_UPLOAD_REPO`. Skip with
`STUDY_SKIP_UPLOAD=1`.

```bash
python 07_upload.py                  # upload after a partial or full run
python 07_upload.py --dry-run        # list files and manifest only
```
