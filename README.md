# ADTC Kuza pipeline

Fine-tune Gemma 4 E2B or Qwen 3.5-4B as Kuza, then merge, quantize, screen, and upload the run to Hugging Face.

Data is already on Hub (`kuzaai/kuza_sft_*`). To rebuild it, see `data/`.

## Train

Needs a CUDA GPU, `HF_TOKEN` (read datasets, write model repos), and a lean copy of this repo (scripts only; no `data/final/*.jsonl`). Screening (`06_screen.py`) also needs the repo root — `experiments/eval_hidden.py` and `data/hidden_prompts.jsonl` — not just the `gemma-4-e2b/` or `qwen-3.5-4b/` subdirectory alone.

```bash
export HF_TOKEN=hf_...
export KUZA_WORK_DIR=/workspace/kuza-pipeline
# export KUZA_EN_DATASET=kuzaai/kuza_sft_english
# export KUZA_SW_DATASET=kuzaai/kuza_sft_swahili
# export KUZA_ADV_DATASET=kuzaai/kuza_sft_adversarial
# export KUZA_MT_DATASET=kuzaai/kuza_sft_multiturn
# export KUZA_UPLOAD_REPO=kuzaai/kuza-gemma-4-e2b   # gemma-4-e2b default
# export KUZA_UPLOAD_REPO=kuzaai/kuza-qwen-3.5-4b  # qwen-3.5-4b default
cd gemma-4-e2b && bash run.sh   # or: cd qwen-3.5-4b && bash run.sh
```

`run.sh` runs install → llama.cpp → SFT → DPO → merge → QAT export → imatrix → quants → screen → provenance → Hub upload.
Skip upload with `KUZA_SKIP_UPLOAD=1`. Skip DPO with `KUZA_SKIP_DPO=1` or `KUZA_DPO_ENABLED=0`. Skip QAT export with `KUZA_SKIP_QAT_EXPORT=1`.
Override repos with `KUZA_UPLOAD_REPO`. Local JSONL instead of Hub: `KUZA_LOCAL_DATA=/path/to/data/final` (put `preference.jsonl` there for DPO).

Artifacts: `$KUZA_WORK_DIR/kuza-gemma-4-e2b/` or `.../kuza-qwen-3.5-4b/`. Model repos are created on upload if missing.

## Download

```bash
huggingface-cli download kuzaai/kuza-gemma-4-e2b --local-dir ./kuza-gemma-4-e2b
huggingface-cli download kuzaai/kuza-qwen-3.5-4b --local-dir ./kuza-qwen-3.5-4b
```
