# ADTC Kuza pipeline

Fine-tune Gemma 4 E2B or Qwen 3.5-4B as Kuza (East Africa agricultural assistant), then merge, calibrate, quantize, and screen GGUFs. Data is prepared on a laptop and pushed to Hugging Face as **four separate datasets**. Training on a cloud GPU (RunPod or similar) fetches those datasets and mixes them by ratio. The GPU also slices `HuggingFaceH4/no_robots` at train time.

## 1. Local data prep

Do this once on a machine with `GROQ_API_KEY` (for Swahili) and a Hugging Face login (to pull the raw English source). You do **not** need a GPU. Push uses the cached Hub login (`huggingface-cli login`); it does not require `HF_TOKEN`.

```bash
export HF_TOKEN=hf_...

# 1. Filter the existing English Hub dataset locally
#    raw source: KUZA_EN_SOURCE (default kuzaai/agri_sft_prod_dedup_25k)
python data/filter_english.py
# writes data/final/english.jsonl and heldout_overlap.jsonl
# optional local file: python data/filter_english.py --source path/to/raw.jsonl

# 2–3. Translate the filtered English set, then QC
export GROQ_API_KEY=gsk_...
python data/kuza-en-to-sw-translate.py
# writes data/final/swahili_raw.jsonl (resumable; smoke with --limit 20)
python data/prepare_swahili.py
# writes data/final/swahili.jsonl

# 4–5. Hand-authored multi-turn and adversarial examples
#     (do not read the cleaned corpora; can run during translation)
python data/generate_multiturn.py
python data/generate_adversarial.py

python data/prepare_cleaned_mix.py
```

`data/run_data.sh` runs English filter, the two generators, and a manifest with `--allow-empty-swahili`. Translate, `prepare_swahili.py`, and push stay commented until you have a Groq key and four Hub repos.

Create four empty Hub dataset repos once, then upload each local file as that repo's `train` split:

```bash
# create:
  # https://huggingface.co/datasets/kuzaai/kuza_sft_english
  # https://huggingface.co/datasets/kuzaai/kuza_sft_swahili
  # https://huggingface.co/datasets/kuzaai/kuza_sft_adversarial
  # https://huggingface.co/datasets/kuzaai/kuza_sft_multiturn
python data/push_to_hub.py \
  --english-repo kuzaai/kuza_sft_english \
  --swahili-repo kuzaai/kuza_sft_swahili \
  --adversarial-repo kuzaai/kuza_sft_adversarial \
  --multiturn-repo kuzaai/kuza_sft_multiturn
```

Override any default with the matching env var (`KUZA_EN_DATASET`, `KUZA_SW_DATASET`, `KUZA_ADV_DATASET`, `KUZA_MT_DATASET`). Raw English for the filter is `KUZA_EN_SOURCE`, not `KUZA_EN_DATASET`. Do not reload `kuzaai/agri_sft_25k_swahili`.

Training mix (hardcoded in both `model.py` files): 100% English train + 35% Swahili + 8% `HuggingFaceH4/no_robots` + 5% adversarial + all multiturn. SFT **fails** if a requested bucket is empty. `no_robots` is fetched on the GPU; it is not uploaded with the four Kuza datasets.

## 2. What `experiments/` is

Not part of training. After the GPU pipeline writes quantized GGUFs:

- `experiments/eval_hidden.py` — rubric scores on `data/hidden_prompts.jsonl`
- `experiments/select_winner.py` — ranks hidden scores and attaches GPU screen KLD/TPS

`07_provenance.py` copies adapter metrics and screen JSON into `experiments/provenance/`.

## 3. Run on RunPod

Upload a **lean** copy of this repo (scripts only). Do not upload `data/final/*.jsonl`, translator checkpoints, or old Hub Swahili.

Needs: CUDA GPU, `cmake`/`git` (llama.cpp build), and the pinned wheels in `*/config.py` on that CUDA index. Set `CUDA_ARCH` if autodetection would pick the wrong SM (default fallback is `80`). One GPU is used (`device_map` device 0).

```bash
export HF_TOKEN=hf_...
export KUZA_WORK_DIR=/workspace/kuza-pipeline
# defaults if unset:
# export KUZA_EN_DATASET=kuzaai/kuza_sft_english
# export KUZA_SW_DATASET=kuzaai/kuza_sft_swahili
# export KUZA_ADV_DATASET=kuzaai/kuza_sft_adversarial
# export KUZA_MT_DATASET=kuzaai/kuza_sft_multiturn
# export KUZA_GENERAL_DATASET=HuggingFaceH4/no_robots

cd gemma-4-e2b && bash run.sh
# or
cd qwen-3.5-4b && bash run.sh
```

`run.sh` checks `HF_TOKEN` and `nvidia-smi`, then runs `00`–`07`: install deps → build llama.cpp → SFT → BF16 merge → imatrix → 3 quants → GPU screen → provenance.

The SFT step mixes the four Hub datasets plus an 8% `no_robots` slice. Imatrix calibration uses the held-out sample written during SFT (English eval + Swahili eval + 250 general rows).

Training is Hub-only unless you set `KUZA_LOCAL_DATA`. A leftover `data/final/` tree on the pod is ignored. Local JSONL override (explicit env required):

```bash
export KUZA_LOCAL_DATA=/workspace/adtc-pipeline/data/final
```

Artifacts land under `$KUZA_WORK_DIR/kuza-gemma-4-e2b/` or `.../kuza-qwen-3.5-4b/` (`adapter/`, `reference/`, `imatrix/`, `quants/`, `screen/`).

Gemma uses QAT int4 + RsLoRA r=32 + LR `2e-5`. Qwen uses BF16 LoRA r=16 + LR `2e-4` (no QAT). Sequence length is 1024 with packing off.

## 4. After training

```bash
export KUZA_WORK_DIR=/workspace/kuza-pipeline
# llama-cli is built by 01_setup_llama_cpp.py
export KUZA_LLAMA_CLI=$KUZA_WORK_DIR/tools/llama.cpp/build/bin/llama-cli

python experiments/eval_hidden.py \
  --gguf $KUZA_WORK_DIR/kuza-gemma-4-e2b/quants/q4_k_m_control/kuza-q4_k_m-control.gguf \
  --name gemma-q4-k-m

python experiments/eval_hidden.py \
  --gguf $KUZA_WORK_DIR/kuza-qwen-3.5-4b/quants/q4_k_m_imatrix/kuza-qwen-q4_k_m.gguf \
  --name qwen-q4-k-m

python experiments/select_winner.py
# writes experiments/results/winner.json
```

## 5. Do not upload to the pod

- `data/incoming/` (removed), `data/quarantine/` (removed)
- `data/final/*.jsonl` and `mix_*` copies — train from Hugging Face
- Translator `*.progress.jsonl` / `*.failed.jsonl` / `*.batch_state.json`
- Laptop kill-gate / Kaggle wrappers (removed)
- `.cursor/` plans
# adtc-pipeline
