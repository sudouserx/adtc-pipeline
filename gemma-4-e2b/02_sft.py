#!/usr/bin/env python3
"""BF16 QAT-aware RsLoRA SFT. Writes adapter/ and heldout.jsonl.

REPLACEMENT (review 2026-09). Changes vs original 02_sft.py:

  1. Per-source train/eval splits BEFORE mixing (was: general had no split and
     held-out general rows were drawn from the same pool used for training —
     direct selection leakage; adversarial pool size throttled the whole eval
     set to 39 rows in the archived run).
  2. Normalized-hash dedup within every split AND cross-split (train vs
     eval/held-out), reported in data_quality.json.
  3. EOS-safe truncation: tail-cropped rows get their termination tokens
     re-appended (the original could train EOS-less suffixes, teaching the
     model to stop mid-sentence without a stop token).
  4. Training config from the new config knobs: LoRA-appropriate LR 1e-4,
     cosine_with_warmup, 4 evals/epoch, patience 4, threshold 1e-4.
  5. New optional mix sources: swahili_native, code_switch, grounding
     (graceful degradation with warnings; recorded in the manifest).
"""

from __future__ import annotations

import unsloth  # noqa: F401  — patch before transformers/trl/peft

import hashlib
import inspect
import json
import math
import os
import re
import shutil

import config
import model
from common import (
    adapter_dir,
    assert_completion_only_labels,
    assert_model_bf16,
    checkpoint_weight_keys,
    deterministic_sample,
    filter_sft_config,
    hf_token,
    load_local_or_hub,
    preserve_best_checkpoint,
    resolve_base_revision,
    resolve_dataset_revisions,
    run_dir,
    seed_everything,
    sha256_bytes,
    split_rows,
    training_dir,
    write_json,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Dedup (review F2.2 / F4.4)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def row_key(row: dict) -> str:
    payload = json.dumps(
        {
            "i": _norm(str(row.get("instruction", "")))[:2000],
            "r": _norm(str(row.get("response", "")))[:4000],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def dedup_rows(rows: list[dict], label: str) -> tuple[list[dict], int]:
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept, len(rows) - len(kept)


def filter_against(rows: list[dict], seen: set[str], label: str) -> tuple[list[dict], int]:
    kept = [row for row in rows if row_key(row) not in seen]
    return kept, len(rows) - len(kept)


def train_only_keys(rows: list[dict]) -> set[str]:
    return {row_key(row) for row in rows}


# ---------------------------------------------------------------------------
# Data mixing with per-source splits BEFORE mixing (review F2.1 / F5)
# ---------------------------------------------------------------------------

def load_training_rows_split(
    dataset_revisions: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Load every source, split each into train/eval pools, THEN mix.

    Fixes the archived-run defects: general leakage into held-out, and the
    adversarial pool throttling the entire trainer eval to 39 rows.
    """
    warnings: list[str] = []

    def load(key: str, language: str, filename: str) -> list[dict]:
        if key not in model.DATASETS:
            return []
        optional = key in config.SFT_OPTIONAL_SOURCES
        hub_fallback = (not optional) or (key in dataset_revisions)
        return load_local_or_hub(
            key, language, filename, dataset_revisions, hub_fallback=hub_fallback
        )

    pools: dict[str, tuple[list[dict], list[dict]]] = {}
    for key in ("english", "swahili", "general", "adversarial", "multiturn"):
        rows = load(key, "swahili" if key == "swahili" else "english", f"{key}.jsonl")
        if key == "multiturn":
            # Multiturn rows carry their own messages; keep them whole but
            # still split so held-out multiturn never appears in training.
            train_rows, eval_rows = split_rows(
                rows, model.MIX["agri_eval_fraction"], config.SEED
            )
            pools[key] = (train_rows, eval_rows)
            continue
        if key == "general":
            rows = [
                row
                for row in rows
                if len(row["instruction"].split()) <= model.MIX["general_max_instruction_words"]
                and len(row["response"].split()) <= model.MIX["general_max_response_words"]
            ]
        if not rows and key == "english":
            raise RuntimeError(
                "No English rows. Push data/final/english.jsonl to "
                f"{model.DATASETS['english']} or set KUZA_LOCAL_DATA."
            )
        fraction = (
            model.MIX["adversarial_eval_fraction"]
            if key == "adversarial"
            else model.MIX["agri_eval_fraction"]
        )
        pools[key] = split_rows(rows, fraction, config.SEED) if rows else ([], [])

    for key in config.SFT_OPTIONAL_SOURCES:
        rows = load(key, "swahili" if key == "swahili_native" else "english", f"{key}.jsonl")
        if not rows:
            warnings.append(f"optional source '{key}' empty; proceeding without it")
            pools[key] = ([], [])
            continue
        fraction = (
            model.MIX["adversarial_eval_fraction"]
            if key == "adversarial"
            else model.MIX["agri_eval_fraction"]
        )
        pools[key] = split_rows(rows, fraction, config.SEED)

    en_train, en_eval = pools["english"]
    sw_train, sw_eval = pools["swahili"]
    gen_train, gen_eval = pools["general"]
    adv_train, adv_eval = pools["adversarial"]
    mt_train, mt_eval = pools["multiturn"]

    sw_target = round(len(en_train) * model.MIX["swahili_of_english"])
    if sw_target > 0 and not sw_train:
        raise RuntimeError(
            "MIX requests Swahili but no swahili rows were found. Translate, "
            f"push to {model.DATASETS['swahili']}, and set KUZA_SW_DATASET."
        )
    general_target = round(len(en_train) * model.MIX["general_of_english"])
    adversarial_target = round(len(en_train) * model.MIX["adversarial_of_english"])
    multiturn_target = round(len(en_train) * model.MIX.get("multiturn_of_english", 0.03))
    if adversarial_target > 0 and len(adv_train) < adversarial_target:
        warnings.append(
            f"adversarial pool {len(adv_train)} < target {adversarial_target}; "
            "safety posture will be thin (review F5)"
        )
    if not mt_train:
        raise RuntimeError(
            "No multiturn rows. Push data/final/multiturn.jsonl to "
            f"{model.DATASETS['multiturn']} or set KUZA_MT_DATASET."
        )

    train = [
        *en_train,
        *deterministic_sample(sw_train, sw_target, config.SEED + 1),
        *deterministic_sample(gen_train, general_target, config.SEED + 2),
        *deterministic_sample(adv_train, adversarial_target, config.SEED + 3),
        *deterministic_sample(mt_train, multiturn_target, config.SEED + 10),
    ]
    for key in config.SFT_OPTIONAL_SOURCES:
        fraction_key = {
            "swahili_native": "swahili_native_of_english",
            "code_switch": "code_switch_of_english",
            "grounding": "grounding_of_english",
        }[key]
        share = model.MIX.get(fraction_key, 0.0)
        target = round(len(en_train) * share)
        if target > 0 and pools[key][0]:
            train.extend(
                deterministic_sample(pools[key][0], target, config.SEED + 11 + len(key))
            )
    random_shuffle(train)

    # Trainer eval: per-source caps instead of a global min throttled by the
    # smallest (adversarial) pool. Fixed budget per language + safety slice.
    eval_group = min(
        model.MIX["eval_group_max"],
        max(1, len(en_eval)),
        max(1, len(sw_eval) or 1),
    )
    adv_eval_cap = min(100, len(adv_eval))
    evaluation = [
        *deterministic_sample(en_eval, eval_group, config.SEED + 4),
        *deterministic_sample(sw_eval, min(eval_group, len(sw_eval)), config.SEED + 5),
        *deterministic_sample(adv_eval, adv_eval_cap, config.SEED + 6),
    ]
    random_shuffle(evaluation)

    # Held-out draws ONLY from eval pools — train rows can never appear here.
    heldout = [
        *en_eval,
        *sw_eval,
        *adv_eval,
        *gen_eval,
        *mt_eval,
        *pools["grounding"][1],
    ]
    return train, evaluation, heldout, warnings


def random_shuffle(rows: list[dict]) -> None:
    import random

    random.Random(config.SEED).shuffle(rows)


# ---------------------------------------------------------------------------
# EOS-safe data preparation (review F4.3)
# ---------------------------------------------------------------------------

def _termination_ids(tokenizer) -> list[int]:
    """Ids that must terminate every kept example: <end_of_turn> then <eos>.

    Resolved via the vocab (not attributes): HF tokenizers expose
    ``eos_token`` but not ``end_of_turn_token``, while Gemma chat turns end
    with <end_of_turn>.
    """
    vocab = tokenizer.get_vocab()
    candidates = ["<end_of_turn>", "<eos>"]
    eos = getattr(tokenizer, "eos_token", None)
    if eos and eos not in candidates:
        candidates.append(eos)
    ids: list[int] = []
    for token in candidates:
        token_id = vocab.get(token)
        if token_id is not None and token_id not in ids:
            ids.append(token_id)
    return ids


def prepare_sft_data(
    tokenizer,
    train_rows: list[dict],
    eval_rows: list[dict],
    heldout_rows: list[dict],
    output_dir,
) -> tuple[object, object, dict]:
    from datasets import Dataset
    from jinja2.exceptions import TemplateError

    max_len = config.TRAIN_MAX_SEQ_LENGTH

    def process(rows: list[dict], split: str) -> tuple[list[dict], dict]:
        valid: list[dict] = []
        render_failed = 0
        missing_marker = 0
        truncated = 0
        eos_restored = 0
        first_error = None
        for row in rows:
            try:
                text = model.render_text(tokenizer, row)
            except (TemplateError, RuntimeError) as exc:
                render_failed += 1
                if first_error is None:
                    first_error = repr(exc)
                continue
            full_ids = model.encode_ids(tokenizer, text)
            if len(full_ids) > max_len:
                # keep_start: never left-slice (drops the system/user turn).
                # NEW: re-append termination tokens so a cropped suffix still
                # teaches the model to STOP; a bare tail-crop trains
                # mid-sentence endings with no stop token.
                term = _termination_ids(tokenizer)
                if term and len(term) < max_len:
                    body = full_ids[: max_len - len(term)]
                    if model.has_response_marker(model.decode_ids(tokenizer, body)):
                        full_ids = body + term
                        eos_restored += 1
                    else:
                        missing_marker += 1
                        continue
                else:
                    missing_marker += 1
                    continue
                truncated += 1
            text = model.decode_ids(tokenizer, full_ids)
            if not model.has_response_marker(text):
                missing_marker += 1
                continue
            valid.append({**row, "text": text})
        invalid = render_failed + missing_marker
        report = {
            "split": split,
            "input_rows": len(rows),
            "valid_rows": len(valid),
            "invalid_rows": invalid,
            "render_failed": render_failed,
            "missing_marker": missing_marker,
            "first_error": first_error,
            "invalid_fraction": invalid / max(1, len(rows)),
            "truncated_rows": truncated,
            "truncated_fraction": truncated / max(1, len(rows)),
            "eos_restored_rows": eos_restored,
            "max_seq_length": max_len,
            "composition": {},
            "languages": {},
        }
        from collections import Counter

        report["composition"] = dict(Counter(row["source"] for row in valid))
        report["languages"] = dict(Counter(row["language"] for row in valid))
        if report["invalid_fraction"] > config.MAX_INVALID_FRACTION:
            raise RuntimeError(f"{split} invalid-label fraction exceeds limit: {report}")
        if report["truncated_fraction"] > config.MAX_TRUNCATED_FRACTION:
            raise RuntimeError(f"{split} truncation fraction exceeds limit: {report}")
        return valid, report

    # Dedup within each split, then enforce cross-split hygiene.
    train_rows, train_dups = dedup_rows(list(train_rows), "train")
    eval_rows, eval_dups = dedup_rows(list(eval_rows), "eval")
    heldout_rows, heldout_dups = dedup_rows(list(heldout_rows), "heldout")
    eval_rows, leak_eval = filter_against(eval_rows, train_only_keys(train_rows), "eval-vs-train")
    heldout_rows, leak_held = filter_against(
        heldout_rows, train_only_keys(train_rows) | train_only_keys(eval_rows), "heldout-vs-train"
    )

    train_valid, train_report = process(train_rows, "train")
    eval_valid, eval_report = process(eval_rows, "eval")
    dedup_report = {
        "within_split_duplicates": {
            "train": train_dups, "eval": eval_dups, "heldout": heldout_dups
        },
        "cross_split_removals": {"eval": leak_eval, "heldout": leak_held},
    }
    train_report["dedup"] = dedup_report
    quality = {"train": train_report, "eval": eval_report}
    write_json(output_dir / "data_quality.json", quality)
    write_jsonl(output_dir / "heldout.jsonl", heldout_rows)
    columns = ["text", "instruction", "response", "language", "source", "source_id"]
    if any("messages" in row for row in train_valid + eval_valid):
        for row in train_valid + eval_valid:
            row.setdefault("messages", [])
        columns.append("messages")
    return (
        Dataset.from_list(train_valid).select_columns(columns),
        Dataset.from_list(eval_valid).select_columns(columns),
        quality,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    hf_token()
    seed_everything(config.SEED)
    for stale in (run_dir() / "adapter-sft", run_dir() / "adapter-dpo"):
        if stale.exists():
            shutil.rmtree(stale)
            print(
                f"sft: cleared stale {stale.name}/ "
                "(SFT re-run invalidates prior DPO snapshot)",
                flush=True,
            )
    base_revision = resolve_base_revision()
    dataset_revisions = resolve_dataset_revisions()
    output_dir = adapter_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

    max_len = config.TRAIN_MAX_SEQ_LENGTH

    loaded, tokenizer = FastModel.from_pretrained(
        model_name=model.BASE_MODEL,
        revision=base_revision,
        max_seq_length=max_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    tokenizer = model.apply_chat_template(tokenizer)
    kv_patch = model.patch_kv_sharing(loaded)
    assert_model_bf16(loaded, "Loaded base model")
    checkpoint_keys = checkpoint_weight_keys(model.BASE_MODEL, base_revision)
    model.validate_kv_sharing(loaded, checkpoint_keys)
    targets = model.lora_target_names(loaded)
    lora_kwargs = {key: value for key, value in model.LORA.items() if value is not None}
    loaded = FastModel.get_peft_model(
        loaded,
        target_modules=targets,
        random_state=config.SEED,
        **lora_kwargs,
    )
    adapter_parameter_names = [
        name for name, _ in loaded.named_parameters() if "lora_" in name
    ]
    if not adapter_parameter_names:
        raise RuntimeError("LoRA attachment produced no adapter parameters")
    model.assert_no_shared_kv_lora(adapter_parameter_names)

    train_rows, eval_rows, heldout_rows, mix_warnings = load_training_rows_split(
        dataset_revisions
    )
    train_dataset, eval_dataset, quality = prepare_sft_data(
        tokenizer, train_rows, eval_rows, heldout_rows, output_dir
    )

    effective_batch = config.TRAIN_BATCH_SIZE * config.GRAD_ACCUMULATION
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    eval_steps = max(1, math.ceil(steps_per_epoch / config.EVALS_PER_EPOCH))
    seed_everything(config.SEED)

    work_dir = training_dir()
    sft_config_kwargs: dict = {
        "output_dir": str(work_dir),
        "per_device_train_batch_size": config.TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRAD_ACCUMULATION,
        "num_train_epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "lr_scheduler_type": config.LR_SCHEDULER,
        "warmup_ratio": config.WARMUP_RATIO,
        "weight_decay": config.WEIGHT_DECAY,
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "neftune_noise_alpha": 5.0,
        "bf16": True,
        "fp16": False,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": eval_steps,
        "save_steps": eval_steps,
        "logging_steps": max(1, eval_steps // 4),
        "save_total_limit": config.SAVE_TOTAL_LIMIT,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": config.SEED,
        "data_seed": config.SEED,
        "report_to": "none",
        "packing": False,
        "dataset_num_proc": min(8, os.cpu_count() or 1),
        "dataset_text_field": "text",
    }
    sft_config_parameters = inspect.signature(SFTConfig).parameters
    length_key = (
        "max_length" if "max_length" in sft_config_parameters else "max_seq_length"
    )
    sft_config_kwargs[length_key] = max_len
    sft_config_kwargs = filter_sft_config(
        sft_config_kwargs,
        sft_config_parameters,
        length_key=length_key,
        max_seq_length=max_len,
    )
    trainer_kwargs: dict = {
        "model": loaded,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": SFTConfig(**sft_config_kwargs),
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD,
            ),
        ],
    }
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer = train_on_responses_only(
        trainer,
        instruction_part=model.INSTRUCTION_PART,
        response_part=model.RESPONSE_PART,
    )
    quality["completion_labels"] = assert_completion_only_labels(trainer)

    result = trainer.train()
    preserve_best_checkpoint(work_dir, trainer.state.best_model_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    write_json(output_dir / "train_metrics.json", result.metrics)
    write_json(output_dir / "data_quality.json", quality)
    write_json(
        output_dir / "kuza_system_prompt.json",
        {
            "system_prompt": model.SYSTEM_PROMPT,
            "sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
        },
    )
    write_json(
        output_dir / "sft_manifest.json",
        {
            "base_model": model.BASE_MODEL,
            "base_revision": base_revision,
            "dataset_revisions": dataset_revisions,
            "max_seq_length": max_len,
            "mix": dict(model.MIX),
            "mix_warnings": mix_warnings,
            "lora": {
                "r": model.LORA["r"],
                "alpha": model.LORA["lora_alpha"],
                "rslora": model.LORA["use_rslora"],
                "qat_scheme": model.LORA["qat_scheme"],
                "target_modules": targets,
            },
            "optimizer": {
                "learning_rate": config.LEARNING_RATE,
                "scheduler": config.LR_SCHEDULER,
                "warmup_ratio": config.WARMUP_RATIO,
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "kv_sharing_patch": kv_patch,
        },
    )
    print(f"adapter: {output_dir}")
    print(f"heldout: {output_dir / 'heldout.jsonl'}")
    print(f"run_dir: {run_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
