#!/usr/bin/env python3
"""Stage 02b — DPO preference alignment on top of the SFT adapter.

FINAL version (2026-09-20). Pairs with config.py and the UNMODIFIED
repo common.py / model.py — the patched helpers the earlier 02b_dpo.py
needed (common.dpo_adapter_dir, model.DATASETS["preference"]) are replaced
by local equivalents, so no other file changes.

Flow:
  1. Snapshot the pristine SFT adapter:  adapter/  ->  adapter-sft/
     (taken once; every (re)run trains DPO FROM this frozen state, so
     re-running DPO never stacks alignment on alignment).
  2. Load `kuzaai/kuza_dpo_preference` (train/validation; local override
     KUZA_LOCAL_DATA/preference.jsonl). The dataset is uploaded MANUALLY by
     the repo owner — this stage only consumes it, never pushes it.
  3. Continue training the SFT LoRA with TRL DPOTrainer (Unsloth-patched),
     ref_model=None (PEFT: reference = adapter disabled), RPO term on.
  4. Save the policy adapter to adapter-dpo/ with a full manifest set.
  5. PROMOTE: replace adapter/ contents with adapter-dpo/ so the UNPATCHED
     downstream stages (03_reference ... 08_upload) automatically merge,
     calibrate, quantize and ship the preference-aligned policy. The SFT
     state stays intact in adapter-sft/ for rollback/provenance.

Controls:
  KUZA_SKIP_DPO=1     skip the stage entirely (adapter/ stays pure SFT)
  KUZA_DPO_ENABLED=0  same as skip (config.DPO["enabled"])
  KUZA_REQUIRE_DPO=1  missing dataset becomes a hard error, not a skip
  KUZA_FORCE_DPO=1    retrain even if adapter-dpo/ already exists

Dataset rows are accepted in either layout:
  - TRL conversational: prompt / chosen / rejected message lists
  - flat mirrors: instruction / chosen_text / rejected_text
The Kuza canonical system prompt is NOT part of the stored prompt; rendering
injects it exactly like SFT (model.SYSTEM_PROMPT via the patched template).
"""

from __future__ import annotations

import shutil
import unsloth  # noqa: F401  — patch before transformers/trl/peft

import inspect
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import config
import model
from common import (
    adapter_dir,
    assert_adapter_tensors_loaded,
    assert_model_bf16,
    cast_lora_to_bf16,
    checkpoint_weight_keys,
    hf_token,
    preserve_best_checkpoint,
    read_jsonl,
    run_dir,
    seed_everything,
    sha256_bytes,
    sha256_file,
    training_dir,
    write_json,
)

GEMMA_ROLES = {"user": "user", "human": "user", "assistant": "model", "model": "model", "gpt": "model"}


# --------------------------------------------------------------------------- #
# local helpers (replace the patched common.py additions of the upgrade pack)
# --------------------------------------------------------------------------- #

def dpo_adapter_dir() -> Path:
    """Policy-adapter output dir (was common.dpo_adapter_dir in the patch)."""
    return run_dir() / "adapter-dpo"


def sft_snapshot_dir() -> Path:
    """Frozen pristine SFT adapter (never modified after creation)."""
    return run_dir() / "adapter-sft"


def _weights_identical(left: Path, right: Path) -> bool:
    """True when both adapter weight files exist and hash identically."""
    marker = "adapter_model.safetensors"
    a, b = left / marker, right / marker
    return a.is_file() and b.is_file() and sha256_file(a) == sha256_file(b)


def promote_policy(dpo_out: Path, target: Path) -> None:
    """Make adapter/ the preference-aligned policy for stages 03-08.

    Idempotent: re-promotes whenever adapter-dpo weights differ from what
    adapter/ currently holds (e.g. after KUZA_FORCE_DPO=1 retrain).
    """
    if not (dpo_out / "adapter_config.json").is_file():
        raise RuntimeError(f"DPO output incomplete, refusing to promote: {dpo_out}")
    if not (dpo_out / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"DPO output has no weights, refusing to promote: {dpo_out}")
    if _weights_identical(dpo_out, target) and (target / "dpo_manifest.json").is_file():
        print(f"promote: adapter/ already holds this DPO policy", flush=True)
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(dpo_out, target)
    write_json(
        target / "dpo_promotion.json",
        {
            "promoted_from": dpo_out.name,
            "sft_snapshot": sft_snapshot_dir().name,
            "weights_sha256": sha256_file(dpo_out / "adapter_model.safetensors"),
        },
    )
    print(f"promote: adapter/ <- {dpo_out} (stages 03-08 now use the aligned policy)", flush=True)


# --------------------------------------------------------------------------- #
# dataset loading
# --------------------------------------------------------------------------- #

def resolve_preference_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    """Return (train_rows, eval_rows, info) or None when unavailable."""
    root = config.LOCAL_DATA_DIR
    if root is not None:
        local = root / "preference.jsonl"
        if local.is_file() and local.stat().st_size > 0:
            rows = read_jsonl(local)
            held = max(1, round(len(rows) * 0.10))
            info = {"origin": "local", "path": str(local), "rows": len(rows)}
            return rows[held:], rows[:held], info
    repo = config.DPO_DATASET  # uploaded manually; consumed only, never pushed
    try:
        from datasets import load_dataset

        loaded = load_dataset(repo, token=hf_token())
    except Exception as exc:  # noqa: BLE001 - graceful skip is the default
        print(
            f"preference dataset not available: {repo} ({exc!r}).\n"
            "Upload download/kuza_dpo_preference/{train,validation}.jsonl to "
            f"{repo} manually, or point KUZA_LOCAL_DATA at a directory containing "
            "preference.jsonl. Skipping DPO (set KUZA_REQUIRE_DPO=1 to fail "
            "instead).",
            flush=True,
        )
        return None
    train_rows = [dict(row) for row in loaded.get("train", [])]
    eval_rows = [dict(row) for row in loaded.get(config.DPO_VALIDATION_SPLIT, [])]
    info = {"origin": "hub", "repo": repo, "rows": len(train_rows) + len(eval_rows)}
    return train_rows, eval_rows, info


def normalize_pair(row: dict[str, Any]) -> dict[str, Any] | None:
    """Accept conversational or flat layouts; return gemma-role dialog."""
    def dialog_from(value: Any) -> list[dict[str, str]] | None:
        if not isinstance(value, list):
            return None
        dialog: list[dict[str, str]] = []
        for turn in value:
            if not isinstance(turn, dict):
                continue
            role = GEMMA_ROLES.get(str(turn.get("role", "")).lower())
            content = str(turn.get("content", "")).strip()
            if role and content:
                dialog.append({"role": role, "content": content})
        return dialog or None

    prompt = dialog_from(row.get("prompt"))
    if prompt is None:
        instruction = str(row.get("instruction", "")).strip()
        if not instruction:
            return None
        prompt = [{"role": "user", "content": instruction}]
    chosen = dialog_from(row.get("chosen"))
    if chosen is None:
        text = str(row.get("chosen_text", "")).strip()
        chosen = [{"role": "model", "content": text}] if text else None
    rejected = dialog_from(row.get("rejected"))
    if rejected is None:
        text = str(row.get("rejected_text", "")).strip()
        rejected = [{"role": "model", "content": text}] if text else None
    if not prompt or not chosen or not rejected:
        return None
    if prompt[-1]["role"] != "user" or chosen[0]["role"] != "model":
        return None
    language = str(row.get("language", "english")).strip().lower() or "english"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "language": language,
        "archetype": str(row.get("archetype", "")),
        "source": str(row.get("source", "")),
        "source_id": str(row.get("source_id", "")),
    }


# --------------------------------------------------------------------------- #
# Kuza-template rendering (identical path to SFT)
# --------------------------------------------------------------------------- #

def render_dialog(tokenizer: Any, dialog: list[dict[str, str]], generate: bool) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": model.SYSTEM_PROMPT}, *dialog],
            tokenize=False,
            add_generation_prompt=generate,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": model.SYSTEM_PROMPT}, *dialog],
            tokenize=False,
            add_generation_prompt=generate,
        )
    model.assert_rendered_chat(tokenizer, rendered)
    return rendered


def render_pair(tokenizer: Any, pair: dict[str, Any]) -> tuple[str, str, str] | None:
    prompt_text = render_dialog(tokenizer, pair["prompt"], generate=True)
    full_chosen = render_dialog(tokenizer, [*pair["prompt"], *pair["chosen"]], generate=False)
    full_rejected = render_dialog(
        tokenizer, [*pair["prompt"], *pair["rejected"]], generate=False
    )
    if not (full_chosen.startswith(prompt_text) and full_rejected.startswith(prompt_text)):
        return None
    return prompt_text, full_chosen[len(prompt_text):], full_rejected[len(prompt_text):]


# --------------------------------------------------------------------------- #
# DPOConfig filtering (version tolerant, mirrors 02_sft.py's approach)
# --------------------------------------------------------------------------- #

def filter_dpo_config(kwargs: dict[str, Any], parameters: Any) -> dict[str, Any]:
    parameter_set = set(parameters)
    filtered = {key: value for key, value in kwargs.items() if key in parameter_set}
    required = ("beta", "loss_type", "max_length")
    missing = [key for key in required if key not in filtered]
    if missing:
        raise RuntimeError(f"DPOConfig dropped required keys: {missing}")
    if filtered.get("bf16") is not True or filtered.get("fp16") is not False:
        raise RuntimeError("DPOConfig lost the BF16 gate")
    return filtered


def json_load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    if os.environ.get("KUZA_SKIP_DPO") == "1":
        print("dpo: skipped (KUZA_SKIP_DPO=1); adapter/ remains the SFT policy")
        return 0
    if not config.DPO.get("enabled", True):
        print("dpo: skipped (KUZA_DPO_ENABLED=0); adapter/ remains the SFT policy")
        return 0
    hf_token()
    seed_everything(config.SEED)

    sft_adapter = adapter_dir()
    if not (sft_adapter / "adapter_config.json").is_file():
        raise RuntimeError("Missing SFT adapter. Run 02_sft.py first.")
    dpo_out = dpo_adapter_dir()
    snapshot = sft_snapshot_dir()

    # ---- pristine SFT snapshot (taken once, frozen forever) --------------- #
    if not snapshot.exists():
        if (sft_adapter / "dpo_manifest.json").is_file():
            raise RuntimeError(
                "adapter/ holds a promoted DPO policy but adapter-sft/ is gone; "
                "the pristine SFT state cannot be reconstructed. Re-run "
                "02_sft.py, then 02b_dpo.py."
            )
        shutil.copytree(sft_adapter, snapshot)
        print(f"snapshot: pristine SFT adapter frozen at {snapshot}", flush=True)

    # ---- skip-fast when a DPO adapter already exists ----------------------- #
    if (dpo_out / "adapter_config.json").is_file() and os.environ.get(
        "KUZA_FORCE_DPO", ""
    ) != "1":
        print(f"dpo: adapter already present, skipping training: {dpo_out}")
        promote_policy(dpo_out, sft_adapter)  # keep adapter/ consistent
        return 0

    payload = resolve_preference_dataset()
    if payload is None:
        if os.environ.get("KUZA_REQUIRE_DPO") == "1":
            raise RuntimeError("KUZA_REQUIRE_DPO=1 but the preference dataset is unavailable")
        print(
            "dpo: skipped — preference dataset unavailable; "
            "adapter/ remains the SFT policy (set KUZA_REQUIRE_DPO=1 to fail instead)",
            flush=True,
        )
        return 0
    raw_train_rows, raw_eval_rows, dataset_info = payload

    # ---- load the PRISTINE SFT adapter for continued training -------------- #
    import torch
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastModel

    base_revision = model.read_base_revision(snapshot)
    loaded, tokenizer = FastModel.from_pretrained(
        model_name=str(snapshot),
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    tokenizer = model.apply_chat_template(tokenizer)
    kv_patch = model.patch_kv_sharing(loaded)
    cast_counts = cast_lora_to_bf16(loaded)
    if cast_counts:
        print(
            f"Loaded SFT adapter for DPO cast LoRA dtypes: {cast_counts}",
            flush=True,
        )
    assert_model_bf16(loaded, "Loaded SFT adapter for DPO")
    model.validate_kv_sharing(loaded, checkpoint_weight_keys(model.BASE_MODEL, base_revision))
    adapter_parameter_names = [name for name, _ in loaded.named_parameters() if "lora_" in name]
    if not adapter_parameter_names:
        raise RuntimeError("Resumed SFT adapter produced no trainable adapter parameters")
    model.assert_no_shared_kv_lora(adapter_parameter_names)
    assert_adapter_tensors_loaded(loaded, snapshot)

    # ---- render + gate the preference pairs ------------------------------- #
    def build(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
        rendered: list[dict[str, str]] = []
        quality: Counter[str] = Counter()
        for row in rows:
            pair = normalize_pair(row)
            if pair is None:
                quality["invalid_row"] += 1
                continue
            texts = render_pair(tokenizer, pair)
            if texts is None:
                quality["render_splice_failed"] += 1
                continue
            prompt_text, chosen_text, rejected_text = texts
            prompt_ids = model.encode_ids(tokenizer, prompt_text)
            chosen_ids = model.encode_ids(tokenizer, chosen_text)
            rejected_ids = model.encode_ids(tokenizer, rejected_text)
            longest = max(len(chosen_ids), len(rejected_ids))
            if len(prompt_ids) + longest > config.MAX_SEQ_LENGTH:
                quality["too_long"] += 1
                continue
            if not chosen_text.strip() or not rejected_text.strip():
                quality["empty_completion"] += 1
                continue
            quality["kept"] += 1
            rendered.append(
                {
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": longest,
                }
            )
        total = len(rows)
        report = {
            "split": split,
            "input_rows": total,
            "valid_rows": len(rendered),
            **{key: value for key, value in sorted(quality.items())},
            "invalid_fraction": (total - len(rendered)) / max(1, total),
            "languages": dict(Counter(row["language"] for row in rows)),
            "archetypes": dict(
                Counter(str(row.get("archetype", "")) for row in rows)
            ),
        }
        if len(rendered) < 32:
            raise RuntimeError(f"DPO {split} split has too few usable pairs: {report}")
        return rendered, report

    train_pairs, train_report = build(raw_train_rows, "train")
    eval_pairs, eval_report = build(raw_eval_rows, "validation")

    from datasets import Dataset

    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset = Dataset.from_list(eval_pairs)

    # ---- trainer ----------------------------------------------------------- #
    effective_batch = config.DPO_TRAIN_BATCH_SIZE * config.DPO_GRAD_ACCUMULATION
    steps_per_epoch = max(1, (len(train_dataset) + effective_batch - 1) // effective_batch)
    save_steps = max(1, (steps_per_epoch + 1) // 2)
    seed_everything(config.SEED)

    work_dir = training_dir() / "dpo"
    dpo_config_kwargs: dict[str, Any] = {
        "output_dir": str(work_dir),
        "per_device_train_batch_size": config.DPO_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.DPO_GRAD_ACCUMULATION,
        "num_train_epochs": config.DPO_EPOCHS,
        "learning_rate": config.DPO_LR,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": config.DPO_WARMUP_RATIO,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "beta": config.DPO_BETA,
        "loss_type": "sigmoid",
        "rpo_alpha": config.DPO_RPO_ALPHA,
        "max_prompt_length": config.DPO_MAX_PROMPT_LENGTH,
        "max_completion_length": config.DPO_MAX_COMPLETION_LENGTH,
        "max_length": config.DPO_MAX_LENGTH,
        "truncation_mode": "keep_start",
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": save_steps,
        "save_steps": save_steps,
        "logging_steps": max(1, save_steps // 10),
        "save_total_limit": 4,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_rewards/accuracies",
        "greater_is_better": True,
        "seed": config.SEED,
        "data_seed": config.SEED,
        "report_to": "none",
        "dataset_num_proc": min(8, os.cpu_count() or 1),
    }
    dpo_config_kwargs = filter_dpo_config(
        dpo_config_kwargs, inspect.signature(DPOConfig).parameters
    )
    trainer_kwargs: dict[str, Any] = {
        "model": loaded,
        "ref_model": None,  # PEFT: reference = adapter disabled
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": DPOConfig(**dpo_config_kwargs),
    }
    trainer_parameters = inspect.signature(DPOTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

    result = trainer.train()
    best = trainer.state.best_model_checkpoint
    if best:
        preserve_best_checkpoint(work_dir, best)
    dpo_out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(dpo_out, safe_serialization=True)
    tokenizer.save_pretrained(dpo_out)
    trainer.state.save_to_json(str(dpo_out / "trainer_state.json"))

    # ---- manifests so stages 03-08 treat the policy exactly like adapter --- #
    for name in ("heldout.jsonl", "kuza_system_prompt.json"):
        source = snapshot / name
        if source.is_file():
            shutil.copy2(source, dpo_out / name)
    sft_manifest_path = snapshot / "sft_manifest.json"
    if sft_manifest_path.is_file():
        manifest = {
            **json_load(sft_manifest_path),
            "dpo": {
                "dataset": dataset_info,
                "beta": config.DPO_BETA,
                "rpo_alpha": config.DPO_RPO_ALPHA,
                "learning_rate": config.DPO_LR,
                "epochs": config.DPO_EPOCHS,
                "train_pairs": len(train_pairs),
                "eval_pairs": len(eval_pairs),
                "best_checkpoint": best,
                "kv_sharing_patch": kv_patch,
            },
        }
        write_json(dpo_out / "sft_manifest.json", manifest)
    write_json(dpo_out / "train_metrics.json", result.metrics)
    write_json(
        dpo_out / "data_quality.json",
        {
            "stage": "dpo",
            "train": train_report,
            "validation": eval_report,
        },
    )
    write_json(
        dpo_out / "dpo_manifest.json",
        {
            "base_model": model.BASE_MODEL,
            "base_revision": base_revision,
            "sft_adapter": str(snapshot),
            "dataset": dataset_info,
            "hyperparameters": {
                key: value
                for key, value in dpo_config_kwargs.items()
                if key in {
                    "beta", "loss_type", "rpo_alpha", "learning_rate",
                    "num_train_epochs", "max_prompt_length",
                    "max_completion_length", "max_length", "truncation_mode",
                }
            },
            "train_pairs": len(train_pairs),
            "eval_pairs": len(eval_pairs),
            "system_prompt_sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
            "kv_sharing_patch": kv_patch,
        },
    )

    # ---- promote: unpatched 03-08 now consume the aligned policy ----------- #
    promote_policy(dpo_out, sft_adapter)
    print(f"dpo adapter: {dpo_out}")
    print(f"run_dir: {run_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
