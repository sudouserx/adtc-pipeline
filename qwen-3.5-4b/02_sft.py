#!/usr/bin/env python3
"""BF16 RsLoRA SFT (no QAT). Writes adapter/ and heldout.jsonl."""

from __future__ import annotations

import unsloth  # noqa: F401  — patch before transformers/trl/peft

import inspect
import math
import os

import config
import model
from common import (
    adapter_dir,
    assert_completion_only_labels,
    assert_model_bf16,
    checkpoint_weight_keys,
    filter_sft_config,
    hf_token,
    load_training_rows,
    prepare_sft_data,
    preserve_best_checkpoint,
    resolve_base_revision,
    resolve_dataset_revisions,
    run_dir,
    seed_everything,
    sha256_bytes,
    training_dir,
    write_json,
)


def main() -> int:
    hf_token()
    seed_everything(config.SEED)
    base_revision = resolve_base_revision()
    dataset_revisions = resolve_dataset_revisions()
    output_dir = adapter_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

    loaded, tokenizer = FastModel.from_pretrained(
        model_name=model.BASE_MODEL,
        revision=base_revision,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    tokenizer = model.apply_chat_template(tokenizer)
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

    train_rows, eval_rows, heldout_rows = load_training_rows(dataset_revisions)
    train_dataset, eval_dataset, quality = prepare_sft_data(
        tokenizer, train_rows, eval_rows, heldout_rows, output_dir
    )

    effective_batch = config.TRAIN_BATCH_SIZE * config.GRAD_ACCUMULATION
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    save_steps = max(1, math.ceil(steps_per_epoch / 2))
    seed_everything(config.SEED)

    work_dir = training_dir()
    sft_config_kwargs: dict = {
        "output_dir": str(work_dir),
        "per_device_train_batch_size": config.TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRAD_ACCUMULATION,
        "num_train_epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "neftune_noise_alpha": 5.0,
        "bf16": True,
        "fp16": False,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": save_steps,
        "save_steps": save_steps,
        "logging_steps": max(1, save_steps // 10),
        "save_total_limit": 8,
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
    sft_config_kwargs[length_key] = config.MAX_SEQ_LENGTH
    sft_config_kwargs = filter_sft_config(
        sft_config_kwargs,
        sft_config_parameters,
        length_key=length_key,
        max_seq_length=config.MAX_SEQ_LENGTH,
    )
    trainer_kwargs: dict = {
        "model": loaded,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": SFTConfig(**sft_config_kwargs),
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=2, early_stopping_threshold=1e-3
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
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "lora": {
                "r": model.LORA["r"],
                "alpha": model.LORA["lora_alpha"],
                "rslora": model.LORA["use_rslora"],
                "qat_scheme": model.LORA["qat_scheme"],
                "target_modules": targets,
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
        },
    )
    print(f"adapter: {output_dir}")
    print(f"heldout: {output_dir / 'heldout.jsonl'}")
    print(f"run_dir: {run_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
