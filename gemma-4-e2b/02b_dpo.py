#!/usr/bin/env python3
"""Stage 02b — DPO preference alignment on top of the SFT adapter.

FINAL version (2026-09-21). Pairs with config.py and the UNMODIFIED
repo common.py / model.py — the patched helpers the earlier 02b_dpo.py
needed (common.dpo_adapter_dir, model.DATASETS["preference"]) are replaced
by local equivalents, so no other file changes.

FIX 2026-09-21 (the crash in the error log):
  The previous version passed a ``_DPOTextProcessingShim`` (a plain object
  delegating via ``__getattr__``) as ``processing_class``. The
  (Unsloth-compiled) TRL DPOTrainer validates
  ``isinstance(processing_class, (PreTrainedTokenizerBase, ProcessorMixin))``
  and rejected it with::

      TypeError: The `processing_class` must be either a
      `PreTrainedTokenizerBase` or a `ProcessorMixin`

  A wrapper can never pass that gate (and ``__getattr__`` delegation cannot
  provide ``__call__``). The tokenizer itself passes the gate — exactly what
  the working SFT stage passes to SFTTrainer — so the shim is removed. The
  concern that motivated it (VLM DPO paths that want
  ``processing_class.tokenizer``) is handled by:
    1. passing the plain (chat-template-patched) tokenizer AND temporarily
       presenting the model as its text backbone (model_type patch) so TRL
       takes the standard TEXT dataset path for the vision-architecture
       model, and
    2. an automatic one-shot retry with a REAL processor whose
       ``.tokenizer`` is this tokenizer (plus an empty ``images`` column)
       if the installed TRL build still refuses a tokenizer.

Additional fixes in this version:
  * pairs whose prompt exceeds DPO_MAX_PROMPT_LENGTH or whose completion
    exceeds DPO_MAX_COMPLETION_LENGTH are dropped up front (counts
    recorded), so TRL prompt truncation can never cut the
    ``<start_of_turn>model`` generation header off a training prompt;
  * after construction the trainer's tokenized first row is verified
    against the rendered pair — a TRL build that prepends BOS to an
    already-``<bos>``-prefixed prompt is detected and the duplicate BOS is
    stripped from every train/eval row;
  * a missing Hub validation split is carved from train instead of
    crashing the stage;
  * dataset rows handed to TRL carry exactly prompt/chosen/rejected;
  * ``_dpo_tokenizer_from_processing`` no longer unwraps
    ``PreTrainedTokenizerFast.tokenizer`` (the Rust backend) by accident.

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

import dataclasses
import inspect
import json
import os
from collections import Counter
from contextlib import contextmanager
from inspect import Parameter
from pathlib import Path
from typing import Any, Iterator

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
    if hasattr(loaded, "keys"):  # DatasetDict
        train_rows = [dict(row) for row in loaded.get("train", [])]
        eval_rows = [dict(row) for row in loaded.get(config.DPO_VALIDATION_SPLIT, [])]
    else:  # single-split Dataset
        train_rows = [dict(row) for row in loaded]
        eval_rows = []
    info = {
        "origin": "hub",
        "repo": repo,
        "rows": len(train_rows) + len(eval_rows),
        "validation_split": config.DPO_VALIDATION_SPLIT,
    }
    # FIX: a repository without a validation split used to reach build() with
    # zero eval rows and die on the "too few usable pairs" gate. Carve a
    # deterministic 10% hold-out from the head of train instead.
    if not eval_rows and train_rows:
        held = max(1, round(len(train_rows) * 0.10))
        eval_rows = train_rows[:held]
        train_rows = train_rows[held:]
        info["validation_carved"] = {"held_out_rows": held, "source": "train_head"}
        print(
            f"dpo: hub dataset has no '{config.DPO_VALIDATION_SPLIT}' split; "
            f"held out {held} train rows for evaluation",
            flush=True,
        )
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
# DPO kwargs routing (version tolerant — Unsloth may move fields to trainer)
# --------------------------------------------------------------------------- #

DPO_OVERFLOW_KEYS = frozenset(
    {
        "loss_type",
        "rpo_alpha",
        "beta",
        "max_length",
        "max_prompt_length",
        "max_completion_length",
        "truncation_mode",
    }
)

DPO_MANIFEST_KEYS = frozenset(
    {
        "beta",
        "loss_type",
        "rpo_alpha",
        "learning_rate",
        "num_train_epochs",
        "max_prompt_length",
        "max_completion_length",
        "max_length",
        "truncation_mode",
    }
)


def _accepts_var_keyword(parameters: Any) -> bool:
    return any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _dataclass_field_names(cls: type) -> set[str]:
    if not dataclasses.is_dataclass(cls):
        return set()
    return {field.name for field in dataclasses.fields(cls)}


def _normalize_loss_type(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def split_dpo_kwargs(
    raw: dict[str, Any],
    config_cls: type,
    trainer_cls: type,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config_params = inspect.signature(config_cls).parameters
    trainer_params = inspect.signature(trainer_cls).parameters
    config_keys = set(config_params) - {"self"}
    trainer_keys = set(trainer_params) - {"self"}
    config_field_names = _dataclass_field_names(config_cls)
    trainer_field_names = _dataclass_field_names(trainer_cls)
    config_var_kw = _accepts_var_keyword(config_params)
    trainer_var_kw = _accepts_var_keyword(trainer_params)

    config_kwargs: dict[str, Any] = {}
    trainer_kwargs: dict[str, Any] = {}
    overflow_routing: dict[str, str] = {}
    deferred: dict[str, Any] = {}

    for key, value in raw.items():
        if key in config_keys:
            config_kwargs[key] = value
            if key in DPO_OVERFLOW_KEYS:
                overflow_routing[key] = "config_named"
        elif key in trainer_keys:
            trainer_kwargs[key] = value
            if key in DPO_OVERFLOW_KEYS:
                overflow_routing[key] = "trainer_named"
        elif key in config_field_names:
            config_kwargs[key] = value
            if key in DPO_OVERFLOW_KEYS:
                overflow_routing[key] = "config_field"
        elif key in DPO_OVERFLOW_KEYS:
            deferred[key] = value

    for key, value in deferred.items():
        if key in config_kwargs or key in trainer_kwargs:
            continue
        if config_var_kw or key in config_field_names:
            config_kwargs[key] = value
            overflow_routing[key] = "config_kwargs"
        elif trainer_var_kw or key in trainer_field_names:
            trainer_kwargs[key] = value
            overflow_routing[key] = "trainer_kwargs"
        elif key == "loss_type" and value == "sigmoid":
            overflow_routing[key] = "default"
            print(
                "dpo: loss_type='sigmoid' not exposed by installed TRL; "
                "using TRL default sigmoid DPO loss",
                flush=True,
            )
        elif key == "rpo_alpha":
            overflow_routing[key] = "unsupported"
            print(
                f"dpo: warning: rpo_alpha={value!r} not supported by installed "
                "DPOConfig/DPOTrainer; RPO term will be inactive",
                flush=True,
            )
        elif key == "loss_type":
            raise RuntimeError(
                f"loss_type={value!r} is not supported by installed DPOConfig or "
                f"DPOTrainer (config_var_kw={config_var_kw}, "
                f"trainer_var_kw={trainer_var_kw})"
            )
        else:
            raise RuntimeError(
                f"DPO hyperparameter {key!r} is not supported by installed "
                f"DPOConfig or DPOTrainer (config_var_kw={config_var_kw}, "
                f"trainer_var_kw={trainer_var_kw})"
            )

    if config_kwargs.get("bf16") is not True or config_kwargs.get("fp16") is not False:
        raise RuntimeError("DPOConfig lost the BF16 gate")

    for key in ("beta", "max_length"):
        if key in raw and key not in config_kwargs and key not in trainer_kwargs:
            raise RuntimeError(
                f"DPO hyperparameter {key!r} is not supported by installed "
                "DPOConfig or DPOTrainer"
            )

    config_overflow = sorted(
        key for key, route in overflow_routing.items() if route == "config_kwargs"
    )
    if config_overflow:
        routed_values = ", ".join(
            f"{key}={config_kwargs[key]!r}" for key in config_overflow
        )
        print(
            f"dpo: routed via DPOConfig **kwargs: {routed_values}",
            flush=True,
        )
    trainer_overflow = sorted(
        key for key, route in overflow_routing.items() if route == "trainer_kwargs"
    )
    if trainer_overflow:
        routed_values = ", ".join(
            f"{key}={trainer_kwargs[key]!r}" for key in trainer_overflow
        )
        print(
            f"dpo: routed via DPOTrainer **kwargs: {routed_values}",
            flush=True,
        )

    return config_kwargs, trainer_kwargs, overflow_routing


def build_dpo_config(
    config_cls: type,
    config_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, str]]:
    routing_updates: dict[str, str] = {}
    try:
        return config_cls(**config_kwargs), routing_updates
    except TypeError as exc:
        if "rpo_alpha" not in config_kwargs:
            raise
        retry_kwargs = dict(config_kwargs)
        retry_kwargs.pop("rpo_alpha", None)
        print(
            f"dpo: warning: DPOConfig rejected rpo_alpha ({exc}); "
            "continuing without RPO term",
            flush=True,
        )
        routing_updates["rpo_alpha"] = "unsupported_runtime"
        return config_cls(**retry_kwargs), routing_updates


def verify_dpo_hparams(
    args: Any,
    trainer: Any,
    intended: dict[str, Any],
    overflow_routing: dict[str, str],
) -> dict[str, dict[str, Any]]:
    routing: dict[str, dict[str, Any]] = {}
    for key in DPO_OVERFLOW_KEYS:
        if key not in intended:
            continue
        intended_value = intended[key]
        applied_via = overflow_routing.get(key, "unknown")
        actual = getattr(args, key, None)
        if actual is None:
            actual = getattr(trainer, key, None)

        entry: dict[str, Any] = {
            "intended": intended_value,
            "applied_via": applied_via,
        }
        if actual is not None:
            entry["actual"] = actual

        if key == "beta" and actual is not None and actual != intended_value:
            print(
                f"dpo: warning: beta intended {intended_value!r} but runtime has {actual!r}",
                flush=True,
            )
        if key == "max_length" and actual is not None and actual != intended_value:
            print(
                f"dpo: warning: max_length intended {intended_value!r} but runtime has {actual!r}",
                flush=True,
            )
        if key == "loss_type" and actual is not None:
            if _normalize_loss_type(actual) != _normalize_loss_type(intended_value):
                print(
                    f"dpo: warning: loss_type intended {intended_value!r} but runtime has {actual!r}",
                    flush=True,
                )
        if key == "rpo_alpha":
            if actual is None and applied_via not in {"unsupported", "unsupported_runtime"}:
                print(
                    f"dpo: warning: rpo_alpha intended {intended_value!r} but runtime has no RPO term",
                    flush=True,
                )
            elif actual is not None and actual != intended_value:
                print(
                    f"dpo: warning: rpo_alpha intended {intended_value!r} but runtime has {actual!r}",
                    flush=True,
                )

        routing[key] = entry
    return routing


# --------------------------------------------------------------------------- #
# processing_class resolution (FIX 2026-09-21)
# --------------------------------------------------------------------------- #

def _is_processor(obj: Any) -> bool:
    try:
        from transformers import ProcessorMixin
    except ImportError:
        return False
    return isinstance(obj, ProcessorMixin)


def _is_plain_tokenizer(obj: Any) -> bool:
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return False
    return isinstance(obj, PreTrainedTokenizerBase) and not _is_processor(obj)


def _is_vlm_model(model: Any) -> bool:
    model_config = getattr(model, "config", None)
    if model_config is None:
        return False
    model_type = str(getattr(model_config, "model_type", "")).lower()
    if model_type in {"gemma4", "gemma3"}:
        return True
    if model_type:
        try:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
            )

            if model_type in {name.lower() for name in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES}:
                return True
        except ImportError:
            pass
    class_name = model.__class__.__name__.lower()
    return "gemma4" in class_name or "imagetext" in class_name


@contextmanager
def _dpo_text_only_model_type(model: Any) -> Iterator[None]:
    """Temporarily present the VLM as its text backbone during trainer init.

    TRL keys its vision-specific DPO dataset path off
    ``model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``.
    We train text-only pre-rendered pairs, so force the composite config's
    model_type to the text config's value for the duration of
    ``DPOTrainer(...)`` so the standard text tokenization path is used.
    """
    model_config = getattr(model, "config", None)
    if model_config is None:
        yield
        return
    original = getattr(model_config, "model_type", None)
    text_type = getattr(getattr(model_config, "text_config", None), "model_type", None)
    if text_type and original and original != text_type:
        model_config.model_type = text_type
        try:
            yield
        finally:
            model_config.model_type = original
        return
    yield


def _processor_for_dpo(tokenizer: Any, base_revision: str) -> Any:
    """A real processor whose .tokenizer is our chat-template-patched tokenizer.

    Fallback for TRL builds that insist on a ProcessorMixin processing_class
    for vision-architecture models even with text-only pairs.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model.BASE_MODEL, revision=base_revision, token=hf_token()
    )
    processor.tokenizer = tokenizer
    template = getattr(tokenizer, "chat_template", None)
    if template:
        try:
            processor.chat_template = template
        except Exception:  # noqa: BLE001 — cosmetic best effort
            pass
    return processor


_DPO_PROCESSING_RETRY_MARKERS = (
    "processing_class",
    "ProcessorMixin",
    "PreTrainedTokenizerBase",
    "tokenizer",
    "images",
)


def _dpo_processing_retryable(exc: BaseException) -> bool:
    if not isinstance(exc, (TypeError, AttributeError, KeyError, ValueError)):
        return False
    text = str(exc)
    return any(marker in text for marker in _DPO_PROCESSING_RETRY_MARKERS)


def construct_dpo_trainer(
    trainer_cls: type,
    trainer_kwargs: dict[str, Any],
    model_obj: Any,
    tokenizer: Any,
    base_revision: str,
    train_pairs: list[dict[str, str]],
    eval_pairs: list[dict[str, str]],
) -> tuple[Any, dict[str, Any]]:
    """Build the DPOTrainer with a processing_class the installed TRL accepts.

    FIX 2026-09-21: the previous run passed a ``_DPOTextProcessingShim`` and
    TRL rejected it with ``TypeError: The `processing_class` must be either a
    `PreTrainedTokenizerBase` or a `ProcessorMixin``` — a delegating wrapper
    is an instance of neither. The tokenizer itself passes that gate.

      attempt 1 — processing_class = tokenizer, with the composite config's
                  model_type temporarily set to the text backbone so TRL
                  takes its standard text path (pairs are pre-rendered
                  strings; no images).
      attempt 2 — only if attempt 1 raises a processing-class-related error:
                  a real processor for the base model with ``.tokenizer``
                  grafted to our patched tokenizer, no model_type patch, and
                  an empty ``images`` column for TRL's vision dataset path.
    """
    metadata: dict[str, Any] = {
        "tokenizer_class": type(tokenizer).__name__,
        "shim_used": False,
        "model_type": getattr(getattr(model_obj, "config", None), "model_type", None),
        "text_config_model_type": getattr(
            getattr(getattr(model_obj, "config", None), "text_config", None),
            "model_type",
            None,
        ),
        "is_vlm_model": _is_vlm_model(model_obj),
    }
    text_patch_active = bool(
        metadata["text_config_model_type"]
        and metadata["model_type"]
        and metadata["model_type"] != metadata["text_config_model_type"]
    )

    # ---- attempt 1: plain tokenizer, text-route model_type ---------------- #
    try:
        with _dpo_text_only_model_type(model_obj):
            trainer = trainer_cls(**{**trainer_kwargs, "processing_class": tokenizer})
    except Exception as exc:  # noqa: BLE001 — classified below
        if not _dpo_processing_retryable(exc):
            raise
        first_error = f"{type(exc).__name__}: {exc}"
        print(
            f"dpo: processing_class=tokenizer rejected ({first_error}); "
            "retrying once with a real processor (VLM route)",
            flush=True,
        )
    else:
        route = {
            **metadata,
            "route": "processing_class_tokenizer_text_patch",
            "model_type_patched_during_init": text_patch_active,
        }
        print(
            f"dpo: processing_class=tokenizer ({type(tokenizer).__name__}); "
            f"text route (model_type patch "
            f"{'active' if text_patch_active else 'inactive'})",
            flush=True,
        )
        return trainer, route

    # ---- attempt 2: real processor, VLM route ----------------------------- #
    try:
        processor = _processor_for_dpo(tokenizer, base_revision)
        from datasets import Dataset

        def with_images(pairs: list[dict[str, str]]) -> Any:
            return Dataset.from_list(
                [
                    {
                        "prompt": pair["prompt"],
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                        "images": [],
                    }
                    for pair in pairs
                ]
            )

        trainer = trainer_cls(
            **{
                **trainer_kwargs,
                "train_dataset": with_images(train_pairs),
                "eval_dataset": with_images(eval_pairs),
                "processing_class": processor,
            }
        )
    except Exception as exc:  # noqa: BLE001 — combined, re-raised with context
        raise RuntimeError(
            "DPOTrainer construction failed with processing_class=tokenizer "
            f"({first_error}) and with a real processor "
            f"({type(exc).__name__}: {exc}). Install a TRL build whose "
            "DPOTrainer accepts a PreTrainedTokenizerBase processing_class "
            "for text-only pairs."
        ) from exc
    route = {
        **metadata,
        "route": "processing_class_processor_vlm",
        "model_type_patched_during_init": False,
        "processor_class": type(processor).__name__,
    }
    print(
        f"dpo: processing_class=processor ({type(processor).__name__}) with "
        "the patched tokenizer grafted in (VLM route)",
        flush=True,
    )
    return trainer, route


def apply_dpo_warmup_steps(
    kwargs: dict[str, Any],
    config_cls: type,
    *,
    steps_per_epoch: int,
    num_train_epochs: float,
) -> dict[str, Any]:
    parameters = inspect.signature(config_cls).parameters
    warmup_ratio = kwargs.get("warmup_ratio")
    if warmup_ratio is None or "warmup_steps" not in parameters:
        return kwargs
    updated = dict(kwargs)
    total_steps = max(1, int(steps_per_epoch * num_train_epochs))
    updated["warmup_steps"] = max(1, int(total_steps * float(warmup_ratio)))
    updated.pop("warmup_ratio", None)
    return updated


def _dpo_tokenizer_from_processing(processing_class: Any) -> Any:
    """HF text tokenizer behind a trainer's processing_class.

    FIX: ``PreTrainedTokenizerFast.tokenizer`` is the *Rust backend*, not an
    HF tokenizer — only unwrap when the inner object is a real tokenizer
    (i.e. the processing class is a processor or wrapper); otherwise return
    the processing class itself.
    """
    inner = getattr(processing_class, "tokenizer", None)
    if inner is not None and _is_plain_tokenizer(inner):
        return inner
    return processing_class


# --------------------------------------------------------------------------- #
# post-construction tokenization verification (double-BOS guard)
# --------------------------------------------------------------------------- #

_PROMPT_ID_COLUMNS = ("prompt_input_ids", "prompt_ids")


def _strip_double_bos_row(
    row: dict[str, Any], column: str, bos_token_id: int
) -> dict[str, Any]:
    ids = list(row[column])
    if len(ids) >= 2 and ids[0] == bos_token_id and ids[1] == bos_token_id:
        return {column: ids[1:]}
    return {column: ids}


def verify_trainer_tokenization(
    trainer: Any,
    tokenizer: Any,
    train_pairs: list[dict[str, str]],
    metadata: dict[str, Any],
) -> None:
    """Verify the trainer's tokenized first row matches the rendered pair.

    Some TRL builds prepend ``bos_token_id`` to ``prompt_input_ids`` even
    when the prompt text already starts with ``<bos>`` from the chat
    template — that would train on a double-BOS prefix (train/inference
    mismatch). Detect it and, if present, strip the duplicate leading BOS
    from every train/eval row.
    """
    if not train_pairs:
        raise RuntimeError("DPO tokenization verification requires at least one train pair")
    dataset = getattr(trainer, "train_dataset", None)
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("DPO tokenization verification requires a prepared train dataset")
    row = dataset[0]
    column = next((name for name in _PROMPT_ID_COLUMNS if name in row), None)
    if column is None:
        metadata["tokenization_check"] = {
            "status": "no_prompt_ids_column",
            "columns": sorted(str(key) for key in row),
        }
        print(
            "dpo: warning: no prompt id column found; tokenization check skipped",
            flush=True,
        )
        return
    inner_tokenizer = model.text_tokenizer(tokenizer)
    bos = getattr(inner_tokenizer, "bos_token_id", None)
    actual = [int(value) for value in row[column]]
    expected = model.encode_ids(inner_tokenizer, train_pairs[0]["prompt"])
    double_bos = bos is not None and actual[:2] == [bos, bos]
    comparison = actual[1:] if double_bos else actual
    shared = min(len(comparison), len(expected))
    if shared and comparison[:shared] != expected[:shared]:
        raise RuntimeError(
            "DPO trainer tokenization does not match the rendered prompt "
            f"(column={column}, actual_head={actual[:10]}, "
            f"expected_head={expected[:10]}); refusing to train on "
            "unexpectedly tokenized pairs"
        )
    if not double_bos:
        metadata["tokenization_check"] = {
            "status": "ok",
            "column": column,
            "prompt_tokens": len(actual),
        }
        return
    print(
        "dpo: TRL prepended a BOS to the already-<bos>-prefixed prompt; "
        "stripping the duplicate from every train/eval row",
        flush=True,
    )
    for attribute in ("train_dataset", "eval_dataset"):
        current = getattr(trainer, attribute, None)
        if current is None:
            continue
        names = getattr(current, "column_names", None) or []
        if column not in names:
            continue
        setattr(
            trainer,
            attribute,
            current.map(
                _strip_double_bos_row,
                fn_kwargs={"column": column, "bos_token_id": int(bos)},
            ),
        )
    metadata["tokenization_check"] = {
        "status": "double_bos_repaired",
        "column": column,
        "prompt_tokens": len(actual) - 1,
    }


def smoke_check_dpo_tokenization(
    trainer: Any,
    train_pairs: list[dict[str, str]],
    processing_metadata: dict[str, Any],
) -> None:
    if not train_pairs:
        raise RuntimeError("DPO smoke check requires at least one train pair")
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is None:
        raise RuntimeError("DPO smoke check requires trainer.processing_class")
    tokenizer = _dpo_tokenizer_from_processing(processing_class)
    row = train_pairs[0]
    failures: list[str] = []
    for key in ("prompt", "chosen", "rejected"):
        text = str(row.get(key, "")).strip()
        if not text:
            failures.append(f"{key}: empty")
            continue
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if not input_ids:
                failures.append(f"{key}: zero-length input_ids")
        except Exception as exc:  # noqa: BLE001 - surface tokenizer failures early
            failures.append(f"{key}: {exc!r}")
    if failures:
        raise RuntimeError(
            "DPO tokenization smoke check failed: "
            f"{failures}; processing={processing_metadata}"
        )
    print("dpo: tokenization smoke check passed", flush=True)


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
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None  # type: ignore[assignment]
    if PeftModel is not None and not isinstance(loaded, PeftModel):
        print(
            "dpo: warning: the resumed model is not a PeftModel; DPOTrainer "
            "will build a full reference copy (extra VRAM) instead of using "
            "the adapter-disabled reference",
            flush=True,
        )

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
            # Drop over-length pairs up front instead of letting TRL truncate:
            # truncating the prompt with truncation_mode="keep_start" cuts the
            # tail — exactly where the <start_of_turn>model generation header
            # lives — and truncating the completion teaches stop-less endings.
            if len(prompt_ids) > config.DPO_MAX_PROMPT_LENGTH:
                quality["prompt_too_long"] += 1
                continue
            if longest > config.DPO_MAX_COMPLETION_LENGTH:
                quality["completion_too_long"] += 1
                continue
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
            "max_prompt_length": config.DPO_MAX_PROMPT_LENGTH,
            "max_completion_length": config.DPO_MAX_COMPLETION_LENGTH,
            "languages": dict(
                Counter(str(row.get("language", "unknown")) for row in rows)
            ),
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

    def pair_columns(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
        # Hand TRL exactly prompt/chosen/rejected — the extra bookkeeping
        # columns are for our reports only and confuse some TRL collators.
        return [
            {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            for pair in pairs
        ]

    train_dataset = Dataset.from_list(pair_columns(train_pairs))
    eval_dataset = Dataset.from_list(pair_columns(eval_pairs))

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
    dpo_config_kwargs = apply_dpo_warmup_steps(
        dpo_config_kwargs,
        DPOConfig,
        steps_per_epoch=steps_per_epoch,
        num_train_epochs=config.DPO_EPOCHS,
    )
    dpo_intended = dict(dpo_config_kwargs)
    config_kwargs, trainer_dpo_kwargs, overflow_routing = split_dpo_kwargs(
        dpo_config_kwargs,
        DPOConfig,
        DPOTrainer,
    )
    dpo_args, runtime_routing = build_dpo_config(DPOConfig, config_kwargs)
    overflow_routing.update(runtime_routing)
    trainer_kwargs: dict[str, Any] = {
        "model": loaded,
        "ref_model": None,  # PEFT: reference = adapter disabled
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": dpo_args,
        **trainer_dpo_kwargs,
    }
    trainer, processing_metadata = construct_dpo_trainer(
        DPOTrainer,
        trainer_kwargs,
        loaded,
        tokenizer,
        base_revision,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
    )
    dpo_routing = verify_dpo_hparams(dpo_args, trainer, dpo_intended, overflow_routing)
    verify_trainer_tokenization(trainer, tokenizer, train_pairs, processing_metadata)
    smoke_check_dpo_tokenization(trainer, train_pairs, processing_metadata)

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
                for key, value in dpo_intended.items()
                if key in DPO_MANIFEST_KEYS
            },
            "hyperparameter_routing": dpo_routing,
            "processing": processing_metadata,
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