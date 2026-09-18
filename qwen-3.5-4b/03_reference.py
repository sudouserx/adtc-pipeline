#!/usr/bin/env python3
"""Merge the adapter to BF16 and convert a BF16 GGUF reference."""

from __future__ import annotations

import gc
import os
import shutil
import sys
from pathlib import Path

import model
from common import (
    adapter_dir,
    assert_adapter_tensors_loaded,
    assert_model_bf16,
    cast_floating_to_bf16,
    checkpoint_weight_keys,
    gguf_inventory,
    help_has,
    hf_token,
    inspect_safetensors_bf16,
    llama_cpp_binaries,
    reference_dir,
    reference_path,
    require_file,
    run,
    run_dir,
    sha256_bytes,
    sha256_file,
    smoke_load,
    write_json,
)


def merge_bf16(base_revision: str, merged: Path) -> dict:
    import torch
    from peft import PeftModel

    shutil.rmtree(merged, ignore_errors=True)
    base, loading_info = model.load_merge_base(base_revision)

    def _loading_keys(field: str) -> list[str]:
        if isinstance(loading_info, dict):
            return list(loading_info.get(field, []) or [])
        return list(getattr(loading_info, field, []) or [])

    unexpected_missing = [
        key for key in _loading_keys("missing_keys") if not model.is_expected_loading_key(key)
    ]
    unexpected_present = [
        key
        for key in _loading_keys("unexpected_keys")
        if not model.is_expected_loading_key(key)
    ]
    if unexpected_missing or unexpected_present:
        raise RuntimeError(
            "Unexpected base loading keys during merge: "
            f"missing={unexpected_missing[:20]}, unexpected={unexpected_present[:20]}"
        )
    assert_model_bf16(base, "Merge base model")
    model.validate_kv_sharing(
        base, checkpoint_weight_keys(model.BASE_MODEL, base_revision)
    )
    loaded = PeftModel.from_pretrained(base, adapter_dir(), is_trainable=False)
    assert_adapter_tensors_loaded(loaded, adapter_dir())
    loaded = loaded.merge_and_unload(safe_merge=True)
    assert_model_bf16(loaded, "Merged model")
    dropped_modalities = []
    for attr in (
        "vision_tower",
        "audio_tower",
        "multi_modal_projector",
        "vision_model",
        "audio_model",
        "vision_encoder",
        "audio_encoder",
        "visual",
        "audio",
    ):
        for obj in (loaded, getattr(loaded, "model", None)):
            if obj is None or getattr(obj, attr, None) is None:
                continue
            setattr(obj, attr, None)
            dropped_modalities.append(attr)
    hf_mtp = model.hf_mtp_keys(loaded.state_dict().keys())
    cast_counts = cast_floating_to_bf16(loaded)
    assert_model_bf16(loaded, "Merged text-only model")
    tokenizer = model.load_tokenizer(adapter_dir())
    merged.mkdir(parents=True, exist_ok=True)
    loaded.save_pretrained(
        merged,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(merged)
    write_json(
        merged / "kuza_system_prompt.json",
        {
            "system_prompt": model.SYSTEM_PROMPT,
            "sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
        },
    )
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (merged / name).exists()]
    if missing:
        raise RuntimeError(f"Merged model is missing required files: {missing}")
    dtype_counts = inspect_safetensors_bf16(merged)
    mtp_reconcile = model.reconcile_mtp_config(merged)
    del loaded, base
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "dtype_counts": dtype_counts,
        "dropped_modalities": sorted(set(dropped_modalities)),
        "cast_to_bf16": cast_counts,
        "text_only": True,
        "hf_mtp_tensors": hf_mtp,
        **mtp_reconcile,
    }


def main() -> int:
    hf_token()
    adapter = adapter_dir()
    require_file(adapter / "adapter_config.json", "Run 02_sft.py first.")
    binaries = llama_cpp_binaries()
    dest_dir = reference_dir()
    dest = reference_path()
    dest_dir.mkdir(parents=True, exist_ok=True)
    merged = run_dir() / "merged_bf16"
    base_revision = model.read_base_revision(adapter)
    merge_details = merge_bf16(base_revision, merged)
    env = os.environ.copy()
    llama_root = binaries["converter"].parent
    env["PYTHONPATH"] = (
        str(llama_root / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    )
    convert_cmd = [
        sys.executable,
        binaries["converter"],
        merged,
        "--outfile",
        dest,
        "--outtype",
        "bf16",
    ]
    try:
        converter_help = run(
            [sys.executable, binaries["converter"], "-h"],
            cwd=llama_root,
            env=env,
            capture=True,
        )
    except RuntimeError as exc:
        converter_help = str(exc)
    has_mtp = bool(merge_details.get("has_mtp"))
    convert_args: list[str] = []
    if not has_mtp:
        if help_has(converter_help, "--no-mtp"):
            convert_args = ["--no-mtp"]
        elif help_has(converter_help, "--no-nextn"):
            convert_args = ["--no-nextn"]
    run([*convert_cmd, *convert_args], cwd=llama_root, env=env)
    inventory = gguf_inventory(dest, llama_root)
    tensor_types = {name.upper() for name in inventory["tensor_type_counts"]}
    forbidden_types = tensor_types - {"BF16", "F32"}
    if "BF16" not in tensor_types or "F16" in tensor_types or forbidden_types:
        raise RuntimeError(
            "Reference GGUF failed BF16 precision gate: "
            f"{inventory['tensor_type_counts']}"
        )
    model.assert_text_only_gguf(inventory)
    model.assert_gguf_mtp_consistency(inventory)
    mtp_tensors = model.mtp_tensor_names(inventory["tensors"])
    if has_mtp and not mtp_tensors:
        hf_mtp = merge_details.get("hf_mtp_tensors") or []
        raise RuntimeError(
            "Merged checkpoint has MTP/nextn tensors but the GGUF dropped them: "
            f"{hf_mtp[:20]}"
        )
    tokenizer = model.load_tokenizer(adapter)
    smoke_load(
        binaries["llama-cli"],
        dest,
        model.render_generation_prompt(
            tokenizer, "How should a farmer diagnose yellow maize leaves?"
        ),
        dest_dir / "smoke.log",
    )
    write_json(
        dest_dir / "reference_manifest.json",
        {
            **merge_details,
            **inventory,
            "tensors": None,
            "text_only": True,
            "mmproj": False,
            "mtp_included": has_mtp,
            "mtp_tensors": mtp_tensors,
        },
    )
    write_json(
        dest_dir / "kuza_system_prompt.json",
        {
            "system_prompt": model.SYSTEM_PROMPT,
            "sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
        },
    )
    print(f"merged: {merged}")
    print(f"reference: {dest}")
    print(f"sha256: {sha256_file(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
