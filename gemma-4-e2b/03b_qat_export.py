#!/usr/bin/env python3
"""Export the merged model onto the QAT int4 (Q4_0) lattice before GGUF conversion.

NEW STAGE (review 2026-09, finding F3).

Why this exists
---------------
The base checkpoint is QAT-trained for the int4 lattice, and 02_sft.py keeps
the base fake-quantized during SFT (unsloth qat_scheme="int4"). But
03_reference.py calls ``merge_and_unload()`` which writes BF16 ``W + BA`` —
the LoRA delta pushes weights OFF the lattice, and ``llama-quantize`` then
re-introduces a fresh round of PTQ error on shifted weights (QA-LoRA, Xu et
al. 2023, exists precisely for this failure mode).

This stage re-applies the target lattice to the merged weights BEFORE
conversion:

  1. Load ``merged_bf16/`` (produced by 03_reference.py).
  2. Fake-quantize language-model linears + token/per-layer embeddings to the
     exact Q4_0 lattice: block-32, scale d = max|w|/8 stored as f16,
     q = roundggml(w/d) in [-8, 7], dequant w' = d*q. (ggml roundf rounds
     half-away-from-zero; torch.round rounds half-to-even — replicated.)
  3. Save HF weights (bf16 values sitting ON the lattice), sanitize tokenizer
     control-token flags (review F8), convert with ``--outtype f32`` (bf16
     values are a subset of f32, so upcast is exact), then
     ``llama-quantize q4_0 --pure`` recovers d and q EXACTLY — the resulting
     GGUF contains the same weights the network saw during QAT+SFT.

Norms stay F32 (llama.cpp never quantizes them). Non-divisible tensors are
skipped with a warning and recorded in the manifest.

Run AFTER 03_reference.py. Produces reference_qat/kuza-qat-q4_0-lattice.gguf,
which 05_quants.py includes as the ``q4_0_qat_export`` candidate as-is.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path

import model
from common import (
    gguf_inventory,
    help_has,
    hf_token,
    llama_cpp_binaries,
    require_file,
    run,
    run_dir,
    sha256_file,
    smoke_load,
    write_json,
)

LATTICE_BLOCK = 32          # q4_0 block size
LATTICE_LOW = -8            # q4_0 min code
LATTICE_HIGH = 7            # q4_0 max code

# Fake-quant applies to language-model linears and (per-layer) embeddings.
LINEAR_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
EMBED_SUFFIXES = ("embed_tokens", "per_layer_token_embd")


def _is_language_linear(name: str) -> bool:
    if "language_model" not in name:
        return False
    leaf = name.removesuffix(".linear").rsplit(".", 1)[-1]
    return leaf in LINEAR_SUFFIXES


def _is_embedding(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return (
        leaf in EMBED_SUFFIXES
        and "language_model" in name
        and not model.SHARED_KV_STATE.search(name + ".")
    )


def fake_quant_q4_0(tensor, name: str) -> tuple[object, dict]:
    """Return a lattice-snapped copy plus error stats. No in-place mutation."""
    import torch

    w = tensor.detach().to(torch.float32)
    stats = {"name": name, "skipped": False}
    rows = w.numel()
    if w.shape[-1] % LATTICE_BLOCK != 0:
        stats["skipped"] = True
        stats["reason"] = f"last dim {w.shape[-1]} not divisible by {LATTICE_BLOCK}"
        return tensor, stats
    blocks = w.reshape(-1, LATTICE_BLOCK)
    d = blocks.abs().amax(dim=1) / 8.0
    d = d.to(torch.float16).to(torch.float32)          # q4_0 scale is f16
    d = torch.where(d == 0, torch.ones_like(d), d)     # guard all-zero blocks
    x = blocks / d.unsqueeze(1)
    # ggml roundf: half away from zero (torch.round is half-to-even)
    q = torch.sign(x) * torch.floor(torch.abs(x) + 0.5)
    q = torch.clamp(q, LATTICE_LOW, LATTICE_HIGH)
    deq = (q * d.unsqueeze(1)).reshape(w.shape)
    err = (deq - w).abs()
    stats.update(
        {
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "zero_scale_blocks": int((blocks.abs().amax(dim=1) == 0).sum()),
        }
    )
    return deq.to(tensor.dtype), stats


def sanitize_tokenizer(directory: Path) -> dict:
    """Declare control-looking tokens as special; fix EOG lists (review F8).

    llama.cpp warns on every load otherwise ("control-looking token ... will
    be overridden") — that correction is version-dependent, so we bake the
    correct declaration into the artifacts instead.
    """
    report: dict = {"patched_tokens": [], "generation_eos": None}
    cfg_path = directory / "tokenizer_config.json"
    gen_path = directory / "generation_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        added = cfg.get("added_tokens_decoder", {})
        control_like = {"</s>", "<|tool_response>", "<|tool_call>", "<|image>", "<|video>"}
        for token_id, entry in added.items():
            content = str(entry.get("content", ""))
            if content in control_like and not entry.get("special"):
                entry["special"] = True
                entry["normalized"] = False
                report["patched_tokens"].append({"id": token_id, "content": content})
        if report["patched_tokens"]:
            cfg["added_tokens_decoder"] = added
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if gen_path.is_file():
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        tok_path = directory / "tokenizer_config.json"
        tok = json.loads(tok_path.read_text(encoding="utf-8")) if tok_path.is_file() else {}
        # Keep the canonical stopping pair: <eos> and <end_of_turn>.
        eos_candidates = []
        for name in ("eos_token",):
            token = tok.get(name)
            if isinstance(token, dict) and token.get("id") is not None:
                eos_candidates.append(int(token["id"]))
        end_of_turn = None
        for token_id, entry in tok.get("added_tokens_decoder", {}).items():
            if entry.get("content") == "<end_of_turn>":
                end_of_turn = int(token_id)
        if end_of_turn is not None:
            eos_candidates.append(end_of_turn)
        if eos_candidates:
            gen["eos_token_id"] = sorted(set(eos_candidates))
            gen.pop("special_eog_ids", None)
            gen_path.write_text(json.dumps(gen, indent=2), encoding="utf-8")
            report["generation_eos"] = gen["eos_token_id"]
    return report


def main() -> int:
    hf_token()
    merged = run_dir() / "merged_bf16"
    require_file(merged / "config.json", "Run 03_reference.py first (missing merged_bf16).")
    binaries = llama_cpp_binaries()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    dest_dir = run_dir() / "reference_qat"
    dest_dir.mkdir(parents=True, exist_ok=True)
    lattice_dir = dest_dir / "merged_qat_lattice"
    lattice_dir.mkdir(parents=True, exist_ok=True)

    print("loading merged BF16 model for lattice export", flush=True)
    loaded = AutoModelForImageTextToText.from_pretrained(
        merged, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    quantized = 0
    skipped: list[dict] = []
    err_max = 0.0
    with torch.no_grad():
        for name, parameter in list(loaded.named_parameters()):
            if model.SHARED_KV_STATE.search(name):
                skipped.append({"name": name, "reason": "kv-shared (absent from checkpoint)"})
                continue
            if _is_language_linear(name) or _is_embedding(name):
                snapped, stats = fake_quant_q4_0(parameter.data, name)
                if stats.get("skipped"):
                    skipped.append(stats)
                    continue
                parameter.data.copy_(snapped)
                quantized += 1
                err_max = max(err_max, stats["max_abs_err"])
    print(f"lattice export: quantized={quantized} skipped={len(skipped)} max_err={err_max:.6f}", flush=True)

    # Text-only: drop vision/audio towers exactly like 03_reference.py.
    dropped = []
    for attr in (
        "vision_tower", "audio_tower", "multi_modal_projector",
        "vision_model", "audio_model", "vision_encoder", "audio_encoder",
    ):
        for obj in (loaded, getattr(loaded, "model", None)):
            if obj is None or getattr(obj, attr, None) is None:
                continue
            setattr(obj, attr, None)
            dropped.append(attr)

    tokenizer = AutoTokenizer.from_pretrained(merged)
    tokenizer.save_pretrained(lattice_dir)
    loaded.config.text_config._attn_implementation = "eager"  # conversion-safe
    loaded.save_pretrained(lattice_dir, safe_serialization=True, max_shard_size="5GB")
    sanitize_report = sanitize_tokenizer(lattice_dir)
    del loaded
    gc.collect()
    torch.cuda.empty_cache()

    env = os.environ.copy()
    llama_root = binaries["converter"].parent
    env["PYTHONPATH"] = str(llama_root / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    f32_path = dest_dir / "kuza-qat-f32.gguf"
    final_path = dest_dir / "kuza-qat-q4_0-lattice.gguf"
    run(
        [
            sys.executable, binaries["converter"], lattice_dir,
            "--outfile", f32_path, "--outtype", "f32",
        ],
        cwd=llama_root,
        env=env,
        log_path=dest_dir / "convert.log",
    )
    run(
        [
            binaries["llama-quantize"], "--pure",
            f32_path, final_path, "q4_0",
        ],
        log_path=dest_dir / "quantize.log",
    )
    f32_path.unlink(missing_ok=True)  # 18GB intermediate; keep the workspace lean

    inventory = gguf_inventory(final_path, llama_root)
    if inventory["tensor_type_counts"].get("Q4_0", 0) < 200:
        raise RuntimeError(
            f"Lattice export looks wrong: {inventory['tensor_type_counts']}"
        )
    model.assert_text_only_gguf(inventory)

    smoke_prompt = model.render_generation_prompt(
        model.load_tokenizer(merged),
        "How should I space maize in a dry season?",
    )
    smoke_load(
        binaries["llama-cli"], final_path, smoke_prompt, dest_dir / "smoke.log"
    )

    write_json(
        dest_dir / "qat_export_manifest.json",
        {
            "lattice": "q4_0",
            "block": LATTICE_BLOCK,
            "code_range": [LATTICE_LOW, LATTICE_HIGH],
            "quantized_tensors": quantized,
            "max_abs_err": err_max,
            "skipped_tensors": skipped[:40],
            "dropped_modalities": sorted(set(dropped)),
            "tokenizer_sanitizer": sanitize_report,
            **{key: value for key, value in inventory.items() if key != "tensors"},
            "sha256": sha256_file(final_path),
            "path": str(final_path),
        },
    )
    print(f"qat lattice reference: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
