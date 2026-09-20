#!/usr/bin/env python3
"""Export the merged model onto the QAT int4 (Q4_0) lattice before GGUF conversion.

NEW STAGE (review 2026-09, finding F3). FIXED 2026-09-21 (rev 3).

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
  2. Fake-quantize language-model linears + token/per-layer embeddings to
     ggml's q4_0 lattice, replicating ``quantize_row_q4_0_ref`` EXACTLY
     (scale d = signed_max/-8 stored as f16, codes floor(x/d + 8.5) clamped
     to 15, dequant q*d).
  3. Save HF weights (values sitting ON the lattice), sanitize tokenizer
     control-token flags (review F8), convert with ``--outtype f32``, then
     ``llama-quantize q4_0 --pure`` recovers the same scale and codes.

Norms stay F32 (llama.cpp never quantizes them). Non-divisible tensors are
skipped with a warning and recorded in the manifest.

Fix history
-----------
rev 1 (2026-09-21):
  * ``patch_kv_sharing`` applied to the merged load — the merged checkpoint
    omits the KV-shared K/V weights (layers 15-34); transformers materializes
    random ones, which must be stripped before export.
  * ggml-exact q4_0 reference quantizer (previous symmetric variant picked
    the opposite clamp side / scale sign, so ``--pure`` did not round-trip).
  * ``sanitize_tokenizer`` EOG fix (<eos> id resolved from
    added_tokens_decoder, not the non-existent ``eos_token["id"]``).

rev 2 (2026-09-21) — the load-report abort:
  merged_bf16/ is a TEXT-ONLY checkpoint: 03_reference.py drops the
  vision/audio towers before saving. Re-instantiating the full
  Gemma4ForConditionalGeneration architecture therefore reports every tower
  parameter as MISSING (newly initialized). The loading-key gate now
  classifies instead of rejecting: modality-tower keys are EXPECTED and
  ignored (never quantized, never exported), TEXT-stack keys remain strictly
  validated, and a positive completeness check verifies every text weight in
  merged_bf16/ actually landed on the loaded model.

rev 3 (2026-09-21) — the validate_kv_sharing abort:
  ``model.SHARED_KV_STATE`` matches ``.layers.(15-34)…(k_proj|v_proj|k_norm)``
  WITHOUT anchoring to the language model. The vision tower has 16 layers
  (0-15), so ``vision_tower.encoder.layers.15.self_attn.k_*`` collided with
  the shared-KV range and ``validate_kv_sharing`` flagged the (irrelevant,
  text-only-export) tower params as "materialized but absent from the
  checkpoint". In 02_sft/03_reference this cannot happen because their
  checkpoints (the base repo) contain the vision weights. Fix: the modality
  towers are now dropped BEFORE ``validate_kv_sharing``, so the state_dict
  under validation is already text-only, matching merged_bf16/'s semantics.
  No model.py change is required (and would be wrong for the other stages).

Run AFTER 03_reference.py. Produces reference_qat/kuza-qat-q4_0-lattice.gguf,
which 05_quants.py includes as the ``q4_0_qat_export`` candidate as-is.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path

import model
from common import (
    gguf_inventory,
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

# Modality submodules that 03_reference.py deliberately drops before saving
# merged_bf16/. When this stage re-instantiates the full multimodal
# architecture, transformers materializes them with random weights and lists
# every parameter as MISSING in the load report. This export is TEXT-ONLY:
# the towers are never quantized, never used, and dropped below — their
# absence from the checkpoint is EXPECTED, not an error.
DROPPED_MODALITY_MODULES = (
    "vision_tower",
    "audio_tower",
    "multi_modal_projector",
    "vision_model",
    "audio_model",
    "vision_encoder",
    "audio_encoder",
    "mmproj",
)
DROPPED_MODALITY_ATTRS = (
    "vision_tower", "audio_tower", "multi_modal_projector",
    "vision_model", "audio_model", "vision_encoder", "audio_encoder",
)
_DROPPED_MODALITY_RE = re.compile(
    r"(?:^|\.)(?:" + "|".join(DROPPED_MODALITY_MODULES) + r")\."
)


def _is_dropped_modality_key(key: str) -> bool:
    return bool(_DROPPED_MODALITY_RE.search(key))


def _modality_keys(keys: list[str]) -> list[str]:
    return [key for key in keys if _is_dropped_modality_key(key)]


def _text_stack_violations(keys: list[str]) -> list[str]:
    """Loading keys that are neither expected-exempt nor dropped modality."""
    return [
        key
        for key in keys
        if not model.is_expected_loading_key(key)
        and not _is_dropped_modality_key(key)
    ]


def _drop_modality_towers(loaded) -> list[str]:
    """Set the vision/audio towers to None, removing them from state_dict().

    Called BEFORE ``validate_kv_sharing`` (rev 3): the shared-KV regex in
    model.py is not anchored to the language model, and the vision tower's
    layer 15 collides with the language model's shared-KV range (15-34), so
    a text-only checkpoint + intact towers false-positives as "materialized
    KV-shared tensors absent from the checkpoint".
    """
    dropped: list[str] = []
    for attr in DROPPED_MODALITY_ATTRS:
        for obj in (loaded, getattr(loaded, "model", None)):
            if obj is None or getattr(obj, attr, None) is None:
                continue
            setattr(obj, attr, None)
            dropped.append(attr)
    return dropped


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


def _safetensors_keys(directory: Path) -> set[str]:
    from safetensors import safe_open

    keys: set[str] = set()
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    if not keys:
        raise RuntimeError(f"No safetensors tensors found in {directory}")
    return keys


def fake_quant_q4_0(tensor, name: str) -> tuple[object, dict]:
    """Return a lattice-snapped copy plus error stats. No in-place mutation.

    Replicates ggml's reference q4_0 quantizer EXACTLY
    (quantize_row_q4_0_ref / dequantize_row_q4_0 in ggml-quants.c):

      * the block scale d is derived from the *signed* value with the
        largest magnitude (first occurrence), d = signed_max / -8 — NOT from
        amax/8;
      * codes are computed against the f32 scale (floor(x*(1/d) + 8.5),
        clamped at 15) while dequantization uses the f16-stored scale —
        ggml's own quirk, replicated so the round trip through
        ``llama-quantize --pure`` is stable.

    (Vectorized SIMD kernels may differ by one code on exact-half values;
    the scale derivation and clamp side match, which is what the lattice
    claim depends on.)
    """
    import torch

    w = tensor.detach().to(torch.float32)
    stats = {"name": name, "skipped": False}
    if w.shape[-1] % LATTICE_BLOCK != 0:
        stats["skipped"] = True
        stats["reason"] = f"last dim {w.shape[-1]} not divisible by {LATTICE_BLOCK}"
        return tensor, stats
    blocks = w.reshape(-1, LATTICE_BLOCK)
    abs_blocks = blocks.abs()
    amax = abs_blocks.max(dim=1).values
    amax_idx = abs_blocks.argmax(dim=1, keepdim=True)   # first occurrence, like ggml
    signed_max = blocks.gather(1, amax_idx).squeeze(1)  # value with largest |w|
    d = signed_max / -8.0                               # f32 scale used for the codes
    d_stored = d.to(torch.float16).to(torch.float32)    # block scale is stored as f16
    id_ = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    xi = torch.clamp(torch.floor(blocks * id_.unsqueeze(1) + 8.5), max=15.0)
    q = xi - 8.0                                        # codes in [-8, 7]
    deq = (q * d_stored.unsqueeze(1)).reshape(w.shape)
    err = (deq - w).abs()
    stats.update(
        {
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "zero_scale_blocks": int((amax == 0).sum()),
        }
    )
    return deq.to(tensor.dtype), stats


def sanitize_tokenizer(directory: Path) -> dict:
    """Declare control-looking tokens as special; fix EOG lists (review F8).

    llama.cpp warns on every load otherwise ("control-looking token ... will
    be overridden") — that correction is version-dependent, so we bake the
    correct declaration into the artifacts instead.

    FIX 2026-09-21: the previous EOG logic read ``eos_token["id"]`` from
    tokenizer_config.json, but that entry carries ``content`` (not ``id``),
    so <eos> was silently dropped from generation_config.eos_token_id and
    only <end_of_turn> survived. Ids are now resolved via
    added_tokens_decoder content lookups.
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
        eos_token = tok.get("eos_token")
        eos_contents = {"<eos>"}
        if isinstance(eos_token, dict) and eos_token.get("content"):
            eos_contents.add(str(eos_token["content"]))
        elif isinstance(eos_token, str) and eos_token:
            eos_contents.add(eos_token)
        eos_candidates: list[int] = []
        end_of_turn = None
        for token_id, entry in tok.get("added_tokens_decoder", {}).items():
            content = str(entry.get("content", ""))
            if content == "<end_of_turn>":
                end_of_turn = int(token_id)
            if content in eos_contents:
                eos_candidates.append(int(token_id))
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
    shutil.rmtree(lattice_dir, ignore_errors=True)  # idempotent re-runs
    lattice_dir.mkdir(parents=True, exist_ok=True)

    print("loading merged BF16 model for lattice export", flush=True)
    result = AutoModelForImageTextToText.from_pretrained(
        merged,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if isinstance(result, tuple) and len(result) == 2:
        loaded, loading_info = result
    else:  # transformers build without loading_info support
        loaded, loading_info = result, None

    def _loading_keys(field: str) -> list[str]:
        if isinstance(loading_info, dict):
            return list(loading_info.get(field, []) or [])
        return list(getattr(loading_info, field, []) or [])

    # ---- loading-key gate with TEXT-ONLY semantics (rev 2) ---------------- #
    # merged_bf16/ omits the modality towers BY DESIGN (03_reference drops
    # them); their "MISSING" entries are expected and ignored. Anything in
    # the TEXT stack that fails to load is still a hard error.
    missing_modality = _modality_keys(_loading_keys("missing_keys"))
    unexpected_modality = _modality_keys(_loading_keys("unexpected_keys"))
    bad_missing = _text_stack_violations(_loading_keys("missing_keys"))
    bad_unexpected = _text_stack_violations(_loading_keys("unexpected_keys"))
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "The TEXT stack did not load cleanly from the merged model "
            f"(missing={bad_missing[:20]}, unexpected={bad_unexpected[:20]}). "
            "Re-run 03_reference.py to rebuild merged_bf16/."
        )
    if missing_modality:
        print(
            f"text-only export: {len(missing_modality)} vision/audio parameters "
            "absent from the merged checkpoint (03_reference.py drops the "
            "modality towers) — expected for a text-only export",
            flush=True,
        )
    if unexpected_modality:
        print(
            f"text-only export: ignoring {len(unexpected_modality)} unused "
            "modality keys reported by the loader",
            flush=True,
        )

    # ---- KV sharing: strip the re-materialized shared K/V modules --------- #
    # The merged checkpoint omits the language model's KV-shared K/V weights
    # (layers 15-34); transformers materializes random tensors for them on
    # load. patch_kv_sharing skips the vision/audio towers itself, so it is
    # safe to call with the towers still attached.
    kv_patch = model.patch_kv_sharing(loaded)
    merged_keys = _safetensors_keys(merged)
    shared_in_checkpoint = sorted(
        key for key in merged_keys if model.SHARED_KV_STATE.search(key)
    )
    if shared_in_checkpoint:
        raise RuntimeError(
            "Merged checkpoint unexpectedly contains KV-shared tensors: "
            f"{shared_in_checkpoint[:10]}"
        )

    # ---- text-only: drop the towers BEFORE KV validation (rev 3) --------- #
    # model.SHARED_KV_STATE is not anchored to the language model, and the
    # vision tower's layer 15 collides with the language shared-KV range
    # (15-34) — validating a text-only checkpoint against a model that still
    # carries the towers false-positives on vision_tower...layers.15.k_*.
    # After this drop, state_dict() is text-only and matches merged_bf16/'s
    # semantics exactly.
    dropped = _drop_modality_towers(loaded)
    if missing_modality and not dropped:
        raise RuntimeError(
            "The loader reported missing modality parameters but no modality "
            "towers were found to drop — architecture mismatch"
        )
    print(
        f"text-only export: dropped modality towers {sorted(set(dropped)) or '(none present)'}",
        flush=True,
    )

    model.validate_kv_sharing(loaded, merged_keys)

    # ---- positive text-stack completeness check --------------------------- #
    # Every text weight in the checkpoint must actually be present on the
    # loaded model. (Expected-exempt keys — tied embeddings, inv_freq — and
    # modality keys are excluded; shared-KV presence in the checkpoint was
    # already ruled out above, so the exemption cannot mask anything.)
    state_keys = set(loaded.state_dict().keys())
    absent_text = sorted(
        key
        for key in merged_keys
        if key not in state_keys
        and not model.is_expected_loading_key(key)
        and not _is_dropped_modality_key(key)
    )
    if absent_text:
        raise RuntimeError(
            "Text-stack weights from merged_bf16/ did not land on the loaded "
            f"model: {absent_text[:20]}"
        )

    # ---- lattice snap ------------------------------------------------------ #
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
    if quantized < 100:
        raise RuntimeError(
            f"Lattice snap touched only {quantized} tensors — the name filters "
            "no longer match this architecture"
        )
    print(f"lattice export: quantized={quantized} skipped={len(skipped)} max_err={err_max:.6f}", flush=True)

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
            "quantizer": "ggml_q4_0_ref_replica",
            "quantizer_note": (
                "scale = signed_max/-8 (f16-stored), codes floor(x/d+8.5) "
                "clamped to 15; SIMD kernels may differ by one code on "
                "exact-half values"
            ),
            "kv_sharing_patch": kv_patch,
            "text_only": True,
            "text_stack_verified": True,
            "modality_params_ignored": len(missing_modality),
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