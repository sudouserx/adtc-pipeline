#!/usr/bin/env python3
"""Quantize new candidate GGUFs from the BF16 reference and domain imatrix."""

from __future__ import annotations

import study_config as config
import paths
import quant_specs
from study_common import (
    gguf_inventory,
    hf_token,
    llama_cpp_binaries,
    load_tokenizer,
    pipeline_model,
    require_file,
    run,
    sha256_file,
    smoke_load,
    write_json,
)
from quant_specs import quant_command, validate_quant_overrides


def main() -> int:
    hf_token()
    require_file(paths.reference_path(), "Run 01_download.py first.")
    require_file(paths.imatrix_path(), "Run 01_download.py first.")
    binaries = llama_cpp_binaries()
    tokenizer = load_tokenizer()
    smoke_prompt = pipeline_model.render_generation_prompt(
        tokenizer,
        "Nipe hatua za kuchunguza majani ya mahindi yenye rangi ya njano.",
    )
    config.QUANTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, spec in quant_specs.NEW_QUANT_CANDIDATES.items():
        output = quant_specs.gguf_path(name, spec)
        candidate_dir = output.parent
        candidate_dir.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        run(
            quant_command(
                binaries["llama-quantize"],
                paths.reference_path(),
                output,
                paths.imatrix_path(),
                spec,
            ),
            log_path=config.QUANTS_DIR / f"{name}.log",
        )
        inventory = gguf_inventory(output, binaries["converter"].parent)
        pipeline_model.assert_text_only_gguf(inventory)
        if any(
            spec.get(key)
            for key in (
                "embedding_type",
                "output_type",
                "ssm_out_type",
                "ssm_gate_type",
                "attention_type",
                "attn_gate_type",
                "bulk_type",
            )
        ):
            validate_quant_overrides(inventory, spec)
        smoke_load(
            binaries["llama-cli"],
            output,
            smoke_prompt,
            config.QUANTS_DIR / f"{name}-smoke.log",
        )
        write_json(
            candidate_dir / "recipe.json",
            {
                "name": name,
                "base_type": spec["base_type"],
                "embedding_type": spec.get("embedding_type"),
                "output_type": spec.get("output_type"),
                "ssm_out_type": spec.get("ssm_out_type"),
                "ssm_gate_type": spec.get("ssm_gate_type"),
                "attention_type": spec.get("attention_type"),
                "attn_gate_type": spec.get("attn_gate_type"),
                "bulk_type": spec.get("bulk_type"),
                "notes": spec.get("notes"),
                "imatrix_sha256": sha256_file(paths.imatrix_path()),
                "reference_sha256": sha256_file(paths.reference_path()),
                **{key: value for key, value in inventory.items() if key != "tensors"},
            },
        )
        print(f"{name}: {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
