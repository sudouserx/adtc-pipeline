#!/usr/bin/env python3
"""Quantize the BF16 reference into the model-defined GGUF candidates."""

from __future__ import annotations

import model
from common import (
    adapter_dir,
    candidate_gguf,
    gguf_inventory,
    hf_token,
    imatrix_path,
    llama_cpp_binaries,
    quants_dir,
    reference_path,
    require_file,
    run,
    sha256_file,
    smoke_load,
    write_json,
)


def main() -> int:
    hf_token()
    require_file(adapter_dir() / "adapter_config.json", "Run 02_sft.py first.")
    require_file(reference_path(), "Run 03_reference.py first.")
    require_file(imatrix_path(), "Run 04_imatrix.py first.")
    binaries = llama_cpp_binaries()
    tokenizer = model.load_tokenizer(adapter_dir())
    smoke_prompt = model.render_generation_prompt(
        tokenizer,
        "Nipe hatua za kuchunguza majani ya mahindi yenye rangi ya njano.",
    )
    root = quants_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in model.QUANT_CANDIDATES.items():
        output = candidate_gguf(name)
        candidate_dir = output.parent
        candidate_dir.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        run(
            model.quant_command(
                binaries["llama-quantize"],
                reference_path(),
                output,
                imatrix_path(),
                spec,
            ),
            log_path=root / f"{name}.log",
        )
        inventory = gguf_inventory(output, binaries["converter"].parent)
        model.validate_quant_overrides(inventory, spec)
        model.assert_text_only_gguf(inventory)
        smoke_load(
            binaries["llama-cli"],
            output,
            smoke_prompt,
            root / f"{name}-smoke.log",
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
                "imatrix_sha256": sha256_file(imatrix_path()),
                "reference_sha256": sha256_file(reference_path()),
                **{key: value for key, value in inventory.items() if key != "tensors"},
            },
        )
        print(f"{name}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
