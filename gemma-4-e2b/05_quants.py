#!/usr/bin/env python3
"""Stage 05 — Quantize the BF16 reference into the GGUF candidate matrix.

FINAL version (2026-09-20). Replaces 05_quants.py; pairs with config.py
and the UNMODIFIED repo common.py / model.py — candidates are written with a
recipe.json each, which 06_screen.py discovers (no model.py edits).

Patched 2026-09-22: validate_result compared Q4_K_M mixtures against the tensor
type "Q4_K_M" (which never exists), so q4_k_m_default always logged a bogus
"bulk is mostly Q4_K, not Q4_K_M" warning. It now maps Q4_K_M/S/L -> Q4_K.

Candidate matrix (imatrix-fixed, see 04_imatrix.py):
  - q4_k_m_default    : the MISSING CONTROL. Plain llama.cpp Q4_K_M mixture
                        (imatrix on, zero tensor-type overrides). The old
                        "ud-style" candidates flatten ffn_* to Q4_K and thus
                        remove the built-in Q6_K upgrades — without this
                        control you cannot tell whether flattening helped.
  - q4_k_m_ud_style   : archived-run candidate (kept for continuity).
  - ud_q4_k_xl        : archived-run candidate (kept).
  - q4_0_imatrix_pure : archived qat-aligned control (kept).
  - iq4_xs            : E8-lattice ~4.25bpw, the CPU-optimal family (REPACK
                        kernels); usually beats Q4_K_M KLD at smaller size.
  - iq4_nl            : non-uniform 4-bit codebook comparison.
  - q8_0_ceiling      : noise-floor ceiling; makes absolute KLD numbers
                        interpretable (fraction of achievable error).
  - q4_0_qat_export   : NOT re-quantized here — the lattice-exact artifact
                        from 03b_qat_export.py, registered AS-IS. This is
                        the answer to "where does the export qat file come
                        in": it enters the race HERE.

Controls:
  KUZA_REQUIRE_QAT_EXPORT=1  missing QAT lattice GGUF becomes a hard error
                             (default: warn and drop that candidate)
"""

from __future__ import annotations

import os
import re
import shutil

import model
from common import (
    adapter_dir,
    gguf_inventory,
    hf_token,
    imatrix_path,
    llama_cpp_binaries,
    quants_dir,
    reference_path,
    require_file,
    run,
    run_dir,
    sha256_file,
    smoke_load,
    write_json,
)

BULK_OVERRIDES = ("ffn_gate", "ffn_up", "ffn_down")

# spec keys:
#   base_type        llama-quantize positional type
#   pure             pass --pure
#   use_imatrix      pass --imatrix (harmless no-op for q4_0/q8_0 scales)
#   overrides        list of (flag, value) pairs, e.g.
#                    ("--token-embedding-type", "q6_k")
#   source           "reference" (default) | "reference_qat"
CANDIDATES: dict[str, dict] = {
    "q4_k_m_default": {
        "base_type": "q4_k_m",
        "filename": "kuza-q4_k_m-default.gguf",
        "pure": False,
        "use_imatrix": True,
        "overrides": [],
        "source": "reference",
    },
    "q4_k_m_ud_style": {
        "base_type": "q4_k_m",
        "filename": "kuza-q4_k_m-ud-style.gguf",
        "pure": False,
        "use_imatrix": True,
        "overrides": [
            ("--token-embedding-type", "q4_k"),
            ("--output-tensor-type", "q4_k"),
            ("--tensor-type", r"per_layer_token_embd\.weight=q4_k"),
            ("--tensor-type", r"per_layer_model_proj\.weight=q4_k"),
            *[("tensor_bulk_q4_k", name) for name in BULK_OVERRIDES],
        ],
        "source": "reference",
    },
    "ud_q4_k_xl": {
        "base_type": "q4_k_m",
        "filename": "kuza-ud-q4_k_xl.gguf",
        "pure": False,
        "use_imatrix": True,
        "overrides": [
            ("--token-embedding-type", "q5_k"),
            ("--output-tensor-type", "q5_k"),
            ("--tensor-type", r"per_layer_token_embd\.weight=q4_k"),
            ("--tensor-type", r"per_layer_model_proj\.weight=q5_k"),
            ("--tensor-type", r"blk\..*\.attn_(q|k|v|output|qkv)\.weight=q5_k"),
            *[("tensor_bulk_q4_k", name) for name in BULK_OVERRIDES],
        ],
        "source": "reference",
    },
    "q4_0_imatrix_pure": {
        "base_type": "q4_0",
        "filename": "kuza-q4_0-imatrix-pure.gguf",
        "pure": True,
        "use_imatrix": True,
        "overrides": [],
        "source": "reference",
    },
    "iq4_xs": {
        "base_type": "iq4_xs",
        "filename": "kuza-iq4_xs.gguf",
        "pure": False,
        "use_imatrix": True,
        "overrides": [],
        "source": "reference",
    },
    "iq4_nl": {
        "base_type": "iq4_nl",
        "filename": "kuza-iq4_nl.gguf",
        "pure": False,
        "use_imatrix": True,
        "overrides": [],
        "source": "reference",
    },
    "q8_0_ceiling": {
        "base_type": "q8_0",
        "filename": "kuza-q8_0-ceiling.gguf",
        "pure": True,
        "use_imatrix": False,
        "overrides": [],
        "source": "reference",
    },
    "q4_0_qat_export": {
        "base_type": "q4_0",
        "filename": "kuza-qat-q4_0-lattice.gguf",
        "pure": True,
        "use_imatrix": False,
        "overrides": [],
        "source": "reference_qat",
    },
}


def build_quant_command(binary, source, output, imatrix, spec) -> list:
    command: list = [binary]
    if spec.get("use_imatrix"):
        command.extend(["--imatrix", imatrix])
    if spec.get("pure"):
        command.append("--pure")
    for flag, value in spec.get("overrides", []):
        if flag == "tensor_bulk_q4_k":
            command.extend(
                ["--tensor-type", rf"blk\..*\.{value}\.weight=q4_k"]
            )
        elif flag == "--tensor-type":
            command.extend(["--tensor-type", value])
        else:
            command.extend([flag, value])
    command.extend([source, output, spec["base_type"]])
    return command


def validate_result(inventory: dict, spec: dict) -> list[str]:
    """Generic per-candidate validation; strict override checks where used."""
    warnings: list[str] = []
    model.assert_text_only_gguf(inventory)
    counts = inventory["tensor_type_counts"]
    base = spec["base_type"].upper().replace(".", "_")
    # llama-quantize mixture names (Q4_K_M/S/L) differ from the per-tensor GGUF
    # type they are built on (Q4_K); compare against the latter.
    base = re.sub(r"^(Q[2-6]_K)_[SML]$", r"\1", base)
    ffn = {
        name
        for name in inventory["tensors"]
        if name.startswith("blk.") and ".ffn_" in name
    }
    if ffn and spec.get("overrides"):
        # Explicit-recipe candidates: bulk must be exactly the requested type.
        allowed = {"Q4_0"} if base == "Q4_0" else {"Q4_K"}
        wrong = [
            name for name in ffn
            if inventory["tensors"][name] not in allowed
        ]
        if wrong:
            raise RuntimeError(
                f"{spec['base_type']} bulk tensors drifted from recipe: "
                f"{dict(list(((n, inventory['tensors'][n]) for n in wrong[:5])))}"
            )
    elif ffn:
        # Default-mixture candidates (Q4_K_M/IQ4_*): bulk must be dominated by
        # the base family; upgrades (Q6_K/Q8_0) are allowed and expected.
        dominant = [name for name in ffn if inventory["tensors"][name] == base]
        if len(dominant) < len(ffn) * 0.5:
            warnings.append(
                f"bulk is mostly {sorted({inventory['tensors'][n] for n in ffn})}, "
                f"not {base}"
            )
    if counts.get("F32", 0) > 0:
        warnings.append(
            f"{counts['F32']} F32 tensors kept (norms/PLE artifacts) — recorded, expected"
        )
    return warnings


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

    qat_source = run_dir() / "reference_qat" / "kuza-qat-q4_0-lattice.gguf"
    sources = {"reference": reference_path(), "reference_qat": qat_source}

    for name, spec in CANDIDATES.items():
        if spec.get("source") == "reference_qat" and not qat_source.is_file():
            if os.environ.get("KUZA_REQUIRE_QAT_EXPORT") == "1":
                raise RuntimeError(
                    "KUZA_REQUIRE_QAT_EXPORT=1 but the QAT lattice reference is "
                    "missing. Run 03b_qat_export.py first (or unset "
                    "KUZA_SKIP_QAT_EXPORT)."
                )
            print(
                f"{name}: skipped — QAT lattice reference missing "
                "(run 03b_qat_export.py; set KUZA_REQUIRE_QAT_EXPORT=1 to "
                "fail instead).",
                flush=True,
            )
            continue
        source = sources[spec.get("source", "reference")]
        output = root / name / spec["filename"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if spec.get("source") == "reference_qat":
            # Register the lattice artifact as-is; no re-quantization.
            if not output.exists() or output.stat().st_size != source.stat().st_size:
                shutil.copy2(source, output)
        else:
            output.unlink(missing_ok=True)
            run(
                build_quant_command(
                    binaries["llama-quantize"],
                    source,
                    output,
                    imatrix_path(),
                    spec,
                ),
                log_path=root / f"{name}.log",
            )
        inventory = gguf_inventory(output, binaries["converter"].parent)
        warnings = validate_result(inventory, spec)
        smoke_load(
            binaries["llama-cli"], output, smoke_prompt, root / f"{name}-smoke.log"
        )
        write_json(
            output.parent / "recipe.json",
            {
                "name": name,
                "base_type": spec["base_type"],
                "pure": spec.get("pure", False),
                "use_imatrix": spec.get("use_imatrix", False),
                "overrides": [
                    [flag, value] for flag, value in spec.get("overrides", [])
                ],
                "source": spec.get("source", "reference"),
                "imatrix_sha256": sha256_file(imatrix_path()),
                "source_sha256": sha256_file(source),
                "warnings": warnings,
                **{key: value for key, value in inventory.items() if key != "tensors"},
            },
        )
        print(f"{name}: {output} ({', '.join(warnings) or 'clean'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())