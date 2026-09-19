"""Extended quantization candidate definitions for the diagnosis study."""

from __future__ import annotations

import re
from typing import Any

# Past-run candidates downloaded from Hugging Face (not re-quantized).
PAST_QUANT_CANDIDATES: dict[str, dict[str, Any]] = {
    "q4_k_m_imatrix": {
        "source": "hf",
        "filename": "kuza-qwen-q4_k_m.gguf",
        "base_type": "q4_k_m",
    },
    "q4_k_xl_ssm": {
        "source": "hf",
        "filename": "kuza-qwen-q4_k_xl.gguf",
        "base_type": "q4_k_m",
    },
    "q4_k_s_ssm": {
        "source": "hf",
        "filename": "kuza-qwen-q4_k_s.gguf",
        "base_type": "q4_k_s",
    },
}

# New candidates built locally from reference/kuza-bf16.gguf + kuza.imatrix.
NEW_QUANT_CANDIDATES: dict[str, dict[str, Any]] = {
    "q4_k_m_plain": {
        "source": "local",
        "filename": "kuza-qwen-q4_k_m_plain.gguf",
        "base_type": "q4_k_m",
        "embedding_type": None,
        "output_type": None,
        "ssm_out_type": None,
        "ssm_gate_type": None,
        "attention_type": None,
        "attn_gate_type": None,
        "bulk_type": None,
        "notes": "Community-style imatrix Q4_K_M without tensor overrides.",
    },
    "q5_k_m_imatrix_ssm": {
        "source": "local",
        "filename": "kuza-qwen-q5_k_m_ssm.gguf",
        "base_type": "q5_k_m",
        "embedding_type": "q6_k",
        "output_type": "q6_k",
        "ssm_out_type": "q6_k",
        "ssm_gate_type": None,
        "attention_type": None,
        "attn_gate_type": None,
        "bulk_type": "q5_k",
        "notes": "Quality step-up per Hob-forge Qwen3.5-4B benchmarks.",
    },
    "q3_k_m_imatrix_ssm": {
        "source": "local",
        "filename": "kuza-qwen-q3_k_m_ssm.gguf",
        "base_type": "q3_k_m",
        "embedding_type": "q6_k",
        "output_type": "q6_k",
        "ssm_out_type": "q6_k",
        "ssm_gate_type": "q5_k",
        "attention_type": "q6_k",
        "attn_gate_type": None,
        "bulk_type": "q3_k",
        "notes": "User-requested 3-bit K-quant with SSM/attention protection.",
    },
    "iq3_xs_imatrix_ssm": {
        "source": "local",
        "filename": "kuza-qwen-iq3_xs_ssm.gguf",
        "base_type": "iq3_xs",
        "embedding_type": "q8_0",
        "output_type": "q8_0",
        "ssm_out_type": "q6_k",
        "ssm_gate_type": "q5_k",
        "attention_type": "q5_k",
        "attn_gate_type": None,
        "bulk_type": None,
        "notes": "User-requested 3-bit I-quant; may trade speed for size.",
    },
    "iq4_xs_imatrix_ssm": {
        "source": "local",
        "filename": "kuza-qwen-iq4_xs_ssm.gguf",
        "base_type": "iq4_xs",
        "embedding_type": "q8_0",
        "output_type": "q8_0",
        "ssm_out_type": "q6_k",
        "ssm_gate_type": "q5_k",
        "attention_type": "q5_k",
        "attn_gate_type": None,
        "bulk_type": None,
        "notes": "Smallest 4-bit I-quant with hybrid SSM protection.",
    },
    "q6_k_imatrix": {
        "source": "local",
        "filename": "kuza-qwen-q6_k.gguf",
        "base_type": "q6_k",
        "embedding_type": None,
        "output_type": None,
        "ssm_out_type": None,
        "ssm_gate_type": None,
        "attention_type": None,
        "attn_gate_type": None,
        "bulk_type": None,
        "notes": "Near-lossless quant reference.",
    },
}

BASELINE_CANDIDATES: dict[str, dict[str, Any]] = {
    "bf16_reference": {
        "source": "hf",
        "filename": "kuza-bf16.gguf",
        "base_type": "bf16",
        "is_reference": True,
    },
}


def all_screen_candidates() -> dict[str, dict[str, Any]]:
    merged = dict(BASELINE_CANDIDATES)
    merged.update(PAST_QUANT_CANDIDATES)
    merged.update(NEW_QUANT_CANDIDATES)
    return merged


def gguf_path(name: str, spec: dict[str, Any]) -> "Path":
    from pathlib import Path

    import paths

    if spec.get("is_reference"):
        return paths.reference_path()
    if spec.get("source") == "hf":
        return paths.past_quant_path(name, spec["filename"])
    return paths.new_quant_path(name, spec["filename"])


def _gguf_type_name(value: str) -> str:
    return value.upper().replace(".", "_")


def quant_command(
    binary: Any,
    reference: Any,
    output: Any,
    imatrix: Any,
    spec: dict[str, Any],
) -> list[Any]:
    base_type = str(spec["base_type"])
    command: list[Any] = [binary, "--imatrix", imatrix]
    if spec.get("embedding_type"):
        command.extend(["--token-embedding-type", str(spec["embedding_type"])])
    if spec.get("output_type"):
        command.extend(["--output-tensor-type", str(spec["output_type"])])
    if spec.get("ssm_out_type"):
        command.extend(
            ["--tensor-type", rf"blk\..*\.ssm_out\.weight={spec['ssm_out_type']}"]
        )
    if spec.get("ssm_gate_type"):
        command.extend(
            [
                "--tensor-type",
                rf"blk\..*\.ssm_alpha\.weight={spec['ssm_gate_type']}",
                "--tensor-type",
                rf"blk\..*\.ssm_beta\.weight={spec['ssm_gate_type']}",
            ]
        )
    if spec.get("attention_type"):
        command.extend(
            [
                "--tensor-type",
                rf"blk\..*\.attn_(q|k|v|output|qkv)\.weight={spec['attention_type']}",
            ]
        )
    if spec.get("attn_gate_type"):
        command.extend(
            [
                "--tensor-type",
                rf"blk\..*\.attn_gate\.weight={spec['attn_gate_type']}",
            ]
        )
    bulk_type = spec.get("bulk_type")
    if bulk_type:
        command.extend(
            [
                "--tensor-type",
                rf"blk\..*\.ffn_gate\.weight={bulk_type}",
                "--tensor-type",
                rf"blk\..*\.ffn_up\.weight={bulk_type}",
                "--tensor-type",
                rf"blk\..*\.ffn_down\.weight={bulk_type}",
            ]
        )
    command.extend([reference, output, base_type])
    return command


def validate_quant_overrides(inventory: dict[str, Any], spec: dict[str, Any]) -> None:
    tensors = inventory["tensors"]
    families: list[tuple[str, list[str], str]] = []
    if spec.get("ssm_out_type"):
        families.append(
            (
                "ssm_out",
                [r"blk\..*\.ssm_out\.weight$"],
                _gguf_type_name(str(spec["ssm_out_type"])),
            )
        )
    if spec.get("ssm_gate_type"):
        families.append(
            (
                "ssm_gates",
                [r"blk\..*\.ssm_(alpha|beta)\.weight$"],
                _gguf_type_name(str(spec["ssm_gate_type"])),
            )
        )
    if spec.get("attention_type"):
        families.append(
            (
                "attention",
                [
                    r"blk\..*\.attn_(q|k|v|output)\.weight$",
                    r"blk\..*\.attn_qkv\.weight$",
                ],
                _gguf_type_name(str(spec["attention_type"])),
            )
        )
    if spec.get("attn_gate_type"):
        families.append(
            (
                "attn_gate",
                [r"blk\..*\.attn_gate\.weight$"],
                _gguf_type_name(str(spec["attn_gate_type"])),
            )
        )
    for family, patterns, expected in families:
        matched: dict[str, str] = {}
        for pattern in patterns:
            compiled = re.compile(pattern)
            found = {
                name: dtype for name, dtype in tensors.items() if compiled.search(name)
            }
            if found:
                matched = found
                break
        if not matched and family != "attn_gate":
            continue
        wrong = {
            name: dtype
            for name, dtype in matched.items()
            if _gguf_type_name(dtype) != expected
        }
        if wrong:
            raise RuntimeError(
                f"Quantizer ignored {expected} override for {family}: {wrong}"
            )
    embedding_names = [
        name
        for name in ("token_embd.weight", "output.weight", "token_embd", "output")
        if name in tensors
    ]
    for name in embedding_names:
        if name in {"output.weight", "output"} and spec.get("output_type"):
            wanted = _gguf_type_name(str(spec["output_type"]))
        elif spec.get("embedding_type"):
            wanted = _gguf_type_name(str(spec["embedding_type"]))
        else:
            continue
        if _gguf_type_name(tensors[name]) != wanted:
            raise RuntimeError(f"{name} should be {wanted}, found {tensors[name]}")
