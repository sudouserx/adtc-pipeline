#!/usr/bin/env python3
"""Evaluate BF16 reference and past HF quants before new quantization."""

from __future__ import annotations

import sys
from pathlib import Path

import study_config as config
import paths
import quant_specs
from study_common import (
    breakdown_hidden,
    hf_token,
    llama_cpp_binaries,
    load_tokenizer,
    parse_mean_kld,
    pipeline_model,
    require_file,
    run,
    smoke_load,
    write_json,
)


def run_hidden_eval(llama_cli: Path, gguf: Path, name: str, output: Path) -> dict:
    require_file(config.EVAL_HIDDEN, "eval_hidden.py missing")
    run(
        [
            sys.executable,
            str(config.EVAL_HIDDEN),
            "--gguf",
            str(gguf),
            "--name",
            name,
            "--llama-cli",
            str(llama_cli),
            "--ngl",
            "999",
            "--output",
            str(output),
        ]
    )
    import json

    return json.loads(output.read_text(encoding="utf-8"))


def run_kld(
    binaries: dict,
    gguf: Path,
    name: str,
    kld_base: Path,
    dest: Path,
    *,
    is_reference: bool = False,
) -> float | None:
    cmd = [
        binaries["llama-perplexity"],
        "-m",
        str(gguf),
        "-f",
        str(paths.eval_path()),
        "-s",
        str(config.SEED),
        "-c",
        str(config.MAX_SEQ_LENGTH),
        "-b",
        "512",
        "--chunks",
        "16",
        "-ngl",
        "999",
        "--kl-divergence-base",
        str(kld_base),
    ]
    if not is_reference:
        cmd.append("--kl-divergence")
    output = run(cmd, capture=True, log_path=dest / f"{name}-kld.log")
    return parse_mean_kld(output)


def run_smoke(
    llama_cli: Path,
    gguf: Path,
    name: str,
    dest: Path,
    tokenizer,
) -> dict:
    results = {}
    for prompt_name, prompt in pipeline_model.SCREEN_SMOKE_PROMPTS.items():
        output = smoke_load(
            llama_cli,
            gguf,
            pipeline_model.render_generation_prompt(tokenizer, prompt),
            dest / f"{name}-{prompt_name}.log",
        )
        results[prompt_name] = {
            "warnings": pipeline_model.smoke_warnings(prompt_name, output),
            "output_preview": output[:400],
        }
    return results


def evaluate_model(
    binaries: dict,
    name: str,
    gguf: Path,
    dest: Path,
    kld_base: Path,
    tokenizer,
    *,
    is_reference: bool = False,
) -> dict:
    require_file(gguf, f"missing GGUF for {name}")
    hidden_path = dest / f"hidden_{name}.json"
    hidden = run_hidden_eval(binaries["llama-cli"], gguf, name, hidden_path)
    breakdown = breakdown_hidden(hidden)
    kld = run_kld(binaries, gguf, name, kld_base, dest, is_reference=is_reference)
    smoke = run_smoke(binaries["llama-cli"], gguf, name, dest, tokenizer)
    return {
        "name": name,
        "gguf": str(gguf),
        "size": gguf.stat().st_size,
        "hidden_mean": hidden.get("mean_score"),
        "safety_failures": hidden.get("safety_failures"),
        "language_failures": hidden.get("language_failures"),
        "hidden_by_language": breakdown["by_language"],
        "hidden_by_category": breakdown["by_category"],
        "mean_kld": kld,
        "smoke": smoke,
        "hidden_report": str(hidden_path),
    }


def main() -> int:
    hf_token()
    dest = config.RESULTS_DIR / "baseline"
    dest.mkdir(parents=True, exist_ok=True)
    require_file(paths.reference_path(), "Run 01_download.py first.")
    require_file(paths.eval_path(), "Run 01_download.py first.")

    binaries = llama_cpp_binaries()
    tokenizer = load_tokenizer()
    kld_base = dest / "kuza-bf16-reference.kld"

    bf16 = evaluate_model(
        binaries,
        "bf16_reference",
        paths.reference_path(),
        dest,
        kld_base,
        tokenizer,
        is_reference=True,
    )
    write_json(dest / "bf16_eval.json", bf16)

    past: dict[str, dict] = {}
    for name, spec in quant_specs.PAST_QUANT_CANDIDATES.items():
        gguf = quant_specs.gguf_path(name, spec)
        past[name] = evaluate_model(
            binaries, name, gguf, dest, kld_base, tokenizer
        )
        if past[name]["mean_kld"] is not None and bf16["mean_kld"] is not None:
            past[name]["delta_kld_vs_bf16"] = round(
                float(past[name]["mean_kld"]) - float(bf16["mean_kld"]), 6
            )
        if past[name]["hidden_mean"] is not None and bf16["hidden_mean"] is not None:
            past[name]["delta_hidden_vs_bf16"] = round(
                float(past[name]["hidden_mean"]) - float(bf16["hidden_mean"]), 3
            )

    write_json(dest / "past_quants_eval.json", past)

    bf16_mean = float(bf16.get("hidden_mean") or 0)
    verdict = (
        "quantization_likely"
        if bf16_mean >= config.HIDDEN_GOOD_THRESHOLD
        else "finetuning_or_template_likely"
    )
    summary = {
        "bf16_hidden_mean": bf16.get("hidden_mean"),
        "bf16_mean_kld": bf16.get("mean_kld"),
        "hidden_good_threshold": config.HIDDEN_GOOD_THRESHOLD,
        "preliminary_verdict": verdict,
        "past_quants": {
            name: {
                "hidden_mean": item.get("hidden_mean"),
                "delta_hidden_vs_bf16": item.get("delta_hidden_vs_bf16"),
                "mean_kld": item.get("mean_kld"),
            }
            for name, item in past.items()
        },
        "note": (
            "If bf16_hidden_mean is good but quants are much lower, quantization is the problem. "
            "If bf16 is also low, investigate finetuning/template."
        ),
    }
    write_json(dest / "summary.json", summary)
    print(f"baseline: bf16 hidden_mean={bf16.get('hidden_mean')} verdict={verdict}")
    print(f"summary: {dest / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
