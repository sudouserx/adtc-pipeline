#!/usr/bin/env python3
"""Screen all models: KLD fidelity, hidden-set accuracy, and GPU throughput."""

from __future__ import annotations

import sys
from pathlib import Path

import study_config as config
import paths
import quant_specs
from study_common import (
    breakdown_hidden,
    generation_tps,
    gguf_inventory,
    hf_token,
    installed_packages,
    llama_cpp_binaries,
    load_tokenizer,
    parse_bench_tps,
    parse_mean_kld,
    pipeline_model,
    require_file,
    run,
    runtime_flags,
    sha256_file,
    smoke_load,
    write_json,
    read_json,
)


def run_hidden(llama_cli: Path, gguf: Path, name: str, output: Path) -> dict:
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
    return read_json(output)


def rank_candidates(candidates: dict[str, dict]) -> dict:
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -float(item[1].get("hidden_mean") if item[1].get("hidden_mean") is not None else -1.0),
            int(item[1].get("size") or 0),
            -generation_tps(item[1].get("bench_values") or []),
        ),
    )
    return {
        "winner": ranked[0][0] if ranked else None,
        "ranked": [
            {
                "name": name,
                "hidden_mean": spec.get("hidden_mean"),
                "size": spec.get("size"),
                "generation_tps": generation_tps(spec.get("bench_values") or []),
                "mean_kld": spec.get("mean_kld"),
            }
            for name, spec in ranked
        ],
        "rule": (
            "hidden_mean desc, then size asc, then generation_tps desc. "
            "KLD is diagnostic only."
        ),
    }


def main() -> int:
    hf_token()
    require_file(paths.reference_path(), "Run 01_download.py first.")
    require_file(paths.eval_path(), "Run 01_download.py first.")
    binaries = llama_cpp_binaries()
    tokenizer = load_tokenizer()
    dest = config.SCREEN_DIR
    dest.mkdir(parents=True, exist_ok=True)
    kld_base = dest / "kuza-bf16-reference.kld"

    # Build or reuse KLD reference from BF16.
    if not kld_base.is_file() or kld_base.stat().st_size == 0:
        run(
            [
                binaries["llama-perplexity"],
                "-m",
                str(paths.reference_path()),
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
            ],
            log_path=dest / "bf16-kld.log",
        )

    results: dict[str, object] = {
        "run_id": config.SOURCE_RUN_ID,
        "study_work_dir": str(config.WORK_DIR),
        "packages": installed_packages(),
        "runtime": {
            "flash_attn": config.FLASH_ATTN,
            "cache_type_k": config.CACHE_TYPE_K,
            "cache_type_v": config.CACHE_TYPE_V,
        },
        "reference": {
            "path": str(paths.reference_path()),
            "sha256": sha256_file(paths.reference_path()),
        },
        "candidates": {},
    }

    all_specs = quant_specs.all_screen_candidates()
    for name, spec in all_specs.items():
        gguf = quant_specs.gguf_path(name, spec)
        if not gguf.is_file():
            print(f"skip {name}: missing {gguf}")
            continue

        is_reference = bool(spec.get("is_reference"))
        kld_cmd = [
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
            kld_cmd.append("--kl-divergence")
        kld_output = run(
            kld_cmd,
            capture=True,
            log_path=dest / f"{name}-kld.log",
        )
        mean_kld = parse_mean_kld(kld_output)

        bench_cmd = [
            binaries["llama-bench"],
            "-m",
            str(gguf),
            "-p",
            "512",
            "-n",
            "128",
            "-ngl",
            "999",
            "-t",
            "4",
            "-r",
            "5",
            *runtime_flags(),
        ]
        bench_output = run(
            bench_cmd,
            capture=True,
            log_path=dest / f"{name}-bench.log",
        )
        bench_values = parse_bench_tps(bench_output)

        smoke_results = {}
        for prompt_name, prompt in pipeline_model.SCREEN_SMOKE_PROMPTS.items():
            output = smoke_load(
                binaries["llama-cli"],
                gguf,
                pipeline_model.render_generation_prompt(tokenizer, prompt),
                dest / f"{name}-{prompt_name}.log",
            )
            smoke_results[prompt_name] = {
                "warnings": pipeline_model.smoke_warnings(prompt_name, output),
            }

        hidden_path = dest / f"hidden_{name}.json"
        hidden = run_hidden(binaries["llama-cli"], gguf, name, hidden_path)
        breakdown = breakdown_hidden(hidden)
        inventory = gguf_inventory(gguf, binaries["converter"].parent)
        recipe_path = gguf.parent / "recipe.json"
        recipe = read_json(recipe_path) if recipe_path.is_file() else None

        entry = {
            "source": spec.get("source"),
            "base_type": spec.get("base_type"),
            "file": str(gguf),
            "size": gguf.stat().st_size,
            "sha256": sha256_file(gguf),
            "recipe": recipe,
            "mean_kld": mean_kld,
            "bench_values": bench_values,
            "generation_tps": generation_tps(bench_values),
            "tensor_type_counts": inventory.get("tensor_type_counts"),
            "smoke": smoke_results,
            "hidden_mean": hidden.get("mean_score"),
            "safety_failures": hidden.get("safety_failures"),
            "language_failures": hidden.get("language_failures"),
            "hidden_by_language": breakdown["by_language"],
            "hidden_by_category": breakdown["by_category"],
            "hidden_report": str(hidden_path),
            "notes": spec.get("notes"),
        }
        results["candidates"][name] = entry
        print(
            f"{name}: hidden={entry['hidden_mean']} kld={entry['mean_kld']} "
            f"tps={entry['generation_tps']:.1f} size={entry['size']}"
        )

    # Deltas vs BF16 reference.
    ref = results["candidates"].get("bf16_reference") or {}
    ref_hidden = ref.get("hidden_mean")
    ref_kld = ref.get("mean_kld")
    for name, entry in results["candidates"].items():
        if name == "bf16_reference":
            continue
        if ref_hidden is not None and entry.get("hidden_mean") is not None:
            entry["delta_hidden_vs_bf16"] = round(
                float(entry["hidden_mean"]) - float(ref_hidden), 3
            )
        if ref_kld is not None and entry.get("mean_kld") is not None:
            entry["delta_kld_vs_bf16"] = round(
                float(entry["mean_kld"]) - float(ref_kld), 6
            )

    results["selection"] = rank_candidates(results["candidates"])
    write_json(dest / "results.json", results)
    write_json(config.RESULTS_DIR / "screen" / "results.json", results)
    print(f"screen: {dest / 'results.json'}")
    print(f"winner: {results['selection']['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
