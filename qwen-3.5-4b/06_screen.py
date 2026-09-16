#!/usr/bin/env python3
"""Screen quantized GGUF candidates against the BF16 reference."""

from __future__ import annotations

import config
import model
from common import (
    adapter_dir,
    candidate_gguf,
    eval_path,
    gguf_inventory,
    hf_token,
    installed_packages,
    llama_cpp_binaries,
    parse_bench_tps,
    parse_mean_kld,
    read_json,
    reference_path,
    require_file,
    run,
    run_dir,
    screen_dir,
    sha256_bytes,
    sha256_file,
    smoke_load,
    write_json,
)


def main() -> int:
    hf_token()
    require_file(adapter_dir() / "adapter_config.json", "Run 02_sft.py first.")
    require_file(reference_path(), "Run 03_reference.py first.")
    require_file(eval_path(), "Run 04_imatrix.py first.")
    candidates = {
        name: candidate_gguf(name) for name in model.QUANT_CANDIDATES
    }
    for name, path in candidates.items():
        require_file(path, f"Run 05_quants.py first (missing {name}).")
    binaries = llama_cpp_binaries()
    tokenizer = model.load_tokenizer(adapter_dir())
    dest = screen_dir()
    dest.mkdir(parents=True, exist_ok=True)
    kld_base = dest / "kuza-bf16-reference.kld"
    reference_output = run(
        [
            binaries["llama-perplexity"],
            "-m",
            reference_path(),
            "-f",
            eval_path(),
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
            kld_base,
        ],
        capture=True,
        log_path=dest / "bf16-kld.log",
    )
    if not kld_base.is_file() or kld_base.stat().st_size == 0:
        raise RuntimeError("BF16 KLD reference file was not created")
    results = {
        "notice": "GPU screening only; not an ADTC laptop measurement.",
        "run_id": config.RUN_ID,
        "work_dir": str(run_dir()),
        "llama_cpp_commit": config.LLAMA_CPP_COMMIT,
        "prompt_sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
        "packages": installed_packages(),
        "reference": {
            "path": reference_path().name,
            "sha256": sha256_file(reference_path()),
            "raw_log": "bf16-kld.log",
            "output_present": bool(reference_output),
        },
        "candidates": {},
    }
    for name, path in candidates.items():
        kld_output = run(
            [
                binaries["llama-perplexity"],
                "-m",
                path,
                "-f",
                eval_path(),
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
                kld_base,
                "--kl-divergence",
            ],
            capture=True,
            log_path=dest / f"{name}-kld.log",
        )
        bench_output = run(
            [
                binaries["llama-bench"],
                "-m",
                path,
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
                "-ctk",
                "q4_0",
                "-ctv",
                "q8_0",
                "-fa",
                "on",
            ],
            capture=True,
            log_path=dest / f"{name}-bench.log",
        )
        prompt_results = {}
        for prompt_name, prompt in model.SCREEN_SMOKE_PROMPTS.items():
            output = smoke_load(
                binaries["llama-cli"],
                path,
                model.render_generation_prompt(tokenizer, prompt),
                dest / f"{name}-{prompt_name}.log",
            )
            prompt_results[prompt_name] = {
                "log": f"{name}-{prompt_name}.log",
                "warnings": model.smoke_warnings(prompt_name, output),
            }
        inventory = gguf_inventory(path, binaries["converter"].parent)
        recipe_path = path.parent / "recipe.json"
        mean_kld = parse_mean_kld(kld_output)
        if mean_kld is None:
            raise RuntimeError(
                f"Could not parse mean KLD for {name}; screening cannot continue"
            )
        results["candidates"][name] = {
            "file": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "recipe": read_json(recipe_path) if recipe_path.exists() else None,
            "mean_kld": mean_kld,
            "bench_values": parse_bench_tps(bench_output),
            "tensor_type_counts": inventory["tensor_type_counts"],
            "smoke": prompt_results,
        }
    results["selection"] = (
        "No winner selected here. Run experiments/eval_hidden.py on the GGUFs, "
        "then experiments/select_winner.py."
    )
    write_json(dest / "results.json", results)
    print(f"screen: {dest / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
