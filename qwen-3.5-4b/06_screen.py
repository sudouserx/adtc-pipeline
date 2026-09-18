#!/usr/bin/env python3
"""Screen quantized GGUF candidates against the BF16 reference.

KLD is a fidelity diagnostic. The winner is ranked by hidden-set score,
then GGUF size, then generation TPS. GPU llama-bench is not an ADTC
laptop measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import config
import model
from common import (
    adapter_dir,
    candidate_gguf,
    eval_path,
    generation_tps,
    gguf_inventory,
    help_has,
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
    tool_help,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_HIDDEN = REPO_ROOT / "experiments" / "eval_hidden.py"
HIDDEN_PROMPTS = REPO_ROOT / "data" / "hidden_prompts.jsonl"
HIDDEN_RESULTS = REPO_ROOT / "experiments" / "results"


def _runtime_flags() -> list[str]:
    return [
        "-fa",
        str(config.FLASH_ATTN),
        "-ctk",
        str(config.CACHE_TYPE_K),
        "-ctv",
        str(config.CACHE_TYPE_V),
    ]


def _llama_cpp_head() -> str:
    checkout = config.TOOLS_DIR / "llama.cpp"
    if not (checkout / ".git").exists() and not (checkout / "HEAD").exists():
        return config.LLAMA_CPP_COMMIT
    try:
        return run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).strip()
    except RuntimeError:
        return config.LLAMA_CPP_COMMIT


def _probe_mtp(binaries: dict[str, Path], dest: Path) -> dict:
    cli_help = tool_help(binaries["llama-cli"])
    bench_help = tool_help(binaries["llama-bench"])
    reference = read_json(reference_path().parent / "reference_manifest.json")
    inventory = gguf_inventory(reference_path(), binaries["converter"].parent)
    mtp_tensors = model.mtp_tensor_names(inventory["tensors"])
    probe = {
        "cli_has_spec_type": help_has(cli_help, "--spec-type"),
        "cli_has_spec_draft_n_max": help_has(cli_help, "--spec-draft-n-max"),
        "cli_has_draft_max": help_has(cli_help, "--draft-max"),
        "bench_has_spec_type": help_has(bench_help, "--spec-type"),
        "bench_has_spec_draft_n_max": help_has(bench_help, "--spec-draft-n-max"),
        "bench_has_draft_max": help_has(bench_help, "--draft-max"),
        "reference_mtp_included": bool(
            reference.get("mtp_included", reference.get("mtp_requested"))
        ),
        "reference_mtp_tensors": mtp_tensors or reference.get("mtp_tensors") or [],
        "speculative_args": [],
        "speculative_enabled": False,
    }
    if probe["reference_mtp_tensors"] and probe["cli_has_spec_type"]:
        args = ["--spec-type", "draft-mtp"]
        if probe["cli_has_spec_draft_n_max"]:
            args.extend(["--spec-draft-n-max", "2"])
        elif probe["cli_has_draft_max"]:
            args.extend(["--draft-max", "2"])
        probe["speculative_args"] = args
        probe["speculative_enabled"] = True
    elif probe["reference_mtp_tensors"] and probe["cli_has_draft_max"]:
        probe["speculative_args"] = ["--draft-max", "2"]
        probe["speculative_enabled"] = True
    write_json(dest / "mtp_probe.json", probe)
    return probe


def _hidden_score(
    llama_cli: Path,
    gguf: Path,
    name: str,
    dest: Path,
) -> dict:
    if not EVAL_HIDDEN.is_file():
        raise RuntimeError(f"Missing {EVAL_HIDDEN}")
    if not HIDDEN_PROMPTS.is_file():
        raise RuntimeError(f"Missing hidden prompt set {HIDDEN_PROMPTS}")
    output = dest / f"hidden_{name}.json"
    HIDDEN_RESULTS.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            EVAL_HIDDEN,
            "--gguf",
            gguf,
            "--name",
            name,
            "--llama-cli",
            llama_cli,
            "--ngl",
            "999",
            "--output",
            output,
        ]
    )
    report = read_json(output)
    shared = HIDDEN_RESULTS / f"hidden_{name}.json"
    if shared.resolve() != output.resolve():
        write_json(shared, report)
    return report


def _rank_candidates(candidates: dict[str, dict]) -> dict:
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -float(item[1].get("hidden_mean") if item[1].get("hidden_mean") is not None else -1.0),
            int(item[1]["size"]),
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
            "hidden_mean desc, then GGUF size asc, then generation TPS desc. "
            "GPU llama-bench TPS is not an ADTC laptop measurement. "
            "KLD is a fidelity diagnostic, not the objective."
        ),
    }


def main() -> int:
    hf_token()
    require_file(adapter_dir() / "adapter_config.json", "Run 02_sft.py first.")
    require_file(reference_path(), "Run 03_reference.py first.")
    require_file(eval_path(), "Run 04_imatrix.py first.")
    candidates = {name: candidate_gguf(name) for name in model.QUANT_CANDIDATES}
    for name, path in candidates.items():
        require_file(path, f"Run 05_quants.py first (missing {name}).")
    binaries = llama_cpp_binaries()
    tokenizer = model.load_tokenizer(adapter_dir())
    dest = screen_dir()
    dest.mkdir(parents=True, exist_ok=True)
    mtp_probe = _probe_mtp(binaries, dest)
    runtime_flags = _runtime_flags()
    speculative = list(mtp_probe.get("speculative_args") or [])
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
        "notice": (
            "GPU screening only; not an ADTC laptop measurement. "
            "Winner is ranked by hidden-set accuracy, then size, then GPU TPS. "
            "KLD is diagnostic."
        ),
        "run_id": config.RUN_ID,
        "work_dir": str(run_dir()),
        "llama_cpp_commit_pin": config.LLAMA_CPP_COMMIT,
        "llama_cpp_commit": _llama_cpp_head(),
        "prompt_sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
        "packages": installed_packages(),
        "runtime": {
            "flash_attn": config.FLASH_ATTN,
            "cache_type_k": config.CACHE_TYPE_K,
            "cache_type_v": config.CACHE_TYPE_V,
            "mtp_probe": mtp_probe,
        },
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
        bench_cmd = [
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
            *runtime_flags,
        ]
        speculative_used = False
        bench_can_speculate = bool(
            speculative
            and (
                (mtp_probe.get("bench_has_spec_type") and "--spec-type" in speculative)
                or mtp_probe.get("bench_has_draft_max")
                or mtp_probe.get("bench_has_spec_draft_n_max")
            )
        )
        if bench_can_speculate:
            try:
                bench_output = run(
                    [*bench_cmd, *speculative],
                    capture=True,
                    log_path=dest / f"{name}-bench.log",
                )
                speculative_used = True
            except RuntimeError:
                bench_output = run(
                    bench_cmd,
                    capture=True,
                    log_path=dest / f"{name}-bench.log",
                )
        else:
            bench_output = run(
                bench_cmd,
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
        hidden = _hidden_score(binaries["llama-cli"], path, name, dest)
        results["candidates"][name] = {
            "file": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "recipe": read_json(recipe_path) if recipe_path.exists() else None,
            "mean_kld": mean_kld,
            "bench_values": parse_bench_tps(bench_output),
            "generation_tps": generation_tps(parse_bench_tps(bench_output)),
            "speculative_used": speculative_used,
            "tensor_type_counts": inventory["tensor_type_counts"],
            "smoke": prompt_results,
            "hidden_mean": hidden.get("mean_score"),
            "safety_failures": hidden.get("safety_failures"),
            "language_failures": hidden.get("language_failures"),
            "hidden_report": str(Path(f"hidden_{name}.json")),
        }
    results["selection"] = _rank_candidates(results["candidates"])
    write_json(dest / "results.json", results)
    print(f"screen: {dest / 'results.json'}")
    print(f"winner: {results['selection']['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
