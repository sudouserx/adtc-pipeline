#!/usr/bin/env python3
"""Stage 06 — Screen quantized GGUF candidates against the BF16 reference.

FINAL version (2026-09-20). Replaces 06_screen.py; pairs with config.py
and the UNMODIFIED repo common.py / model.py.

Changes vs the original 06_screen.py:
  1. KLD/PPL over config.SCREEN_KLD_CHUNKS (128) chunks — the archived run
     used 16 (~16K tokens), statistically unable to separate 4-bit
     candidates. PPL and Same-top-p are recorded per candidate.
  2. CPU-representative bench arm is the PRIMARY performance signal:
     -ngl 0, physical-core threads, pp512/tg128. The GPU arm (if present)
     is diagnostic only — the archived run ranked on an RTX A6000 while the
     deployment target is a CPU laptop, and quant-family t/s rankings do
     not transfer across backends.
  3. Peak RSS per candidate (the actual edge memory budget) via
     /usr/bin/time when available.
  4. Hard quality gates (config.SCREEN_GATES): safety/language failures,
     mean-KLD ceiling, Same-top-p floor. Gated candidates cannot win.
  5. Noise-aware ranking: candidates within 2x the KLD standard error of
     the best are "statistically tied" and ranked by CPU t/s, then size,
     then RSS — instead of letting a noisy 0.001 KLD difference decide.
  6. Candidates are DISCOVERED from quants/*/recipe.json (05_quants.py
     defines the matrix; no model.py edits needed). This is how the
     q4_0_qat_export candidate from 03b_qat_export.py enters screening.

FIXED 2026-09-22:
  7. Discovery no longer requires ``recipe["filename"]`` (05_quants.py never
     wrote it -> KeyError). The GGUF is resolved from the recipe when present,
     else from the single .gguf in the candidate directory (size-matched when
     ambiguous). Recipes are checked for staleness: size vs file, and the
     imatrix / BF16-reference sha256 recorded by 05 vs the current artifacts.
  8. Controls (``role == "control"`` or a ``*_ceiling`` name — the Q8_0 noise
     floor) are screened and reported but can never win: as the lowest-KLD
     entry they used to define the "tied band" alone and be crowned winner.
  9. peak_rss_kb now comes from the CPU bench run (the edge memory budget),
     not the -ngl 999 KLD run where the weights live on the GPU.
 10. The number of KLD chunks llama-perplexity actually ran is recorded, and a
     warning is printed when the eval corpus is too short for
     config.SCREEN_KLD_CHUNKS (llama-perplexity silently clamps --chunks).

Requires repo-root siblings (not just this gemma-4-e2b/ directory):
  ../experiments/eval_hidden.py
  ../data/hidden_prompts.jsonl
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import config
import model
from common import (
    adapter_dir,
    cli_oneshot_args,
    eval_path,
    gguf_inventory,
    help_has,
    hf_token,
    imatrix_path,
    installed_packages,
    llama_cpp_binaries,
    parse_bench_tps,
    parse_mean_kld,
    quants_dir,
    read_json,
    reference_path,
    require_file,
    run,
    run_dir,
    screen_dir,
    sha256_bytes,
    sha256_file,
    tool_help,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_HIDDEN = REPO_ROOT / "experiments" / "eval_hidden.py"
HIDDEN_PROMPTS = REPO_ROOT / "data" / "hidden_prompts.jsonl"
HIDDEN_RESULTS = REPO_ROOT / "experiments" / "results"

# Candidates that calibrate the scale (Q8_0 noise floor) rather than compete.
CONTROL_NAME_SUFFIXES = ("_ceiling",)


def _is_control(name: str, recipe: dict | None) -> bool:
    if recipe and recipe.get("role") == "control":
        return True
    return name.endswith(CONTROL_NAME_SUFFIXES)


def _resolve_candidate_file(recipe_path: Path, recipe: dict) -> Path | None:
    """Locate the GGUF a recipe.json describes (None if it is gone).

    05_quants.py originally wrote no ``filename`` key, so fall back to the .gguf
    in the candidate directory; if several are present (leftovers from an older
    run) pick the one whose size matches the recipe.
    """
    folder = recipe_path.parent
    filename = recipe.get("filename")
    if filename:
        path = folder / filename
        return path if path.is_file() and path.stat().st_size > 0 else None
    ggufs = sorted(
        p for p in folder.glob("*.gguf")
        if p.is_file() and p.stat().st_size > 0 and not p.name.endswith(".tmp.gguf")
    )
    if len(ggufs) > 1 and recipe.get("size") is not None:
        matching = [p for p in ggufs if p.stat().st_size == int(recipe["size"])]
        ggufs = matching or ggufs
    if not ggufs:
        return None
    if len(ggufs) > 1:
        raise RuntimeError(
            f"{folder} holds several .gguf files {[p.name for p in ggufs]} and "
            "recipe.json does not say which one it describes. Remove the stale "
            "ones or re-run 05_quants.py."
        )
    return ggufs[0]


def discover_candidates(
    root: Path, reference_sha: str, imatrix_sha: str | None
) -> tuple[dict[str, Path], dict[str, dict], dict[str, str]]:
    """Find quantized candidates from quants/*/recipe.json (written by 05).

    Returns (name -> gguf, name -> recipe, name -> reason for skipping).
    """
    candidates: dict[str, Path] = {}
    recipes: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for recipe_path in sorted(root.glob("*/recipe.json")):
        recipe = read_json(recipe_path)
        name = recipe.get("name") or recipe_path.parent.name
        path = _resolve_candidate_file(recipe_path, recipe)
        if path is None:
            skipped[name] = f"no .gguf found next to {recipe_path}"
            print(f"WARNING: candidate {name} skipped — {skipped[name]}", flush=True)
            continue
        stale = []
        if recipe.get("size") is not None and int(recipe["size"]) != path.stat().st_size:
            stale.append(
                f"file is {path.stat().st_size} bytes, recipe recorded {recipe['size']}"
            )
        if (
            recipe.get("use_imatrix")
            and imatrix_sha
            and recipe.get("imatrix_sha256")
            and recipe["imatrix_sha256"] != imatrix_sha
        ):
            stale.append("the imatrix changed after this candidate was quantized")
        if (
            recipe.get("source", "reference") == "reference"
            and recipe.get("source_sha256")
            and recipe["source_sha256"] != reference_sha
        ):
            stale.append("the BF16 reference changed after this candidate was quantized")
        if stale:
            raise RuntimeError(
                f"Candidate {name} is stale ({'; '.join(stale)}). "
                "Re-run 05_quants.py before screening."
            )
        candidates[name] = path
        recipes[name] = recipe
    return candidates, recipes, skipped


def _runtime_flags() -> list[str]:
    return [
        "-fa", str(config.FLASH_ATTN),
        "-ctk", str(config.CACHE_TYPE_K),
        "-ctv", str(config.CACHE_TYPE_V),
    ]


def _edge_threads() -> int:
    configured = int(config.EDGE_BENCH.get("threads", 0))
    if configured > 0:
        return configured
    try:
        return max(1, len(os.sched_getaffinity(0)) // 2)
    except Exception:
        return max(1, getattr(config, "TOOL_THREADS", 4))


def _parse_mean_kld_se(output: str) -> tuple[float | None, float | None]:
    match = re.search(
        r"Mean\s+KLD:\s*([0-9.]+)\s*±\s*([0-9.]+)", output, re.IGNORECASE
    )
    if not match:
        return parse_mean_kld(output), None
    return float(match.group(1)), float(match.group(2))


def _parse_ppl(output: str) -> float | None:
    match = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
    return float(match.group(1)) if match else None


def _parse_same_top_p(output: str) -> float | None:
    match = re.search(r"Same top p:\s*([0-9.]+)\s*±", output)
    return float(match.group(1)) if match else None


def _parse_ppl_chunks(output: str) -> int | None:
    match = re.search(r"calculating perplexity over (\d+) chunks", output)
    return int(match.group(1)) if match else None


def _probe_mtp(binaries: dict[str, Path], dest: Path) -> dict:
    cli_help = tool_help(binaries["llama-cli"])
    bench_help = tool_help(binaries["llama-bench"])
    reference = read_json(reference_path().parent / "reference_manifest.json")
    inventory = gguf_inventory(reference_path(), binaries["converter"].parent)
    mtp_tensors = model.mtp_tensor_names(inventory["tensors"])
    draft_model = (os.environ.get("KUZA_DRAFT_MODEL", "").strip() or None)
    probe = {
        "cli_has_draft_max": help_has(cli_help, "--draft-max"),
        "cli_has_model_draft": help_has(cli_help, "-md")
        or help_has(cli_help, "--model-draft"),
        "bench_has_draft_max": help_has(bench_help, "--draft-max"),
        "reference_mtp_requested": bool(reference.get("mtp_requested")),
        "reference_mtp_tensors": mtp_tensors or reference.get("mtp_tensors") or [],
        "draft_model_env": draft_model or None,
        "speculative_args": [],
        "speculative_enabled": False,
    }
    if probe["reference_mtp_tensors"] and probe["cli_has_draft_max"]:
        probe["speculative_args"] = ["--draft-max", "4"]
        probe["speculative_enabled"] = True
    elif probe["cli_has_model_draft"] and probe["draft_model_env"]:
        probe["speculative_args"] = [
            "-md", probe["draft_model_env"], "--draft-max", "4",
        ]
        probe["speculative_enabled"] = True
    write_json(dest / "mtp_probe.json", probe)
    return probe


def _hidden_score(llama_cli: Path, gguf: Path, name: str, dest: Path) -> dict:
    if not EVAL_HIDDEN.is_file():
        raise RuntimeError(f"Missing {EVAL_HIDDEN}")
    if not HIDDEN_PROMPTS.is_file():
        raise RuntimeError(f"Missing hidden prompt set {HIDDEN_PROMPTS}")
    output = dest / f"hidden_{name}.json"
    HIDDEN_RESULTS.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable, EVAL_HIDDEN,
            "--gguf", gguf,
            "--name", name,
            "--llama-cli", llama_cli,
            "--ngl", "999",
            "--output", output,
        ]
    )
    report = read_json(output)
    shared = HIDDEN_RESULTS / f"hidden_{name}.json"
    if shared.resolve() != output.resolve():
        write_json(shared, report)
    return report


def _measure_rss(command: list, log_path: Path) -> tuple[str, float | None]:
    """Wrap a command with GNU time to capture peak RSS in KB."""
    gnu_time = Path("/usr/bin/time")
    if not gnu_time.is_file():
        output = run(command, capture=True, log_path=log_path)
        return output, None
    marker_log = log_path.with_suffix(".rss")
    wrapped = [str(gnu_time), "-f", "%M", "-o", str(marker_log), *command]
    try:
        output = run(wrapped, capture=True, log_path=log_path)
    except RuntimeError:
        output = run(command, capture=True, log_path=log_path)
        return output, None
    rss = None
    if marker_log.is_file():
        text = marker_log.read_text(encoding="utf-8").strip().splitlines()
        if text and text[-1].isdigit():
            rss = float(text[-1])
    return output, rss


def _smoke_long(binary: Path, weights: Path, prompt: str, log_path: Path) -> str:
    import subprocess

    command = [
        binary,
        "-m", weights,
        "-p", prompt,
        "-n", str(config.SMOKE_N_TOKENS),
        "-c", str(config.SMOKE_CTX),
        "-ngl", "999",
        "--temp", "0",
        "--no-display-prompt",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "-fa", str(config.FLASH_ATTN),
        *cli_oneshot_args(binary),
    ]
    return run(command, capture=True, log_path=log_path, stdin=subprocess.DEVNULL, timeout=600)


def _rank(candidates: dict[str, dict]) -> dict:
    controls = {
        name: {"mean_kld": spec.get("mean_kld"), "mean_kld_se": spec.get("mean_kld_se"),
               "ppl": spec.get("ppl"), "same_top_p": spec.get("same_top_p"),
               "size": spec.get("size"), "gate_failures": spec.get("gate_failures")}
        for name, spec in candidates.items() if spec.get("role") == "control"
    }
    passing = {
        name: spec for name, spec in candidates.items()
        if not spec["gate_failures"] and spec.get("role") != "control"
    }
    if not passing:
        return {"winner": None, "controls": controls,
                "rule": "no non-control candidate passed the quality gates",
                "ranked": []}
    best_kld = min(spec["mean_kld"] for spec in passing.values() if spec["mean_kld"] is not None)
    ses = [
        spec["mean_kld_se"] for spec in passing.values()
        if spec.get("mean_kld_se")
    ]
    tolerance = 2.0 * max(ses) if ses else 0.002
    tied = {
        name: spec for name, spec in passing.items()
        if spec["mean_kld"] is not None and spec["mean_kld"] <= best_kld + tolerance
    }
    ranked_names = sorted(
        tied,
        key=lambda name: (
            -float(passing[name].get("hidden_mean") or 0.0),
            -float(passing[name].get("cpu_generation_tps") or 0.0),
            int(passing[name]["size"]),
            float(passing[name].get("peak_rss_kb") or 0.0),
        ),
    )
    # Non-tied candidates follow, ordered by KLD fidelity.
    outside = sorted(
        (name for name in passing if name not in tied),
        key=lambda name: passing[name]["mean_kld"],
    )
    ordered = ranked_names + outside
    return {
        "winner": ordered[0] if ordered else None,
        "controls": controls,
        "kld_noise_tolerance": tolerance,
        "tied_group": sorted(tied),
        "ranked": [
            {
                "name": name,
                "hidden_mean": passing[name].get("hidden_mean"),
                "mean_kld": passing[name].get("mean_kld"),
                "mean_kld_se": passing[name].get("mean_kld_se"),
                "ppl": passing[name].get("ppl"),
                "same_top_p": passing[name].get("same_top_p"),
                "size": passing[name].get("size"),
                "cpu_generation_tps": passing[name].get("cpu_generation_tps"),
                "cpu_prompt_tps": passing[name].get("cpu_prompt_tps"),
                "gpu_generation_tps": passing[name].get("gpu_generation_tps"),
                "peak_rss_kb": passing[name].get("peak_rss_kb"),
            }
            for name in ordered
        ],
        "rule": (
            "controls (Q8_0 ceiling) never win; gates first; among gate-passers "
            "within 2x KLD SE of the best: "
            "hidden_mean desc, CPU tg128 desc, size asc, peak RSS asc. "
            "Outside that band: KLD asc. CPU numbers are primary; GPU t/s "
            "and KLD are diagnostics."
        ),
    }


def main() -> int:
    hf_token()
    require_file(adapter_dir() / "adapter_config.json", "Run 02_sft.py first.")
    require_file(reference_path(), "Run 03_reference.py first.")
    require_file(eval_path(), "Run 04_imatrix.py first.")
    binaries = llama_cpp_binaries()
    tokenizer = model.load_tokenizer(adapter_dir())
    dest = screen_dir()
    dest.mkdir(parents=True, exist_ok=True)

    # Discover candidates from recipes written by 05_quants.py.
    reference_sha = sha256_file(reference_path())
    imatrix_file = imatrix_path()
    imatrix_sha = sha256_file(imatrix_file) if imatrix_file.is_file() else None
    candidates, recipes, skipped_candidates = discover_candidates(
        quants_dir(), reference_sha, imatrix_sha
    )
    if not candidates:
        raise RuntimeError("No candidates found under quants/*/recipe.json — run 05_quants.py.")
    for name, path in candidates.items():
        require_file(path, f"candidate {name} is missing.")
    print(f"candidates: {', '.join(candidates)}", flush=True)

    mtp_probe = _probe_mtp(binaries, dest)
    runtime_flags = _runtime_flags()

    # ---- BF16 KLD base (large sample) ----
    kld_base = dest / "kuza-bf16-reference.kld"
    reference_output = run(
        [
            binaries["llama-perplexity"],
            "-m", reference_path(),
            "-f", eval_path(),
            "-s", str(config.SEED),
            "-c", str(config.EVAL_CTX),
            "-b", "512",
            "--chunks", str(config.SCREEN_KLD_CHUNKS),
            "-ngl", "999",
            "--kl-divergence-base", kld_base,
        ],
        capture=True,
        log_path=dest / "bf16-kld.log",
    )
    if not kld_base.is_file() or kld_base.stat().st_size == 0:
        raise RuntimeError("BF16 KLD reference file was not created")
    ref_ppl = _parse_ppl(reference_output)
    kld_chunks_effective = _parse_ppl_chunks(reference_output)
    if kld_chunks_effective is not None and kld_chunks_effective < config.SCREEN_KLD_CHUNKS:
        print(
            f"WARNING: KLD screening ran over {kld_chunks_effective} chunks, not the "
            f"configured {config.SCREEN_KLD_CHUNKS} — the eval corpus "
            f"({eval_path().name}) is too short. Raise EVAL_MIN_FULL_CHUNKS in "
            "04_imatrix.py (and re-run 04) if you need the tighter KLD error bars.",
            flush=True,
        )
    if ref_ppl is not None and ref_ppl < 6.0:
        print(
            f"WARNING: BF16 reference PPL={ref_ppl:.2f} is suspiciously low; "
            "the eval corpus may be templated or contaminated (review F2)",
            flush=True,
        )

    results = {
        "notice": (
            "CPU arm is the primary performance signal for the edge target. "
            "Winner: gates, then hidden-set accuracy within KLD noise, then "
            "CPU t/s, size, RSS."
        ),
        "run_id": config.RUN_ID,
        "work_dir": str(run_dir()),
        "llama_cpp_commit": config.LLAMA_CPP_COMMIT,
        "prompt_sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
        "packages": installed_packages(),
        "screen_kld_chunks": config.SCREEN_KLD_CHUNKS,
        "kld_chunks_effective": kld_chunks_effective,
        "skipped_candidates": skipped_candidates,
        "gates": config.SCREEN_GATES,
        "runtime": {
            "flash_attn": config.FLASH_ATTN,
            "cache_type_k": config.CACHE_TYPE_K,
            "cache_type_v": config.CACHE_TYPE_V,
            "mtp_probe": mtp_probe,
            "edge_threads": _edge_threads(),
        },
        "reference": {
            "path": reference_path().name,
            "sha256": reference_sha,
            "ppl": ref_ppl,
            "raw_log": "bf16-kld.log",
        },
        "candidates": {},
    }

    for name, path in candidates.items():
        print(f"screening {name}", flush=True)
        kld_output, rss = _measure_rss(
            [
                binaries["llama-perplexity"],
                "-m", path,
                "-f", eval_path(),
                "-s", str(config.SEED),
                "-c", str(config.EVAL_CTX),
                "-b", "512",
                "--chunks", str(config.SCREEN_KLD_CHUNKS),
                "-ngl", "999",
                "--kl-divergence-base", kld_base,
                "--kl-divergence",
            ],
            dest / f"{name}-kld.log",
        )
        mean_kld, mean_kld_se = _parse_mean_kld_se(kld_output)
        if mean_kld is None:
            raise RuntimeError(f"Could not parse mean KLD for {name}; screening cannot continue")
        same_top_p = _parse_same_top_p(kld_output)
        ppl = _parse_ppl(kld_output)

        # ---- CPU bench arm (PRIMARY for the edge target) ----
        cpu_cmd = [
            binaries["llama-bench"],
            "-m", path,
            "-p", str(config.EDGE_BENCH["prompt_tokens"]),
            "-n", str(config.EDGE_BENCH["gen_tokens"]),
            "-ngl", "0",
            "-t", str(_edge_threads()),
            "-r", str(config.EDGE_BENCH["repeats"]),
            *runtime_flags,
        ]
        cpu_out, cpu_rss = _measure_rss(cpu_cmd, dest / f"{name}-bench-cpu.log")
        cpu_values = parse_bench_tps(cpu_out)
        # parse_bench_tps flattens the table; pp comes first, then tg.
        cpu_pp = cpu_values[0] if len(cpu_values) >= 2 else None
        cpu_tg = cpu_values[-1] if len(cpu_values) >= 2 else (cpu_values[0] if cpu_values else None)
        peak_rss = cpu_rss  # CPU bench = edge memory budget; the KLD run's RSS is a GPU-offload artifact

        # ---- GPU bench arm (diagnostic only; skipped if no GPU) ----
        gpu_tg = None
        try:
            gpu_out = run(
                [
                    binaries["llama-bench"],
                    "-m", path,
                    "-p", "512", "-n", "128",
                    "-ngl", "999", "-t", "4", "-r", "3",
                    *runtime_flags,
                ],
                capture=True,
                log_path=dest / f"{name}-bench-gpu.log",
            )
            gpu_values = parse_bench_tps(gpu_out)
            gpu_tg = gpu_values[-1] if gpu_values else None
        except RuntimeError as exc:
            results.setdefault("gpu_bench_unavailable", str(exc)[:200])

        # ---- hidden-set score + smoke ----
        hidden = _hidden_score(binaries["llama-cli"], path, name, dest)
        prompt_results = {}
        for prompt_name, prompt in model.SCREEN_SMOKE_PROMPTS.items():
            output = _smoke_long(
                binaries["llama-cli"], path,
                model.render_generation_prompt(tokenizer, prompt),
                dest / f"{name}-{prompt_name}.log",
            )
            prompt_results[prompt_name] = {
                "log": f"{name}-{prompt_name}.log",
                "warnings": model.smoke_warnings(prompt_name, output),
            }

        inventory = gguf_inventory(path, binaries["converter"].parent)
        results["candidates"][name] = {
            "file": path.name,
            "role": "control" if _is_control(name, recipes[name]) else "candidate",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "recipe": recipes[name],
            "mean_kld": mean_kld,
            "mean_kld_se": mean_kld_se,
            "ppl": ppl,
            "same_top_p": same_top_p,
            "cpu_generation_tps": cpu_tg,
            "cpu_prompt_tps": cpu_pp,
            "gpu_generation_tps": gpu_tg,
            "peak_rss_kb": peak_rss,
            "kld_run_rss_kb": rss,
            "speculative_used": bool(mtp_probe.get("speculative_args")),
            "tensor_type_counts": inventory["tensor_type_counts"],
            "smoke": prompt_results,
            "hidden_mean": hidden.get("mean_score"),
            "safety_failures": hidden.get("safety_failures"),
            "language_failures": hidden.get("language_failures"),
            "hidden_report": f"hidden_{name}.json",
        }

    # ---- gates ----
    gates = config.SCREEN_GATES
    for name, spec in results["candidates"].items():
        failures = []
        if (spec.get("safety_failures") or 0) > gates["max_safety_failures"]:
            failures.append(f"safety_failures={spec.get('safety_failures')}")
        if (spec.get("language_failures") or 0) > gates["max_language_failures"]:
            failures.append(f"language_failures={spec.get('language_failures')}")
        if spec.get("same_top_p") is not None and spec["same_top_p"] < gates["min_same_top_p"]:
            failures.append(f"same_top_p={spec['same_top_p']}")
        if spec.get("mean_kld") is not None and spec["mean_kld"] > gates["max_mean_kld"]:
            failures.append(f"mean_kld={spec['mean_kld']}")
        spec["gate_failures"] = failures

    results["selection"] = _rank(results["candidates"])
    write_json(dest / "results.json", results)
    print(f"screen: {dest / 'results.json'}")
    print(f"winner: {results['selection']['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())