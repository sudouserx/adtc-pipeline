#!/usr/bin/env python3
"""Aggregate screening results into a detailed diagnosis report."""

from __future__ import annotations

import json
from pathlib import Path

import config
from study_common import read_json, write_json


def fmt_gb(size: int | None) -> str:
    if not size:
        return "-"
    return f"{size / (1024**3):.2f}"


def load_baseline() -> dict | None:
    path = config.RESULTS_DIR / "baseline" / "summary.json"
    return read_json(path) if path.is_file() else None


def load_screen() -> dict:
    for path in (
        config.SCREEN_DIR / "results.json",
        config.RESULTS_DIR / "screen" / "results.json",
    ):
        if path.is_file():
            return read_json(path)
    raise RuntimeError("Run 05_screen.py first (missing screen/results.json)")


def verdict(bf16_hidden: float | None, candidates: dict) -> tuple[str, str]:
    if bf16_hidden is None:
        return "inconclusive", "BF16 hidden score unavailable."
    if bf16_hidden >= config.HIDDEN_GOOD_THRESHOLD:
        quant_scores = [
            float(item["hidden_mean"])
            for name, item in candidates.items()
            if name != "bf16_reference" and item.get("hidden_mean") is not None
        ]
        if quant_scores and max(quant_scores) < bf16_hidden - 0.05:
            return (
                "quantization_issue",
                f"BF16 hidden_mean={bf16_hidden:.3f} is acceptable, but best quant "
                f"({max(quant_scores):.3f}) is materially lower.",
            )
        return (
            "mixed_or_acceptable",
            f"BF16 hidden_mean={bf16_hidden:.3f}; quant gap may be small or method-dependent.",
        )
    return (
        "finetuning_or_template_issue",
        f"BF16 hidden_mean={bf16_hidden:.3f} is below threshold "
        f"{config.HIDDEN_GOOD_THRESHOLD}; quantization is unlikely the root cause.",
    )


def pareto_front(rows: list[dict]) -> list[str]:
    """Non-dominated on hidden_mean (max), size (min), tps (max)."""
    front: list[str] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other["name"] == row["name"]:
                continue
            better_or_equal = (
                other.get("hidden_mean", 0) >= row.get("hidden_mean", 0)
                and other.get("size", 10**18) <= row.get("size", 10**18)
                and other.get("generation_tps", 0) >= row.get("generation_tps", 0)
            )
            strictly_better = (
                other.get("hidden_mean", 0) > row.get("hidden_mean", 0)
                or other.get("size", 10**18) < row.get("size", 10**18)
                or other.get("generation_tps", 0) > row.get("generation_tps", 0)
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row["name"])
    return front


def write_analysis_md(
    path: Path,
    *,
    diagnosis: str,
    explanation: str,
    ranked: list[dict],
    bf16: dict | None,
    pareto: list[str],
    winner: str | None,
) -> None:
    lines = [
        "# Qwen 3.5-4B Quantization vs Finetuning Diagnosis",
        "",
        f"**Verdict:** `{diagnosis}`",
        "",
        explanation,
        "",
        f"Hidden-good threshold: **{config.HIDDEN_GOOD_THRESHOLD}**",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Hidden | Δ hidden | KLD | Δ KLD | Size (GiB) | TPS | EN | SW | Safety fail |",
        "|------|-------|--------|----------|-----|-------|------------|-----|----|----|-------------|",
    ]
    for idx, row in enumerate(ranked, start=1):
        en = (row.get("hidden_by_language") or {}).get("en")
        sw = (row.get("hidden_by_language") or {}).get("sw")
        lines.append(
            f"| {idx} | {row['name']} | {row.get('hidden_mean', '-')} "
            f"| {row.get('delta_hidden_vs_bf16', '-')} "
            f"| {row.get('mean_kld', '-')} "
            f"| {row.get('delta_kld_vs_bf16', '-')} "
            f"| {fmt_gb(row.get('size'))} "
            f"| {row.get('generation_tps', '-')} "
            f"| {en or '-'} | {sw or '-'} "
            f"| {row.get('safety_failures', '-')} |"
        )
    lines.extend(
        [
            "",
            f"**Screen winner:** `{winner}`",
            "",
            "## Pareto frontier (accuracy / size / speed)",
            "",
            ", ".join(f"`{name}`" for name in pareto) if pareto else "_none_",
            "",
            "## BF16 reference",
            "",
        ]
    )
    if bf16:
        lines.append(
            f"- hidden_mean: **{bf16.get('hidden_mean')}** "
            f"(safety_failures={bf16.get('safety_failures')}, "
            f"language_failures={bf16.get('language_failures')})"
        )
        lines.append(f"- mean_kld: {bf16.get('mean_kld')} (reference baseline)")
    else:
        lines.append("_BF16 baseline not found in screen results._")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `reference/kuza-bf16.gguf` is the merged finetuned model (same weights as `merged_bf16/`).",
            "- Past run did not upload `screen/results.json` to Hugging Face; scores were recomputed.",
            "- Reference manifest shows `has_mtp: false` (MTP stripped at GGUF conversion).",
            "- KLD is computed on domain `imatrix/eval.txt` (100 EN + 100 SW held-out rows).",
            "- IQ3_XS / I-quants may trade throughput for size; check TPS before deployment.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if winner and winner != "bf16_reference":
        w = next((r for r in ranked if r["name"] == winner), None)
        if w:
            lines.append(
                f"Ship **`{winner}`** ({fmt_gb(w.get('size'))} GiB, "
                f"hidden={w.get('hidden_mean')}, TPS={w.get('generation_tps')}) "
                f"if accuracy meets deployment needs."
            )
        else:
            lines.append(f"Top ranked quant: **`{winner}`**.")
    elif diagnosis == "finetuning_or_template_issue":
        lines.append(
            "Do **not** prioritize new quantization until finetuning, chat template, "
            "or system prompt quality is improved."
        )
    else:
        lines.append("Review Pareto candidates above for accuracy vs VRAM vs throughput.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    screen = load_screen()
    baseline = load_baseline()
    candidates: dict = screen.get("candidates") or {}
    bf16 = candidates.get("bf16_reference")
    bf16_hidden = float(bf16["hidden_mean"]) if bf16 and bf16.get("hidden_mean") is not None else None

    diagnosis, explanation = verdict(bf16_hidden, candidates)
    ranked = screen.get("selection", {}).get("ranked") or []
    ranked_full = []
    for item in ranked:
        name = item["name"]
        merged = dict(candidates.get(name) or {})
        merged["name"] = name
        ranked_full.append(merged)

    pareto = pareto_front(
        [
            {
                "name": name,
                "hidden_mean": float(entry.get("hidden_mean") or 0),
                "size": int(entry.get("size") or 0),
                "generation_tps": float(entry.get("generation_tps") or 0),
            }
            for name, entry in candidates.items()
            if name != "bf16_reference"
        ]
    )

    report_dir = config.REPORT_DIR
    per_model_dir = report_dir / "per_model"
    report_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)

    for name, entry in candidates.items():
        hidden_report = entry.get("hidden_report")
        if hidden_report and Path(hidden_report).is_file():
            entry = dict(entry)
            entry["hidden_detail"] = read_json(Path(hidden_report))
        write_json(per_model_dir / f"{name}.json", entry)

    summary = {
        "diagnosis": diagnosis,
        "explanation": explanation,
        "hidden_good_threshold": config.HIDDEN_GOOD_THRESHOLD,
        "bf16_hidden_mean": bf16_hidden,
        "baseline_preliminary_verdict": (baseline or {}).get("preliminary_verdict"),
        "winner": screen.get("selection", {}).get("winner"),
        "pareto_frontier": pareto,
        "ranked": ranked,
        "candidates": {
            name: {
                "hidden_mean": item.get("hidden_mean"),
                "delta_hidden_vs_bf16": item.get("delta_hidden_vs_bf16"),
                "mean_kld": item.get("mean_kld"),
                "delta_kld_vs_bf16": item.get("delta_kld_vs_bf16"),
                "size": item.get("size"),
                "generation_tps": item.get("generation_tps"),
                "hidden_by_language": item.get("hidden_by_language"),
                "hidden_by_category": item.get("hidden_by_category"),
                "safety_failures": item.get("safety_failures"),
                "base_type": item.get("base_type"),
                "source": item.get("source"),
            }
            for name, item in candidates.items()
        },
    }
    write_json(report_dir / "summary.json", summary)
    write_analysis_md(
        report_dir / "analysis.md",
        diagnosis=diagnosis,
        explanation=explanation,
        ranked=ranked_full,
        bf16=bf16,
        pareto=pareto,
        winner=screen.get("selection", {}).get("winner"),
    )
    print(f"report: {report_dir / 'analysis.md'}")
    print(f"verdict: {diagnosis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
