#!/usr/bin/env python3
"""Pick a submission GGUF from hidden scores and GPU screen results."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PROVENANCE = ROOT / "experiments" / "provenance"
WORK_DIR = Path(os.environ.get("KUZA_WORK_DIR", "/workspace/kuza-pipeline"))
RUN_IDS = ("kuza-gemma-4-e2b", "kuza-qwen-3.5-4b")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def hidden_scores() -> list[dict]:
    scores = []
    for path in RESULTS.glob("hidden_*.json"):
        data = read_json(path)
        if data:
            scores.append(data)
    return scores


def screen_payloads() -> list[dict]:
    rows = []
    seen: set[str] = set()

    def add(path: Path, default_name: str) -> None:
        key = str(path.resolve()) if path.is_file() else ""
        if not key or key in seen:
            return
        data = read_json(path)
        if not data:
            return
        data.setdefault("name", data.get("run_id") or default_name)
        data["source"] = str(path)
        seen.add(key)
        rows.append(data)

    for path in RESULTS.glob("screen_*.json"):
        add(path, path.stem)
    for run_id in RUN_IDS:
        add(WORK_DIR / run_id / "screen" / "results.json", run_id)
    if PROVENANCE.is_dir():
        for path in PROVENANCE.glob("*/screen_results.json"):
            add(path, path.parent.name)
    return rows


def attach_screen(hidden: dict, screens: list[dict]) -> dict | None:
    gguf = str(hidden.get("gguf") or "")
    name = str(hidden.get("name") or "")
    for screen in screens:
        if name and name in {screen.get("name"), screen.get("run_id")}:
            return screen
        for spec in (screen.get("candidates") or {}).values():
            file_name = str(spec.get("file") or "")
            if file_name and file_name in gguf:
                return screen
    return None


def rank_candidates() -> list[dict]:
    screens = screen_payloads()
    ranked = []
    for hidden in hidden_scores():
        screen = attach_screen(hidden, screens)
        candidates = (screen or {}).get("candidates") or {}
        ranked.append(
            {
                "name": hidden.get("name"),
                "gguf": hidden.get("gguf"),
                "hidden_mean": hidden.get("mean_score", 0),
                "safety_failures": hidden.get("safety_failures", 0),
                "language_failures": hidden.get("language_failures", 0),
                "screen_run": (screen or {}).get("run_id") or (screen or {}).get("name"),
                "screen_candidates": {
                    key: {
                        "mean_kld": spec.get("mean_kld"),
                        "bench_values": spec.get("bench_values"),
                    }
                    for key, spec in candidates.items()
                }
                if candidates
                else None,
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["hidden_mean"]),
            int(row["safety_failures"]),
            int(row["language_failures"]),
        )
    )
    return ranked


def main() -> int:
    ranked = rank_candidates()
    winner = ranked[0] if ranked else None
    payload = {
        "ranked": ranked,
        "winner": winner,
        "note": (
            "Accuracy (hidden rubric) is the primary rank. "
            "Use GPU screen KLD/TPS on the top candidates before submitting."
        ),
    }
    dest = RESULTS / "winner.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
