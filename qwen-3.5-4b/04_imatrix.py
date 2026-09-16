#!/usr/bin/env python3
"""Build a bilingual imatrix and held-out eval corpus from the BF16 reference."""

from __future__ import annotations

import config
import model
from common import (
    adapter_dir,
    eval_path,
    heldout_path,
    hf_token,
    imatrix_dir,
    imatrix_path,
    llama_cpp_binaries,
    read_jsonl,
    reference_path,
    render_corpus,
    require_file,
    run,
    select_balanced,
    select_mixcal,
    sha256_bytes,
    write_json,
)


def main() -> int:
    hf_token()
    require_file(adapter_dir() / "adapter_config.json", "Run 02_sft.py first.")
    require_file(heldout_path(), "Run 02_sft.py first.")
    require_file(reference_path(), "Run 03_reference.py first.")
    binaries = llama_cpp_binaries()

    dest_dir = imatrix_dir()
    dest = imatrix_path()
    evaluation = eval_path()
    dest_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(heldout_path())
    calibration_rows = select_mixcal(
        rows,
        config.CALIBRATION_PER_LANGUAGE,
        getattr(config, "CALIBRATION_GENERIC", 200),
        config.SEED,
    )
    calibration_ids = {(row["source"], row["source_id"]) for row in calibration_rows}
    remaining = [
        row
        for row in rows
        if (row["source"], row["source_id"]) not in calibration_ids
    ]
    eval_rows = select_balanced(remaining, config.EVAL_PER_LANGUAGE, config.SEED + 20)
    if not calibration_rows or not eval_rows:
        raise RuntimeError("Insufficient balanced held-out rows for calibration/eval")

    tokenizer = model.load_tokenizer(adapter_dir())
    calibration_file = dest_dir / "calibration.txt"
    calib_stats = render_corpus(
        tokenizer, calibration_rows, calibration_file, config.MAX_SEQ_LENGTH
    )
    eval_stats = render_corpus(
        tokenizer, eval_rows, evaluation, config.MAX_SEQ_LENGTH
    )
    run(
        [
            binaries["llama-imatrix"],
            "-m",
            reference_path(),
            "-f",
            calibration_file,
            "-o",
            dest,
            "-ngl",
            "999",
            "-c",
            str(config.MAX_SEQ_LENGTH),
            "-b",
            "512",
            "--chunks",
            "200",
            "-t",
            "4",
        ],
        log_path=dest_dir / "imatrix.log",
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("llama-imatrix did not produce a valid artifact")
    write_json(
        dest_dir / "corpus_manifest.json",
        {
            "calibration": calib_stats,
            "evaluation": eval_stats,
            "prompt_sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
            "overlap": sorted(
                calibration_ids
                & {(row["source"], row["source_id"]) for row in eval_rows}
            ),
        },
    )
    print(f"imatrix: {dest}")
    print(f"eval: {evaluation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
