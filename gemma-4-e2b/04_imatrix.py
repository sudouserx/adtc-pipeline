#!/usr/bin/env python3
"""Build a bilingual imatrix and held-out eval corpus from the BF16 reference.

REPLACEMENT (review 2026-09, finding F1 — Critical). FIXED 2026-09-21.

What went wrong in the archived run
-----------------------------------
corpus_manifest.json recorded ``rows: 1000, tokens: 2000`` for a 1.9 MB
calibration file (i.e. 2 tokens/row — the token accounting in
``render_corpus``/``encode_ids`` was broken), so ``chunks`` resolved to
``ceil(2000/1024)+8 = 10`` and llama-imatrix logged
``computing over 10 chunks`` — the importance matrix was estimated from
~10,240 tokens instead of the intended ~1M. Every quantized candidate
inherited a high-variance, system-prompt-biased importance matrix.

What this replacement does
--------------------------
  1. HARDENED token accounting: structural checks raise immediately on
     unwrap bugs; legitimately short rows are SKIPPED (counted) rather than
     crashing the stage; a final cross-check enforces
     config.CALIBRATION_MIN_TOKENS (default 512K) — a 10-chunk imatrix can
     never be produced silently again.
  2. Corpus composition: 75% of chunks are raw user/model turns rendered
     WITHOUT the fixed system prompt; 25% are full chat renders. FIX
     2026-09-21: the Kuza-patched chat template INJECTS the system prompt
     whenever the first turn is not a system turn, so raw renders used to
     fail 100% of the time and the stage died with "No calibration rows
     could be rendered". Raw renders now temporarily swap in the STOCK
     gemma-4 template (no injection).
  3. Chunks are derived from the VERIFIED token count with a safety margin,
     and the produced imatrix is verified for tensor coverage before handoff.
  4. FIX: the English top-up for small held-out pools now also excludes rows
     selected into the KLD EVAL corpus (calibration must not overlap the
     screening corpus), and the eval corpus is truncated at EVAL_CTX (the
     context the perplexity stage actually runs at), not CALIBRATION_CTX.
"""

from __future__ import annotations

import math
from typing import Any

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
    load_local_or_hub,
    read_jsonl,
    reference_path,
    render_corpus,
    require_file,
    run,
    select_balanced,
    sha256_bytes,
    write_json,
)

MIN_ROW_TOKENS = 8
# Gemma tokenizers average ~3-4 chars/token; a row using >2 tokens/char means
# the BatchEncoding unwrap broke (the archived run's root cause).
CHARS_PER_TOKEN_FLOOR = 2.0

_STOCK_TEMPLATE_CACHE: str | None = None
_STOCK_TEMPLATE_PROBED = False


def _stock_gemma4_template(tokenizer: Any) -> str:
    """The stock gemma-4 chat template (no Kuza system-prompt injection).

    FIX 2026-09-21: the Kuza-patched template injects the canonical system
    prompt whenever the first turn is not a system turn, which makes
    system-free renders impossible through it. Probe unsloth for the stock
    template (on a COPY so the working tokenizer is never mutated).
    """
    global _STOCK_TEMPLATE_CACHE, _STOCK_TEMPLATE_PROBED
    if not _STOCK_TEMPLATE_PROBED:
        _STOCK_TEMPLATE_PROBED = True
        try:
            import copy as _copy

            from unsloth.chat_templates import get_chat_template

            probe = _copy.copy(model.text_tokenizer(tokenizer))
            wrapped = get_chat_template(probe, chat_template=model.CHAT_TEMPLATE_NAME)
            template = getattr(wrapped, "chat_template", None)
            if not template:
                inner = model.text_tokenizer(wrapped)
                template = getattr(inner, "chat_template", None)
            if isinstance(template, str) and template:
                _STOCK_TEMPLATE_CACHE = template
        except Exception as exc:  # noqa: BLE001 — surfaced below
            print(
                f"imatrix: could not obtain the stock "
                f"{model.CHAT_TEMPLATE_NAME} template ({exc!r})",
                flush=True,
            )
    if _STOCK_TEMPLATE_CACHE is None:
        raise RuntimeError(
            "The stock gemma-4 chat template is unavailable, so raw "
            "(system-free) calibration chunks cannot be rendered. Ensure "
            "unsloth is importable in this environment, then re-run."
        )
    return _STOCK_TEMPLATE_CACHE


def verify_ids(tokenizer, text: str, ids: list, where: str) -> int:
    """Structural sanity on encode_ids output (review F1) — raises on breakage.

    Length handling is the CALLER's job: legitimately tiny rows are skipped,
    not treated as accounting bugs (the old MIN_ROW_TOKENS raise here used to
    kill the whole stage on one short row).
    """
    if not isinstance(ids, list) or (ids and isinstance(ids[0], list)):
        raise RuntimeError(f"{where}: encode_ids did not unwrap to a flat id list")
    if len(ids) * CHARS_PER_TOKEN_FLOOR > len(text) + 64:
        raise RuntimeError(
            f"{where}: {len(ids)} tokens for {len(text)} chars exceeds the "
            f"{CHARS_PER_TOKEN_FLOOR} chars/token sanity floor; the tokenizer "
            "call is returning an unexpected structure (review F1)"
        )
    return len(ids)


def render_raw(tokenizer, row: dict) -> str:
    """Chat-style user/model turns WITHOUT the fixed system prompt.

    FIX 2026-09-21: temporarily swap the Kuza-patched template for the stock
    gemma-4 template — the patched one injects the system prompt into any
    system-less dialog, which used to fail every raw render.
    """
    dialog = model.dialog_messages(row)
    if not dialog:
        raise RuntimeError("Row has no alternating user/model turns")
    inner = model.text_tokenizer(tokenizer)
    patched = getattr(inner, "chat_template", None)
    inner.chat_template = _stock_gemma4_template(tokenizer)
    try:
        rendered = model._apply_chat_template(
            inner, dialog, tokenize=False, add_generation_prompt=False
        )
    finally:
        if patched is not None:
            inner.chat_template = patched
    if model.SYSTEM_PROMPT in rendered:
        raise RuntimeError("Raw render leaked the system prompt")
    return rendered


def build_calibration(
    tokenizer, rows: list[dict], dest_dir, system_prompt_fraction: float
) -> tuple[dict, dict]:
    """Interleave raw (75%) and chat (25%) renders until the token floor is met."""
    raw_rows = []
    chat_rows = []
    skipped = 0
    for row in rows:
        try:
            raw_rows.append(render_raw(tokenizer, row))
            chat_rows.append(model.render_text(tokenizer, row))
        except RuntimeError:
            skipped += 1
    if not raw_rows:
        raise RuntimeError("No calibration rows could be rendered")

    chat_share = max(0.0, min(system_prompt_fraction, 0.5))
    target = config.CALIBRATION_MIN_TOKENS
    total = 0
    texts: list[str] = []
    short_rows = 0
    raw_index = 0
    chat_index = 0
    # Interleave 1 chat per (1/chat_share - 1) raw rows so ANY prefix of the
    # corpus (llama-imatrix consumes leading chunks) is composition-correct.
    raw_per_chat = max(1, round((1.0 - chat_share) / chat_share)) if chat_share else 10**9
    while total < target and (raw_index < len(raw_rows) or chat_index < len(chat_rows)):
        for _ in range(raw_per_chat):
            if raw_index >= len(raw_rows):
                break
            text = raw_rows[raw_index]
            raw_index += 1
            count = verify_ids(tokenizer, text, model.encode_ids(tokenizer, text), "raw")
            if count < MIN_ROW_TOKENS:
                short_rows += 1
                continue
            texts.append(text)
            total += count
        if chat_index < len(chat_rows):
            text = chat_rows[chat_index]
            chat_index += 1
            count = verify_ids(tokenizer, text, model.encode_ids(tokenizer, text), "chat")
            if count < MIN_ROW_TOKENS:
                short_rows += 1
                continue
            texts.append(text)
            total += count

    if total < target:
        raise RuntimeError(
            f"Calibration corpus reached only {total} tokens (< "
            f"config.CALIBRATION_MIN_TOKENS={target}). Enlarge the held-out "
            "pool or set KUZA_CALIB_TOKENS lower — never ship an "
            "under-calibrated imatrix (review F1)."
        )

    stats = {"rows_rendered": len(texts), "raw": raw_index, "chat": chat_index,
             "skipped_rows": skipped, "short_rows": short_rows, "tokens": total}
    file = dest_dir / "calibration.txt"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    import common

    stats["sha256"] = common.sha256_file(file)
    stats["size"] = file.stat().st_size
    return stats, file


def verify_imatrix_coverage(path, llama_root) -> dict:
    """The imatrix is a GGUF of per-tensor moments; make sure it covers the model."""
    import contextlib
    import sys

    sys.path.insert(0, str(llama_root / "gguf-py"))
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path), "r")
        fields = list(reader.fields.keys())
        has_ffn = any("ffn_down" in name for name in fields)
        has_attn = any("attn_q" in name or "attn_v" in name for name in fields)
        report = {
            "tensor_entries": len(fields),
            "covers_ffn": has_ffn,
            "covers_attention": has_attn,
        }
        if len(fields) < 100 or not (has_ffn and has_attn):
            raise RuntimeError(
                f"imatrix coverage looks wrong: {report} — regenerate before "
                "quantizing (review F1)"
            )
        return report
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(llama_root / "gguf-py"))


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
    calibration_rows = _select_mixcal(rows)
    calibration_ids = {(row["source"], row["source_id"]) for row in calibration_rows}
    remaining = [
        row for row in rows if (row["source"], row["source_id"]) not in calibration_ids
    ]
    eval_rows = select_balanced(remaining, config.EVAL_PER_LANGUAGE, config.SEED + 20)
    if not calibration_rows or not eval_rows:
        raise RuntimeError("Insufficient balanced held-out rows for calibration/eval")
    # FIX: the KLD screening corpus must stay disjoint from calibration rows.
    eval_ids = {(row["source"], row["source_id"]) for row in eval_rows}

    # Top up from the local English pool if the held-out corpus is too small
    # (calibration is not evaluation; training-distribution rows are fine here
    # and exclude held-out AND eval-corpus identities).
    if len(calibration_rows) * 600 < config.CALIBRATION_MIN_TOKENS:
        extra = load_local_or_hub(
            "english", "english", "english.jsonl", {"english": "main"}
        )
        extra = [
            row for row in extra
            if (row["source"], row["source_id"]) not in calibration_ids
            and (row["source"], row["source_id"]) not in eval_ids
        ]
        calibration_rows = calibration_rows + extra
        print(
            f"topped up calibration pool with {len(extra)} english rows "
            "to reach the token floor",
            flush=True,
        )

    tokenizer = model.load_tokenizer(adapter_dir())
    calib_stats, calibration_file = build_calibration(
        tokenizer,
        calibration_rows,
        dest_dir,
        config.CALIBRATION_SYSTEM_PROMPT_MAX_FRACTION,
    )
    # FIX: truncate the eval corpus at EVAL_CTX — the context the perplexity
    # stage actually runs at — not CALIBRATION_CTX.
    eval_stats = render_corpus(
        tokenizer, eval_rows, evaluation, config.EVAL_CTX
    )
    # Eval corpus token sanity as well (screening reads this file).
    if eval_stats["tokens"] < 100 * config.EVAL_CTX:
        raise RuntimeError(
            f"Eval corpus only {eval_stats['tokens']} tokens; screening KLD "
            "needs >= 100 full-context chunks (review F2)."
        )

    chunks = max(1, math.ceil(calib_stats["tokens"] / config.CALIBRATION_CTX) + 8)
    if calib_stats["tokens"] / chunks < config.CALIBRATION_CTX * 0.9:
        raise RuntimeError("chunk math inconsistent with verified token count")
    run(
        [
            binaries["llama-imatrix"],
            "-m", reference_path(),
            "-f", calibration_file,
            "-o", dest,
            "-ngl", "999",
            "-c", str(config.CALIBRATION_CTX),
            "-b", "512",
            "--chunks", str(chunks),
            "-t", str(getattr(config, "TOOL_THREADS", 4)),
        ],
        log_path=dest_dir / "imatrix.log",
    )
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("llama-imatrix did not produce a valid artifact")
    coverage = verify_imatrix_coverage(dest, binaries["converter"].parent)
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
            "imatrix_chunks": chunks,
            "imatrix_coverage": coverage,
            "calibration_token_floor": config.CALIBRATION_MIN_TOKENS,
            "system_prompt_max_fraction": config.CALIBRATION_SYSTEM_PROMPT_MAX_FRACTION,
        },
    )
    print(f"imatrix: {dest}")
    print(f"eval: {evaluation}")
    print(f"chunks: {chunks} (verified {calib_stats['tokens']} tokens)")
    return 0


def _select_mixcal(rows: list[dict]) -> list[dict]:
    """Local fallback mirroring common.select_mixcal with new quotas."""
    from common import deterministic_sample

    english = [
        row for row in rows
        if row.get("language") == "english" and row.get("source") != "general"
    ]
    swahili = [row for row in rows if row.get("language") == "swahili"]
    generic = [row for row in rows if row.get("source") == "general"]
    return [
        *deterministic_sample(english, config.CALIBRATION_PER_LANGUAGE, config.SEED),
        *deterministic_sample(swahili, config.CALIBRATION_PER_LANGUAGE, config.SEED + 1),
        *deterministic_sample(generic, config.CALIBRATION_GENERIC, config.SEED + 2),
    ]


if __name__ == "__main__":
    raise SystemExit(main())