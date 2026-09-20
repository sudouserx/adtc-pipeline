#!/usr/bin/env python3
"""Build a bilingual imatrix and held-out eval corpus from the BF16 reference.

REPLACEMENT (review 2026-09, finding F1 — Critical). FIXED 2026-09-21.
FIXED 2026-09-22 (coverage verifier false positive): see "FIX 2026-09-22" below.

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
  5. FIX 2026-09-22: ``verify_imatrix_coverage`` inspected ``reader.fields``
     (GGUF *metadata*: 3 built-in GGUF.* keys + general.type + 3 imatrix.*
     keys = exactly 7 entries) instead of ``reader.tensors``, where
     llama-imatrix stores the statistics as ``<weight>.in_sum2`` /
     ``<weight>.counts`` pairs. It therefore rejected every valid imatrix
     ("tensor_entries: 7, covers_ffn: False") AFTER a ~10 minute run. The
     verifier now reads the tensors, cross-checks them against the BF16
     reference's ``blk.*`` matmul weights, and validates the numbers
     (finite, non-zero counts, chunk_count * chunk_size vs the token floor).
  6. llama-imatrix now writes ``<name>.gguf`` (silences the "GGUF format with a
     different suffix" warning) and the file is atomically renamed to
     imatrix_path() only when the run succeeded, so a killed run can never
     leave a half-calibrated ``kuza.imatrix`` behind (llama-imatrix rewrites
     the output every 10 chunks). A stale imatrix is removed before a fresh
     run. Set KUZA_REUSE_IMATRIX=1 to skip the ~10 min llama-imatrix step and
     just re-verify an existing imatrix (e.g. after the verifier fix).
"""

from __future__ import annotations

import json
import math
import os
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

# KLD screening must have enough complete contexts for a stable comparison.
# The configured EVAL_PER_LANGUAGE sample can be smaller than this floor.
EVAL_MIN_FULL_CHUNKS = 50
EVAL_TOKEN_MARGIN = 1.10

# --- imatrix verification (llama.cpp GGUF imatrix layout) -------------------
# llama-imatrix stores, per weight ``blk.N.<fam>.weight``, two tensors:
#   ``blk.N.<fam>.weight.in_sum2`` (sum of squared activations) and
#   ``blk.N.<fam>.weight.counts``. Only ``blk.*`` matmul weights are collected
# (token_embd / per_layer_token_embd are get_rows lookups and never appear).
IMATRIX_SUM_SUFFIX = ".in_sum2"
IMATRIX_COUNT_SUFFIX = ".counts"
# Families that every dense transformer block runs through mul_mat. Missing
# these means calibration did not exercise the model. attn_k/attn_v/attn_output
# are reported but not gated: Gemma-4 E2B KV-shared layers never compute K/V, so
# those legitimately have no statistics.
REQUIRED_FAMILIES = ("ffn_down", "ffn_gate", "ffn_up", "attn_q")
MIN_FAMILY_COVERAGE = 0.90
# chunk_count * chunk_size vs the token floor: below WARN -> warning, below
# HARD -> the archived "10 chunks" failure mode (2% of the floor) -> error.
CHUNK_TOKENS_WARN_FRACTION = 0.90
CHUNK_TOKENS_HARD_FRACTION = 0.25

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


def _gguf_scalar(reader, key: str) -> int | None:
    """Read a scalar GGUF metadata value across gguf-py versions."""
    field = reader.fields.get(key)
    if field is None:
        return None
    try:
        return int(field.contents())
    except Exception:  # noqa: BLE001 — older gguf-py has no .contents()
        try:
            return int(field.parts[field.data[0]][0])
        except Exception:  # noqa: BLE001
            return None


def _weight_family(name: str) -> str | None:
    """``blk.12.ffn_down.weight`` -> ``ffn_down`` (None for non-block names)."""
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blk" and parts[-1] == "weight":
        return ".".join(parts[2:-1])
    return None


def verify_imatrix_coverage(
    path,
    llama_root,
    reference=None,
    min_tokens: int | None = None,
    expected_ctx: int | None = None,
) -> dict:
    """Validate a llama-imatrix GGUF against the BF16 reference.

    FIX 2026-09-22: the statistics are GGUF *tensors* (``<w>.in_sum2`` /
    ``<w>.counts``), not metadata *fields*. The previous version counted
    ``reader.fields`` (always 7 for an imatrix) and rejected every good file.

    Hard errors (unambiguous corruption / wrong model): no statistics, any
    non-finite value, a required family (ffn_*, attn_q) below
    MIN_FAMILY_COVERAGE of the reference's weights, or a token budget far below
    the floor. Softer conditions land in ``report["warnings"]``.
    """
    import contextlib
    import sys

    import numpy as np

    sys.path.insert(0, str(llama_root / "gguf-py"))
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path), "r")
        sums: dict[str, Any] = {}
        counts: dict[str, Any] = {}
        for tensor in reader.tensors:
            if tensor.name.endswith(IMATRIX_SUM_SUFFIX):
                sums[tensor.name[: -len(IMATRIX_SUM_SUFFIX)]] = tensor
            elif tensor.name.endswith(IMATRIX_COUNT_SUFFIX):
                counts[tensor.name[: -len(IMATRIX_COUNT_SUFFIX)]] = tensor
        if not sums:
            raise RuntimeError(
                f"{path} contains no '<weight>{IMATRIX_SUM_SUFFIX}' tensors "
                f"({len(reader.fields)} metadata fields, {len(reader.tensors)} "
                "tensors) — not a GGUF-format imatrix, or llama-imatrix wrote "
                "an empty one. Regenerate it."
            )

        problems: list[str] = []
        warnings: list[str] = []

        # -- numeric sanity -----------------------------------------------------
        nonfinite: list[str] = []
        zero_counts: list[str] = []
        count_min = count_max = None
        for name, tensor in sums.items():
            if not bool(np.isfinite(tensor.data).all()):
                nonfinite.append(name)
            counter = counts.get(name)
            if counter is None:
                warnings.append(f"{name}: has in_sum2 but no counts tensor")
                continue
            cmin = float(np.min(counter.data))
            cmax = float(np.max(counter.data))
            if not (math.isfinite(cmin) and math.isfinite(cmax)):
                nonfinite.append(name + IMATRIX_COUNT_SUFFIX)
                continue
            if cmin <= 0:
                zero_counts.append(name)
            count_min = cmin if count_min is None else min(count_min, cmin)
            count_max = cmax if count_max is None else max(count_max, cmax)
        if nonfinite:
            problems.append(
                f"{len(nonfinite)} tensors hold NaN/Inf, e.g. {nonfinite[:5]}"
            )
        if zero_counts:
            warnings.append(
                f"{len(zero_counts)} tensors saw zero activations, "
                f"e.g. {zero_counts[:5]}"
            )

        # -- coverage vs the reference -----------------------------------------
        expected: set[str] = set()
        if reference is not None:
            ref_reader = GGUFReader(str(reference), "r")
            expected = {
                t.name
                for t in ref_reader.tensors
                if t.name.startswith("blk.")
                and t.name.endswith(".weight")
                and len(t.shape) >= 2
            }
            del ref_reader
        families: dict[str, dict[str, int]] = {}
        for name in expected:
            fam = _weight_family(name) or "other"
            bucket = families.setdefault(fam, {"expected": 0, "covered": 0})
            bucket["expected"] += 1
            bucket["covered"] += int(name in sums)
        for fam in REQUIRED_FAMILIES:
            if expected:
                bucket = families.get(fam)
                if not bucket:
                    continue  # this architecture has no such weight
                ratio = bucket["covered"] / bucket["expected"]
                if ratio < MIN_FAMILY_COVERAGE:
                    problems.append(
                        f"{fam}: imatrix covers {bucket['covered']}/"
                        f"{bucket['expected']} reference weights "
                        f"({ratio:.0%} < {MIN_FAMILY_COVERAGE:.0%})"
                    )
            elif not any(_weight_family(n) == fam for n in sums):
                problems.append(f"{fam}: no imatrix statistics at all")
        unmatched = sorted(name for name in sums if expected and name not in expected)
        if unmatched:
            warnings.append(
                f"{len(unmatched)} imatrix entries do not exist in the "
                f"reference GGUF, e.g. {unmatched[:5]} (different model?)"
            )
        uncovered = sorted(expected - set(sums))

        # -- calibration budget recorded by llama-imatrix ---------------------
        chunk_count = _gguf_scalar(reader, "imatrix.chunk_count")
        chunk_size = _gguf_scalar(reader, "imatrix.chunk_size")
        if expected_ctx and chunk_size and chunk_size != expected_ctx:
            warnings.append(
                f"imatrix chunk_size={chunk_size} != CALIBRATION_CTX={expected_ctx}"
            )
        tokens_seen = None
        if chunk_count and chunk_size:
            tokens_seen = chunk_count * chunk_size
            if min_tokens:
                if tokens_seen < CHUNK_TOKENS_HARD_FRACTION * min_tokens:
                    problems.append(
                        f"imatrix saw only {chunk_count} chunks x {chunk_size} = "
                        f"{tokens_seen} tokens (floor {min_tokens}) — the "
                        "archived under-calibration failure (review F1)"
                    )
                elif tokens_seen < CHUNK_TOKENS_WARN_FRACTION * min_tokens:
                    warnings.append(
                        f"imatrix saw {tokens_seen} tokens, below "
                        f"{CHUNK_TOKENS_WARN_FRACTION:.0%} of the {min_tokens} floor"
                    )
        elif min_tokens:
            warnings.append(
                "imatrix has no chunk_count/chunk_size metadata; "
                "token budget not verifiable"
            )

        report = {
            "tensor_entries": len(sums),
            "covers_ffn": any(
                (_weight_family(n) or "").startswith("ffn_") for n in sums
            ),
            "covers_attention": any(
                (_weight_family(n) or "").startswith("attn_") for n in sums
            ),
            "reference_weights": len(expected),
            "covered_reference_weights": len(expected) - len(uncovered),
            "family_coverage": {k: families[k] for k in sorted(families)},
            "uncovered_sample": uncovered[:10],
            "chunk_count": chunk_count,
            "chunk_size": chunk_size,
            "tokens_seen": tokens_seen,
            "count_min": count_min,
            "count_max": count_max,
            "warnings": warnings,
        }
        if problems:
            raise RuntimeError(
                "imatrix coverage looks wrong: "
                + "; ".join(problems)
                + f" — report: {json.dumps(report, default=str)}"
            )
        return report
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(llama_root / "gguf-py"))


def _eval_row_key(row: dict) -> tuple[Any, Any]:
    return row.get("source"), row.get("source_id")


def _render_eval_token_count(tokenizer, row: dict) -> int:
    """Render one held-out row exactly as the eval corpus does and count tokens."""
    rendered = model.render_text(tokenizer, row)
    return verify_ids(
        tokenizer,
        rendered,
        model.encode_ids(tokenizer, rendered),
        "eval",
    )


def _expand_eval_rows(
    tokenizer,
    rows: list[dict],
    initial_rows: list[dict],
    target_tokens: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Expand a balanced held-out eval set until the verified token floor is met.

    Calibration rows have already been removed by the caller, so expansion is
    restricted to the remaining held-out pool. No training rows are introduced
    into the KLD screening corpus.
    """
    from common import deterministic_sample

    selected: list[dict] = []
    selected_ids: set[tuple[Any, Any]] = set()

    def add_if_new(row: dict) -> bool:
        key = _eval_row_key(row)
        if key in selected_ids:
            return False
        selected_ids.add(key)
        selected.append(row)
        return True

    for row in initial_rows:
        add_if_new(row)

    buckets: dict[str, list[dict]] = {"english": [], "swahili": [], "other": []}
    for row in rows:
        key = _eval_row_key(row)
        if key in selected_ids:
            continue
        language = str(row.get("language", "")).lower()
        bucket = (
            "english"
            if language == "english"
            else "swahili"
            if language == "swahili"
            else "other"
        )
        buckets[bucket].append(row)

    sampled: dict[str, list[dict]] = {}
    for offset, bucket_name in enumerate(("english", "swahili", "other")):
        bucket = buckets[bucket_name]
        sampled[bucket_name] = (
            deterministic_sample(bucket, len(bucket), seed + offset + 100)
            if bucket
            else []
        )

    indices = {name: 0 for name in sampled}
    token_total = 0
    valid_rows = 0
    skipped_rows = 0

    # Count the initial selected rows using the same render/token path as eval.
    for row in selected:
        try:
            token_total += _render_eval_token_count(tokenizer, row)
            valid_rows += 1
        except RuntimeError:
            skipped_rows += 1

    # Round-robin over language buckets so top-up remains approximately balanced.
    while token_total < target_tokens:
        added_any = False
        for bucket_name in ("english", "swahili", "other"):
            bucket = sampled[bucket_name]
            index = indices[bucket_name]
            if index >= len(bucket):
                continue

            row = bucket[index]
            indices[bucket_name] += 1
            add_if_new(row)
            added_any = True

            try:
                token_total += _render_eval_token_count(tokenizer, row)
                valid_rows += 1
            except RuntimeError:
                skipped_rows += 1

            if token_total >= target_tokens:
                break

        if not added_any:
            break

    stats = {
        "target_tokens": target_tokens,
        "selected_rows": len(selected),
        "valid_rows": valid_rows,
        "skipped_rows": skipped_rows,
        "verified_tokens": token_total,
    }
    return selected, stats


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
        row for row in rows
        if (row["source"], row["source_id"]) not in calibration_ids
    ]
    if not calibration_rows or not remaining:
        raise RuntimeError("Insufficient held-out rows for calibration/eval")

    tokenizer = model.load_tokenizer(adapter_dir())

    # FIX F2: select enough held-out rows for at least 100 complete EVAL_CTX
    # chunks. Start from the configured balanced set, then expand deterministically
    # inside the remaining held-out pool until the actual rendered token count
    # reaches the floor with a 10% safety margin.
    initial_eval_rows = select_balanced(
        remaining, config.EVAL_PER_LANGUAGE, config.SEED + 20
    )
    target_eval_tokens = math.ceil(
        EVAL_MIN_FULL_CHUNKS * config.EVAL_CTX * EVAL_TOKEN_MARGIN
    )
    eval_rows, eval_selection_stats = _expand_eval_rows(
        tokenizer,
        remaining,
        initial_eval_rows,
        target_eval_tokens,
        config.SEED + 20,
    )
    if not eval_rows:
        raise RuntimeError("No held-out evaluation rows could be rendered")

    # KLD screening must stay disjoint from calibration.
    eval_ids = {_eval_row_key(row) for row in eval_rows}

    # Top up calibration from the local English pool if the held-out calibration
    # pool is too small. These rows remain excluded from the evaluation corpus.
    if len(calibration_rows) * 600 < config.CALIBRATION_MIN_TOKENS:
        extra = load_local_or_hub(
            "english", "english", "english.jsonl", {"english": "main"}
        )
        extra = [
            row for row in extra
            if _eval_row_key(row) not in calibration_ids
            and _eval_row_key(row) not in eval_ids
        ]
        calibration_rows = calibration_rows + extra
        print(
            f"topped up calibration pool with {len(extra)} english rows "
            "to reach the token floor",
            flush=True,
        )

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
    # Eval corpus token sanity as well (screening reads this file). The
    # selection stage targets a 10% margin, but the final rendered artifact is
    # authoritative because render_corpus may skip malformed rows.
    eval_floor = EVAL_MIN_FULL_CHUNKS * config.EVAL_CTX
    if eval_stats["tokens"] < eval_floor:
        raise RuntimeError(
            f"Eval corpus only {eval_stats['tokens']} tokens; screening KLD "
            f"needs >= {EVAL_MIN_FULL_CHUNKS} full-context chunks "
            f"({eval_floor} tokens). The remaining held-out pool was "
            "insufficient after rendering/skips."
        )

    chunks = max(1, math.ceil(calib_stats["tokens"] / config.CALIBRATION_CTX) + 8)
    if calib_stats["tokens"] / chunks < config.CALIBRATION_CTX * 0.9:
        raise RuntimeError("chunk math inconsistent with verified token count")

    # llama-imatrix rewrites its output every 10 chunks, so write to a ``.gguf``
    # temp name (the format is GGUF; a different suffix only triggers a warning)
    # and rename into place after the run succeeds.
    meta_path = dest_dir / "imatrix_meta.json"
    tmp_dest = dest.with_name(dest.name + ".tmp.gguf")
    reused = False
    if os.environ.get("KUZA_REUSE_IMATRIX") == "1":
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError(
                "KUZA_REUSE_IMATRIX=1 but there is no existing imatrix at "
                f"{dest}; unset it to generate one."
            )
        if meta_path.is_file():
            previous = json.loads(meta_path.read_text()).get("calibration_sha256")
            if previous and previous != calib_stats["sha256"]:
                raise RuntimeError(
                    "KUZA_REUSE_IMATRIX=1 but the calibration corpus changed "
                    "since this imatrix was generated; unset it to regenerate."
                )
        else:
            print(
                "imatrix: no imatrix_meta.json (imatrix predates it); reusing "
                "on the strength of the verification below.",
                flush=True,
            )
        reused = True
        print(f"imatrix: reusing existing {dest}", flush=True)
    else:
        dest.unlink(missing_ok=True)      # never leave a stale imatrix behind
        meta_path.unlink(missing_ok=True)
        tmp_dest.unlink(missing_ok=True)
        run(
            [
                binaries["llama-imatrix"],
                "-m", reference_path(),
                "-f", calibration_file,
                "-o", tmp_dest,
                "-ngl", "999",
                "-c", str(config.CALIBRATION_CTX),
                "-b", "512",
                "--chunks", str(chunks),
                "-t", str(getattr(config, "TOOL_THREADS", 4)),
            ],
            log_path=dest_dir / "imatrix.log",
        )
        if not tmp_dest.exists() or tmp_dest.stat().st_size == 0:
            raise RuntimeError("llama-imatrix did not produce a valid artifact")
        os.replace(tmp_dest, dest)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("llama-imatrix did not produce a valid artifact")
    coverage = verify_imatrix_coverage(
        dest,
        binaries["converter"].parent,
        reference=reference_path(),
        min_tokens=config.CALIBRATION_MIN_TOKENS,
        expected_ctx=config.CALIBRATION_CTX,
    )
    for warning in coverage.get("warnings", []):
        print(f"imatrix warning: {warning}", flush=True)
    print(
        f"imatrix verified: {coverage['tensor_entries']} tensors, "
        f"{coverage['covered_reference_weights']}/{coverage['reference_weights']} "
        f"reference block weights, {coverage['tokens_seen']} tokens",
        flush=True,
    )
    write_json(
        meta_path,
        {
            "calibration_sha256": calib_stats["sha256"],
            "chunks_requested": chunks,
            "calibration_ctx": config.CALIBRATION_CTX,
        },
    )
    write_json(
        dest_dir / "corpus_manifest.json",
        {
            "calibration": calib_stats,
            "evaluation": eval_stats,
            "evaluation_selection": eval_selection_stats,
            "eval_min_full_chunks": EVAL_MIN_FULL_CHUNKS,
            "eval_token_margin": EVAL_TOKEN_MARGIN,
            "prompt_sha256": sha256_bytes(model.SYSTEM_PROMPT.encode()),
            "overlap": sorted(
                calibration_ids
                & {(row["source"], row["source_id"]) for row in eval_rows}
            ),
            "imatrix_chunks": chunks,
            "imatrix_reused": reused,
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