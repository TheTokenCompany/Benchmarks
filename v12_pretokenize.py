#!/usr/bin/env python3
"""v12 pretokenization: REAL full-document windows, far negatives, dense mix.

v11 stated its own limitation and this file exists to remove it. Every v11 window is
a sub-window of an evidence-CENTERED ~2k chunk that v9 had already cut out of the
filing, so:

  * the model never sees a window whose surrounding context the chunker did not
    already pre-select as answer-adjacent, and
  * every negative is drawn from the evidence's own neighbourhood -- a hard negative,
    which is the useful half of the distribution but not the half that dominates
    serving. At serve time most windows of a 10-K are a different STATEMENT, a
    different SECTION, or boilerplate hundreds of lines from the answer.

v12 windows the FULL document. The source is the v12 corpus (22k validated QA over
2201 SEC filings, FinanceBench filers blocklisted at fetch time), where each QA
carries `evidence_abs_lines` -- absolute line numbers in the filing. Windows are cut
from the whole filing's token stream:

  1. POSITIVE WINDOW. The evidence's relative position r is sampled uniformly inside a
     uniformly chosen third, then CLAMPED so the whole evidence span still fits: with
     evidence of width W_ev in a window of width W, full containment needs
     r in [W_ev/2W, 1 - W_ev/2W]. v11 needed seam machinery because its chunk edges
     forced cuts through evidence rows; here the document is 10-100x the budget, so a
     window that contains the evidence always exists and the honest thing is to take
     it. Items whose evidence span cannot fit in the budget at all are dropped, not
     truncated.

  2. FAR NEGATIVE (prob --negative-frac). A window from the SAME document at least
     --negative-min-line-gap lines away from every evidence line -- a different part
     of the filing entirely, not the next table down. The question is kept (that is
     the serving case: question present, answer absent) and the targets are all-drop
     at weight 1.0. The region is still run through v10's rare-fact rule and rejected
     if anything in it scores as evidence, so a restatement of the same figures
     elsewhere in the filing never ships as a negative.

  3. DENSE MIX. v11b collapses in the oracle regime (52% @ 33% budget): asked to keep
     a THIRD of an already evidence-dense context it still keeps one row, because
     every training target it ever saw was a sparse needle. Items whose evidence spans
     >= --dense-min-span lines, or whose window is >= --dense-min-density evidence
     lines, get keep-most semantics: v10's labels as usual, PLUS every TABLE line
     between the first and last gold evidence line (bounded by the window) co-kept.
     A dense region is a table block, and reading a table block means reading its rows.

Labels are v10's policy verbatim (v10_build_targets.build_labels) on the window's own
word list, plus the dense co-keep above. No new label tiers: the overnight E54/E55
experiments proved minimal/necessity-style targets LOSE to this scaffold.

Sampling: --n-qa items, validated only, equal thirds across qtype, at most
--ticker-cap QA per ticker, FinanceBench filers excluded (by CIK and by ticker, from
the corpus's own exclusions.json -- the same name->CIK resolution the fetcher used).
Train/val split is at the DOCUMENT level so no filing leaks across the split.

Outputs <out-dir>/: same tensor contract as v11 (v8_train reads it unchanged):
  {train,val}.pt          input_ids int32, attention_mask uint8, word_id int16,
                          line_type int8, n_words int32, source_id int8, qa_id
  {train,val}_targets.pt  targets float16, loss_mask uint8, loss_weight float16
  {train,val}_meta.jsonl  one row per WINDOW
  meta.json               provenance + counts + distributions

Run:
    .venv/bin/python v12_pretokenize.py --selftest
    .venv/bin/python v12_pretokenize.py --n-qa 120 --no-save    # quick stats
    .venv/bin/python v12_pretokenize.py                         # full build
    .venv/bin/python v12_pretokenize.py --upload                # + modal put
"""

import argparse
import json
import random
import subprocess
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from v8_build_masks import is_table_line
from v9_rl_prep import (
    N_SPECIAL, QTYPE_OTHER, QTYPES, encode_question, gold_facts, segment_words,
)
from v10_build_targets import W_DROPPED_NUM, build_labels
from v11_pretokenize import (
    LINE_TYPE_NAMES, N_LINE_TYPES, POSITION_BUCKETS, WORD_BOUNDARY_GOLDEN,
    check_dangling, check_number_atomicity, check_window, check_word_boundaries,
    evidence_geometry, item_rng, line_type_per_token, looks_like_evidence, pack,
    position_window, prepare_source, run_fact_checks, slice_window, snap_to_words,
    summarize, surviving_evidence_facts, third_of, verify_tensors,
)

SCRATCH = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-"
           "b409-cb03154f2355-financebench-check/838fe3de-3a8f-4846-ba75-9e663316210c/"
           "scratchpad")
DEFAULT_CORPUS = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-"
                  "c76f-41f6-b409-cb03154f2355-financebench-check/"
                  "3224cbfc-fe21-4c45-9714-a9878f8f7d10/scratchpad/v12corpus")
DEFAULT_OUT = f"{SCRATCH}/v12-sft"

VOLUME = "otso-v8-data"
VOLUME_SUBDIR = "v12-sft"


# --------------------------------------------------------------------------- #
# document geometry
# --------------------------------------------------------------------------- #
def newline_positions(text):
    """Char offsets of every '\\n', ascending. C-speed; a 200 KB filing is instant."""
    out, i = [], text.find("\n")
    while i >= 0:
        out.append(i)
        i = text.find("\n", i + 1)
    return out


def doc_line_of_word(text, spans):
    """ABSOLUTE document line index (0-based) for each word span.

    v10's word_lines() cannot be used for this. It derives the line number by
    accumulating newlines from the FIRST WORD onward, so any blank lines before the
    document's first word shift every line number, and it is only ever correct
    relative to a chunk. `evidence_abs_lines` is absolute against text.split("\\n"),
    so the index is taken from the char offset directly."""
    nl = newline_positions(text)
    return [bisect_right(nl, s) for s, _e in spans]


def doc_line_words(line_of_word):
    """line index -> [word idx, ...]. Only lines that actually carry a word appear."""
    out = defaultdict(list)
    for k, line in enumerate(line_of_word):
        out[line].append(k)
    return dict(out)


def prepare_doc(text, tok):
    """Tokenize one FULL filing once and add absolute-line geometry to it.

    prepare_source() is v11's, imported rather than copied: the word segmentation,
    the token->word mapping and the first/last-token tables must be bit-identical to
    the build v11's self-checks were written against, because v12 reuses v11's
    window slicing and packing on top of them."""
    src = prepare_source({"chunk_text": text}, tok)
    if src is None:
        return None
    src["line_of_word"] = doc_line_of_word(text, src["spans"])
    src["line_words"] = doc_line_words(src["line_of_word"])
    src["n_lines"] = (src["line_of_word"][-1] + 1) if src["line_of_word"] else 0
    return src


def evidence_words_of_lines(src, lines):
    """Word indices on the gold evidence lines. -> sorted list (may be empty)."""
    out = []
    for line in lines:
        out.extend(src["line_words"].get(line, ()))
    return sorted(out)


def token_span_of_words(src, word_idx):
    """(lo, hi) half-open token span covering `word_idx`, or None."""
    first, last = src["first_tok"], src["last_tok"]
    lo = [first[k] for k in word_idx if first[k] >= 0]
    hi = [last[k] for k in word_idx if last[k] >= 0]
    if not lo or not hi:
        return None
    return min(lo), max(hi) + 1


# --------------------------------------------------------------------------- #
# window selection
# --------------------------------------------------------------------------- #
def containment_clamp(r, ev_width, budget, pad=0.02):
    """Clamp requested position r so an evidence span of `ev_width` fits whole.

    -> (r_clamped, margin) or (None, margin) when the span cannot fit at all.

    The evidence is placed by its CENTRE. A centre at r puts the span's edges at
    r +- W_ev/2W, so full containment needs r within [m, 1-m] for m = W_ev/2W. This
    is what replaces v11's seam machinery: v11 had to decide what to do with a row
    cut in half because its source chunk was only budget-sized and there was nowhere
    else to put the window. A full filing always has room, so the window is moved
    instead of the label being renegotiated."""
    m = ev_width / (2.0 * budget) + pad
    if m >= 0.5:
        return None, m
    return min(max(r, m), 1.0 - m), m


def far_negative_spans(src, ev_lines, budget, min_gap, rng, tries=10):
    """Yield candidate token spans >= `min_gap` LINES from every evidence line.

    Line distance, not token distance, on purpose: the corpus gives evidence in line
    space and a filing's line density varies by two orders of magnitude between a
    dense table and a wall of risk-factor prose, so a token-distance rule would put
    the "far" negative three rows below the answer in one document and in a different
    section in the next. 200 lines is a different statement or a different item.

    Several candidates rather than one, because the CALLER's cleanliness test rejects
    most first draws: gold_facts includes the answer's own content words, and a filing
    repeats "revenue" and a fiscal year on nearly every page, so two of them landing
    on one line is enough to score. Rejecting a location must not mean rejecting the
    document -- the strict test stays, the search around it widens."""
    if not ev_lines:
        return
    lo_line, hi_line = min(ev_lines), max(ev_lines)
    n_lines = src["n_lines"]
    regions = []
    if lo_line - min_gap > 0:
        regions.append((0, lo_line - min_gap))
    if hi_line + min_gap < n_lines:
        regions.append((hi_line + min_gap, n_lines))
    if not regions:
        return
    lw = src["line_words"]
    cands = []
    for a, b in regions:
        words = [k for line in range(a, b) for k in lw.get(line, ())]
        if not words:
            continue
        span = token_span_of_words(src, words)
        if span and span[1] - span[0] >= budget // 2:
            cands.append(span)
    if not cands:
        return
    seen = set()
    for _ in range(tries):
        lo, hi = cands[rng.randrange(len(cands))]
        take = min(budget, hi - lo)
        start = lo if hi - lo <= take else rng.randint(lo, hi - take)
        s, e = snap_to_words(src, start, start + take)
        if e - s >= budget // 2 and (s, e) not in seen:
            seen.add((s, e))
            yield s, e


# --------------------------------------------------------------------------- #
# dense mix
# --------------------------------------------------------------------------- #
def dense_cokeep(src, window, keep, weight, info, gold_ev_lines):
    """Co-keep every TABLE line between the first and last gold evidence line.

    Returns the number of words flipped drop -> keep. The weight of a flipped word is
    reset the way build_labels itself would weight a KEPT non-evidence word (1.0):
    leaving W_DROPPED_NUM on it would train the model to be extra sure about dropping
    a number it is now being told to keep.

    Bounded by the window on both sides, so an evidence block that runs past the
    window edge contributes only the rows the window can actually show."""
    if not gold_ev_lines:
        return 0
    lo, hi = min(gold_ev_lines), max(gold_ev_lines)
    w0 = window["word_start"]
    words = window["words"]
    numeric = info["numeric_word"]
    by_line = defaultdict(list)
    for i in range(len(words)):
        by_line[src["line_of_word"][w0 + i]].append(i)
    added = 0
    for line in range(lo, hi + 1):
        idxs = by_line.get(line)
        if not idxs:
            continue
        if not is_table_line([words[i] for i in idxs]):
            continue
        for i in idxs:
            if not keep[i]:
                keep[i] = True
                weight[i] = 1.0 if numeric[i] else 1.0
                added += 1
    return added


def is_dense(info, gold_ev_lines, min_span, min_density):
    """-> (bool, reason). Evidence that is a BLOCK rather than a needle."""
    span = (max(gold_ev_lines) - min(gold_ev_lines) + 1) if gold_ev_lines else 0
    if span >= min_span:
        return True, "span"
    n_content = max(1, info["n_content_lines"])
    if info["n_evidence_lines"] / n_content >= min_density:
        return True, "density"
    return False, ""


# --------------------------------------------------------------------------- #
# per-window record
# --------------------------------------------------------------------------- #
def make_window_record(rec, src, window, kind, bucket, q_ids, q_trunc, facts,
                       ev_words=(), gold_ev_lines=(), requested_pos=None,
                       n_fact_hits=None, n_numeric_hits=None, args=None):
    """One window -> the record that gets packed and written to *_meta.jsonl.

    Same shape as v11's, so summarize()/pack()/check_window() read it unchanged. The
    v11-only seam fields are carried as constants: v12 never ships a cut evidence
    span, and check_window's seam assertions are written against those fields."""
    words, nl_after = window["words"], window["nl_after"]
    n_words = len(words)
    dense = False
    dense_reason = ""
    n_dense_added = 0

    if kind == "negative":
        keep = [False] * n_words
        weight = [1.0] * n_words
        info = {"no_evidence": False, "evidence_lines": [], "n_evidence_lines": 0,
                "n_header_lines": 0, "n_floor_lines": 0, "rowlabel_added": 0,
                "n_numeric": 0, "n_numeric_keep": 0, "keep_frac_words": 0.0,
                "n_rare_facts": 0, "n_clause_scoped_lines": 0, "n_content_lines": 0}
        inert = False
        policy_hits_gold = None
    else:
        keep, info = build_labels(words, nl_after, facts, rec["qtype"])
        weight = info["weight"]
        inert = bool(info["no_evidence"])
        # Did v10's rare-fact rule land on the lines the corpus says hold the answer?
        # Diagnostic only -- the labels are NOT overwritten with the gold lines. The
        # policy's header/floor/row-label scaffold is what wins on FinanceBench, and
        # forcing gold lines in would be a new label tier.
        w0 = window["word_start"]
        policy_lines = {src["line_of_word"][w0 + i]
                        for line in info["evidence_lines"]
                        for i in info["line_words"].get(line, ())}
        policy_hits_gold = bool(policy_lines & set(gold_ev_lines))
        if not inert and args is not None:
            dense, dense_reason = is_dense(info, gold_ev_lines, args.dense_min_span,
                                           args.dense_min_density)
            if dense:
                n_dense_added = dense_cokeep(src, window, keep, weight, info,
                                             gold_ev_lines)

    suffix = {"positive": "p", "negative": "n"}[kind]
    ev_start, ev_centre, ev_width = (
        (None, None, None) if kind == "negative"
        else evidence_geometry(src, window, ev_words))
    return {
        "qa_id": f"{rec['qa_id']}#{suffix}",
        "source_qa_id": rec["qa_id"],
        "file": rec.get("file", ""),
        "ticker": rec.get("ticker", ""),
        "cik": rec.get("cik", 0),
        "question": rec["question"],
        "answer": rec.get("answer", ""),
        "qtype": rec.get("qtype", ""),
        "qtype_id": (QTYPES.index(rec["qtype"]) if rec.get("qtype") in QTYPES
                     else QTYPE_OTHER),
        "window_kind": ("negative" if kind == "negative"
                        else ("dense" if dense else "positive")),
        "is_negative": kind == "negative",
        "is_dense": dense,
        "dense_reason": dense_reason,
        "n_dense_cokeep_words": n_dense_added,
        "policy_hits_gold_lines": policy_hits_gold,
        "inert": inert,
        "had_chunk_evidence": bool(ev_words),
        "is_seam": False,                       # v12 never cuts an evidence span
        "n_evidence_fact_hits": n_fact_hits,
        "n_evidence_numeric_hits": n_numeric_hits,
        "gold_evidence_lines": list(gold_ev_lines),
        "position_bucket": bucket,
        "requested_pos": requested_pos,
        "primary_rel_pos": ev_centre,
        "evidence_rel_pos": ev_centre,
        "evidence_start_pos": ev_start,
        "evidence_width_frac": ev_width,
        "window_offset": window["tok_start"],
        "window_tokens": window["tok_end"] - window["tok_start"],
        "window_word_range": [window["word_start"], window["word_end"]],
        "source_chunk_tokens": src["n_tok"],
        "doc_tokens": src["n_tok"],
        "doc_lines": src["n_lines"],
        "words": words,
        "nl_after": nl_after,
        "gold_fact_words": facts,
        "n_words": n_words,
        "question_truncated": q_trunc,
        "question_tokens": len(q_ids),
        "keep": keep,
        "_q_ids": q_ids,
        "_c_ids": window["ids"],
        "_word_id": window["word_id"],
        "_line_type": line_type_per_token(window),
        "_weight": weight,
        "_info": {k: v for k, v in info.items()
                  if k not in ("line_of_word", "line_words", "numeric_word", "weight")},
    }


# --------------------------------------------------------------------------- #
# per-item build
# --------------------------------------------------------------------------- #
def build_item(rec, src, tok, args, want_negative):
    """One QA over one FULL document -> its window records + a reason dict."""
    reasons = defaultdict(int)
    q_ids, q_trunc = encode_question(rec["question"], tok, args.question_budget)
    if not q_ids:
        reasons["empty_question"] += 1
        return [], reasons
    budget = args.max_len - len(q_ids) - N_SPECIAL
    if budget <= 0:
        reasons["question_eats_budget"] += 1
        return [], reasons

    gold_lines = [l for l in rec["evidence_abs_lines"] if l in src["line_words"]]
    if not gold_lines:
        reasons["evidence_lines_have_no_words"] += 1
        return [], reasons
    ev_words = evidence_words_of_lines(src, gold_lines)
    ev_span = token_span_of_words(src, ev_words)
    if ev_span is None:
        reasons["evidence_has_no_tokens"] += 1
        return [], reasons
    ev_lo, ev_hi = ev_span
    ev_width = ev_hi - ev_lo

    facts = gold_facts([rec["_lines"][l] for l in gold_lines], rec.get("answer", ""))
    if not facts:
        reasons["no_gold_facts"] += 1
        return [], reasons

    rng = item_rng(args.seed, rec["qa_id"])
    bucket, lo, hi = POSITION_BUCKETS[rng.randrange(len(POSITION_BUCKETS))]
    r_req = rng.uniform(lo, hi)
    r, margin = containment_clamp(r_req, ev_width, budget)
    if r is None:
        reasons["evidence_wider_than_budget"] += 1
        return [], reasons

    e_c = (ev_lo + ev_hi) / 2.0
    s, e = position_window(src["n_tok"], e_c, budget, r, args.min_window_tokens)
    s, e = snap_to_words(src, s, e)
    win = slice_window(src, s, e) if e > s else None
    if win is None:
        reasons["empty_window"] += 1
        return [], reasons
    if not (win["word_start"] <= ev_words[0] and ev_words[-1] <= win["word_end"]):
        # The clamp should make this unreachable; a snap at a pathological edge can
        # still shave a word. Dropped rather than relabelled -- v12's contract is that
        # a positive window holds the WHOLE evidence span.
        reasons["evidence_clipped_by_snap"] += 1
        return [], reasons

    n_hits, n_num_hits = surviving_evidence_facts(win, set(ev_words), facts)
    if not n_hits:
        # The evidence line carries none of the facts derived from it -- normalization
        # disagreement between gold_facts and fact_hits. Rare; dropped so
        # check_window's positive-window invariant stays a real check.
        reasons["evidence_matches_no_fact"] += 1
        return [], reasons

    out = [make_window_record(rec, src, win, "positive", bucket, q_ids, q_trunc, facts,
                              ev_words=ev_words, gold_ev_lines=gold_lines,
                              requested_pos=r, n_fact_hits=n_hits,
                              n_numeric_hits=n_num_hits, args=args)]
    reasons["clamped_position"] += int(abs(r - r_req) > 1e-9)

    if want_negative:
        ev_set = set(ev_words)
        n_tried = 0
        for span in far_negative_spans(src, gold_lines, budget,
                                       args.negative_min_line_gap, rng):
            n_tried += 1
            nwin = slice_window(src, *span)
            if nwin is None:
                continue
            if ev_set & set(range(nwin["word_start"], nwin["word_end"] + 1)):
                reasons["negative_overlaps_evidence"] += 1
                continue
            if looks_like_evidence(nwin["words"], nwin["nl_after"], facts):
                reasons["negative_location_not_clean"] += 1
                continue
            out.append(make_window_record(rec, src, nwin, "negative", "none",
                                          q_ids, q_trunc, facts, args=args))
            reasons["negative_built"] += 1
            break
        else:
            reasons["negative_no_far_region" if not n_tried
                    else "negative_no_clean_location"] += 1
    return out, reasons


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def load_exclusions(corpus):
    """FinanceBench filers, as the corpus fetcher resolved them.

    The fetcher already mapped FinanceBench's company names to CIKs against SEC's own
    company index and BLOCKED them at download time (verify_report.json:
    financebench_overlap = 0). Re-deriving the mapping here from the HF dataset would
    add a second, weaker name->ticker guess on top of a stronger name->CIK one; the
    exclusion is re-applied instead, by CIK and by ticker, as a standing assertion
    rather than a new decision."""
    ex = json.loads((Path(corpus) / "exclusions.json").read_text())
    ciks = set(ex.get("cik", []))
    tickers = {v["ticker"] for v in ex.get("report", {}).get("resolved", {}).values()
               if v.get("ticker")}
    return ciks, tickers


def load_qa(corpus, ex_ciks, ex_tickers):
    qa, skipped = [], defaultdict(int)
    with open(Path(corpus) / "qa.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("validated"):
                skipped["not_validated"] += 1
                continue
            if r.get("cik") in ex_ciks:
                skipped["financebench_cik"] += 1
                continue
            if r.get("ticker") in ex_tickers:
                skipped["financebench_ticker"] += 1
                continue
            if not r.get("evidence_abs_lines"):
                skipped["no_evidence_lines"] += 1
                continue
            qa.append(r)
    return qa, dict(skipped)


def sample_qa(qa, n_total, ticker_cap, seed):
    """Equal thirds across qtype, at most `ticker_cap` QA per ticker.

    Round-robin across the qtypes rather than filling one at a time: the ticker cap is
    GLOBAL, so a sequential fill would let whichever qtype went first spend every
    ticker's budget and starve the others onto a narrower set of filers."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for r in qa:
        pools[r["qtype"]].append(r)
    order = sorted(pools)
    for qt in order:
        pools[qt].sort(key=lambda r: r["qa_id"])
        rng.shuffle(pools[qt])
    quota = {qt: n_total // len(order) for qt in order}
    for qt in order[:n_total - sum(quota.values())]:
        quota[qt] += 1

    taken, per_ticker, cursor = [], defaultdict(int), {qt: 0 for qt in order}
    got = defaultdict(int)
    progress = True
    while progress and len(taken) < n_total:
        progress = False
        for qt in order:
            if got[qt] >= quota[qt]:
                continue
            pool, i = pools[qt], cursor[qt]
            while i < len(pool):
                r = pool[i]
                i += 1
                if per_ticker[r["ticker"]] < ticker_cap:
                    per_ticker[r["ticker"]] += 1
                    taken.append(r)
                    got[qt] += 1
                    progress = True
                    break
            cursor[qt] = i
    return taken, dict(got), per_ticker


def split_by_doc(items, val_frac, seed):
    """Doc-level split: every QA on a filing lands on the same side."""
    docs = sorted({r["file"] for r in items})
    rng = random.Random(seed + 3)
    rng.shuffle(docs)
    n_val = max(1, int(round(len(docs) * val_frac)))
    val_docs = set(docs[:n_val])
    train = [r for r in items if r["file"] not in val_docs]
    val = [r for r in items if r["file"] in val_docs]
    return {"train": train, "val": val}, val_docs, len(docs)


# --------------------------------------------------------------------------- #
# split build
# --------------------------------------------------------------------------- #
def build_split(recs, corpus, tok, args, cls_id, sep_id, split):
    """QA records (grouped by document) -> (windows, reasons, timing)."""
    vocab = len(tok)
    by_doc = defaultdict(list)
    for r in recs:
        by_doc[r["file"]].append(r)
    want_neg = {r["qa_id"]: item_rng(args.seed + 1, r["qa_id"]).random()
                < args.negative_frac for r in recs}

    docs_dir = Path(corpus) / "docs"
    windows, reasons = [], defaultdict(int)
    atomicity_bad = []
    t_tok = t_win = 0.0
    t0 = time.time()
    for d, (fname, group) in enumerate(sorted(by_doc.items())):
        text = (docs_dir / fname).read_text()
        ta = time.time()
        src = prepare_doc(text, tok)
        t_tok += time.time() - ta
        if src is None:
            reasons["unsegmentable_doc"] += len(group)
            continue
        if args.check_atomicity:
            bad = check_number_atomicity(src)
            if bad:
                atomicity_bad.append((fname, bad[:3]))
        lines = text.split("\n")
        ta = time.time()
        for rec in group:
            rec = dict(rec, answer=rec.get("gold_answer", ""), _lines=lines)
            wins, rs = build_item(rec, src, tok, args, want_neg[rec["qa_id"]])
            for k, v in rs.items():
                reasons[k] += v
            for w in wins:
                check_window(w, args.max_len, vocab, cls_id, sep_id)
                check_dangling(src, w)
            windows.extend(wins)
        t_win += time.time() - ta
        if (d + 1) % 100 == 0:
            print(f"  {split}: {d+1}/{len(by_doc)} docs -> {len(windows)} windows "
                  f"({time.time()-t0:.0f}s: {t_tok:.0f}s tokenize, {t_win:.0f}s window)")

    if atomicity_bad:
        raise AssertionError(
            f"{split}: number atomicity violated on {len(atomicity_bad)} docs, "
            f"first: {atomicity_bad[:2]}")
    timing = {"docs": len(by_doc), "total_s": time.time() - t0,
              "tokenize_s": t_tok, "window_s": t_win}
    print(f"  {split}: {len(by_doc)} docs in {timing['total_s']:.0f}s "
          f"({t_tok:.0f}s tokenize, {t_win:.0f}s window/label)")
    return windows, dict(reasons), timing


def summarize_v12(windows, split):
    """v11's summary plus the three things v12 exists to change."""
    agg = summarize(windows, split)
    dense = [w for w in windows if w.get("is_dense")]
    pos = [w for w in windows if not w["is_negative"] and not w["inert"]]
    neg = [w for w in windows if w["is_negative"]]

    def keep_frac(ws):
        out = []
        for w in ws:
            wid = np.asarray(w["_word_id"])
            c = wid[wid >= 0]
            if c.size:
                out.append(float(np.asarray(w["keep"], dtype=bool)[c].mean()))
        return float(np.mean(out)) if out else 0.0

    checked = [w for w in pos if w.get("policy_hits_gold_lines") is not None]
    agg.update({
        "n_dense": len(dense),
        "dense_frac_of_positive": len(dense) / max(1, len(pos)),
        "dense_reasons": {k: sum(1 for w in dense if w["dense_reason"] == k)
                          for k in ("span", "density")},
        "mean_dense_cokeep_words": float(np.mean([w["n_dense_cokeep_words"]
                                                  for w in dense])) if dense else 0.0,
        "keep_frac_dense": keep_frac(dense),
        "keep_frac_sparse_positive": keep_frac([w for w in pos if not w["is_dense"]]),
        "negative_frac_of_windows": len(neg) / max(1, len(windows)),
        "policy_hits_gold_lines_frac": (sum(1 for w in checked
                                            if w["policy_hits_gold_lines"])
                                        / max(1, len(checked))),
        "n_docs": len({w["file"] for w in windows}),
        "n_tickers": len({w["ticker"] for w in windows}),
        "mean_doc_tokens": float(np.mean([w["doc_tokens"] for w in windows])),
        "median_doc_tokens": float(np.median([w["doc_tokens"] for w in windows])),
    })
    print(f"  docs={agg['n_docs']}  tickers={agg['n_tickers']}  "
          f"doc tokens median={agg['median_doc_tokens']:.0f} "
          f"mean={agg['mean_doc_tokens']:.0f}")
    print(f"  dense windows: {len(dense)} ({agg['dense_frac_of_positive']:.3f} of "
          f"positive)  by {agg['dense_reasons']}  "
          f"+{agg['mean_dense_cokeep_words']:.1f} co-kept words each")
    print(f"  keep_frac: dense={agg['keep_frac_dense']:.3f}  "
          f"sparse={agg['keep_frac_sparse_positive']:.3f}")
    print(f"  far negatives: {len(neg)} "
          f"({agg['negative_frac_of_windows']:.3f} of windows)")
    print(f"  v10 policy landed on a GOLD evidence line: "
          f"{agg['policy_hits_gold_lines_frac']:.3f}")
    return agg


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def selftest():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: got {got!r} want {want!r}")

    print("word-boundary golden (inherited from v11; v12 depends on it identically)")
    try:
        check_word_boundaries()
        check("segment_words unchanged", True, True)
    except AssertionError as exc:
        check("segment_words unchanged", str(exc)[:60], True)

    print("absolute line indexing survives leading blank lines")
    text = "\n\n\nCONSOLIDATED BALANCE SHEETS\nTotal Current Assets $5,825.8\n"
    spans = segment_words(text)
    lines = doc_line_of_word(text, spans)
    check("first word is on line 3", lines[0], 3)
    check("second line's words are on line 4", lines[-1], 4)
    from v10_build_targets import word_lines as _wl
    from v9_rl_prep import newlines_after as _nl
    words = [text[a:b] for a, b in spans]
    rel, _ = _wl(words, _nl(text, spans))
    check("v10 word_lines is RELATIVE (off by the blank prefix)", rel[0], 0)

    print("containment clamp")
    r, m = containment_clamp(0.05, 400, 2000)
    check("narrow evidence at the start is pushed just inside", round(r, 3), 0.12)
    r, m = containment_clamp(0.5, 400, 2000)
    check("a centred request is untouched", r, 0.5)
    r, m = containment_clamp(0.9, 400, 2000)
    check("end-third request is pulled just inside", round(r, 3), 0.88)
    r, m = containment_clamp(0.5, 2100, 2000)
    check("evidence wider than the budget is rejected", r, None)
    r, m = containment_clamp(0.5, 1900, 2000)
    check("evidence filling 95% of the budget still fits, centred", r, 0.5)
    for width in (10, 200, 800, 1600):
        r, m = containment_clamp(0.02, width, 2000)
        if r is None:
            continue
        lo = r - width / (2 * 2000)
        hi = r + width / (2 * 2000)
        check(f"span width {width} stays inside [0,1]", (lo >= 0, hi <= 1), (True, True))

    print("dense detection")
    info = {"n_content_lines": 40, "n_evidence_lines": 2}
    check("8-line evidence block is dense by span",
          is_dense(info, list(range(100, 108)), 8, 0.30)[0], True)
    check("3-line evidence in a sparse window is not dense",
          is_dense(info, [100, 101, 102], 8, 0.30)[0], False)
    info_d = {"n_content_lines": 10, "n_evidence_lines": 4}
    check("4/10 evidence lines is dense by density",
          is_dense(info_d, [100, 101], 8, 0.30), (True, "density"))

    print("far-negative line-gap geometry on a synthetic filing")
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    body = "\n".join(f"Line {i} Revenue {1000 + i} Total {2000 + i}" for i in range(900))
    src = prepare_doc(body, tok)
    check("absolute lines cover the document", src["n_lines"], 900)
    rng = random.Random(0)
    spans = list(far_negative_spans(src, [450], 600, 200, rng))
    check("far regions exist", len(spans) > 0, True)
    worst = 10 ** 9
    for span in spans:
        ws = [w for w in src["word_id"][span[0]:span[1]] if w >= 0]
        got = {src["line_of_word"][w] for w in ws}
        worst = min(worst, min(abs(l - 450) for l in got))
    check("EVERY candidate stays >= 200 lines from the evidence", worst >= 200, True)
    check("no far region in a short document",
          list(far_negative_spans(src, [450], 600, 600, rng)), [])

    print("dense co-keep flips table rows between the evidence lines")
    rows = ["CONSOLIDATED BALANCE SHEETS", "(Millions) 2025 2024",
            "Total Current Assets 5,825.8 6,363.0"]
    rows += [f"Item{i} {1000+i}.1 {2000+i}.7" for i in range(20)]
    rows += ["Goodwill 3,110.0 3,090.4",
             "We caution readers that these statements are not guarantees of the future"]
    text = "\n".join(rows)
    src2 = prepare_doc(text, tok)
    win = slice_window(src2, 0, src2["n_tok"])
    facts = ["5,825.8", "6,363.0", "assets", "3,110.0"]
    keep, info = build_labels(win["words"], win["nl_after"], facts, "metrics-extraction")
    before = sum(keep)
    kept_lines_before = {src2["line_of_word"][win["word_start"] + i]
                         for i, k in enumerate(keep) if k}
    check("the 20 filler rows between the evidence rows start OUT",
          bool(kept_lines_before & set(range(3, 23))), False)
    added = dense_cokeep(src2, win, keep, info["weight"], info, [2, 23])
    check("co-keep added table rows", added > 0, True)
    check("keep count grew by exactly the additions", sum(keep), before + added)
    lows = {src2["line_of_word"][win["word_start"] + i]
            for i, k in enumerate(keep) if k}
    check("every table row between the evidence rows is now kept",
          set(range(2, 24)) <= lows, True)
    check("the prose line AFTER the evidence block was NOT co-kept",
          len(rows) - 1 in lows, False)
    flipped_w = {info["weight"][i] for i, k in enumerate(keep)
                 if k and src2["line_of_word"][win["word_start"] + i] in range(3, 23)}
    check("flipped words carry weight 1.0, not the dropped-numeral 1.5",
          flipped_w, {1.0})

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", default="jhu-clsp/mmBERT-small")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--question-budget", type=int, default=192)
    ap.add_argument("--n-qa", type=int, default=6000)
    ap.add_argument("--ticker-cap", type=int, default=30)
    ap.add_argument("--val-doc-frac", type=float, default=0.08)
    ap.add_argument("--negative-frac", type=float, default=0.25)
    ap.add_argument("--negative-min-line-gap", type=int, default=200)
    ap.add_argument("--dense-min-span", type=int, default=8)
    ap.add_argument("--dense-min-density", type=float, default=0.30)
    ap.add_argument("--min-window-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-fact-checks", type=int, default=20)
    ap.add_argument("--fact-survival-min", type=float, default=0.85)
    ap.add_argument("--prevalence-lo", type=float, default=0.05)
    ap.add_argument("--prevalence-hi", type=float, default=0.30)
    ap.add_argument("--check-atomicity", action="store_true", default=True)
    ap.add_argument("--no-check-atomicity", dest="check_atomicity",
                    action="store_false")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument("--volume-subdir", default=VOLUME_SUBDIR)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    t_start = time.time()
    check_word_boundaries()
    print(f"word-boundary golden: OK ({len(WORD_BOUNDARY_GOLDEN)} fixtures)")

    ex_ciks, ex_tickers = load_exclusions(args.corpus)
    qa, skipped = load_qa(args.corpus, ex_ciks, ex_tickers)
    print(f"corpus: {len(qa)} eligible QA  (skipped {skipped})")
    print(f"financebench exclusion: {len(ex_ciks)} CIKs / {len(ex_tickers)} tickers")

    taken, got, per_ticker = sample_qa(qa, args.n_qa, args.ticker_cap, args.seed)
    print(f"sampled {len(taken)} QA  by qtype={dict(sorted(got.items()))}  "
          f"tickers={len(per_ticker)}  max/ticker={max(per_ticker.values())}")
    splits, val_docs, n_docs = split_by_doc(taken, args.val_doc_frac, args.seed)
    print(f"doc-level split: {n_docs} docs -> {len(val_docs)} val "
          f"({len(splits['train'])} train QA / {len(splits['val'])} val QA)")
    if set(r["file"] for r in splits["train"]) & set(r["file"] for r in splits["val"]):
        raise AssertionError("document leaked across the train/val split")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError(f"tokenizer missing CLS/SEP: {cls_id}/{sep_id}")
    print(f"tokenizer={args.tokenizer} cls/sep/pad={cls_id}/{sep_id}/{pad_id}  "
          f"max_len={args.max_len}  q_cap={args.question_budget}  seed={args.seed}")

    out_dir = Path(args.out_dir)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "built_by": "v12_pretokenize.py",
        "source": f"v12 corpus @ {args.corpus}",
        "source_note": (
            "FULL SEC filings. Unlike v9/v10/v11 there is no evidence-centered chunk "
            "between the filing and the window: windows are cut from the whole "
            "document's token stream and negatives come from a different part of the "
            "filing, not from the evidence's neighbourhood."),
        "max_len": args.max_len,
        "tokenizer": args.tokenizer,
        "layout": "[CLS] question [SEP] window [SEP]",
        "word_id_semantics": "word index on window tokens, -1 = non-content",
        "qtype_source_ids": QTYPES,
        "source_id_other": QTYPE_OTHER,
        "seed": args.seed,
        "sampling": {
            "n_requested": args.n_qa, "n_taken": len(taken),
            "by_qtype": dict(sorted(got.items())),
            "ticker_cap": args.ticker_cap, "n_tickers": len(per_ticker),
            "validated_only": True,
            "financebench_excluded_ciks": sorted(ex_ciks),
            "financebench_excluded_tickers": sorted(ex_tickers),
            "skipped": skipped,
        },
        "split": {"level": "document", "val_doc_frac": args.val_doc_frac,
                  "n_docs": n_docs, "n_val_docs": len(val_docs)},
        "budget_policy": {
            "question": f"actual length, hard cap {args.question_budget} tokens",
            "content": f"{args.max_len} - len(question_ids) - {N_SPECIAL}",
        },
        "window_policy": {
            "positioning": ("evidence relative position sampled uniformly inside a "
                            "uniformly chosen third of the window, then CLAMPED to "
                            "[W_ev/2W, 1-W_ev/2W] so the whole evidence span fits"),
            "thirds": {name: [lo, hi] for name, lo, hi in POSITION_BUCKETS},
            "min_window_tokens": args.min_window_tokens,
            "edges": "both ends snapped to whole words (no dangling subwords)",
            "seam": "none -- a window that cannot hold the whole evidence is dropped",
            "negatives": {
                "frac_of_items": args.negative_frac,
                "min_line_gap": args.negative_min_line_gap,
                "region": "same document, >= min_line_gap LINES from any evidence line",
                "clean_test": "no line scores as evidence under the v10 rare-fact rule",
                "labels": "all-drop, loss_weight 1.0, question KEPT",
            },
            "dense_mix": {
                "trigger": (f"gold evidence spans >= {args.dense_min_span} lines OR "
                            f">= {args.dense_min_density} of the window's content "
                            f"lines score as evidence"),
                "action": ("v10 build_labels as usual, PLUS every TABLE line between "
                           "the first and last gold evidence line (window-bounded) "
                           "co-kept at weight 1.0"),
            },
        },
        "line_type": {
            "dtype": "int8, separate tensor [N, max_len]",
            "codes": {str(k): v for k, v in LINE_TYPE_NAMES.items()},
            "precedence": "period_header > title > table > prose",
        },
        "label_policy": ("v10_build_targets.build_labels VERBATIM on the window words "
                         "+ the dense table co-keep; no new label tiers"),
    }

    for split in ("train", "val"):
        recs = splits[split]
        print(f"\n{split}: {len(recs)} QA over "
              f"{len({r['file'] for r in recs})} documents")
        windows, reasons, timing = build_split(recs, args.corpus, tok, args,
                                               cls_id, sep_id, split)
        if not windows:
            raise AssertionError(f"{split}: no windows built  reasons={reasons}")
        print(f"  -> {len(windows)} windows   reasons={dict(sorted(reasons.items()))}")
        agg = summarize_v12(windows, split)
        agg["build_reasons"] = reasons
        agg["n_source_items"] = len(recs)
        agg["timing"] = timing
        inputs, tgts = pack(windows, args.max_len, cls_id, sep_id, pad_id)
        prevalence = verify_tensors(inputs, tgts, windows, split, args.max_len)
        agg["pos_prevalence"] = prevalence
        if not (args.prevalence_lo <= prevalence <= args.prevalence_hi):
            raise AssertionError(
                f"{split}: pos_prevalence {prevalence:.4f} outside "
                f"[{args.prevalence_lo}, {args.prevalence_hi}] -- the label "
                f"distribution moved, do not train on this build")
        agg["fact_checks"] = run_fact_checks(windows, args, split)
        meta[f"n_{split}"] = len(windows)
        meta[f"{split}_stats"] = agg
        meta["pos_prevalence" if split == "val" else "pos_prevalence_train"] = prevalence

        if args.no_save:
            continue
        torch.save(inputs, out_dir / f"{split}.pt")
        torch.save(tgts, out_dir / f"{split}_targets.pt")
        with open(out_dir / f"{split}_meta.jsonl", "w") as f:
            for w in windows:
                f.write(json.dumps({k: v for k, v in w.items()
                                    if not k.startswith("_")}) + "\n")

    meta["build_seconds"] = time.time() - t_start
    print(f"\ntotal build time: {meta['build_seconds']:.0f}s")
    if args.no_save:
        print("--no-save: nothing written")
        return 0

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")

    if args.upload:
        cmd = [".venv/bin/modal", "volume", "put", "--force", args.volume,
               str(out_dir), args.volume_subdir]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"uploaded -> {args.volume}:/{args.volume_subdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
