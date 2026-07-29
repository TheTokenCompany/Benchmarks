#!/usr/bin/env python3
"""v10 "rule-retiring" SFT targets: teach natively what safenum/safetab patch.

The serving rules keep EVERY number (safenum) and EVERY table row label (safetab)
regardless of the question. That buys fact survival by inflating retention. v10
teaches the same competence as a QUESTION-CONDITIONED SELECTION: keep the numbers
the question needs, and confidently DROP the rest. There is deliberately NO blanket
keep-numbers prior -- the non-evidence numerals are the negative class, and they are
up-weighted so the model has to learn the distinction instead of hedging.

Targets are built ON TOP of the existing v9-rl tensors (no re-tokenization): labels
are decided per WORD and broadcast to tokens through word_id (>=0 = chunk content,
which is also the v8 loss_mask).

Label policy, per item:
  (a) EVIDENCE LINES   lines that hit the item's gold_fact_words -> full keep.
      Fact hits are scored, not counted: a fact that matches many lines of this
      chunk (a year, "income", "total") identifies nothing, so only RARE facts
      (<= MAX_FACT_LINES matching lines) count, numeric ones double. A line needs
      score >= 2, i.e. one rare number OR two rare label words.
  (b) HEADERS          nearest statement-title line and nearest period-header line
      above each evidence line (40-line scan cap, as v8_build_masks) -> full keep.
      A stripped table cell is unreadable without "CONSOLIDATED BALANCE SHEETS"
      and the "2025 2024 2023" column header.
  (c) ROW-LABEL CO-KEEP  any table line holding a kept number keeps words
      [0, first_numeric) capped at 12 -- the row label that names the number.
  (d) CONTEXT FLOOR    qtype-graded, mirroring v8_build_masks QUAL_PARA:
      domain-qualitative +-14 lines, multistep-numerical +-7, metrics-extraction 0.
      Radius is an upper bound; lines are added nearest-first and stop at the
      qtype keep-fraction cap, because these chunks run 6-60 words per line and a
      flat +-14 would swallow a prose chunk whole.
  (e) EVERYTHING ELSE  drop, including every non-evidence numeral.

Loss weights (a third tensor, same shape as targets):
  4.0  numeric token on an evidence line with target=keep   (the answer cells)
  1.5  numeric token with target=drop                       (selectivity)
  1.0  everything else
  0.0  wherever loss_mask == 0

Outputs <out-dir>/ : train.pt val.pt (copied v9-rl inputs, unchanged),
{train,val}_targets.pt (targets float16, loss_mask uint8, loss_weight float16,
qa_id) and meta.json (v8-trainer compatible: max_len, tokenizer,
qtype_source_ids, pos_prevalence + the v10 policy block).

Run:
    .venv/bin/python v10_build_targets.py --selftest
    .venv/bin/python v10_build_targets.py --limit 40 --no-save     # quick stats
    .venv/bin/python v10_build_targets.py                          # full build
    .venv/bin/python v10_build_targets.py --upload                 # + modal put
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# single source of truth for the line heuristics
from v8_build_masks import (
    is_numeric_word, is_period_header, is_table_line, is_title_line,
    row_label_span,
)
from v9_rl_prep import fact_survival, render_mask, strip_edges

SCRATCH = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-"
           "b409-cb03154f2355-financebench-check/838fe3de-3a8f-4846-ba75-9e663316210c/"
           "scratchpad")
DEFAULT_IN = f"{SCRATCH}/v9rl"
DEFAULT_OUT = f"{SCRATCH}/v10-sft"

VOLUME = "otso-v8-data"
VOLUME_SUBDIR = "v10-sft"

HEADER_SCAN = 40          # lines to look up for a title / period header (v8 value)
LONG_LINE_WORDS = 60      # above this a "line" is a paragraph, not a table row
CLAUSE_PAD = 20           # words kept either side of a fact hit inside such a line
MAX_FACT_LINES = 4        # a fact matching more lines than this is not identifying
EVIDENCE_MIN_SCORE = 2    # 1 rare number (2) or 2 rare label words (1+1)
MAX_EVIDENCE_LINES = 10   # runaway guard before the context floor is applied
ROW_LABEL_CAP = 12
NUM_TOL = 0.02            # relative tolerance for numeric fact matching

# qtype -> (context-floor line radius, keep-fraction cap for the floor)
QTYPE_FLOOR = {
    "domain-qualitative":  (14, 0.28),
    "multistep-numerical": (7,  0.16),
    "metrics-extraction":  (0,  0.10),
}
DEFAULT_FLOOR = (7, 0.16)

W_EVIDENCE_NUM = 4.0
W_DROPPED_NUM = 1.5

DIGIT = re.compile(r"\d")
WS = re.compile(r"\s+")
NUM_CHARS = re.compile(r"[^\d.\-]")


# --------------------------------------------------------------------------- #
# normalization / matching
# --------------------------------------------------------------------------- #
def norm_word(w):
    """Word -> comparison form: whitespace squeezed out, edges stripped, lowered.

    v9's segment_words merges a number with its symbol across whitespace, so a
    single "word" can be "$\\n\\n119" -- the squeeze is what makes it comparable to
    the fact "$119" that gold_facts lifted from a one-line rendering."""
    return strip_edges(WS.sub("", w)).lower()


def num_value(s):
    """Numeric value of a money/percent cell, or None. "(349.4)" -> -349.4."""
    if not DIGIT.search(s):
        return None
    neg = s.startswith("(") and s.endswith(")")
    body = NUM_CHARS.sub("", WS.sub("", s).replace(",", ""))
    if not body or not DIGIT.search(body):
        return None
    try:
        v = float(body.rstrip(".-").lstrip("."))
    except ValueError:
        return None
    return -v if neg else v


def _year_like(v):
    return float(v).is_integer() and 1900 <= abs(v) <= 2100


def num_match(a, b):
    """Do two numeric surface forms denote the same figure, within tolerance?

    Sign-insensitive (a filing writes "(349.4)" for the answer's "349.4"). Two
    guards keep the 2% window from manufacturing evidence:
      - two year-like values must be EQUAL, or 2024 matches 2025 and every page
        header in the chunk becomes an answer cell;
      - anything under 100 must be EQUAL, because at that magnitude 2% is smaller
        than the reporting precision (30 would match 30.4, 12.5 match 12.6).
    Tolerance is also only ever REACHED as a fallback -- see fact_hits."""
    if a is None or b is None:
        return False
    x, y = abs(a), abs(b)
    if x == y:
        return True
    if _year_like(a) and _year_like(b):
        return False
    hi = max(x, y)
    if hi < 100:
        return False
    return abs(x - y) / hi <= NUM_TOL


# --------------------------------------------------------------------------- #
# line bookkeeping over the stored word list
# --------------------------------------------------------------------------- #
def word_lines(words, nl_after):
    """-> (line_of_word[list[int]], line_words[dict line -> [word idx, ...]]).

    Line index of a word = newlines seen before its first character, which is the
    running sum of newlines INSIDE previous words plus nl_after between them. A
    merged word that straddles a line break is attributed to the line it starts on,
    so the word indices in line_words stay contiguous and align with `words`."""
    line_of_word, line_words = [], defaultdict(list)
    line = 0
    for k, w in enumerate(words):
        line_of_word.append(line)
        line_words[line].append(k)
        line += w.count("\n") + nl_after[k]
    return line_of_word, dict(line_words)


def fact_hits(words, facts):
    """-> (hit_facts[list[set[fact_idx]]], numeric_word, n_tolerance_facts).

    Which gold facts each word matches. Matching is TIERED: exact normalized form
    first, and the numeric tolerance is applied ONLY to facts that nothing in the
    chunk matches exactly. v9_rl_prep already filtered gold_fact_words to those the
    all-keep chunk contains, so the exact tier resolves nearly everything; letting
    tolerance run alongside it instead pulled in neighbouring cells (a "$354.4" row
    is within 2% of the answer's "349.4" and would be labelled evidence)."""
    exact = defaultdict(set)
    numeric_facts = []
    for fi, f in enumerate(facts):
        nf = norm_word(f)
        if nf:
            exact[nf].add(fi)
        v = num_value(f)
        if v is not None:
            numeric_facts.append((fi, v))

    numeric_word = [is_numeric_word(w) for w in words]
    norms = [norm_word(w) for w in words]
    hit_facts = [set(exact.get(nw, ())) for nw in norms]

    matched = {fi for hs in hit_facts for fi in hs}
    unmatched_nums = [(fi, v) for fi, v in numeric_facts if fi not in matched]
    if unmatched_nums:
        for k, w in enumerate(words):
            if not numeric_word[k]:
                continue
            v = num_value(w)
            if v is None:
                continue
            hit_facts[k] |= {fi for fi, fv in unmatched_nums if num_match(v, fv)}
    return hit_facts, numeric_word, len(unmatched_nums)


def score_lines(line_words, hit_facts, numeric_word, facts):
    """-> (scores[line -> int], rare[set of fact idx]).

    Rarity is measured inside this chunk: a fact matching more than MAX_FACT_LINES
    lines is a year / "total" / "income" and identifies nothing, so it cannot
    contribute. A rare figure is worth 2, a rare label word 1, and a rare YEAR is
    worth 1 rather than 2 -- a line whose only rare hits are years is a column
    header, not an answer row, and headers already come in via propagation."""
    fact_lines = defaultdict(set)
    for line, idxs in line_words.items():
        for k in idxs:
            for fi in hit_facts[k]:
                fact_lines[fi].add(line)
    rare = {fi for fi, ls in fact_lines.items() if len(ls) <= MAX_FACT_LINES}
    year_fact = {fi for fi in rare
                 if (v := num_value(facts[fi])) is not None and _year_like(v)}

    scores = {}
    for line, idxs in line_words.items():
        figures, labels, years = set(), set(), set()
        for k in idxs:
            for fi in hit_facts[k] & rare:
                if fi in year_fact:
                    years.add(fi)
                elif numeric_word[k]:
                    figures.add(fi)
                else:
                    labels.add(fi)
        s = 2 * len(figures) + len(labels) + len(years)
        if s >= EVIDENCE_MIN_SCORE and not (figures or labels):
            s = 1                       # year-only line: never evidence on its own
        if s:
            scores[line] = s
    return scores, rare


# --------------------------------------------------------------------------- #
# label policy
# --------------------------------------------------------------------------- #
def build_labels(words, nl_after, facts, qtype):
    """-> (keep[list[bool]], info dict) for one item."""
    line_of_word, line_words = word_lines(words, nl_after)
    hit_facts, numeric_word, n_tol_facts = fact_hits(words, facts)
    scores, rare = score_lines(line_words, hit_facts, numeric_word, facts)
    n_words = len(words)
    lines_sorted = sorted(line_words)

    # ---- (a) evidence lines ----
    cand = [(s, l) for l, s in scores.items() if s >= EVIDENCE_MIN_SCORE]
    cand.sort(key=lambda t: (-t[0], t[1]))
    evidence = sorted(l for _s, l in cand[:MAX_EVIDENCE_LINES])
    if not evidence and scores:                 # never label an item all-drop
        evidence = [max(scores, key=lambda l: (scores[l], -l))]
    keep_lines = set(evidence)

    # ---- (b) header propagation ----
    headers, got_title, got_period = set(), 0, 0
    for ev in evidence:
        found_t = found_p = False
        for line in range(ev - 1, max(-1, ev - HEADER_SCAN - 1), -1):
            idxs = line_words.get(line)
            if not idxs:
                continue
            lw = [words[k] for k in idxs]
            if not found_t and is_title_line(lw):
                headers.add(line)
                found_t = True
            if not found_p and is_period_header(lw):
                headers.add(line)
                found_p = True
            if found_t and found_p:
                break
        got_title += int(found_t)
        got_period += int(found_p)
    keep_lines |= headers

    # ---- evidence keeps: full line, or clause-scoped when the "line" is a paragraph
    # A converted filing puts a whole legal-proceedings paragraph on one line; keeping
    # 234 words to expose one damages figure is what the rules do wrong, in miniature.
    keep = [False] * n_words
    clause_scoped = 0
    for line in evidence:
        idxs = line_words[line]
        if len(idxs) <= LONG_LINE_WORDS:
            for k in idxs:
                keep[k] = True
            continue
        anchors = [k for k in idxs if hit_facts[k] & rare] or [idxs[0]]
        clause_scoped += 1
        for a in anchors:
            lo = max(idxs[0], a - CLAUSE_PAD)
            hi = min(idxs[-1], a + CLAUSE_PAD)
            for k in range(lo, hi + 1):
                keep[k] = True

    for line in headers:
        for k in line_words[line]:
            keep[k] = True

    # ---- (d) qtype-graded context floor, nearest-first up to the keep cap ----
    pad, cap = QTYPE_FLOOR.get(qtype, DEFAULT_FLOOR)
    floor_lines = set()
    if pad and evidence:
        kept_w = sum(keep)
        ranked = sorted((min(abs(line - e) for e in evidence), line)
                        for line in lines_sorted
                        if line not in keep_lines
                        and min(abs(line - e) for e in evidence) <= pad)
        for _d, line in ranked:
            add = len(line_words[line])
            if (kept_w + add) / max(1, n_words) > cap:
                continue          # skip this one, a nearer-but-smaller line may fit
            keep_lines.add(line)
            floor_lines.add(line)
            kept_w += add
            for k in line_words[line]:
                keep[k] = True

    # ---- (c) row-label co-keep on table lines holding a kept number ----
    rowlabel_lines, rowlabel_ok, rowlabel_added = 0, 0, 0
    for line, idxs in line_words.items():
        lw = [words[k] for k in idxs]
        if not is_table_line(lw):
            continue
        if not any(keep[k] and numeric_word[k] for k in idxs):
            continue
        rowlabel_lines += 1
        span = row_label_span(lw)
        if span is None or span[1] <= span[0]:
            rowlabel_ok += 1                     # nothing to co-keep (leading number)
            continue
        span_idx = idxs[span[0]:span[1]]
        if all(keep[k] for k in span_idx):
            rowlabel_ok += 1
        for k in span_idx:
            if not keep[k]:
                keep[k] = True
                rowlabel_added += 1

    # ---- loss weights ----
    ev_set = set(evidence)
    weight = [1.0] * n_words
    for k in range(n_words):
        if not numeric_word[k]:
            continue
        if keep[k]:
            if line_of_word[k] in ev_set:
                weight[k] = W_EVIDENCE_NUM
        else:
            weight[k] = W_DROPPED_NUM

    n_num = sum(numeric_word)
    n_num_keep = sum(1 for k in range(n_words) if numeric_word[k] and keep[k])
    matched = [k for k in range(n_words) if hit_facts[k] & rare]
    info = {
        # No rare fact hit anywhere -> we cannot say WHICH lines answer this question.
        # The row is masked out of the loss entirely rather than labelled all-drop,
        # which would teach "drop everything" on a chunk that does hold the answer.
        "no_evidence": not evidence,
        "n_lines": (max(lines_sorted) + 1) if lines_sorted else 0,
        "n_content_lines": len(line_words),
        "evidence_lines": evidence,
        "n_evidence_lines": len(evidence),
        "n_header_lines": len(headers),
        "header_title_frac": got_title / max(1, len(evidence)),
        "header_period_frac": got_period / max(1, len(evidence)),
        "n_floor_lines": len(floor_lines),
        "n_clause_scoped_lines": clause_scoped,
        "rowlabel_lines": rowlabel_lines,
        "rowlabel_ok": rowlabel_ok,
        "rowlabel_added": rowlabel_added,
        "n_words": n_words,
        "n_keep_words": sum(keep),
        "keep_frac_words": sum(keep) / max(1, n_words),
        "n_numeric": n_num,
        "n_numeric_keep": n_num_keep,
        "n_rare_facts": len(rare),
        "n_tolerance_facts": n_tol_facts,
        "n_matched_words": len(matched),
        "line_of_word": line_of_word,
        "line_words": line_words,
        "numeric_word": numeric_word,
        "weight": weight,
    }
    return keep, info


# --------------------------------------------------------------------------- #
# tensor assembly
# --------------------------------------------------------------------------- #
def build_split(inputs, recs, split, verbose_n=0):
    """-> (out dict of tensors, per-item stats list)."""
    word_id = inputs["word_id"]
    n, max_len = word_id.shape
    assert len(recs) == n, f"{split}: {len(recs)} meta rows vs {n} tensor rows"
    if isinstance(inputs.get("qa_id"), list):
        bad = [i for i in range(n) if inputs["qa_id"][i] != recs[i]["qa_id"]]
        assert not bad, f"{split}: qa_id misalignment at rows {bad[:5]}"

    targets = torch.zeros((n, max_len), dtype=torch.float16)
    loss_mask = torch.zeros((n, max_len), dtype=torch.uint8)
    loss_weight = torch.zeros((n, max_len), dtype=torch.float16)
    stats, inert = [], []

    for i, rec in enumerate(recs):
        words, nl_after = rec["words"], rec["nl_after"]
        keep, info = build_labels(words, nl_after, rec["gold_fact_words"],
                                  rec["qtype"])
        wid = word_id[i].numpy()
        content = wid >= 0
        idx = wid[content].astype(np.int64)
        if idx.size:
            assert int(idx.max()) < len(words), (
                f"{split} row {i}: word_id {int(idx.max())} >= len(words) {len(words)}")
        kp = np.asarray(keep, dtype=np.float32)
        wt = np.asarray(info["weight"], dtype=np.float32)
        t_row = np.zeros(max_len, dtype=np.float32)
        w_row = np.zeros(max_len, dtype=np.float32)
        t_row[content] = kp[idx]
        w_row[content] = wt[idx]
        if info["no_evidence"]:
            content = np.zeros_like(content)
            t_row[:] = 0.0
            w_row[:] = 0.0
        targets[i] = torch.from_numpy(t_row).half()
        loss_mask[i] = torch.from_numpy(content.astype(np.uint8))
        loss_weight[i] = torch.from_numpy(w_row).half()

        n_tok = int(content.sum())
        if info["no_evidence"]:
            inert.append(rec["qa_id"])
            continue                       # excluded from the stats, not from the file
        tok_keep = int((t_row[content] >= 0.5).sum())
        num_tok = np.zeros(len(words), dtype=bool)
        num_tok[:] = info["numeric_word"]
        ntok_num = int(num_tok[idx].sum()) if idx.size else 0
        ntok_num_keep = int((num_tok[idx] & (kp[idx] >= 0.5)).sum()) if idx.size else 0
        stats.append({
            "qa_id": rec["qa_id"], "qtype": rec["qtype"],
            "keep_frac_words": info["keep_frac_words"],
            "keep_frac_tokens": tok_keep / max(1, n_tok),
            "n_tokens": n_tok, "n_evidence_lines": info["n_evidence_lines"],
            "n_header_lines": info["n_header_lines"],
            "header_title_frac": info["header_title_frac"],
            "header_period_frac": info["header_period_frac"],
            "n_floor_lines": info["n_floor_lines"],
            "rowlabel_lines": info["rowlabel_lines"],
            "rowlabel_ok": info["rowlabel_ok"],
            "rowlabel_added": info["rowlabel_added"],
            "n_numeric_words": info["n_numeric"],
            "n_numeric_keep": info["n_numeric_keep"],
            "n_numeric_tokens": ntok_num, "n_numeric_tokens_keep": ntok_num_keep,
            "n_rare_facts": info["n_rare_facts"], "n_facts": len(rec["gold_fact_words"]),
            "n_tolerance_facts": info["n_tolerance_facts"],
            "n_clause_scoped_lines": info["n_clause_scoped_lines"],
            **survival_stats(rec["gold_fact_words"],
                             render_mask(words, nl_after, keep)),
            "n_matched_words": info["n_matched_words"],
        })
        if i < verbose_n:
            show_item(rec, keep, info)

    if inert:
        print(f"  {split}: {len(inert)} rows masked out (no rare fact hit): "
              f"{inert[:5]}{' ...' if len(inert) > 5 else ''}")
    out = {"targets": targets, "loss_mask": loss_mask, "loss_weight": loss_weight,
           "qa_id": [r["qa_id"] for r in recs], "inert_qa_id": inert}
    return out, stats, inert


# --------------------------------------------------------------------------- #
# eyeball / verification helpers
# --------------------------------------------------------------------------- #
def survival_stats(facts, kept_text):
    """Gold-fact survival in the kept text, overall and for FIGURES only.

    The numeric-only number is the one that matters: a dropped label word costs the
    reader little, a dropped answer figure costs the answer."""
    nums = [f for f in facts if num_value(f) is not None]
    return {"fact_survival": fact_survival(facts, kept_text),
            "fact_survival_numeric": fact_survival(nums, kept_text),
            "n_numeric_facts": len(nums)}


def show_item(rec, keep, info, max_chars=1400):
    kept = render_mask(rec["words"], rec["nl_after"], keep)
    from v9_rl_prep import fact_survival
    surv = fact_survival(rec["gold_fact_words"], kept)
    print(f"\n{'-'*78}\n[{rec['qa_id']}] {rec['qtype']}  "
          f"keep_words={info['keep_frac_words']:.3f}  "
          f"ev_lines={info['n_evidence_lines']} hdr={info['n_header_lines']} "
          f"floor={info['n_floor_lines']}  numeric keep/total="
          f"{info['n_numeric_keep']}/{info['n_numeric']}  fact_survival={surv:.2f}")
    print(f"Q: {rec['question'][:160]}")
    print(f"A: {str(rec.get('answer', ''))[:120]}")
    print(f"facts: {rec['gold_fact_words'][:12]}")
    print("--- KEPT TEXT ---")
    print(kept[:max_chars] + ("\n[...]" if len(kept) > max_chars else ""))


def summarize(stats, name):
    def pcts(vals):
        v = sorted(vals)
        if not v:
            return "n/a"
        q = lambda f: v[min(len(v) - 1, int(len(v) * f))]
        return f"p10={q(.1):.3f} p50={q(.5):.3f} p90={q(.9):.3f}"

    print(f"\n=== {name}: {len(stats)} items ===")
    by_q = defaultdict(list)
    for s in stats:
        by_q[s["qtype"]].append(s)
    for qt in sorted(by_q):
        ss = by_q[qt]
        print(f"  {qt:22s} n={len(ss):4d}  "
              f"keep_tok {pcts([s['keep_frac_tokens'] for s in ss])}  "
              f"mean={np.mean([s['keep_frac_tokens'] for s in ss]):.3f}")
        print(f"  {'':22s} keep_words mean="
              f"{np.mean([s['keep_frac_words'] for s in ss]):.3f}  "
              f"ev_lines mean={np.mean([s['n_evidence_lines'] for s in ss]):.1f}  "
              f"floor mean={np.mean([s['n_floor_lines'] for s in ss]):.1f}  "
              f"numeric kept={np.sum([s['n_numeric_tokens_keep'] for s in ss]) / max(1, np.sum([s['n_numeric_tokens'] for s in ss])):.3f}")
        print(f"  {'':22s} header any={np.mean([s['n_header_lines'] > 0 for s in ss]):.3f} "
              f"(title {np.mean([s['header_title_frac'] > 0 for s in ss]):.3f} / "
              f"period {np.mean([s['header_period_frac'] > 0 for s in ss]):.3f})  "
              f"fact_survival mean={np.mean([s['fact_survival'] for s in ss]):.3f} "
              f"(=1.0 on {np.mean([s['fact_survival'] >= 0.999 for s in ss]):.3f})")

    tok = sum(s["n_tokens"] for s in stats)
    keep_tok = sum(s["keep_frac_tokens"] * s["n_tokens"] for s in stats)
    num_tok = sum(s["n_numeric_tokens"] for s in stats)
    num_keep = sum(s["n_numeric_tokens_keep"] for s in stats)
    hdr_items = [s for s in stats if s["n_evidence_lines"]]
    row_items = [s for s in stats if s["rowlabel_lines"]]
    agg = {
        "pos_prevalence": keep_tok / max(1, tok),
        "numeric_tokens_keep_frac": num_keep / max(1, num_tok),
        "numeric_tokens_drop_frac": 1 - num_keep / max(1, num_tok),
        "items_with_header_title": np.mean([s["header_title_frac"] > 0
                                            for s in hdr_items]),
        "items_with_header_period": np.mean([s["header_period_frac"] > 0
                                             for s in hdr_items]),
        "items_with_any_header": np.mean([s["n_header_lines"] > 0 for s in hdr_items]),
        # every kept-number table line ends up label-carrying BY CONSTRUCTION (rule c);
        # these two say how often the line keeps already covered it vs co-keep having
        # to add the label itself.
        "rowlabel_lines_labeled_after_cokeep": 1.0,
        "rowlabel_lines_already_labeled": (
            sum(s["rowlabel_ok"] for s in row_items)
            / max(1, sum(s["rowlabel_lines"] for s in row_items))),
        "items_needing_cokeep": float(
            np.mean([s["rowlabel_added"] > 0 for s in row_items])) if row_items else 0.0,
        "items_with_kept_number_lines": len(row_items) / max(1, len(stats)),
        "mean_rare_facts": float(np.mean([s["n_rare_facts"] for s in stats])),
        "mean_facts": float(np.mean([s["n_facts"] for s in stats])),
        "items_using_numeric_tolerance": float(
            np.mean([s["n_tolerance_facts"] > 0 for s in stats])),
        "fact_survival_mean": float(np.mean([s["fact_survival"] for s in stats])),
        "fact_survival_full_frac": float(
            np.mean([s["fact_survival"] >= 0.999 for s in stats])),
        "fact_survival_numeric_mean": float(
            np.mean([s["fact_survival_numeric"] for s in stats])),
        "fact_survival_numeric_full_frac": float(
            np.mean([s["fact_survival_numeric"] >= 0.999 for s in stats])),
        "items_with_clause_scoping": float(
            np.mean([s["n_clause_scoped_lines"] > 0 for s in stats])),
        "items_zero_evidence": int(sum(1 for s in stats if not s["n_evidence_lines"])),
    }
    print(f"  ALL: keep_tok(prevalence)={agg['pos_prevalence']:.4f}  "
          f"numeric tokens kept={agg['numeric_tokens_keep_frac']:.3f} / "
          f"dropped={agg['numeric_tokens_drop_frac']:.3f}")
    print(f"  headers: title on {agg['items_with_header_title']:.3f} of items, "
          f"period on {agg['items_with_header_period']:.3f}, any "
          f"{agg['items_with_any_header']:.3f}")
    print(f"  row-labels: 1.000 of kept-number table lines carry their label after "
          f"co-keep ({agg['rowlabel_lines_already_labeled']:.3f} already did; co-keep "
          f"added labels on {agg['items_needing_cokeep']:.3f} of items).  Items with "
          f"kept-number table lines: {agg['items_with_kept_number_lines']:.3f}")
    print(f"  fact_survival: all mean={agg['fact_survival_mean']:.3f} "
          f"(full on {agg['fact_survival_full_frac']:.3f});  FIGURES mean="
          f"{agg['fact_survival_numeric_mean']:.3f} "
          f"(full on {agg['fact_survival_numeric_full_frac']:.3f});  clause-scoped "
          f"paragraphs on {agg['items_with_clause_scoping']:.3f} of items")
    print(f"  facts: {agg['mean_facts']:.1f} per item, "
          f"{agg['mean_rare_facts']:.1f} rare;  tolerance tier used on "
          f"{agg['items_using_numeric_tolerance']:.3f} of items;  "
          f"items with no evidence line: {agg['items_zero_evidence']}")
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in agg.items()}


def sanity_facts(recs, tensors, n=2):
    """Reconstruct keep-text for n items and report gold-fact survival."""
    from v9_rl_prep import fact_survival
    out = []
    for rec in recs[:n]:
        keep, _info = build_labels(rec["words"], rec["nl_after"],
                                   rec["gold_fact_words"], rec["qtype"])
        kept = render_mask(rec["words"], rec["nl_after"], keep)
        facts = rec["gold_fact_words"]
        surv = fact_survival(facts, kept)
        missing = [f for f in facts if fact_survival([f], kept) < 1.0]
        out.append({"qa_id": rec["qa_id"], "qtype": rec["qtype"],
                    "fact_survival": surv, "n_facts": len(facts),
                    "missing": missing[:8]})
        print(f"  sanity {rec['qa_id']} ({rec['qtype']}): fact_survival="
              f"{surv:.3f} of {len(facts)} facts"
              + (f"  missing={missing[:6]}" if missing else "  ALL PRESENT"))
    return out


# --------------------------------------------------------------------------- #
# self-test on synthetic lines
# --------------------------------------------------------------------------- #
def selftest():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: got {got!r} want {want!r}")

    print("normalization / numeric matching")
    check("norm merged word", norm_word("$\n\n1,904"), "$1,904")
    check("num_value parens", num_value("(349.4)"), -349.4)
    check("num_value money", num_value("$41,059.5"), 41059.5)
    check("num_value pct", num_value("12.5%"), 12.5)
    check("num_value none", num_value("Revenue"), None)
    check("num exact via format", num_match(num_value("$4,137.1"),
                                            num_value("4137.10")), True)
    check("num 1% rounding", num_match(num_value("1,904.5"),
                                       num_value("1,905.0")), True)
    check("num 5% apart", num_match(num_value("100.0"), num_value("105.0")), False)
    check("years never fuzzy", num_match(num_value("2024"), num_value("2025")), False)
    check("equal values match", num_match(num_value("12"), num_value("12")), True)
    check("sub-100 exact only", num_match(num_value("30"), num_value("30.4")), False)
    check("sign-insensitive", num_match(num_value("(349.4)"), num_value("349.4")), True)
    check("1.4% apart is inside the window",      # why tiering, not the window, saves us
          num_match(num_value("($354.4)"), num_value("349.4")), True)

    print("tiered matching: tolerance only for facts with no exact hit")
    hf, _nw, n_tol = fact_hits(["($354.4)", "(349.4)", "Revenue"], ["349.4"])
    check("exact hit only", [sorted(h) for h in hf], [[], [0], []])
    check("no tolerance tier needed", n_tol, 0)
    hf, _nw, n_tol = fact_hits(["($354.4)", "Revenue"], ["349.4"])
    check("tolerance tier fires when nothing matches exactly",
          [sorted(h) for h in hf], [[0], []])
    check("tolerance fact counted", n_tol, 1)

    print("line heuristics (v8_build_masks, on this data's geometry)")
    check("title line", is_title_line("CONSOLIDATED BALANCE SHEETS".split()), True)
    check("title rejects numbers",
          is_title_line("CONSOLIDATED BALANCE SHEETS 2025".split()), False)
    check("period header",
          is_period_header("(Millions of U.S. Dollars) 2025 2024 2023".split()), True)
    check("prose is not a period header",
          is_period_header("We caution that these statements are not guarantees".split()),
          False)
    check("table line",
          is_table_line("Trade receivables, net 1,901.2 1,821.6 1,700.0".split()), True)
    check("row label span",
          row_label_span("Total Current Assets $5,825.8 $6,363.0".split()), (0, 3))

    print("word -> line bookkeeping")
    words = ["Alpha", "Beta", "CONSOLIDATED", "BALANCE", "SHEETS", "Cash", "$\n\n119"]
    nl = [0, 2, 0, 0, 2, 0, 1]
    low, lw = word_lines(words, nl)
    check("line_of_word", low, [0, 0, 2, 2, 2, 4, 4])
    check("line_words keys", sorted(lw), [0, 2, 4])

    print("policy on a synthetic filing")
    lines = [
        "Air Products and Chemicals, Inc. and Subsidiaries",          # 0
        "CONSOLIDATED COMPREHENSIVE INCOME STATEMENTS",               # 1 title
        "(Millions of U.S. Dollars) 2025 2024 2023",                  # 2 period hdr
        "Net Income (Loss) ($354.4) $3,862.4 $2,338.6",               # 3 distractor
        "Total Other Comprehensive Income 5.0 274.7 521.0",           # 4 distractor
        "Comprehensive Income (Loss) (349.4) 4,137.1 2,859.6",        # 5 EVIDENCE
        "Net Income Attributable to Noncontrolling Interests 40.1 34.2 38.4",  # 6
    ]
    text = "\n\n".join(lines)
    from v9_rl_prep import newlines_after, segment_words
    spans = segment_words(text)
    words = [text[s:e] for s, e in spans]
    nl_after = newlines_after(text, spans)
    facts = ["4,137.1", "349.4", "2,859.6", "comprehensive", "income", "2025", "2024"]
    keep, info = build_labels(words, nl_after, facts, "metrics-extraction")
    low, lw = word_lines(words, nl_after)
    kept_lines = sorted({low[k] for k in range(len(words)) if keep[k]})
    # blank-line separated, so filing row r sits on line index 2*r
    check("evidence line found", info["evidence_lines"], [10])
    check("title+period propagated", [l for l in kept_lines if l in (2, 4)], [2, 4])
    check("distractor rows dropped", [l for l in kept_lines if l in (6, 8, 12)], [])
    kept_txt = render_mask(words, nl_after, keep)
    check("answer figure kept", "4,137.1" in kept_txt, True)
    check("distractor figure dropped", "3,862.4" in kept_txt, False)
    w = info["weight"]
    ev_num_w = {w[k] for k in range(len(words))
                if info["numeric_word"][k] and keep[k] and low[k] == 10}
    drop_num_w = {w[k] for k in range(len(words))
                  if info["numeric_word"][k] and not keep[k]}
    check("evidence numeric weight", sorted(ev_num_w), [W_EVIDENCE_NUM])
    check("dropped numeric weight", sorted(drop_num_w), [W_DROPPED_NUM])
    check("metrics floor is 0", info["n_floor_lines"], 0)

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_split(in_dir, split, limit=0):
    inputs = torch.load(Path(in_dir) / f"{split}.pt", map_location="cpu",
                        weights_only=False)
    recs = [json.loads(l) for l in open(Path(in_dir) / f"{split}_meta.jsonl")]
    if limit:
        recs = recs[:limit]
        inputs = {k: (v[:limit] if hasattr(v, "__getitem__") else v)
                  for k, v in inputs.items()}
    return inputs, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=DEFAULT_IN, help="local copy of v9-rl")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="cap items per split (debug)")
    ap.add_argument("--show", type=int, default=0, help="print N rendered keep-masks")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--upload", action="store_true",
                    help=f"modal volume put -> {VOLUME}:/{VOLUME_SUBDIR}")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    src_meta = json.loads((in_dir / "meta.json").read_text())
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "built_by": "v10_build_targets.py",
        "source": f"{VOLUME}:/v9-rl",
        "max_len": src_meta["max_len"],
        "tokenizer": src_meta["tokenizer"],
        "layout": src_meta["layout"],
        "qtype_source_ids": src_meta["qtype_source_ids"],
        "source_id_other": src_meta.get("source_id_other", 3),
        "policy": {
            "goal": ("teach safenum/safetab competence as question-conditioned "
                     "SELECTION; no blanket keep-numbers prior"),
            "evidence": (f"lines scoring >= {EVIDENCE_MIN_SCORE} on rare gold-fact "
                         f"hits (numeric x2, rare = <= {MAX_FACT_LINES} matching "
                         f"lines), max {MAX_EVIDENCE_LINES} lines -> full keep"),
            "headers": f"nearest title + period header above, {HEADER_SCAN}-line scan",
            "row_label": f"words [0, first_numeric) cap {ROW_LABEL_CAP} on table "
                         f"lines with a kept number",
            "context_floor": {k: {"lines": v[0], "keep_frac_cap": v[1]}
                              for k, v in QTYPE_FLOOR.items()},
            "everything_else": "drop, including all non-evidence numerals",
            "loss_weights": {"evidence_numeric_keep": W_EVIDENCE_NUM,
                             "numeric_drop": W_DROPPED_NUM, "other": 1.0,
                             "outside_loss_mask": 0.0},
            "numeric_tolerance": NUM_TOL,
            "numeric_tolerance_exceptions": "years (1900-2100) and integers < 100",
        },
    }

    for split in ("train", "val"):
        inputs, recs = load_split(in_dir, split, args.limit)
        out, stats, inert = build_split(inputs, recs, split,
                                        verbose_n=args.show if split == "val" else 0)
        agg = summarize(stats, split)
        agg["n_inert_rows"] = len(inert)
        print(f"  sanity: gold facts in reconstructed keep-text ({split})")
        agg["fact_sanity"] = sanity_facts(recs, out, n=2)
        meta[f"n_{split}"] = len(recs)
        meta[f"{split}_stats"] = agg
        if split == "val":
            meta["pos_prevalence"] = agg["pos_prevalence"]
        if args.no_save:
            continue
        torch.save(out, out_dir / f"{split}_targets.pt")
        shutil.copy(in_dir / f"{split}.pt", out_dir / f"{split}.pt")
        shutil.copy(in_dir / f"{split}_meta.jsonl", out_dir / f"{split}_meta.jsonl")
        qt = defaultdict(int)
        for s in stats:
            qt[s["qtype"]] += 1
        meta[f"{split}_qtypes"] = dict(qt)
        meta[f"{split}_keep_frac_by_qtype"] = {
            qtype: float(np.mean([s["keep_frac_tokens"] for s in stats
                                  if s["qtype"] == qtype]))
            for qtype in qt}

    if args.no_save:
        print("\n--no-save: nothing written")
        return 0
    meta["pos_prevalence_train"] = meta["train_stats"]["pos_prevalence"]
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved -> {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")

    if args.upload:
        cmd = ["modal", "volume", "put", "--force", VOLUME, str(out_dir),
               VOLUME_SUBDIR]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"uploaded -> {VOLUME}:/{VOLUME_SUBDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
