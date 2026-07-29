#!/usr/bin/env python3
"""v11 pretokenization: repositioned windows + negatives + a line-type channel.

v10 taught the right LABELS on the wrong DISTRIBUTION. Every v9-rl chunk is an
evidence-CENTERED window, so the model can score "middle of the window" and collect
most of the reward; and every window contains the answer, so it never has to decide
that a window holds nothing worth keeping. Serving does neither: the evidence lands
wherever the tiling puts it, and most windows of a 10-K are irrelevant to the
question. v11 rebuilds the inputs to remove both freebies, and adds the one feature
the encoder cannot recover from token ids alone -- whether a token sits on a table
row, a statement title, a period header, or prose.

HONEST LIMITATION, stated once: the source is v9-rl's chunk_text, an evidence-
centered ~1949-token window, NOT the full filing. Everything here re-windows WITHIN
that text. A window therefore cannot show the model context the chunk never had, and
the negatives are drawn from the same neighbourhood as the evidence rather than from
a random page of the 10-K -- they are HARD negatives (same statement, same company,
adjacent tables), which is the useful half of the distribution but not all of it.

What each item produces:

  1. ONE REPOSITIONED WINDOW.  The evidence's relative position r is sampled
     uniformly from a third (start / middle / end), and the window is the LONGEST
     window <= budget that actually puts the evidence at r:

         W = min(budget, e_c / r, (n_tok - e_c) / (1 - r)),  start = e_c - r*W

     Sliding a full-budget window inside a chunk that is itself only budget-sized
     would move nothing, so the window is allowed to be SHORTER than the budget when
     the chunk edge runs out. Asking for evidence at 20% of a chunk with 975 tokens
     to its right yields a ~1200-token window that keeps all of that right context
     and drops the left. This costs context and buys position variance; the achieved
     position is measured, not assumed.

  2. OPTIONALLY ONE NEGATIVE WINDOW (--negative-frac of items). Drawn from the far
     side of the evidence, >= --min-negative-tokens long, and only if that region is
     CLEAN: no evidence word in it, and no line in it scores as evidence under the
     v10 rare-fact rule. Region too small or not clean -> the item gets no negative.
     Nothing is fabricated. Negatives keep the QUESTION (that is the serving case:
     question present, answer absent) and carry all-drop targets at weight 1.0.

  A repositioned window that loses ALL of its evidence to the reposition is itself
  relabelled a negative, and so is a SEAM window whose surviving evidence words no
  longer match a single gold fact -- a cut that takes "5,825.8 6,363.0" and leaves
  "Total Current Assets" leaves a row label that still scores as evidence over a
  fragment carrying none of the figures the question asked for. Positivity is decided
  by OVERLAP with the full-chunk evidence words rather than by re-running the policy:
  v10's build_labels always picks a best line rather than returning nothing, so
  re-running it on an evidence-free window would invent an evidence line.

Labels are v10's policy verbatim (v10_build_targets.build_labels), applied to the
WINDOW's word list: rarity-scored gold-fact matching, header propagation, row-label
co-keep, qtype context floors, +-20-word clause windows on paragraph lines, and the
4.0 / 1.5 / 1.0 loss weights. Facts outside the window simply stop hitting.

Outputs <out-dir>/:
  {train,val}.pt          input_ids int32, attention_mask uint8, word_id int16,
                          line_type int8, n_words int32, source_id int8, qa_id
  {train,val}_targets.pt  targets float16, loss_mask uint8, loss_weight float16
  {train,val}_meta.jsonl  one row per WINDOW (question, answer, facts, words,
                          nl_after, window_offset, is_negative, ...)
  meta.json               full provenance + counts + distributions

Run:
    .venv/bin/python v11_pretokenize.py --selftest
    .venv/bin/python v11_pretokenize.py --limit 40 --no-save     # quick stats
    .venv/bin/python v11_pretokenize.py                          # full build
    .venv/bin/python v11_pretokenize.py --upload                 # + modal put
"""

import argparse
import json
import random
import re
import subprocess
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from v8_build_masks import is_period_header, is_table_line, is_title_line
from v9_rl_prep import (
    N_SPECIAL, QTYPE_OTHER, QTYPES, encode_question, fact_survival,
    map_tokens_to_words, newlines_after, render_mask, segment_words,
)
from v10_build_targets import (
    EVIDENCE_MIN_SCORE, build_labels, fact_hits, score_lines, word_lines,
)

SCRATCH = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-"
           "b409-cb03154f2355-financebench-check/838fe3de-3a8f-4846-ba75-9e663316210c/"
           "scratchpad")
DEFAULT_IN = f"{SCRATCH}/v9rl"
DEFAULT_OUT = f"{SCRATCH}/v11-sft"

VOLUME = "otso-v8-data"
VOLUME_SUBDIR = "v11-sft"

# line_type channel. 0 is reserved for "not filing content" so the zero-initialised
# embedding row is the one that pad / special / question tokens hit.
LT_NONE, LT_PROSE, LT_TABLE, LT_TITLE, LT_PERIOD = 0, 1, 2, 3, 4
N_LINE_TYPES = 5
LINE_TYPE_NAMES = {LT_NONE: "none", LT_PROSE: "prose", LT_TABLE: "table",
                   LT_TITLE: "title", LT_PERIOD: "period_header"}

# thirds the evidence position is sampled from, as (lo, hi) relative positions
POSITION_BUCKETS = (("start", 0.05, 0.33), ("middle", 0.33, 0.67), ("end", 0.67, 0.95))

# qtypes whose seam survivors must retain an actual FIGURE, not merely some gold fact.
# For these the answer IS a number: "Total Current Assets" with its cells cut off is
# not answerable content, however well the row label matches. domain-qualitative is
# excluded on purpose -- its evidence is prose and its answers (a list of countries, a
# risk factor) are legitimately non-numeric, so a figure requirement would throw away
# correct labels.
FIGURE_REQUIRED_QTYPES = frozenset({"metrics-extraction", "multistep-numerical"})

# Golden character spans for filing geometry that has broken before. The
# number-atomicity check verifies token->WORD ASSIGNMENT and is satisfied by ANY
# self-consistent segmentation, so it passed clean on the build where a bare "$" bound
# backward to the preceding year ("2026 $" + "9,353"). Only pinning the exact word
# strings catches a change in segment_words itself, which v11 imports rather than owns.
WORD_BOUNDARY_GOLDEN = (
    # currency binds FORWARD, percent binds BACKWARD, in one line
    ("2026 $ 9,353 12.5 %", ["2026", "$ 9,353", "12.5 %"]),
    # the regression that shipped a stale build
    ("Contributions 2026 $ 9,353", ["Contributions", "2026", "$ 9,353"]),
    # two amounts in one row: each symbol takes the amount to its RIGHT
    ("Issuance $\n\n119\n\n$\n\n439", ["Issuance", "$\n\n119", "$\n\n439"]),
    # thousands separator split across whitespace
    ("Revenue 1, 904", ["Revenue", "1, 904"]),
    # already-attached symbols are left alone
    ("Total Current Assets $5,825.8 $6,363.0",
     ["Total", "Current", "Assets", "$5,825.8", "$6,363.0"]),
)

DIGIT = re.compile(r"\d")


def check_word_boundaries():
    """Assert segment_words still produces the golden character spans. Raises.

    Run at the START of every build, not only from --selftest: v11 imports
    segment_words from v9_rl_prep, so this file's correctness depends on a module it
    does not own, and a silent change there produces a plausible-looking dataset with
    mis-split currency columns."""
    bad = []
    for text, want in WORD_BOUNDARY_GOLDEN:
        got = [text[a:b] for a, b in segment_words(text)]
        if got != want:
            bad.append(f"  {text!r}\n    got  {got}\n    want {want}")
    if bad:
        raise AssertionError(
            "v9_rl_prep.segment_words no longer produces the golden word boundaries "
            "-- v11 windows and labels would be built on a different segmentation "
            "than the one these self-checks were written against:\n" + "\n".join(bad))


# --------------------------------------------------------------------------- #
# geometry over one source chunk
# --------------------------------------------------------------------------- #
def word_token_ranges(word_id, n_words):
    """-> (first_token[list], last_token[list]) per word; -1 when a word has none."""
    first = [-1] * n_words
    last = [-1] * n_words
    for t, w in enumerate(word_id):
        if w < 0:
            continue
        if first[w] < 0:
            first[w] = t
        last[w] = t
    return first, last


def prepare_source(rec, tok):
    """Tokenize one stored chunk_text ONCE and return its word/token geometry.

    The stored `words` list is a PREFIX of a fresh segmentation (v9 truncated it to
    that build's token budget while storing chunk_text whole), so v11 re-segments
    from the text: the extra tail is real chunk content that a repositioned window
    is allowed to use."""
    text = rec["chunk_text"]
    spans = segment_words(text)
    if not spans:
        return None
    words = [text[s:e] for s, e in spans]
    nl_after = newlines_after(text, spans)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return None
    wid = map_tokens_to_words(offs, spans)
    first, last = word_token_ranges(wid, len(words))
    return {"text": text, "spans": spans, "words": words, "nl_after": nl_after,
            "ids": ids, "offs": offs, "word_id": wid, "first_tok": first,
            "last_tok": last, "n_tok": len(ids), "n_words": len(words)}


def evidence_of_chunk(src, facts, qtype):
    """Full-chunk evidence, used ONLY to decide where to put the window. -> dict|None.

    Two different things come back, and they are used for two different jobs:
      ev_words   every word on every evidence line. Decides POSITIVITY -- a window
                 that retains none of them no longer answers the question.
      primary    the single highest-scoring evidence line, i.e. the answer row under
                 v10's own candidate ordering. Decides PLACEMENT.
    Placement anchors on the primary line and not on the evidence span because the
    span is wide (a third of a window on average, 80%+ at p90) and scattered across
    up to MAX_EVIDENCE_LINES lines: its centre is pinned near 0.5 by its own width, so
    positioning by it cannot produce the start/end windows this build exists to make.
    """
    _keep, info = build_labels(src["words"], src["nl_after"], facts, qtype)
    if info["no_evidence"]:
        return None
    lw = info["line_words"]
    ev_lines = info["evidence_lines"]
    hits, numeric, _n_tol = fact_hits(src["words"], facts)
    scores, _rare = score_lines(lw, hits, numeric, facts)
    primary = min(ev_lines, key=lambda l: (-scores.get(l, 0), l))   # v10's ordering
    return {"ev_words": {k for line in ev_lines for k in lw.get(line, ())},
            "ev_lines": ev_lines, "primary_line": primary,
            "primary_words": sorted(lw.get(primary, ()))}


# --------------------------------------------------------------------------- #
# window selection
# --------------------------------------------------------------------------- #
def position_window(n_tok, e_c, budget, r, min_tokens):
    """Longest window <= budget that puts token position `e_c` at relative pos `r`.

    Returns (start, end) in token space. Shorter than `budget` whenever the chunk
    runs out of text on the side the requested position needs -- which is the whole
    point: the source chunk is only budget-sized, so a budget-length window has
    nowhere to slide and the evidence would stay centered forever.

    `min_tokens` is a floor on that shrinking. v9's center_window ANCHORS instead of
    centering when the evidence is wider than the budget, so some chunks carry their
    evidence right against an edge; asking for such evidence at the far end of the
    window solves to a 60-token window, which is not a training example of anything.
    Below the floor the length wins and the position is clamped to whatever that
    length can reach -- the caller measures where the evidence actually landed."""
    r = min(max(r, 0.02), 0.98)
    w = float(budget)
    w = min(w, e_c / r)
    w = min(w, (n_tok - e_c) / (1.0 - r))
    w = int(max(1, min(w, n_tok, budget)))
    w = min(max(w, min(min_tokens, n_tok, budget)), n_tok, budget)
    start = int(round(e_c - r * w))
    start = max(0, min(start, n_tok - w))
    return start, start + w


def snap_to_words(src, s, e):
    """Shrink token window [s, e) until both edges fall on whole words.

    A window that starts mid-word would feed the encoder a dangling subword whose
    word_id points at a word the window only partly contains -- the keep/drop
    decision for that word would then be trained on half its evidence.

    Walking the edges token-by-token is NOT enough: segment_words merges a number
    with its symbol across whitespace ("$\\n\\n119" is one word), so a word_id == -1
    whitespace token can sit INSIDE a word, and an edge scan would stop on it and
    admit the dangling half. The edges are resolved through the word table instead,
    and the result is trimmed to the first/last token of the surviving words."""
    wid, first, last = src["word_id"], src["first_tok"], src["last_tok"]
    s = max(0, s)
    e = min(src["n_tok"], e)
    if e <= s:
        return s, s
    touched = [w for w in wid[s:e] if w >= 0]
    if not touched:
        return s, s
    w0, w1 = touched[0], touched[-1]
    if first[w0] < s:                       # window starts inside this word
        w0 += 1
    if w1 < len(last) and last[w1] >= e:    # window ends inside this word
        w1 -= 1
    while w0 <= w1 and first[w0] < 0:       # a word the tokenizer gave no token
        w0 += 1
    while w1 >= w0 and last[w1] < 0:
        w1 -= 1
    if w0 > w1:
        return s, s
    return first[w0], last[w1] + 1


def slice_window(src, s, e):
    """Token span -> the window's own words / nl_after / ids / word_id.

    -> dict or None when the span holds no content word."""
    wid = src["word_id"]
    seg = wid[s:e]
    content = [w for w in seg if w >= 0]
    if not content:
        return None
    w0, w1 = content[0], content[-1]
    if content != sorted(content):
        raise AssertionError(f"word_id not monotonic in window [{s},{e})")
    if set(content) != set(range(w0, w1 + 1)):
        missing = sorted(set(range(w0, w1 + 1)) - set(content))
        raise AssertionError(f"window [{s},{e}) skips word(s) {missing[:5]}")
    return {
        "tok_start": s, "tok_end": e,
        "word_start": w0, "word_end": w1,
        "ids": list(src["ids"][s:e]),
        "word_id": [(w - w0) if w >= 0 else -1 for w in seg],
        "words": src["words"][w0:w1 + 1],
        "nl_after": src["nl_after"][w0:w1 + 1],
    }


def negative_span(src, ev_words, budget, min_tokens):
    """Token span of the largest evidence-free region, or None.

    'Far side' means the side of the evidence span with the most room; inside that
    region the window is pushed as far from the evidence as it fits, so a negative
    is never a near-miss of the answer row."""
    first, last = src["first_tok"], src["last_tok"]
    ev_tok = [first[k] for k in ev_words if first[k] >= 0]
    ev_tok += [last[k] for k in ev_words if last[k] >= 0]
    if not ev_tok:
        return None
    lo, hi = min(ev_tok), max(ev_tok)
    left = (0, lo)
    right = (hi + 1, src["n_tok"])
    region = max((left, right), key=lambda ab: ab[1] - ab[0])
    span = region[1] - region[0]
    if span < min_tokens:
        return None
    take = min(budget, span)
    if region is left:                      # hug the chunk start, away from evidence
        s, e = region[0], region[0] + take
    else:                                   # hug the chunk end
        s, e = region[1] - take, region[1]
    s, e = snap_to_words(src, s, e)
    return (s, e) if e - s >= min_tokens else None


def looks_like_evidence(words, nl_after, facts):
    """Does the v10 rare-fact rule score any line here as evidence?

    build_labels cannot answer this -- it falls back to the best-scoring line rather
    than returning nothing -- so the scoring is called directly. A negative window
    that trips this is discarded, never relabelled."""
    _low, lw = word_lines(words, nl_after)
    hits, numeric, _n_tol = fact_hits(words, facts)
    scores, _rare = score_lines(lw, hits, numeric, facts)
    return bool(scores) and max(scores.values()) >= EVIDENCE_MIN_SCORE


# --------------------------------------------------------------------------- #
# line-type channel
# --------------------------------------------------------------------------- #
def line_type_of_lines(words, nl_after):
    """-> (line index -> LT_*, line_of_word, line_words).

    Precedence period-header > title > table > prose. A column header "2025 2024
    2023" satisfies is_table_line too (three numerics), and period-header is the
    more specific statement about it; a title line can never be a table line because
    is_title_line rejects any numeric word."""
    low, lw = word_lines(words, nl_after)
    types = {}
    for line, idxs in lw.items():
        ws = [words[k] for k in idxs]
        if is_period_header(ws):
            types[line] = LT_PERIOD
        elif is_title_line(ws):
            types[line] = LT_TITLE
        elif is_table_line(ws):
            types[line] = LT_TABLE
        else:
            types[line] = LT_PROSE
    return types, low, lw


def line_type_per_token(window):
    """int8 line type for each of the window's chunk tokens (0 on word_id -1)."""
    types, low, _lw = line_type_of_lines(window["words"], window["nl_after"])
    return [LT_NONE if w < 0 else types[low[w]] for w in window["word_id"]]


# --------------------------------------------------------------------------- #
# per-item build
# --------------------------------------------------------------------------- #
def item_rng(seed, qa_id):
    """Per-item RNG keyed by qa_id: decisions do not move when the input order or
    --limit changes, which is what makes a partial rebuild comparable to a full one."""
    return random.Random(f"{seed}:{qa_id}")


def evidence_geometry(src, window, ev_words):
    """Where the PLACED evidence sits in the finished window -> (start, centre, width).

    Measured against the evidence the placement used (the full-chunk evidence), not
    against whatever build_labels re-derives inside the window: rarity is recomputed
    per window, so the window's own evidence set is a different -- and usually wider
    -- thing, and scoring the placement with it would report the label policy's
    spread as the placement's error.

    Both ends are reported because the CENTRE is a poor descriptor of a wide span: an
    evidence set covering half the window has its centre near 0.5 whatever the
    placement did, so the centre histogram alone reads as a middle-bias that is really
    just span width."""
    s, e = window["tok_start"], window["tok_end"]
    inside = [k for k in ev_words if window["word_start"] <= k <= window["word_end"]]
    toks = [src["first_tok"][k] for k in inside if src["first_tok"][k] >= 0]
    toks += [src["last_tok"][k] for k in inside if src["last_tok"][k] >= 0]
    toks = [t for t in toks if s <= t < e]
    if not toks or e <= s:
        return None, None, None
    lo, hi = min(toks), max(toks) + 1
    span = float(e - s)
    return (lo - s) / span, ((lo + hi) / 2.0 - s) / span, (hi - lo) / span


def surviving_evidence_facts(window, kept_ev, facts):
    """-> (n gold-fact hits, n NUMERIC gold-fact hits) on the surviving evidence words.

    A seam cut takes the right-hand side of a table row away and leaves the row LABEL
    behind. "Total Current Assets" still matches the label-word half of
    gold_fact_words, so the line keeps scoring and the window keeps its keep-labels --
    over a fragment that no longer carries a single figure the question asked for.
    Counting the surviving fact hits is what separates that from a real seam."""
    w0 = window["word_start"]
    hits, numeric, _n_tol = fact_hits(window["words"], facts)
    n_hit = n_num = 0
    for k in kept_ev:
        i = k - w0
        if 0 <= i < len(hits) and hits[i]:
            n_hit += 1
            n_num += int(numeric[i])
    return n_hit, n_num


# ---------------------------------------------------------------------------
# E54 discovered labels (optional --labels-json): reader-verified minimal keep-sets
# replace the v10 policy on items the search covered. Three tiers, chosen so budget
# serving keeps a full ranking instead of a cliff: verified-breaker lines (removing
# them flipped the reader's answer) keep at weight 4.0; the rest of the minimal set
# keeps at 2.0; lines v10 kept but the search proved DROPPABLE get a soft 0.25
# target at weight 0.5 — droppable is not the same as junk, and hard-zeroing the
# headers/qualifiers the search happened not to need is how a compressor learns to
# strand numbers (the "decision distortion" failure). Everything else follows v10's
# drop weighting (numerals 1.5).
DISCOVERED = {}
DISCOVERED_DROP = set()
# "replace": E55/E55b tiered targets. "repair": v10/v11 labels VERBATIM, plus the
# discovered keep-set ADDED as keeps (fixes the 24% of items whose labels could not
# answer) and verified breakers up-weighted to 4.0. Nothing is ever demoted.
DISCOVERED_MODE = "replace"


def load_discovered(path):
    n_full = 0
    for line in open(path):
        r = json.loads(line)
        if r["status"] in ("ok", "v10_insufficient_full_ok"):
            DISCOVERED[r["qa_id"]] = r
            n_full += r["status"] == "v10_insufficient_full_ok"
        else:
            DISCOVERED_DROP.add(r["qa_id"])
    print(f"discovered labels: {len(DISCOVERED)} items "
          f"({n_full} searched from full chunk), {len(DISCOVERED_DROP)} dropped "
          f"(unanswerable/error)")


def discovered_window_labels(rec, src, window, dis, v10_weight_win,
                             window_v10_keep=()):
    """Map chunk-word-indexed discovered labels into one window's word list.

    Tiers (targets chosen so a CALIBRATED tau=0.5 recovers the optimal subset):
    primary minimal set 1.0 (breaker lines weight 4.0, rest 2.0); alternate-pass
    minimal words 0.6 (interchangeable evidence — kept at 0.5, ranked below core);
    v10-kept-but-proven-droppable 0.25 @ 0.5; else v10 drop weighting. Items the
    leakage probe flagged train at 0.3x weight: the reader answered them with no
    context, so their search carries little selection signal."""
    cache = dis.get("_cache")
    if cache is None:
        _lo, line_words = word_lines(src["words"], src["nl_after"])
        keep_set = set(dis["keep_words"])
        v10_chunk, _ = build_labels(src["words"], src["nl_after"],
                                    rec.get("gold_fact_words", []), rec["qtype"])
        cache = dis["_cache"] = {
            "keep": keep_set,
            "alt": set(dis.get("alt_keep_words", [])),
            "breaker": {w for l in dis.get("breaker_lines", [])
                        for w in line_words.get(l, [])},
            "v10_removed": ({i for i, k in enumerate(v10_chunk) if k}
                            - keep_set - set(dis.get("alt_keep_words", []))),
            "wmul": 0.3 if dis.get("leaked") else 1.0,
        }
    w0 = window["word_start"]
    wmul = cache["wmul"]
    keep, weight = [], []
    n_dis = 0
    if DISCOVERED_MODE == "repair":
        v10_keep_win = list(window_v10_keep)
        for i in range(len(window["words"])):
            g = w0 + i
            k = float(v10_keep_win[i])
            w = v10_weight_win[i]
            if g in cache["keep"] and k < 1.0:
                k = 1.0
                w = max(w, 2.0)
            if g in cache["breaker"] and k >= 0.5:
                w = max(w, 4.0)
            keep.append(k)
            weight.append(w * wmul)
            n_dis += int(k >= 0.5)
        return keep, weight, n_dis == 0
    for i in range(len(window["words"])):
        g = w0 + i
        if g in cache["keep"]:
            keep.append(1.0)
            weight.append((4.0 if g in cache["breaker"] else 2.0) * wmul)
            n_dis += 1
        elif g in cache["alt"]:
            keep.append(0.8)
            weight.append(1.0 * wmul)
            n_dis += 1
        elif g in cache["v10_removed"]:
            keep.append(0.5)
            weight.append(1.0 * wmul)
        else:
            keep.append(0.0)
            weight.append(v10_weight_win[i] * wmul)
    return keep, weight, n_dis == 0


def make_window_record(rec, src, window, kind, bucket, q_ids, q_trunc, args,
                       ev_words=(), primary_words=(), requested_pos=None,
                       n_evidence_fact_hits=None, n_evidence_numeric_hits=None,
                       is_seam=False):
    """One window -> the record that gets packed and written to *_meta.jsonl."""
    words, nl_after = window["words"], window["nl_after"]
    facts = rec.get("gold_fact_words", [])
    n_words = len(words)

    if kind == "negative":
        keep = [False] * n_words
        weight = [1.0] * n_words
        info = {"no_evidence": False, "evidence_lines": [], "n_evidence_lines": 0,
                "n_header_lines": 0, "n_floor_lines": 0, "rowlabel_added": 0,
                "n_numeric": 0, "n_numeric_keep": 0, "keep_frac_words": 0.0,
                "n_rare_facts": 0, "n_clause_scoped_lines": 0}
        inert = False
    else:
        keep, info = build_labels(words, nl_after, facts, rec["qtype"])
        weight = info["weight"]
        inert = bool(info["no_evidence"])
        dis = DISCOVERED.get(rec["qa_id"])
        if dis is not None:
            keep, weight, inert = discovered_window_labels(rec, src, window, dis,
                                                           weight, keep)

    suffix = {"positive": "p", "negative": "n"}[kind]
    ev_start, ev_centre, ev_width = (
        (None, None, None) if kind == "negative"
        else evidence_geometry(src, window, ev_words))
    _ps, primary_pos, _pw = (
        (None, None, None) if kind == "negative"
        else evidence_geometry(src, window, primary_words))
    return {
        "qa_id": f"{rec['qa_id']}#{suffix}",
        "source_qa_id": rec["qa_id"],
        "file": rec.get("file", ""),
        "question": rec["question"],
        "answer": rec.get("answer", ""),
        "qtype": rec.get("qtype", ""),
        "qtype_id": (QTYPES.index(rec["qtype"]) if rec.get("qtype") in QTYPES
                     else QTYPE_OTHER),
        "is_negative": kind == "negative",
        "inert": inert,
        "had_chunk_evidence": bool(ev_words),
        "is_seam": is_seam,
        "n_evidence_fact_hits": n_evidence_fact_hits,
        "n_evidence_numeric_hits": n_evidence_numeric_hits,
        "position_bucket": bucket,
        "requested_pos": requested_pos,
        "primary_rel_pos": primary_pos,
        "evidence_rel_pos": ev_centre,
        "evidence_start_pos": ev_start,
        "evidence_width_frac": ev_width,
        "window_offset": window["tok_start"],
        "window_tokens": window["tok_end"] - window["tok_start"],
        "window_word_range": [window["word_start"], window["word_end"]],
        "source_chunk_tokens": src["n_tok"],
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
        "_info": info,
    }


def build_item(rec, src, tok, args, want_negative):
    """One source item -> its window records + a per-item reason dict."""
    reasons = defaultdict(int)
    q_ids, q_trunc = encode_question(rec["question"], tok, args.question_budget)
    if not q_ids:
        reasons["empty_question"] += 1
        return [], reasons
    budget = args.max_len - len(q_ids) - N_SPECIAL
    if budget <= 0:
        reasons["question_eats_budget"] += 1
        return [], reasons

    facts = rec.get("gold_fact_words", [])
    ev = evidence_of_chunk(src, facts, rec.get("qtype", ""))
    ev_words = ev["ev_words"] if ev else set()
    primary_words = ev["primary_words"] if ev else []

    rng = item_rng(args.seed, rec["qa_id"])
    bucket, lo, hi = POSITION_BUCKETS[rng.randrange(len(POSITION_BUCKETS))]
    r = rng.uniform(lo, hi)

    out = []
    # ---- the repositioned window ----
    if ev_words:
        first, last = src["first_tok"], src["last_tok"]
        anchor = [first[k] for k in primary_words if first[k] >= 0]
        anchor += [last[k] for k in primary_words if last[k] >= 0]
        if not anchor:                       # primary line produced no token
            anchor = [first[k] for k in ev_words if first[k] >= 0]
            anchor += [last[k] for k in ev_words if last[k] >= 0]
        e_c = (min(anchor) + max(anchor) + 1) / 2.0
        s, e = position_window(src["n_tok"], e_c, budget, r, args.min_window_tokens)
    else:
        # No evidence anywhere in the chunk: the policy cannot say which lines answer
        # this question, so there is nothing to position against. Take the head of the
        # chunk and let the label pass mark the row inert.
        reasons["chunk_without_evidence"] += 1
        bucket = "none"
        s, e = 0, min(budget, src["n_tok"])

    s, e = snap_to_words(src, s, e)
    win = slice_window(src, s, e) if e > s else None
    if win is None:
        reasons["empty_window"] += 1
        return [], reasons

    kept_ev = ev_words & set(range(win["word_start"], win["word_end"] + 1))
    n_hits = n_num_hits = None
    if ev_words:
        n_hits, n_num_hits = surviving_evidence_facts(win, kept_ev, facts)

    qtype = rec.get("qtype", "")
    needs_figure = qtype in FIGURE_REQUIRED_QTYPES
    is_seam = bool(ev_words) and 0 < len(kept_ev) < len(ev_words)
    # What the seam survivor has to carry: a FIGURE for the numeric qtypes, any gold
    # fact for domain-qualitative.
    seam_survived = (n_num_hits if needs_figure else n_hits) if is_seam else None

    if ev_words and not kept_ev:
        # The reposition pushed every evidence word out of frame. The window really
        # does not answer the question any more, so it is a negative -- not an item
        # to throw away and not one to keep labelling as if the answer were still in
        # there.
        kind, bucket, is_seam = "negative", "none", False
        reasons["repositioned_into_negative"] += 1
    elif is_seam and not seam_survived:
        # A seam that left evidence WORDS but nothing answerable: no gold fact at all,
        # or (for the numeric qtypes) a row label whose figures were cut off.
        # Keep-labelling it would teach the model to keep table furniture whose
        # numbers are gone. Treated as the evidence-free case it actually is.
        kind, bucket = "negative", "none"
        reasons["seam_factless_negative"] += 1
        reasons["seam_factless_" + (qtype or "unknown")] += 1
        is_seam = False
    else:
        kind = "positive"
        if is_seam:
            reasons["seam_straddling"] += 1
            if not n_num_hits:
                # Survives only under the qualitative rule: gold fact present, no
                # figure. Counted so the split stays visible.
                reasons["seam_labels_only_no_figure"] += 1

    out.append(make_window_record(rec, src, win, kind, bucket, q_ids, q_trunc, args,
                                  ev_words=ev_words, primary_words=primary_words,
                                  requested_pos=r, n_evidence_fact_hits=n_hits,
                                  n_evidence_numeric_hits=n_num_hits, is_seam=is_seam))

    # ---- the optional extra negative ----
    if want_negative and ev_words and kind == "positive":
        span = negative_span(src, ev_words, budget, args.min_negative_tokens)
        if span is None:
            reasons["negative_no_region"] += 1
        else:
            nwin = slice_window(src, *span)
            if nwin is None:
                reasons["negative_no_region"] += 1
            elif ev_words & set(range(nwin["word_start"], nwin["word_end"] + 1)):
                reasons["negative_overlaps_evidence"] += 1
            elif looks_like_evidence(nwin["words"], nwin["nl_after"], facts):
                reasons["negative_not_clean"] += 1
            else:
                out.append(make_window_record(rec, src, nwin, "negative", "none",
                                              q_ids, q_trunc, args))
                reasons["negative_built"] += 1
    return out, reasons


# --------------------------------------------------------------------------- #
# self-checks (run on every window; these raise)
# --------------------------------------------------------------------------- #
def check_number_atomicity(src):
    """Every token lying inside a digit-carrying word must map to THAT word.

    A number is one keep/drop decision (otsofier PR #714). The way that breaks is not
    a word splitting -- it is map_tokens_to_words handing one of a number's subwords
    to the neighbouring label word, after which half the digits can be dropped while
    the other half is kept. Checked against char offsets on the whole chunk, so it
    covers every window sliced out of it."""
    text, spans, wid, offs = src["text"], src["spans"], src["word_id"], src["offs"]
    starts = [s for s, _ in spans]
    bad = []
    digit_word = [bool(DIGIT.search(text[s:e])) for s, e in spans]
    for t, (a, b) in enumerate(offs):
        if b <= a:
            continue
        core_a = a + (len(text[a:b]) - len(text[a:b].lstrip()))
        if core_a >= b:                     # whitespace-only token
            continue
        i = bisect_right(starts, core_a) - 1
        if i < 0 or not digit_word[i]:
            continue
        ws, we = spans[i]
        if core_a >= ws and b <= we and wid[t] != i:
            bad.append((t, wid[t], i, text[ws:we]))
    return bad


def check_window(win, max_len, vocab_size, cls_id, sep_id):
    """Structural invariants of one packed window. Raises on violation."""
    qa = win["qa_id"]
    q_ids, c_ids, wid = win["_q_ids"], win["_c_ids"], win["_word_id"]
    lt = win["_line_type"]
    n_pre = 1 + len(q_ids) + 1
    total = n_pre + len(c_ids) + 1
    if total > max_len:
        raise AssertionError(f"{qa}: packed length {total} > max_len {max_len}")
    if len(wid) != len(c_ids) or len(lt) != len(c_ids):
        raise AssertionError(f"{qa}: word_id/line_type length != chunk ids")
    ids_max = max([cls_id, sep_id] + list(q_ids) + list(c_ids))
    if ids_max >= vocab_size:
        raise AssertionError(f"{qa}: token id {ids_max} >= vocab {vocab_size}")
    if min([cls_id, sep_id] + list(q_ids) + list(c_ids)) < 0:
        raise AssertionError(f"{qa}: negative token id")

    content = [w for w in wid if w >= 0]
    if not content:
        raise AssertionError(f"{qa}: window has no content token")
    if content != sorted(content):
        raise AssertionError(f"{qa}: word_id is not non-decreasing")
    if content[0] != 0 or content[-1] != len(win["words"]) - 1:
        raise AssertionError(f"{qa}: word_id spans {content[0]}..{content[-1]}, "
                             f"words has {len(win['words'])}")
    if set(content) != set(range(len(win["words"]))):
        missing = sorted(set(range(len(win["words"]))) - set(content))
        raise AssertionError(f"{qa}: word_id has gaps at {missing[:5]}")
    for w, t in zip(wid, lt):
        if w < 0 and t != LT_NONE:
            raise AssertionError(f"{qa}: line_type {t} on a non-content token")
        if w >= 0 and not (LT_PROSE <= t <= LT_PERIOD):
            raise AssertionError(f"{qa}: line_type {t} out of range on content")
    if len(win["nl_after"]) != len(win["words"]):
        raise AssertionError(f"{qa}: nl_after/words length mismatch")
    if win["is_negative"] and any(win["keep"]):
        raise AssertionError(f"{qa}: negative window has {sum(win['keep'])} keeps")
    # Every POSITIVE window drawn from a chunk that had evidence must still hold at
    # least one gold fact on its surviving evidence, and a SEAM window on a numeric
    # qtype must still hold a FIGURE. Anything else is a factless fragment and should
    # have been relabelled negative upstream.
    if not win["is_negative"] and not win["inert"] and win["had_chunk_evidence"]:
        if not win["n_evidence_fact_hits"]:
            raise AssertionError(
                f"{qa}: positive window whose surviving evidence matches no gold "
                f"fact -- the seam relabel did not fire")
        if (win["is_seam"] and win["qtype"] in FIGURE_REQUIRED_QTYPES
                and not win["n_evidence_numeric_hits"]):
            raise AssertionError(
                f"{qa}: seam window on {win['qtype']} kept label words but no "
                f"surviving FIGURE -- the seam relabel did not fire")


def check_dangling(src, win):
    """Boundary words must have brought ALL their tokens into the window."""
    w0, w1 = win["window_word_range"]
    s, e = win["window_offset"], win["window_offset"] + win["window_tokens"]
    for w in (w0, w1):
        if not (s <= src["first_tok"][w] and src["last_tok"][w] < e):
            raise AssertionError(f"{win['qa_id']}: word {w} is dangling "
                                 f"(tokens {src['first_tok'][w]}..{src['last_tok'][w]} "
                                 f"vs window [{s},{e}))")


def fact_check_window(win, args):
    """Gold-fact survival for one positive window, against its IN-WINDOW fact set.

    Facts the reposition pushed out of frame are excluded before scoring: they are
    unreachable by construction, and counting them would report the window policy's
    losses as the label policy's."""
    words, nl_after = win["words"], win["nl_after"]
    facts = win["gold_fact_words"]
    full = render_mask(words, nl_after, [True] * len(words))
    in_window = [f for f in facts if fact_survival([f], full) == 1.0]
    kept = render_mask(words, nl_after, win["keep"])
    surv = fact_survival(in_window, kept)
    missing = [f for f in in_window if fact_survival([f], kept) < 1.0]
    return {"qa_id": win["qa_id"], "qtype": win["qtype"], "n_facts": len(facts),
            "n_in_window": len(in_window), "fact_survival": surv,
            "missing": missing[:8]}


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #
def pack(windows, max_len, cls_id, sep_id, pad_id):
    """Windows -> (inputs dict, targets dict). v8/v9 tensor conventions + line_type."""
    n = len(windows)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.int32)
    attn = torch.zeros((n, max_len), dtype=torch.uint8)
    word_id = torch.full((n, max_len), -1, dtype=torch.int16)
    line_type = torch.zeros((n, max_len), dtype=torch.int8)
    n_words = torch.zeros((n,), dtype=torch.int32)
    source_id = torch.zeros((n,), dtype=torch.int8)

    targets = torch.zeros((n, max_len), dtype=torch.float16)
    loss_mask = torch.zeros((n, max_len), dtype=torch.uint8)
    loss_weight = torch.zeros((n, max_len), dtype=torch.float16)

    for i, w in enumerate(windows):
        q_ids, c_ids = w["_q_ids"], w["_c_ids"]
        n_pre = 1 + len(q_ids) + 1
        seq = [cls_id] + list(q_ids) + [sep_id] + list(c_ids) + [sep_id]
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
        attn[i, :len(seq)] = 1
        wid = np.asarray(w["_word_id"], dtype=np.int64)
        word_id[i, n_pre:n_pre + len(c_ids)] = torch.from_numpy(wid.astype(np.int16))
        line_type[i, n_pre:n_pre + len(c_ids)] = torch.tensor(w["_line_type"],
                                                              dtype=torch.int8)
        n_words[i] = len(w["words"])
        source_id[i] = w["qtype_id"]

        keep = np.asarray(w["keep"], dtype=np.float32)
        weight = np.asarray(w["_weight"], dtype=np.float32)
        content = wid >= 0
        idx = wid[content]
        seg_t = np.zeros(len(c_ids), dtype=np.float32)
        seg_w = np.zeros(len(c_ids), dtype=np.float32)
        seg_m = np.zeros(len(c_ids), dtype=np.uint8)
        # An inert row keeps its inputs but leaves the loss mask empty: we cannot say
        # which lines answer this question, and labelling it all-drop would teach
        # "drop everything" on a window that does hold the answer.
        if idx.size and not w["inert"]:
            seg_t[content] = keep[idx]
            seg_w[content] = weight[idx]
            seg_m[content] = 1
        t_row = np.zeros(max_len, dtype=np.float32)
        w_row = np.zeros(max_len, dtype=np.float32)
        m_row = np.zeros(max_len, dtype=np.uint8)
        t_row[n_pre:n_pre + len(c_ids)] = seg_t
        w_row[n_pre:n_pre + len(c_ids)] = seg_w
        m_row[n_pre:n_pre + len(c_ids)] = seg_m
        targets[i] = torch.from_numpy(t_row).half()
        loss_weight[i] = torch.from_numpy(w_row).half()
        loss_mask[i] = torch.from_numpy(m_row)

    inputs = {"input_ids": input_ids, "attention_mask": attn, "word_id": word_id,
              "line_type": line_type, "n_words": n_words, "source_id": source_id,
              "qa_id": [w["qa_id"] for w in windows]}
    tgts = {"targets": targets, "loss_mask": loss_mask, "loss_weight": loss_weight,
            "qa_id": [w["qa_id"] for w in windows],
            "inert_qa_id": [w["qa_id"] for w in windows if w["inert"]]}
    return inputs, tgts


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def third_of(pos):
    return "start" if pos < 1 / 3 else ("middle" if pos < 2 / 3 else "end")


def summarize(windows, split):
    pos = [w for w in windows if not w["is_negative"] and not w["inert"]]
    neg = [w for w in windows if w["is_negative"]]
    inert = [w for w in windows if w["inert"]]

    def keep_frac_tokens(w):
        wid = np.asarray(w["_word_id"])
        content = wid[wid >= 0]
        if not content.size:
            return 0.0
        keep = np.asarray(w["keep"], dtype=bool)
        return float(keep[content].mean())

    thirds, starts, requested = defaultdict(int), defaultdict(int), defaultdict(int)
    span_thirds = defaultdict(int)
    widths = []
    for w in pos:
        p = w.get("primary_rel_pos")
        if p is not None:
            thirds[third_of(p)] += 1
        if w.get("evidence_rel_pos") is not None:
            span_thirds[third_of(w["evidence_rel_pos"])] += 1
        if w.get("evidence_start_pos") is not None:
            starts[third_of(w["evidence_start_pos"])] += 1
            widths.append(w["evidence_width_frac"])
        if w.get("requested_pos") is not None:
            requested[third_of(w["requested_pos"])] += 1
    by_q = defaultdict(list)
    for w in pos:
        by_q[w["qtype"]].append(w)

    lt_counts = defaultdict(int)
    for w in windows:
        for t in w["_line_type"]:
            lt_counts[int(t)] += 1

    agg = {
        "n_windows": len(windows), "n_positive": len(pos), "n_negative": len(neg),
        "n_inert": len(inert),
        "position_thirds": {k: thirds.get(k, 0) for k in ("start", "middle", "end")},
        "position_thirds_frac": {k: thirds.get(k, 0) / max(1, len(pos))
                                 for k in ("start", "middle", "end")},
        "position_thirds_requested": {k: requested.get(k, 0)
                                      for k in ("start", "middle", "end")},
        "position_thirds_evidence_start": {k: starts.get(k, 0)
                                           for k in ("start", "middle", "end")},
        "position_thirds_evidence_span_centre": {k: span_thirds.get(k, 0)
                                                 for k in ("start", "middle", "end")},
        "evidence_width_frac_mean": float(np.mean(widths)) if widths else 0.0,
        "evidence_width_frac_p90": (float(sorted(widths)[int(len(widths) * .9)])
                                    if widths else 0.0),
        "mean_window_tokens": float(np.mean([w["window_tokens"] for w in windows])),
        "mean_question_tokens": float(np.mean([w["question_tokens"] for w in windows])),
        "n_question_truncated": sum(1 for w in windows if w["question_truncated"]),
        "mean_keep_frac_tokens_positive": float(np.mean([keep_frac_tokens(w)
                                                         for w in pos])) if pos else 0.0,
        "keep_frac_tokens_by_qtype": {
            qt: float(np.mean([keep_frac_tokens(w) for w in ws]))
            for qt, ws in sorted(by_q.items())},
        "n_by_qtype": {qt: len(ws) for qt, ws in sorted(by_q.items())},
        "line_type_token_frac": {
            LINE_TYPE_NAMES[k]: v / max(1, sum(lt_counts.values()))
            for k, v in sorted(lt_counts.items())},
    }
    kf = [keep_frac_tokens(w) for w in pos]
    if kf:
        kf.sort()
        agg["keep_frac_tokens_p10"] = float(kf[int(len(kf) * .1)])
        agg["keep_frac_tokens_p50"] = float(kf[int(len(kf) * .5)])
        agg["keep_frac_tokens_p90"] = float(kf[min(len(kf) - 1, int(len(kf) * .9))])

    print(f"\n=== {split}: {len(windows)} windows "
          f"({len(pos)} positive, {len(neg)} negative, {len(inert)} inert) ===")
    print(f"  window tokens mean={agg['mean_window_tokens']:.0f}  "
          f"question tokens mean={agg['mean_question_tokens']:.1f} "
          f"({agg['n_question_truncated']} truncated)")
    print(f"  ANSWER ROW position achieved: " + "  ".join(
        f"{k}={agg['position_thirds'][k]} ({agg['position_thirds_frac'][k]:.2f})"
        for k in ("start", "middle", "end"))
        + "   requested: " + "/".join(str(agg["position_thirds_requested"][k])
                                      for k in ("start", "middle", "end")))
    print(f"  full evidence span: start third " + "/".join(
        str(agg["position_thirds_evidence_start"][k])
        for k in ("start", "middle", "end"))
        + "   centre third " + "/".join(
            str(agg["position_thirds_evidence_span_centre"][k])
            for k in ("start", "middle", "end"))
        + f"   width mean={agg['evidence_width_frac_mean']:.2f} "
          f"p90={agg['evidence_width_frac_p90']:.2f} of the window")
    print(f"  keep_frac_tokens (positive): p10={agg.get('keep_frac_tokens_p10', 0):.3f} "
          f"p50={agg.get('keep_frac_tokens_p50', 0):.3f} "
          f"p90={agg.get('keep_frac_tokens_p90', 0):.3f} "
          f"mean={agg['mean_keep_frac_tokens_positive']:.3f}")
    for qt, v in agg["keep_frac_tokens_by_qtype"].items():
        print(f"    {qt:22s} n={agg['n_by_qtype'][qt]:4d}  keep_frac={v:.3f}")
    print("  line_type token share: " + "  ".join(
        f"{k}={v:.3f}" for k, v in agg["line_type_token_frac"].items()))
    return agg


# --------------------------------------------------------------------------- #
# split build
# --------------------------------------------------------------------------- #
def load_source(in_dir, split, limit=0):
    path = Path(in_dir) / f"{split}_meta.jsonl"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
            if limit and len(recs) >= limit:
                break
    return recs


def build_split(recs, tok, args, cls_id, sep_id, split):
    """Source records -> (windows, reasons, checks)."""
    vocab = len(tok)
    # Which items get an extra negative: a deterministic per-item draw, so the choice
    # does not move with --limit or with the order the file happens to be in.
    want_neg = {r["qa_id"]: item_rng(args.seed + 1, r["qa_id"]).random()
                < args.negative_frac for r in recs}

    windows, reasons = [], defaultdict(int)
    atomicity_bad = []
    for i, rec in enumerate(recs):
        if rec["qa_id"] in DISCOVERED_DROP:
            reasons["discovered_unanswerable"] += 1
            continue
        src = prepare_source(rec, tok)
        if src is None:
            reasons["unsegmentable_chunk"] += 1
            continue
        bad = check_number_atomicity(src)
        if bad:
            atomicity_bad.append((rec["qa_id"], bad[:3]))
        wins, rs = build_item(rec, src, tok, args, want_neg[rec["qa_id"]])
        for k, v in rs.items():
            reasons[k] += v
        for w in wins:
            check_window(w, args.max_len, vocab, cls_id, sep_id)
            check_dangling(src, w)
        windows.extend(wins)
        if (i + 1) % 250 == 0:
            print(f"  {split}: {i+1}/{len(recs)} items -> {len(windows)} windows")

    if atomicity_bad:
        raise AssertionError(
            f"{split}: number atomicity violated on {len(atomicity_bad)} chunks, "
            f"first: {atomicity_bad[:2]}")
    return windows, dict(reasons)


def run_fact_checks(windows, args, split):
    """Fact-survival reconstruction on N random positive windows. Raises if low."""
    pos = [w for w in windows if not w["is_negative"] and not w["inert"]
           and w["gold_fact_words"]]
    if not pos:
        print(f"  {split}: no positive window with facts to check")
        return []
    rng = random.Random(args.seed + 7)
    sample = rng.sample(pos, min(args.n_fact_checks, len(pos)))
    out = [fact_check_window(w, args) for w in sample]
    surv = [c["fact_survival"] for c in out]
    full = sum(1 for s in surv if s >= 0.999)
    print(f"  {split}: fact survival on {len(out)} random positive windows: "
          f"mean={np.mean(surv):.3f}  full on {full}/{len(out)}  "
          f"min={min(surv):.3f}")
    for c in out[:3]:
        print(f"    {c['qa_id']} ({c['qtype']}): {c['fact_survival']:.3f} of "
              f"{c['n_in_window']} in-window facts"
              + (f"  missing={c['missing'][:4]}" if c["missing"] else "  ALL PRESENT"))
    if float(np.mean(surv)) < args.fact_survival_min:
        raise AssertionError(
            f"{split}: mean in-window fact survival {np.mean(surv):.3f} < "
            f"{args.fact_survival_min} -- the label policy is losing answer figures")
    return out


def verify_tensors(inputs, tgts, windows, split, max_len):
    """Post-pack invariants on the actual tensors that ship."""
    wid = inputs["word_id"]
    lm = tgts["loss_mask"]
    content = wid >= 0
    inert_rows = torch.tensor([w["inert"] for w in windows])
    expect = content.clone()
    expect[inert_rows] = False
    if not torch.equal(lm.bool(), expect):
        n_bad = int((lm.bool() != expect).sum())
        raise AssertionError(f"{split}: loss_mask != (word_id>=0 & not inert) on "
                             f"{n_bad} positions -- the question or the specials "
                             f"would enter the loss")
    if (~content).any() and float(tgts["targets"][~content].abs().max()) != 0.0:
        raise AssertionError(f"{split}: non-zero target outside the chunk region")
    if not lm.any():
        raise AssertionError(f"{split}: loss mask is empty")
    if float(tgts["loss_weight"][lm.bool()].min()) <= 0.0:
        raise AssertionError(f"{split}: zero loss_weight inside the loss mask")
    neg_rows = [i for i, w in enumerate(windows) if w["is_negative"]]
    if neg_rows:
        t = tgts["targets"][torch.tensor(neg_rows)]
        if float(t.max()) != 0.0:
            raise AssertionError(f"{split}: a negative window has a keep target")
        lw = tgts["loss_weight"][torch.tensor(neg_rows)]
        m = lm[torch.tensor(neg_rows)].bool()
        if m.any() and (float(lw[m].min()) != 1.0 or float(lw[m].max()) != 1.0):
            raise AssertionError(f"{split}: negative window loss_weight != 1.0")
    lt = inputs["line_type"]
    if int(lt.min()) < 0 or int(lt.max()) >= N_LINE_TYPES:
        raise AssertionError(f"{split}: line_type out of range "
                             f"[{int(lt.min())}, {int(lt.max())}]")
    if not torch.equal((lt != 0), content):
        raise AssertionError(f"{split}: line_type nonzero off the content tokens")
    if inputs["input_ids"].shape[1] != max_len:
        raise AssertionError(f"{split}: seq len {inputs['input_ids'].shape[1]}")
    print(f"  {split}: tensor checks OK  "
          f"content_tokens={int(content.sum())}  "
          f"loss_tokens={int(lm.sum())}  "
          f"pos_prevalence={float((tgts['targets'][lm.bool()] >= 0.5).float().mean()):.4f}")
    return float((tgts["targets"][lm.bool()] >= 0.5).float().mean())


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

    print("word-boundary golden (character spans, not token->word assignment)")
    for text, want in WORD_BOUNDARY_GOLDEN:
        got = [text[a:b] for a, b in segment_words(text)]
        check(f"segment {text[:34]!r}", got, want)

    print("position_window: the evidence lands where it was asked to")
    for r in (0.1, 0.2, 0.5, 0.8, 0.9):
        s, e = position_window(2000, 1000.0, 1949, r, 0)
        got = (1000.0 - s) / (e - s)
        check(f"r={r} achieved", round(got, 3), round(r, 3))
    s, e = position_window(2000, 1000.0, 1949, 0.5, 0)
    check("centred window uses the whole budget", e - s, 1949)
    s, e = position_window(2000, 1000.0, 1949, 0.2, 0)
    check("start-third window is shorter (chunk runs out on the right)",
          (e - s) < 1949, True)
    check("start-third window ends at the chunk end", e, 2000)
    s, e = position_window(500, 250.0, 1949, 0.5, 0)
    check("short chunk clamps to the chunk", (s, e), (0, 500))

    print("line-type precedence")
    from v9_rl_prep import newlines_after as _nl, segment_words as _sw
    lines = ["CONSOLIDATED BALANCE SHEETS",
             "(Millions of U.S. Dollars) 2025 2024 2023",
             "Total Current Assets $5,825.8 $6,363.0 $5,100.0",
             "We caution readers that these statements are not guarantees of future"]
    text = "\n".join(lines)
    spans = _sw(text)
    words = [text[a:b] for a, b in spans]
    nl = _nl(text, spans)
    types, low, lw = line_type_of_lines(words, nl)
    check("title", types[0], LT_TITLE)
    check("period header beats table", types[1], LT_PERIOD)
    check("table", types[2], LT_TABLE)
    check("prose", types[3], LT_PROSE)

    print("snap_to_words / slice_window on a real tokenization")
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    src = prepare_source({"chunk_text": text + "\n" + text}, tok)
    n = src["n_tok"]
    bad_edges = 0
    for s0 in range(0, n, 3):
        e0 = min(n, s0 + 17)
        s1, e1 = snap_to_words(src, s0, e0)
        if e1 <= s1:
            continue
        w = slice_window(src, s1, e1)
        if w is None:
            continue
        w0, w1 = w["word_start"], w["word_end"]
        if not (src["first_tok"][w0] >= s1 and src["last_tok"][w1] < e1):
            bad_edges += 1
    check("no dangling subword over a sliding sweep", bad_edges, 0)
    check("number atomicity on the fixture", check_number_atomicity(src), [])

    print("seam survivor fact accounting")
    row = "Total Current Assets 5,825.8 6,363.0".split()
    win_full = {"words": row, "word_start": 0}
    facts_row = ["5,825.8", "6,363.0", "assets"]
    n_hit, n_num = surviving_evidence_facts(win_full, set(range(len(row))), facts_row)
    check("intact row hits facts incl. figures", (n_hit, n_num), (3, 2))
    # the seam case: the figures were cut, only the row label survives
    win_cut = {"words": "Total Current Assets".split(), "word_start": 0}
    n_hit, n_num = surviving_evidence_facts(win_cut, {0, 1, 2}, facts_row)
    check("label-only survivor keeps a fact but no figure", (n_hit, n_num), (1, 0))
    win_none = {"words": "Total Current".split(), "word_start": 0}
    n_hit, n_num = surviving_evidence_facts(win_none, {0, 1}, facts_row)
    check("factless survivor scores zero -> relabelled negative", n_hit, 0)

    print("negative window rejection")
    facts = ["5,825.8", "assets"]
    check("evidence-bearing text is not clean",
          looks_like_evidence(words, nl, facts), True)
    prose = "We caution readers that these statements are not guarantees".split()
    check("prose is clean", looks_like_evidence(prose, [0] * len(prose), facts), False)

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=DEFAULT_IN,
                    help="local copy of the v9-rl build (source of chunk_text)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", default="",
                    help="default: whatever the source meta.json used")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--question-budget", type=int, default=192,
                    help="HARD CAP on question tokens; the actual question length is "
                         "used when shorter, and the content budget follows it")
    ap.add_argument("--negative-frac", type=float, default=0.20,
                    help="share of items that also get an evidence-free window")
    ap.add_argument("--min-window-tokens", type=int, default=768,
                    help="floor on the repositioned window; below it the length wins "
                         "and the requested position is clamped")
    ap.add_argument("--min-negative-tokens", type=int, default=600,
                    help="a negative region shorter than this is skipped, not padded")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap source items per split")
    ap.add_argument("--n-fact-checks", type=int, default=20)
    ap.add_argument("--fact-survival-min", type=float, default=0.85)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--upload", action="store_true",
                    help=f"modal volume put -> {VOLUME}:/{VOLUME_SUBDIR}")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument("--volume-subdir", default=VOLUME_SUBDIR)
    ap.add_argument("--labels-mode", default="replace",
                    choices=["replace", "repair"],
                    help="repair: v10 labels verbatim + discovered keeps ADDED + "
                         "breakers up-weighted (nothing demoted)")
    ap.add_argument("--labels-json", default="",
                    help="e54 labels.jsonl: discovered minimal-sufficient keep-sets "
                         "override the v10 policy on covered items; unanswerable "
                         "items are dropped")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    # Before anything is read or written: the imported segmentation must still be the
    # one this build's self-checks were written against.
    check_word_boundaries()
    print("word-boundary golden: OK "
          f"({len(WORD_BOUNDARY_GOLDEN)} fixtures, segment_words unchanged)")
    if args.labels_json:
        global DISCOVERED_MODE
        DISCOVERED_MODE = args.labels_mode
        print(f"discovered-labels mode: {DISCOVERED_MODE}")
        load_discovered(args.labels_json)

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    src_meta = json.loads((in_dir / "meta.json").read_text())
    tokenizer_name = args.tokenizer or src_meta["tokenizer"]
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError(f"tokenizer missing CLS/SEP: {cls_id}/{sep_id}")
    print(f"tokenizer={tokenizer_name} cls/sep/pad={cls_id}/{sep_id}/{pad_id}  "
          f"max_len={args.max_len}  q_cap={args.question_budget}  seed={args.seed}")

    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "built_by": "v11_pretokenize.py",
        "source": f"{VOLUME}:/v9-rl",
        "source_meta": {k: src_meta.get(k) for k in
                        ("max_len", "tokenizer", "layout", "seed", "n_train", "n_val")},
        "source_limitation": (
            "chunk_text is an evidence-centered ~2048-token window of the filing, not "
            "the full filing; all v11 windows are sub-windows of it, so negatives are "
            "HARD negatives from the evidence's own neighbourhood"),
        "max_len": args.max_len,
        "tokenizer": tokenizer_name,
        "layout": "[CLS] question [SEP] window [SEP]",
        "word_id_semantics": "word index on window tokens, -1 = non-content",
        "qtype_source_ids": QTYPES,
        "source_id_other": QTYPE_OTHER,
        "labels_json": args.labels_json or None,
        "seed": args.seed,
        "budget_policy": {
            "question": f"actual question length, hard cap {args.question_budget} "
                        f"tokens, truncated on a word boundary",
            "content": f"{args.max_len} - len(question_ids) - {N_SPECIAL}",
        },
        "window_policy": {
            "positioning": ("evidence relative position r sampled uniformly inside a "
                            "uniformly chosen third; window = longest window <= budget "
                            "with the evidence at r, i.e. "
                            "W = min(budget, e_c/r, (n_tok-e_c)/(1-r))"),
            "thirds": {name: [lo, hi] for name, lo, hi in POSITION_BUCKETS},
            "min_window_tokens": args.min_window_tokens,
            "edges": "both ends snapped to whole words (no dangling subwords)",
            "seam": ("a window that cuts an evidence line keeps it, PROVIDED the "
                     "surviving evidence is still answerable; a window that loses "
                     "all evidence, or fails that test, is relabelled a negative"),
            "seam_survivor_requirement": {
                "metrics-extraction": "at least one surviving NUMERIC gold fact",
                "multistep-numerical": "at least one surviving NUMERIC gold fact",
                "domain-qualitative": ("at least one surviving gold fact of any kind "
                                       "(prose evidence is legitimately non-numeric)"),
            },
            "word_boundary_golden": [t for t, _w in WORD_BOUNDARY_GOLDEN],
            "negatives": {
                "frac_of_items": args.negative_frac,
                "min_tokens": args.min_negative_tokens,
                "region": "far side of the evidence span, pushed away from it",
                "clean_test": ("no evidence word inside, and no line scoring >= "
                               f"{EVIDENCE_MIN_SCORE} under the v10 rare-fact rule"),
                "labels": "all-drop, loss_weight 1.0, question KEPT",
            },
        },
        "line_type": {
            "dtype": "int8, separate tensor [N, max_len]",
            "codes": {str(k): v for k, v in LINE_TYPE_NAMES.items()},
            "precedence": "period_header > title > table > prose",
            "note": "0 on pad / special / question AND on word_id == -1 tokens",
        },
        "label_policy": "v10_build_targets.build_labels, applied to the WINDOW words",
    }

    all_checks = {}
    for split in ("train", "val"):
        recs = load_source(in_dir, split, args.limit)
        print(f"\n{split}: {len(recs)} source items")
        windows, reasons = build_split(recs, tok, args, cls_id, sep_id, split)
        print(f"  -> {len(windows)} windows   reasons={dict(sorted(reasons.items()))}")
        agg = summarize(windows, split)
        agg["build_reasons"] = reasons
        agg["n_source_items"] = len(recs)
        inputs, tgts = pack(windows, args.max_len, cls_id, sep_id, pad_id)
        prevalence = verify_tensors(inputs, tgts, windows, split, args.max_len)
        agg["pos_prevalence"] = prevalence
        all_checks[split] = run_fact_checks(windows, args, split)
        agg["fact_checks"] = all_checks[split]
        meta[f"n_{split}"] = len(windows)
        meta[f"{split}_stats"] = agg
        if split == "val":
            meta["pos_prevalence"] = prevalence
        else:
            meta["pos_prevalence_train"] = prevalence

        if args.no_save:
            continue
        torch.save(inputs, out_dir / f"{split}.pt")
        torch.save(tgts, out_dir / f"{split}_targets.pt")
        with open(out_dir / f"{split}_meta.jsonl", "w") as f:
            for w in windows:
                f.write(json.dumps({k: v for k, v in w.items()
                                    if not k.startswith("_")}) + "\n")

    if args.no_save:
        print("\n--no-save: nothing written")
        return 0

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved -> {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")

    if args.upload:
        cmd = ["modal", "volume", "put", "--force", args.volume, str(out_dir),
               args.volume_subdir]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"uploaded -> {args.volume}:/{args.volume_subdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
