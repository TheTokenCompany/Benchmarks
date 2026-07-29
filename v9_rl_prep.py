#!/usr/bin/env python3
"""v9 RL prep: synthetic QA -> evidence-centered RL chunks (runs LOCALLY, no Modal).

The RWR trainer needs something the v8 SFT pretokenizer never produced: for each
question, ONE chunk that actually contains the answer, plus everything needed to
render a word-level keep-mask back into readable text and score it.

Per synthetic_qa.jsonl item this emits:

    input_ids   [CLS] question [SEP] chunk [SEP] [PAD...]      (v8 layout, verbatim)
    word_id     per-token word index on chunk tokens, -1 on question/specials/pad
                (so it doubles as the v8 loss_mask -- content tokens are word_id>=0)
    words       the chunk's word strings, in order
    nl_after    newlines that follow each word (rendering keeps table/line geometry)
    gold_fact_words  numeric tokens + adjacent label words from the evidence lines,
                     the free dense half of the reward

Chunk selection: the filing is tokenized ONCE (offset mapping), the evidence lines
are resolved to a char span, that span's token range is CENTERED in a
content_budget = max_len - len(question) - 3 window, and the window is sliced back
out of the RAW TEXT by char offset. Slicing text (not detokenizing ids) is what
keeps the newlines and column spacing that make a 10-Q table readable -- the reward
model reads this text, so its layout is load-bearing.

Word segmentation is whitespace spans plus a merge pass for split numbers
("1, 904" -> one word, "$ 1,904" -> one word), mirroring otsofier PR #714: a
number is ONE atomic keep/drop decision, never a partially-kept string of digits.
This also matches serving's "one word per SentencePiece '_'-prefixed group" for
every realistic money/number token.

Run:
    .venv/bin/python v9_rl_prep.py                       # full build (~2200 items)
    .venv/bin/python v9_rl_prep.py --limit 20 --out-dir /tmp/v9smoke   # smoke
    .venv/bin/python v9_rl_prep.py --upload               # build + modal volume put
"""

import argparse
import json
import random
import re
import subprocess
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

SCRATCH = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-"
           "b409-cb03154f2355-financebench-check/838fe3de-3a8f-4846-ba75-9e663316210c/"
           "scratchpad")
DEFAULT_DATA = f"{SCRATCH}/v8data"

N_SPECIAL = 3                  # [CLS] + [SEP] (after question) + [SEP] (after chunk)
QTYPES = ["metrics-extraction", "multistep-numerical", "domain-qualitative"]
QTYPE_OTHER = 3

VOLUME = "otso-v8-data"
VOLUME_SUBDIR = "v9-rl"

DIGIT = re.compile(r"\d")
WS = re.compile(r"\s+")
CURRENCY = {"$", "€", "£", "¥"}
UNITS = {"%", "$", "€", "£", "¥", "bps"}
# label words next to a number that carry no identifying information on their own
FACT_STOP = {
    "the", "and", "for", "with", "was", "were", "are", "our", "its", "his", "her",
    "from", "that", "this", "these", "those", "than", "into", "per", "not", "but",
    "all", "any", "has", "had", "have", "been", "which", "such", "also", "other",
    "ended", "months", "three", "nine", "six", "twelve", "year", "years", "period",
    "periods", "respectively", "compared", "including", "approximately", "primarily",
    "increase", "decrease", "increased", "decreased", "million", "billion",
    "thousands", "millions", "billions", "total",
}


# --------------------------------------------------------------------------- #
# word segmentation
# --------------------------------------------------------------------------- #
def segment_words(text):
    """[(start, end)] char spans of atomic words in `text`.

    Base pass is whitespace-run splitting (identical to v8's word_char_spans and to
    line.split() indexing). Then three merges keep numbers atomic, per PR #714:
      "1, 904"  -> one word (a thousands separator that got split by whitespace)
      "$ 1,904" -> one word (a bare currency symbol is never its own decision)
      "12.5 %"  -> one word (a unit that only means something attached to a number)
    Currency binds forward, units bind backward; a symbol that is both (`$`) uses
    the forward rule only.
    """
    spans, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        spans.append((i, j))
        i = j

    merged = []
    for s, e in spans:
        w = text[s:e]
        if merged:
            ps, pe = merged[-1]
            prev = text[ps:pe]
            # thousands separator split across whitespace: "...1," + "904..."
            split_number = (len(prev) >= 2 and prev[-1] in ",." and prev[-2].isdigit()
                            and w[:1].isdigit())
            # bare currency symbol glued onto the amount that follows it
            bare_currency = prev in CURRENCY and (w[:1].isdigit() or w[:1] in CURRENCY)
            # a unit that only means something attached to its number: "12.5" + "%".
            # Currency symbols are excluded: they bind FORWARD (bare_currency), and
            # merging one backward both attaches it to the wrong amount and blocks
            # the forward merge -- "2026 $ 9,353" became "2026 $" + "9,353".
            unit = strip_edges(w)
            trailing_unit = (unit in UNITS and unit not in CURRENCY and prev
                             and (prev[-1].isdigit() or prev[-1] in ")%"))
            if split_number or bare_currency or trailing_unit:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    return merged


def newlines_after(text, spans):
    """How many '\\n' follow each word before the next one starts (tail included)."""
    out = []
    for k, (_s, e) in enumerate(spans):
        nxt = spans[k + 1][0] if k + 1 < len(spans) else len(text)
        out.append(text.count("\n", e, nxt))
    return out


def render_mask(words, nl_after, keep):
    """Word keep-mask -> the text the reward model reads.

    Kept words are space-joined; dropped words vanish; newlines survive wherever
    they were, so a table that keeps its numbers still reads as a table."""
    parts = []
    for w, nl, k in zip(words, nl_after, keep):
        if k:
            parts.append(w)
            parts.append("\n" * nl if nl else " ")
        elif nl:
            parts.append("\n" * nl)
    out = "".join(parts)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# --------------------------------------------------------------------------- #
# gold facts
# --------------------------------------------------------------------------- #
def strip_edges(w):
    return w.strip("()[]{}<>.,;:!?\"'*|")


def gold_facts(evidence_lines, gold_answer, max_facts=40):
    """Numeric tokens from the evidence lines + the label words next to them.

    Kept in surface form (commas and all): rendering re-emits the original word
    strings, so substring containment is the right test (see fact_survival for the
    one normalization that matters).

    The answer's own content words go in too, not just its numbers. Without them a
    domain-qualitative item -- whose evidence is prose and whose answer is something
    like a list of countries -- gets a fact set of whatever page-header years the
    numeric scan happened to find, which is signal about nothing."""
    facts, seen = [], set()

    def add(w):
        w = strip_edges(w).lower()
        if len(w) < 2 or w in seen:
            return
        seen.add(w)
        facts.append(w)

    for line in evidence_lines:
        ws = line.split()
        for i, w in enumerate(ws):
            if not DIGIT.search(w):
                continue
            add(w)
            for j in (i - 2, i - 1, i + 1):     # adjacent label words
                if 0 <= j < len(ws):
                    cand = strip_edges(ws[j])
                    if (len(cand) > 2 and cand.replace("-", "").isalpha()
                            and cand.lower() not in FACT_STOP):
                        add(cand)
    for w in gold_answer.split():
        cand = strip_edges(w)
        if DIGIT.search(cand) or (len(cand) > 3 and cand.replace("-", "").isalpha()
                                  and cand.lower() not in FACT_STOP):
            add(cand)
    return facts[:max_facts]


def fact_survival(facts, rendered):
    """Fraction of gold_fact_words present in the rendered text. Free, no LLM.

    Numeric facts are also tried against a whitespace-squeezed copy: filing tables
    routinely split an amount from its symbol ("$   1,155,199", "0.1 %"), so a fact
    lifted from the answer text as "$1,155,199" must still match the column it came
    from. Digits themselves are never squeezed together across a DROP, so this
    cannot credit a fact whose number was removed."""
    if not facts:
        return 1.0
    low = rendered.lower()
    squeezed = None
    hit = 0
    for f in facts:
        if f in low:
            hit += 1
        elif DIGIT.search(f):
            if squeezed is None:
                squeezed = WS.sub("", low)
            hit += int(WS.sub("", f) in squeezed)
    return hit / len(facts)


# --------------------------------------------------------------------------- #
# filing tokenization / chunk selection
# --------------------------------------------------------------------------- #
class TokenizedFiling:
    """A filing tokenized once: ids + char offsets + per-line char geometry.

    Same construction as v8_pretokenize.TokenizedFiling; reused across every QA on
    the filing because tokenizing an 80k-token 10-Q per question is the whole cost."""

    def __init__(self, text, tok):
        self.text = text
        self.lines = text.split("\n")
        starts, pos = [], 0
        for ln in self.lines:
            starts.append(pos)
            pos += len(ln) + 1
        self.line_start = starts

        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids = np.asarray(enc["input_ids"], dtype=np.int64)
        offs = enc["offset_mapping"]
        self._ts = [o[0] for o in offs]
        self._te = [o[1] for o in offs]

    def line_char_span(self, line_idx):
        k = line_idx
        if not (0 <= k < len(self.lines)):
            return None
        base = self.line_start[k]
        return base, base + len(self.lines[k])

    def token_range(self, cs, ce):
        """Half-open token range [lo, hi) overlapping char span [cs, ce)."""
        lo = bisect_right(self._te, cs)
        hi = bisect_left(self._ts, ce)
        return lo, max(lo, hi)

    def char_span_of_tokens(self, lo, hi):
        return self._ts[lo], self._te[hi - 1]


def center_window(tf, evidence_lines, budget):
    """Char span of a `budget`-token window centered on the evidence.

    Returns (cs, ce, n_evidence_lines_inside). When the evidence itself is wider
    than the budget the window is ANCHORED at its first line instead of centered,
    so the primary evidence is never the half that falls off the edge."""
    n_tok = len(tf.ids)
    if n_tok == 0:
        return None
    ev_spans = [s for s in (tf.line_char_span(k) for k in evidence_lines) if s]
    if not ev_spans:
        return None
    cs_ev = min(s for s, _ in ev_spans)
    ce_ev = max(e for _, e in ev_spans)
    lo, hi = tf.token_range(cs_ev, ce_ev)
    lo = min(lo, n_tok - 1)
    hi = max(min(hi, n_tok), lo + 1)

    if hi - lo >= budget:
        start = lo
    else:
        start = (lo + hi) // 2 - budget // 2
    start = max(0, min(start, max(0, n_tok - budget)))
    end = min(n_tok, start + budget)
    cs, ce = tf.char_span_of_tokens(start, end)
    inside = sum(1 for s, e in ev_spans if s >= cs and e <= ce)
    return cs, ce, inside


# --------------------------------------------------------------------------- #
# token -> word mapping
# --------------------------------------------------------------------------- #
def map_tokens_to_words(offsets, word_spans):
    """word index per token, -1 for tokens covering only whitespace/newline.

    mmBERT offsets can include the leading space of a '_'-prefixed token, so a token
    may straddle a word boundary; the word with the LARGEST overlap wins."""
    starts = [s for s, _ in word_spans]
    ends = [e for _, e in word_spans]
    out = []
    for s, e in offsets:
        if e <= s:
            out.append(-1)
            continue
        i = bisect_right(starts, s) - 1
        best, best_ov = -1, 0
        for cand in (i, i + 1):
            if 0 <= cand < len(starts):
                ov = min(e, ends[cand]) - max(s, starts[cand])
                if ov > best_ov:
                    best, best_ov = cand, ov
        out.append(best)
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def encode_question(question, tok, budget):
    """Question ids truncated at a word boundary to <= budget (v8 rule, verbatim)."""
    ids = tok(question, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return ids, False
    words = question.split()
    lo, hi, best = 1, len(words), None
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = tok(" ".join(words[:mid]), add_special_tokens=False)["input_ids"]
        if len(cand) <= budget:
            best, lo = cand, mid + 1
        else:
            hi = mid - 1
    return (best if best is not None else ids[:budget]), True


def build_item(tf, qa, tok, max_len, question_budget):
    """One QA -> the RL record, or None if it can't be placed."""
    q_ids, trunc = encode_question(qa["question"], tok, question_budget)
    if not q_ids:
        return None
    budget = max_len - len(q_ids) - N_SPECIAL
    if budget <= 0:
        return None

    win = center_window(tf, qa.get("evidence_abs_lines", []), budget)
    if win is None:
        return None
    cs, ce, ev_inside = win
    chunk_text = tf.text[cs:ce]
    if not chunk_text.strip():
        return None

    word_spans = segment_words(chunk_text)
    if not word_spans:
        return None

    enc = tok(chunk_text, add_special_tokens=False, return_offsets_mapping=True)
    c_ids, c_offs = enc["input_ids"], enc["offset_mapping"]
    if len(c_ids) > budget:                      # retokenization drift: cut on a word
        c_ids, c_offs = c_ids[:budget], c_offs[:budget]
        cutoff = c_offs[-1][1]
        word_spans = [(s, e) for s, e in word_spans if e <= cutoff]
        if not word_spans:
            return None

    word_id = map_tokens_to_words(c_offs, word_spans)
    n_words = len(word_spans)
    if not any(w >= 0 for w in word_id):
        return None

    words = [chunk_text[s:e] for s, e in word_spans]
    nl_after = newlines_after(chunk_text, word_spans)
    ev_lines = [tf.lines[k] for k in qa.get("evidence_abs_lines", [])
                if 0 <= k < len(tf.lines)]
    facts = gold_facts(ev_lines, qa.get("gold_answer", ""))
    # Keep only facts that the ALL-KEEP chunk contains. A fact the chunk cannot
    # supply (a multistep answer's computed total, evidence that fell outside the
    # window) would pin fact_survival below 1.0 for every candidate alike -- a
    # constant that cancels in the within-item reward softmax but flattens the part
    # of the reward that is supposed to discriminate. Filtering also makes
    # fact_survival == 1.0 an achievable target, which is what the val curve reads.
    full = render_mask(words, nl_after, [True] * len(words))
    facts = [f for f in facts if fact_survival([f], full) == 1.0]

    return {
        "qa_id": qa["qa_id"], "file": qa["file"],
        "question": qa["question"], "answer": qa.get("gold_answer", ""),
        "qtype": qa.get("qtype", ""),
        "qtype_id": QTYPES.index(qa["qtype"]) if qa.get("qtype") in QTYPES else QTYPE_OTHER,
        "chunk_text": chunk_text, "words": words, "nl_after": nl_after,
        "gold_fact_words": facts, "n_words": n_words,
        "evidence_lines_in_chunk": ev_inside,
        "n_evidence_lines": len(ev_lines),
        "question_truncated": trunc,
        "_q_ids": q_ids, "_c_ids": c_ids, "_word_id": word_id,
    }


def rebuild_item(rec, tok, max_len, question_budget, resegment=False):
    """Re-tokenize a STORED record with a different tokenizer, same chunk and words.

    This is how the ettin variant gets byte-identical items to the mmBERT build:
    chunk_text, words, nl_after and gold_fact_words are tokenizer-INDEPENDENT (they are
    char-level facts about the filing), so only the question ids, the chunk ids and the
    token->word map are rebuilt. Re-running the full selection under a different
    tokenizer would instead re-window every filing and select a different 2000 items,
    which makes the two RL runs incomparable.

    Ettin's BPE is not mmBERT's: the same chunk can exceed the token budget, so the
    tail is cut at a word boundary and any gold fact that leaves with it is dropped
    (keeping fact_survival == 1.0 reachable, as in the original build)."""
    q_ids, trunc = encode_question(rec["question"], tok, question_budget)
    if not q_ids:
        return None
    budget = max_len - len(q_ids) - N_SPECIAL
    if budget <= 0:
        return None

    chunk_text = rec["chunk_text"]
    spans_full = segment_words(chunk_text)
    if not spans_full:
        return None
    # chunk_text was stored WHOLE but `words` was truncated to the source tokenizer's
    # budget, so the stored list is a PREFIX of a fresh segmentation, not equal to it.
    # Cap the rebuild at that prefix: both builds then cover exactly the same words,
    # which is the whole point of --reuse-from. Segmentation itself is deterministic and
    # tokenizer-independent, so a prefix MISMATCH means the word rules changed since
    # that build and every downstream index would be quietly misaligned -- fail loudly.
    if resegment:
        # --reuse-resegment: same items (chunk_text identity), FRESH word rules.
        # For rebuilding a dataset after a deliberate segmentation fix (e.g. the
        # $-binding correction): item selection and text are preserved, word
        # boundaries are re-derived, so downstream labels/facts re-anchor to the
        # corrected words. Cover the same CHAR span the stored words covered.
        stored_end = None
        if rec["words"]:
            # find where the stored prefix ended in chunk_text via cumulative search
            pos = 0
            for w in rec["words"]:
                pos = chunk_text.find(w, pos)
                if pos < 0:
                    break
                pos += len(w)
            stored_end = pos if pos > 0 else None
        if stored_end:
            word_spans = [(s, e) for s, e in spans_full if e <= stored_end]
        else:
            word_spans = spans_full
        if not word_spans:
            return None
        words = [chunk_text[s:e] for s, e in word_spans]
    else:
        n_stored = len(rec["words"])
        if len(spans_full) < n_stored:
            raise ValueError(f"{rec['qa_id']}: re-segmentation yields {len(spans_full)} "
                             f"words, fewer than the {n_stored} stored")
        word_spans = spans_full[:n_stored]
        words = [chunk_text[s:e] for s, e in word_spans]
        if words != rec["words"]:
            bad = next((i for i, (a, b) in enumerate(zip(words, rec["words"])) if a != b), -1)
            raise ValueError(f"{rec['qa_id']}: re-segmentation disagrees with the stored "
                             f"words at index {bad} ({words[bad]!r} vs "
                             f"{rec['words'][bad]!r}); the prep's word rules changed since "
                             f"that build")

    # Tokenize only the span the stored words cover, so the new tokenizer does not spend
    # budget on text the source build had already dropped. Offsets stay comparable
    # because the slice shares chunk_text's origin.
    enc = tok(chunk_text[:word_spans[-1][1]], add_special_tokens=False,
              return_offsets_mapping=True)
    c_ids, c_offs = enc["input_ids"], enc["offset_mapping"]
    n_words_before = len(word_spans)
    if len(c_ids) > budget:
        c_ids, c_offs = c_ids[:budget], c_offs[:budget]
        cutoff = c_offs[-1][1]
        word_spans = [(s, e) for s, e in word_spans if e <= cutoff]
        if not word_spans:
            return None
    dropped = n_words_before - len(word_spans)

    word_id = map_tokens_to_words(c_offs, word_spans)
    if not any(w >= 0 for w in word_id):
        return None
    words = [chunk_text[s:e] for s, e in word_spans]
    nl_after = newlines_after(chunk_text, word_spans)
    facts = rec["gold_fact_words"]
    if dropped:                       # the tail took some facts with it
        full = render_mask(words, nl_after, [True] * len(words))
        facts = [f for f in facts if fact_survival([f], full) == 1.0]

    out = dict(rec)
    out.update(words=words, nl_after=nl_after, gold_fact_words=facts,
               n_words=len(word_spans), question_truncated=trunc,
               words_dropped_by_budget=dropped)
    out["_q_ids"], out["_c_ids"], out["_word_id"] = q_ids, c_ids, word_id
    return out


def pack(items, max_len, cls_id, sep_id, pad_id):
    """Records -> the dense tensors the trainer loads (v8 .pt conventions)."""
    n = len(items)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.int32)
    attn = torch.zeros((n, max_len), dtype=torch.uint8)
    word_id = torch.full((n, max_len), -1, dtype=torch.int16)
    n_words = torch.zeros((n,), dtype=torch.int32)
    source_id = torch.zeros((n,), dtype=torch.int8)
    for i, it in enumerate(items):
        q_ids, c_ids = it["_q_ids"], it["_c_ids"]
        n_pre = 1 + len(q_ids) + 1
        seq = [cls_id] + q_ids + [sep_id] + list(c_ids) + [sep_id]
        assert len(seq) <= max_len, (len(seq), max_len)
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
        attn[i, :len(seq)] = 1
        word_id[i, n_pre:n_pre + len(c_ids)] = torch.tensor(it["_word_id"],
                                                            dtype=torch.int16)
        n_words[i] = it["n_words"]
        source_id[i] = it["qtype_id"]
    return {"input_ids": input_ids, "attention_mask": attn, "word_id": word_id,
            "n_words": n_words, "source_id": source_id,
            "qa_id": [it["qa_id"] for it in items]}


def stratified_take(items, quota, rng):
    """Take up to `quota` items, balanced across qtypes then filled round-robin."""
    if quota <= 0 or not items:
        return []
    buckets = defaultdict(list)
    for it in items:
        buckets[it["qtype"]].append(it)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets)
    out, exhausted = [], False
    while len(out) < quota and not exhausted:
        exhausted = True
        for qt in order:
            if buckets[qt] and len(out) < quota:
                out.append(buckets[qt].pop())
                exhausted = False
    return out


def write_splits(splits, meta, out_dir, args, tok, cls_id, sep_id, pad_id):
    """Pack, save and report both splits. Shared by the build and --reuse-from paths."""
    for name in ("train", "val"):
        items = splits.get(name) or []
        if not items:
            print(f"  WARN {name} split is EMPTY")
            meta[f"n_{name}"] = 0
            continue
        d = pack(items, args.max_len, cls_id, sep_id, pad_id)
        torch.save(d, out_dir / f"{name}.pt")
        with open(out_dir / f"{name}_meta.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps({k: v for k, v in it.items()
                                    if not k.startswith("_")}) + "\n")
        n_facts = [len(it["gold_fact_words"]) for it in items]
        n_words = [it["n_words"] for it in items]
        content = int((d["word_id"] >= 0).sum())
        got_ev = sum(1 for it in items if it.get("evidence_lines_in_chunk", 0) > 0)
        dropped = [it.get("words_dropped_by_budget", 0) for it in items]
        qt = defaultdict(int)
        for it in items:
            qt[it["qtype"]] += 1
        meta[f"n_{name}"] = len(items)
        meta[f"{name}_content_tokens"] = content
        meta[f"{name}_mean_words"] = float(np.mean(n_words))
        meta[f"{name}_mean_gold_facts"] = float(np.mean(n_facts))
        meta[f"{name}_evidence_in_chunk_frac"] = got_ev / len(items)
        meta[f"{name}_zero_fact_items"] = sum(1 for c in n_facts if c == 0)
        meta[f"{name}_qtypes"] = dict(qt)
        meta[f"{name}_max_token_id"] = int(d["input_ids"].max())
        line = (f"  {name}.pt: {len(items)} items  {content} content tokens  "
                f"mean_words={np.mean(n_words):.0f}  "
                f"mean_gold_facts={np.mean(n_facts):.1f} "
                f"({sum(1 for c in n_facts if c == 0)} with none)  "
                f"evidence_in_chunk={got_ev/len(items):.3f}  {dict(qt)}")
        if any(dropped):
            n_tr = sum(1 for x in dropped if x)
            meta[f"{name}_items_word_truncated"] = n_tr
            meta[f"{name}_mean_words_dropped"] = float(np.mean(dropped))
            line += (f"\n    budget truncation: {n_tr}/{len(items)} items lost words "
                     f"(mean {np.mean(dropped):.1f}, max {max(dropped)})")
        print(line)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved -> {out_dir}")

    ev_frac = meta.get("train_evidence_in_chunk_frac", 0.0)
    if ev_frac < 0.95:
        print(f"  WARN only {ev_frac:.3f} of train chunks fully contain their evidence "
              f"lines -- answer_survival is capped at that, check the centering")
    else:
        print(f"  OK evidence contained in {ev_frac:.3f} of train chunks")


def maybe_upload(args, out_dir):
    if not args.upload:
        return
    target = args.volume_subdir or Path(out_dir).name
    cmd = ["modal", "volume", "put", "--force", VOLUME, str(out_dir), target]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"uploaded -> {VOLUME}:/{target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default="", help="default: <data-dir>/v9-rl")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--question-budget", type=int, default=96)
    ap.add_argument("--tokenizer", default="jhu-clsp/mmBERT-small")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--val-file-frac", type=float, default=0.12,
                    help="share of FILINGS reserved for val (no filing crosses)")
    ap.add_argument("--limit", type=int, default=0, help="cap QA processed (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-from", default="",
                    help="dir holding {train,val}_meta.jsonl from a previous build: "
                         "reuse the SAME items and chunk_text, re-tokenizing only. This "
                         "is how the ettin variant stays comparable to the mmBERT one.")
    ap.add_argument("--reuse-resegment", action="store_true",
                    help="with --reuse-from: keep the same items but re-derive word "
                         "boundaries with the CURRENT segment_words (for rebuilding "
                         "after a deliberate segmentation fix). Facts/labels re-anchor "
                         "to the corrected words.")
    ap.add_argument("--exclude-train-from", default="",
                    help="dir holding train_meta.jsonl from a previous build: exclude "
                         "those qa_ids from the TRAIN pool (fresh-data RL round — the "
                         "fixed reward set gets exploited across rounds). Same seed "
                         "keeps the val set identical for comparability.")
    ap.add_argument("--upload", action="store_true",
                    help=f"modal volume put -> {VOLUME}:/<out-dir name>")
    ap.add_argument("--volume-subdir", default="",
                    help=f"upload target under {VOLUME} (default: out-dir's name)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    exclude_train_ids = set()
    if args.exclude_train_from:
        with open(Path(args.exclude_train_from) / "train_meta.jsonl") as _f:
            exclude_train_ids = {json.loads(l)["qa_id"] for l in _f}
        print(f"excluding {len(exclude_train_ids)} previously-used train qa_ids")
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / VOLUME_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise ValueError(f"tokenizer missing CLS/SEP: {cls_id}/{sep_id}")
    print(f"tokenizer={args.tokenizer} cls/sep/pad={cls_id}/{sep_id}/{pad_id} "
          f"max_len={args.max_len} q_budget={args.question_budget}")

    # ---- reuse path: same items, different tokenizer ----
    if args.reuse_from:
        src = Path(args.reuse_from)
        splits, meta = {}, {
            "max_len": args.max_len, "question_budget": args.question_budget,
            "tokenizer": args.tokenizer, "seed": args.seed,
            "qtype_source_ids": QTYPES, "source_id_other": QTYPE_OTHER,
            "layout": "[CLS] question [SEP] chunk [SEP]",
            "word_id_semantics": "word index on chunk tokens, -1 = non-content",
            "reused_from": str(src),
        }
        for name in ("train", "val"):
            mp = src / f"{name}_meta.jsonl"
            if not mp.exists():
                raise FileNotFoundError(f"missing {mp} (--reuse-from must point at a "
                                        f"previous build's output dir)")
            recs = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
            if args.limit:
                recs = recs[:args.limit]
            out, failed = [], 0
            for rec in recs:
                it = rebuild_item(rec, tok, args.max_len, args.question_budget,
                                  resegment=args.reuse_resegment)
                if it is None:
                    failed += 1
                    continue
                out.append(it)
            splits[name] = out
            meta[f"n_{name}_source"] = len(recs)
            meta[f"n_{name}_unplaceable"] = failed
            print(f"  {name}: {len(recs)} source records -> {len(out)} rebuilt "
                  f"({failed} unplaceable)")
        if not splits["train"]:
            raise RuntimeError("no train items rebuilt")
        write_splits(splits, meta, out_dir, args, tok, cls_id, sep_id, pad_id)
        maybe_upload(args, out_dir)
        return

    qa_path = data_dir / "synthetic_qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(f"missing {qa_path}")
    qas = []
    with open(qa_path) as f:
        for line in f:
            line = line.strip()
            if line:
                qas.append(json.loads(line))
    qas = [q for q in qas if q.get("evidence_abs_lines") and q.get("gold_answer")]
    print(f"{len(qas)} validated QA with evidence + gold answer")

    by_file = defaultdict(list)
    for q in qas:
        by_file[q["file"]].append(q)

    # FILE-level val reservation, so a val chunk never comes from a filing the policy
    # trained on (the same guard v8_pretokenize applies).
    files = sorted(by_file)
    rng.shuffle(files)
    n_val_files = max(1, int(round(len(files) * args.val_file_frac)))
    val_files = set(files[:n_val_files])
    print(f"{len(files)} filings -> {len(val_files)} val / "
          f"{len(files) - len(val_files)} train")

    # Cap work up front: only tokenize filings we will actually sample from. Each
    # filing costs a full 80k-token tokenize pass, so oversample per file rather
    # than reading all 300.
    per_file_cap = max(4, (args.n_train + args.n_val) * 3 // max(1, len(files)))
    train_pool, val_pool = [], []
    n_seen = n_failed = n_trunc = 0
    for fi, fname in enumerate(sorted(by_file)):
        fpath = data_dir / "filings" / fname
        if not fpath.exists():
            print(f"  WARN missing filing {fname}")
            continue
        picks = list(by_file[fname])
        if exclude_train_ids and fname not in val_files:
            picks = [q for q in picks if q["qa_id"] not in exclude_train_ids]
        rng.shuffle(picks)
        picks = picks[:per_file_cap]
        tf = TokenizedFiling(fpath.read_text(errors="replace"), tok)
        dest = val_pool if fname in val_files else train_pool
        for qa in picks:
            n_seen += 1
            it = build_item(tf, qa, tok, args.max_len, args.question_budget)
            if it is None:
                n_failed += 1
                continue
            n_trunc += int(it["question_truncated"])
            dest.append(it)
            if args.limit and n_seen >= args.limit:
                break
        del tf
        if args.limit and n_seen >= args.limit:
            break
        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(by_file)} filings -> "
                  f"{len(train_pool)} train / {len(val_pool)} val candidates")

    print(f"built {len(train_pool) + len(val_pool)} candidates from {n_seen} QA "
          f"({n_failed} unplaceable, {n_trunc} questions truncated)")
    if not train_pool:
        raise RuntimeError("no train candidates produced")

    n_train = args.n_train if not args.limit else min(args.n_train, len(train_pool))
    n_val = args.n_val if not args.limit else min(args.n_val, len(val_pool))
    splits = {"train": stratified_take(train_pool, n_train, rng),
              "val": stratified_take(val_pool, n_val, rng)}

    meta = {"max_len": args.max_len, "question_budget": args.question_budget,
            "tokenizer": args.tokenizer, "seed": args.seed,
            "qtype_source_ids": QTYPES, "source_id_other": QTYPE_OTHER,
            "n_qa_available": len(qas), "n_candidates_built": n_seen - n_failed,
            "n_unplaceable": n_failed, "n_questions_truncated": n_trunc,
            "n_val_files": len(val_files), "val_file_frac": args.val_file_frac,
            "layout": "[CLS] question [SEP] chunk [SEP]",
            "word_id_semantics": "word index on chunk tokens, -1 = non-content"}

    write_splits(splits, meta, out_dir, args, tok, cls_id, sep_id, pad_id)
    maybe_upload(args, out_dir)


if __name__ == "__main__":
    main()
