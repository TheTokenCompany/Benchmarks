#!/usr/bin/env python3
"""v8 pretokenize: line/word keep-masks -> per-token targets (runs LOCALLY, no Modal).

Consumes the v8 datagen outputs (filings/*.txt + synthetic_qa.jsonl + masks.jsonl)
and emits train.pt / val.pt of fixed-length cross-encoder sequences in the same
format taaha's focus trainer eats:

    [CLS] question_tokens [SEP] filing_window_tokens [SEP] [PAD...]

- Reserve `question_budget` (96) tokens for the question; content = max_len - q - 3.
  Our questions are LONGER than taaha's 64-token focus budget, so over-budget
  questions are TRUNCATED at a word boundary rather than dropped (taaha's rule).
- masks.jsonl is LINE/WORD indexed (keep_lines + keep_spans word ranges, where
  words == line.split()). Those are resolved to absolute char spans in the filing
  text, then to token indices via the tokenizer's offset_mapping. Target 1.0 for
  tokens overlapping a kept word, else 0.0 (binary — the mask builder already
  encoded its graded policy as which words it keeps).
- Filings are 70-90k tokens, so each file is tokenized ONCE and reused across all
  of its QAs (targets are the only per-QA part). QAs are processed grouped by file.
- Each QA TILES its whole filing into consecutive content_budget windows, all
  carrying the same question. Windows with zero kept tokens are NEGATIVES; only
  `empty_window_keep` of them are kept, seeded.
  On empty_window_keep: taaha's value was 0.15, but his rows were pre-chunked, while
  each of our QAs tiles a FULL ~80k-token filing -> ~40 windows of which 1-2 hold the
  evidence. At 0.15 negatives are 80% of the dataset and dilute pos_prevalence to
  0.033, under the learnable band. Measured on the 4141-QA / 300-filing corpus:
  0.15 -> 0.033 (out of band), 0.08 -> 0.052, 0.05 -> 0.069, 0.03 -> 0.088.
  Default 0.05 sits in the band with margin and still leaves ~56% of windows all-drop,
  which is the signal that teaches "this window holds nothing, drop all of it" -- the
  behaviour real full-doc retention depends on, so the negatives are not just filler.
- loss_mask is 1 ONLY on filing-content tokens (question + specials + pad = 0), so
  the model is conditioned on the question but never trained to label it.
- Train/val split at the FILE level, so no filing leaks across the split.

pos_prevalence (share of content tokens with target>=0.5) is asserted into the
drop-majority band 0.05-0.30 — taaha's learnability finding: a KEEP-majority label
set makes the all-keep basin a strong attractor and collapses these runs.

Run:
    .venv/bin/python v8_pretokenize.py --max-len 2048
    .venv/bin/python v8_pretokenize.py --max-len 512
    # r8 (ettin-400m) needs its own vocab -- different tokenizer, so its own build:
    .venv/bin/python v8_pretokenize.py --max-len 2048 \
        --tokenizer jhu-clsp/ettin-encoder-400m --out-dir <dir>/pretok-ettin-ctx2048
"""

import argparse
import json
import random
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

N_SPECIAL = 3            # [CLS] + [SEP] (after question) + [SEP] (after content)
# source_id: qtype, so the trainer can log per-qtype keep-rates (int8 in the .pt)
QTYPES = ["metrics-extraction", "multistep-numerical", "domain-qualitative"]
QTYPE_OTHER = 3


# --------------------------------------------------------------------------- #
# char-offset helpers
# --------------------------------------------------------------------------- #
def word_char_spans(line):
    """[(start, end)] per word, char offsets WITHIN the line, matching the index
    order of line.split() (whitespace-run splitting, no empty words)."""
    out, i, n = [], 0, len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not line[j].isspace():
            j += 1
        out.append((i, j))
        i = j
    return out


class TokenizedFiling:
    """A filing tokenized ONCE: token ids + offsets + per-line char geometry.

    Reused across every QA on that filing (only the targets are per-QA)."""

    def __init__(self, text, tok):
        self.lines = text.split("\n")
        # absolute char offset where each line starts ("\n" is 1 char)
        starts, pos = [], 0
        for ln in self.lines:
            starts.append(pos)
            pos += len(ln) + 1
        self.line_start = starts
        self._wspans = {}

        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids = np.asarray(enc["input_ids"], dtype=np.int64)
        offs = enc["offset_mapping"]
        self.tok_start = np.asarray([o[0] for o in offs], dtype=np.int64)
        self.tok_end = np.asarray([o[1] for o in offs], dtype=np.int64)
        # python lists for bisect (much faster than np searchsorted per-span here)
        self._ts = self.tok_start.tolist()
        self._te = self.tok_end.tolist()

    def words(self, k):
        """Cached word char-spans for line k."""
        if k not in self._wspans:
            self._wspans[k] = word_char_spans(self.lines[k])
        return self._wspans[k]

    def token_range(self, cs, ce):
        """Half-open token index range [lo, hi) overlapping char span [cs, ce).

        A token overlaps iff tok_end > cs and tok_start < ce. Offsets are sorted
        and non-decreasing, so both edges are a bisect."""
        lo = bisect_right(self._te, cs)      # first token whose end > cs
        hi = bisect_left(self._ts, ce)       # first token whose start >= ce
        return lo, max(lo, hi)


def qa_char_spans(tf, keep_lines, keep_spans):
    """mask (line indices + word-index spans) -> absolute char spans in the filing."""
    spans = []
    n_lines = len(tf.lines)
    for k in keep_lines:
        if not (0 <= k < n_lines) or not tf.lines[k].strip():
            continue
        base = tf.line_start[k]
        ws = tf.words(k)
        if ws:      # trim to real text so the span never covers trailing whitespace
            spans.append((base + ws[0][0], base + ws[-1][1]))
    for k_str, wranges in keep_spans.items():
        k = int(k_str)
        if not (0 <= k < n_lines):
            continue
        base, ws = tf.line_start[k], tf.words(k)
        for w0, w1 in wranges:
            w0, w1 = max(0, int(w0)), min(len(ws), int(w1))
            if w1 > w0:
                spans.append((base + ws[w0][0], base + ws[w1 - 1][1]))
    return spans


# --------------------------------------------------------------------------- #
# question budget
# --------------------------------------------------------------------------- #
def encode_question(question, tok, budget):
    """Token ids for the question, truncated at a WORD boundary to <= budget.

    Taaha DROPPED over-budget focuses (his were short phrases); our questions are
    full sentences that routinely exceed the budget, so dropping them would throw
    away most of the dataset. Truncation keeps the question prefix, which carries
    the metric/entity/period the mask was built around."""
    ids = tok(question, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return ids, False
    words = question.split()
    lo, hi, best = 1, len(words), None
    while lo <= hi:                            # longest word prefix that fits
        mid = (lo + hi) // 2
        cand = tok(" ".join(words[:mid]), add_special_tokens=False)["input_ids"]
        if len(cand) <= budget:
            best, lo = cand, mid + 1
        else:
            hi = mid - 1
    if best is None:                           # single word over budget -> hard cut
        best = ids[:budget]
    return best, True


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def qa_windows(tf, q_ids, char_spans, max_len, empty_window_keep, rng):
    """TILE the filing into content_budget windows; yield compact window dicts."""
    content_budget = max_len - len(q_ids) - N_SPECIAL
    if content_budget <= 0 or tf.ids.size == 0:
        return []

    tgt = np.zeros(tf.ids.size, dtype=np.float16)
    for cs, ce in char_spans:
        lo, hi = tf.token_range(cs, ce)
        if hi > lo:
            tgt[lo:hi] = 1.0

    out = []
    for w0 in range(0, tf.ids.size, content_budget):
        w_ids = tf.ids[w0:w0 + content_budget]
        w_tgt = tgt[w0:w0 + content_budget]
        has_pos = bool((w_tgt >= 0.5).any())
        if not has_pos and rng.random() > empty_window_keep:
            continue                            # drop most all-drop windows
        out.append({"ids": w_ids, "tgt": w_tgt, "has_pos": has_pos})
    return out


def pack(windows, q_ids_by_idx, max_len, pad_id):
    """Compact windows -> the dense tensor dict the trainer loads."""
    n = len(windows)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.int32)
    targets = torch.zeros((n, max_len), dtype=torch.float16)
    attn = torch.zeros((n, max_len), dtype=torch.uint8)
    loss_mask = torch.zeros((n, max_len), dtype=torch.uint8)
    source_id = torch.zeros((n,), dtype=torch.int8)
    for i, w in enumerate(windows):
        q_ids = q_ids_by_idx[w["qa_id"]]
        n_pre = 1 + len(q_ids) + 1                      # CLS + question + SEP
        nc = len(w["ids"])
        seq = [CLS_ID] + q_ids + [SEP_ID] + w["ids"].tolist() + [SEP_ID]
        assert len(seq) <= max_len, (len(seq), max_len)
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
        targets[i, n_pre:n_pre + nc] = torch.from_numpy(w["tgt"].astype(np.float16))
        loss_mask[i, n_pre:n_pre + nc] = 1              # loss ONLY on filing content
        attn[i, :len(seq)] = 1
        source_id[i] = w["src"]
    return {"input_ids": input_ids, "attention_mask": attn, "targets": targets,
            "loss_mask": loss_mask, "source_id": source_id,
            "qa_id": [w["qa_id"] for w in windows],
            "file": [w["file"] for w in windows],
            "is_negative": [not w["has_pos"] for w in windows]}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA,
                    help="dir holding filings/, synthetic_qa.jsonl, masks.jsonl")
    ap.add_argument("--out-dir", default="",
                    help="default: <data-dir>/pretok-ctx<max_len>")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--masks-file", default="masks.jsonl")
    ap.add_argument("--question-budget", type=int, default=96)
    ap.add_argument("--tokenizer", default="jhu-clsp/mmBERT-small")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--empty-window-keep", type=float, default=0.05,
                    help="fraction of all-drop windows kept as negatives (see module "
                         "docstring: 0.15 dilutes prevalence out of the learnable band)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / f"pretok-ctx{args.max_len}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    global CLS_ID, SEP_ID
    CLS_ID, SEP_ID = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if CLS_ID is None or SEP_ID is None:
        raise ValueError(f"tokenizer missing CLS/SEP: {CLS_ID}/{SEP_ID}")
    print(f"tokenizer={args.tokenizer} cls/sep/pad={CLS_ID}/{SEP_ID}/{pad_id} "
          f"max_len={args.max_len} question_budget={args.question_budget}")

    # ---- load masks + QA meta, group by file ----
    masks_path = data_dir / getattr(args, "masks_file", "masks.jsonl")
    if not masks_path.exists():
        raise FileNotFoundError(f"missing {masks_path} (run v8_build_masks.py first)")
    by_file = defaultdict(list)
    n_masks = 0
    with open(masks_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            by_file[m["file"]].append(m)
            n_masks += 1
    print(f"{n_masks} masks over {len(by_file)} filings")

    # ---- FILE-level train/val split (no filing crosses the split) ----
    files = sorted(by_file)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * args.val_frac))) if len(files) > 1 else 0
    val_files = set(files[:n_val])
    print(f"split: {len(files) - len(val_files)} train / {len(val_files)} val filings")

    splits = {"train": [], "val": []}
    q_ids_by_idx = {}
    n_trunc = n_neg = n_skipped = 0
    for fi, fname in enumerate(sorted(by_file)):
        fpath = data_dir / "filings" / fname
        if not fpath.exists():
            n_skipped += len(by_file[fname])
            print(f"  WARN missing filing {fname}: skipped {len(by_file[fname])} QA")
            continue
        tf = TokenizedFiling(fpath.read_text(errors="replace"), tok)
        which = "val" if fname in val_files else "train"
        for m in by_file[fname]:
            q_ids, trunc = encode_question(m["question"], tok, args.question_budget)
            if not q_ids:
                n_skipped += 1
                continue
            n_trunc += int(trunc)
            q_ids_by_idx[m["qa_id"]] = q_ids
            spans = qa_char_spans(tf, m.get("keep_lines", []), m.get("keep_spans", {}))
            src = QTYPES.index(m["qtype"]) if m.get("qtype") in QTYPES else QTYPE_OTHER
            for w in qa_windows(tf, q_ids, spans, args.max_len,
                                args.empty_window_keep, rng):
                w.update(qa_id=m["qa_id"], file=fname, src=src)
                splits[which].append(w)
                n_neg += 0 if w["has_pos"] else 1
        del tf
        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(by_file)} filings -> "
                  f"{len(splits['train'])+len(splits['val'])} windows")

    n_tot = len(splits["train"]) + len(splits["val"])
    print(f"{n_masks} masks ({n_skipped} skipped, {n_trunc} questions truncated to "
          f"{args.question_budget} tok) -> {n_tot} windows ({n_neg} negative)")
    if n_tot == 0:
        raise RuntimeError("no windows produced")

    # ---- pack + save ----
    meta = {"max_len": args.max_len, "question_budget": args.question_budget,
            "tokenizer": args.tokenizer, "n_masks": n_masks,
            "n_skipped": n_skipped, "n_questions_truncated": n_trunc,
            "n_windows": n_tot, "n_negative_windows": n_neg,
            "empty_window_keep": args.empty_window_keep, "seed": args.seed,
            "val_frac": args.val_frac, "n_train_files": len(files) - len(val_files),
            "n_val_files": len(val_files), "qtype_source_ids": QTYPES,
            "source_id_other": QTYPE_OTHER}
    kept_tot = content_tot = 0
    for name in ("train", "val"):
        d = pack(splits[name], q_ids_by_idx, args.max_len, pad_id)
        torch.save(d, out_dir / f"{name}.pt")
        lm = d["loss_mask"].bool()
        kept = int(((d["targets"] >= 0.5) & lm).sum())
        content = int(lm.sum())
        kept_tot += kept
        content_tot += content
        meta[f"n_{name}"] = len(splits[name])
        meta[f"{name}_content_tokens"] = content
        meta[f"{name}_pos_prevalence"] = kept / max(content, 1)
        print(f"  {name}.pt: {len(splits[name])} windows  {content} content tokens  "
              f"pos_prevalence={kept/max(content,1):.4f}")

    prevalence = kept_tot / max(content_tot, 1)
    meta["token_keep_rate_at_0.5"] = prevalence
    meta["pos_prevalence"] = prevalence
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved -> {out_dir}")
    print(f"pos_prevalence = {prevalence:.4f}  (target band 0.05-0.30, drop-majority)")
    if prevalence < 0.05:
        print("  WARN prevalence BELOW 0.05: labels too sparse, expect all-drop collapse "
              "-- widen the mask halo or raise the qualitative context floor")
    elif prevalence > 0.30:
        print("  WARN prevalence ABOVE 0.30: drifting toward KEEP-majority, where the "
              "all-keep basin becomes the attractor (taaha's v7.0 collapse) -- tighten "
              "the mask policy")
    else:
        print("  OK prevalence in the drop-majority learnable band")
    print(json.dumps({k: v for k, v in meta.items() if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
