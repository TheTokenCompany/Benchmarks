#!/usr/bin/env python3
"""Assert a v8 pretokenized build is well-formed AND correctly aligned.

Shape/dtype checks are cheap; the one that actually matters is ALIGNMENT: decode
the tokens whose target>=0.5 back to text and require the evidence line's own
words to show up there. That catches an off-by-one in the line->word->char->token
chain, which no shape assertion would notice.

Run: .venv/bin/python v8_verify_pretok.py --pretok-dir <dir> --data-dir <dir>
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

EXPECT_DTYPE = {"input_ids": torch.int32, "attention_mask": torch.uint8,
                "targets": torch.float16, "loss_mask": torch.uint8,
                "source_id": torch.int8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretok-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    pd_, dd = Path(args.pretok_dir), Path(args.data_dir)

    meta = json.loads((pd_ / "meta.json").read_text())
    L = meta["max_len"]
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    masks = {json.loads(l)["qa_id"]: json.loads(l)
             for l in open(dd / "masks.jsonl") if l.strip()}

    splits, all_files = {}, {}
    for name in ("train", "val"):
        d = torch.load(pd_ / f"{name}.pt", map_location="cpu", weights_only=False)
        splits[name] = d
        n = d["input_ids"].shape[0]
        assert n == meta[f"n_{name}"], f"{name}: n mismatch {n} vs meta {meta[f'n_{name}']}"

        for k, dt in EXPECT_DTYPE.items():
            assert k in d, f"{name}: missing key {k}"
            assert d[k].dtype == dt, f"{name}.{k}: dtype {d[k].dtype} != {dt}"
        for k in ("input_ids", "attention_mask", "targets", "loss_mask"):
            assert tuple(d[k].shape) == (n, L), f"{name}.{k}: shape {tuple(d[k].shape)} != {(n, L)}"
        assert tuple(d["source_id"].shape) == (n,), f"{name}.source_id shape"
        assert d["source_id"].min() >= 0 and d["source_id"].max() <= 3, "source_id range"

        ids, attn, tgt, lm = d["input_ids"], d["attention_mask"].bool(), \
            d["targets"].float(), d["loss_mask"].bool()

        assert ((tgt >= 0.0) & (tgt <= 1.0)).all(), f"{name}: targets outside [0,1]"
        assert (tgt[~lm] == 0).all(), f"{name}: nonzero target outside loss_mask"
        assert (lm & ~attn).sum() == 0, f"{name}: loss_mask outside attention_mask"
        assert (ids[:, 0] == cls_id).all(), f"{name}: row does not start with CLS"
        # pad region: attention 0 and the pad id
        assert (ids[~attn] == pad_id).all(), f"{name}: non-pad id in unattended region"

        for i in range(n):
            row_ids, row_lm, row_attn = ids[i].tolist(), lm[i], attn[i]
            nz = row_lm.nonzero().flatten()
            assert nz.numel() > 0, f"{name}[{i}]: empty loss_mask"
            lo, hi = int(nz[0]), int(nz[-1])
            assert hi - lo + 1 == nz.numel(), f"{name}[{i}]: loss_mask not contiguous"
            # layout: CLS q SEP <content> SEP
            assert row_ids[lo - 1] == sep_id, f"{name}[{i}]: no SEP before content"
            assert row_ids[hi + 1] == sep_id, f"{name}[{i}]: no SEP after content"
            n_q = lo - 2
            assert 1 <= n_q <= meta["question_budget"], \
                f"{name}[{i}]: question len {n_q} outside 1..{meta['question_budget']}"
            assert int(row_attn.sum()) == hi + 2, f"{name}[{i}]: attention != seq len"
            assert hi + 2 <= L, f"{name}[{i}]: sequence exceeds max_len"

        all_files[name] = set(d["file"])
        print(f"{name}.pt OK: {n} windows, {int(lm.sum())} content tokens, "
              f"pos_prevalence={float(((tgt>=0.5)&lm).sum())/max(int(lm.sum()),1):.4f}")

    leak = all_files["train"] & all_files["val"]
    assert not leak, f"FILE LEAK across split: {leak}"
    print(f"no file leakage: {len(all_files['train'])} train / {len(all_files['val'])} val files")

    # ---- ALIGNMENT ----
    # Windowing-independent, both directions:
    #  PRECISION: every positive token's text must be part of some word the mask kept
    #    (a positive outside the kept-word set = the line/word->char->token chain slipped).
    #    Tolerance exists because a token can straddle the space between a dropped and a
    #    kept word, so its decoded text spans both -- a real tokenizer artifact, not a bug.
    #  RECALL: across all of a QA's windows, the positives must cover most of the DISTINCT
    #    words the mask kept (catches silently-dropped or truncated spans).
    file_lines = {}
    kept_blob, kept_words = {}, {}
    for qa_id, m in masks.items():
        if m["file"] not in file_lines:
            file_lines[m["file"]] = (dd / "filings" / m["file"]).read_text(
                errors="replace").split("\n")
        lines = file_lines[m["file"]]
        words = []
        for k in m["keep_lines"]:
            if 0 <= k < len(lines):
                words += lines[k].split()
        for k_str, wranges in m.get("keep_spans", {}).items():
            k = int(k_str)
            if 0 <= k < len(lines):
                lw = lines[k].split()
                for w0, w1 in wranges:
                    words += lw[max(0, int(w0)):min(len(lw), int(w1))]
        kept_words[qa_id] = {w for w in words if len(w) > 3}
        kept_blob[qa_id] = " ".join(words)

    checked = n_tok = n_viol = 0
    covered = {q: set() for q in masks}
    for name, d in splits.items():
        ids, tgt, lm = d["input_ids"], d["targets"].float(), d["loss_mask"].bool()
        for i in range(ids.shape[0]):
            if d["is_negative"][i]:
                continue
            qa_id = d["qa_id"][i]
            blob = kept_blob[qa_id]
            pos_ids = ids[i][(tgt[i] >= 0.5) & lm[i]].tolist()
            viol = []
            for t in pos_ids:
                s = tok.decode([t]).strip()
                if not s:
                    continue
                n_tok += 1
                if s not in blob:
                    n_viol += 1
                    viol.append(s)
            frac = len(viol) / max(len(pos_ids), 1)
            assert frac <= 0.10, (
                f"{name}[{i}] {qa_id}: {frac:.1%} of positive tokens are NOT in the "
                f"mask's kept words -> misalignment. offenders={viol[:12]}")
            txt = tok.decode(pos_ids)
            covered[qa_id] |= {w for w in kept_words[qa_id] if w in txt}
            checked += 1

    print(f"alignment precision OK: {checked} positive windows, {n_tok} positive tokens, "
          f"{n_viol} ({n_viol/max(n_tok,1):.2%}) straddling a boundary (tol 10%/window)")

    recalls = [len(covered[q]) / len(kept_words[q]) for q in masks if kept_words[q]]
    mean_recall = sum(recalls) / max(len(recalls), 1)
    assert mean_recall >= 0.50, (
        f"alignment RECALL too low: positives cover only {mean_recall:.1%} of the "
        f"distinct kept words -> spans are being lost")
    print(f"alignment recall OK: positives cover {mean_recall:.1%} of distinct kept "
          f"words (min {min(recalls):.1%} over {len(recalls)} QA)")
    print(f"\nALL CHECKS PASSED  (max_len={L}, pos_prevalence={meta['pos_prevalence']:.4f})")


if __name__ == "__main__":
    main()
