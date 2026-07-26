#!/usr/bin/env python3
"""Build a tiny synthetic v8 fixture (5 QA on 2 fake filings) for the smoke test.

Emits <out>/filings/*.txt, synthetic_qa.jsonl, masks.jsonl in the real v8 formats.
Masks come from v8_build_masks.build_mask, so the fixture exercises the same label
policy (and the same keep_lines / keep_spans shapes) as the real pipeline.

Run: .venv/bin/python v8_make_fixture.py --out <dir>
"""

import argparse
import json
import random
from pathlib import Path

from v8_build_masks import build_mask


def make_filing(ticker, seed, n_blocks=26):
    """A fake 10-K-ish text: prose paragraphs + titled statement tables."""
    rng = random.Random(seed)
    lines = [f"{ticker} INCORPORATED", "FORM 10-K", "", "PART I", ""]
    for b in range(n_blocks):
        lines += [f"Item {b+1}. Management Discussion And Analysis", ""]
        for _ in range(6):
            lines.append(" ".join(
                rng.choice(["the", "company", "recorded", "revenue", "growth",
                            "across", "segments", "driven", "by", "demand",
                            "pricing", "volume", "and", "currency", "effects",
                            "during", "the", "reported", "period", "management",
                            "believes", "operating", "results", "reflect"])
                for _ in range(rng.randint(14, 26))))
        lines += ["", "CONSOLIDATED STATEMENTS OF OPERATIONS", "",
                  "September 2025 September 2024", ""]
        for row in ("Net revenues", "Cost of products sold", "Research and development",
                    "Operating earnings", "Net earnings"):
            lines.append(f"{row} $ {rng.randint(1000, 99999):,} $ {rng.randint(1000, 99999):,}")
        lines += ["", "The accompanying notes are an integral part of these statements.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "filings").mkdir(parents=True, exist_ok=True)

    specs = [("FAKEA", "FAKEA_10K_2026-01-15.txt", 11),
             ("FAKEB", "FAKEB_10Q_2026-02-20.txt", 22)]
    texts = {}
    for ticker, fname, seed in specs:
        texts[fname] = make_filing(ticker, seed)
        (out / "filings" / fname).write_text(texts[fname])

    # 5 QA: spread across both files and all three qtypes, evidence on real table rows
    qa = []
    for fname, text in texts.items():
        lines = text.split("\n")
        rows = [i for i, l in enumerate(lines) if l.startswith("Net revenues")]
        prose = [i for i, l in enumerate(lines) if l.startswith("the ") or l.startswith("management ")]
        qa += [
            {"qa_id": f"{fname[:5]}_q000", "file": fname,
             "question": f"What were {fname[:5]}'s net revenues for the three months "
                         f"ended September 30, 2025 as reported in the consolidated "
                         f"statements of operations?",
             "gold_answer": "$1,234 million", "qtype": "metrics-extraction",
             "evidence_abs_lines": [rows[1]]},
            {"qa_id": f"{fname[:5]}_q001", "file": fname,
             "question": "What was the combined total of net revenues across the two "
                         "reported periods?",
             "gold_answer": "$2,468 million", "qtype": "multistep-numerical",
             "evidence_abs_lines": [rows[2], rows[2] + 3]},
            {"qa_id": f"{fname[:5]}_q002", "file": fname,
             "question": "According to management, what factors drove the change in "
                         "operating results during the reported period, and how does "
                         "management characterize the outlook given segment demand, "
                         "pricing actions, volume trends and currency effects?",
             "gold_answer": "demand, pricing and volume", "qtype": "domain-qualitative",
             "evidence_abs_lines": [prose[3]]},
        ]
    qa = qa[:5]

    with open(out / "synthetic_qa.jsonl", "w") as f:
        for q in qa:
            f.write(json.dumps({**q, "ticker": q["file"][:5], "validated": True}) + "\n")

    n_span_lines = 0
    with open(out / "masks.jsonl", "w") as f:
        for q in qa:
            lw = [l.split() for l in texts[q["file"]].split("\n")]
            keep_lines, keep_spans = build_mask(lw, q["evidence_abs_lines"], q["qtype"])
            total_w = sum(len(w) for w in lw) or 1
            kept_w = (sum(len(lw[k]) for k in keep_lines)
                      + sum(b - a for sp in keep_spans.values() for a, b in sp))
            n_span_lines += len(keep_spans)
            f.write(json.dumps({
                "qa_id": q["qa_id"], "file": q["file"], "question": q["question"],
                "qtype": q["qtype"], "keep_lines": sorted(keep_lines),
                "keep_spans": {str(k): sorted(set(map(tuple, v))) for k, v in keep_spans.items()},
                "n_lines": len(lw), "mask_token_frac": round(kept_w / total_w, 4),
            }) + "\n")

    print(f"fixture -> {out}")
    print(f"  {len(texts)} filings ({', '.join(f'{k}:{len(v.splitlines())}L' for k, v in texts.items())})")
    print(f"  {len(qa)} QA, {n_span_lines} lines carrying partial word-spans")


if __name__ == "__main__":
    main()
