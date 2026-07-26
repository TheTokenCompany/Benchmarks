#!/usr/bin/env python3
"""Build sufficiency keep-masks for v8 training from synthetic QA + filings.

Encodes the 24-25.07 study findings as label policy:
  - evidence lines kept FULLY (answer rows with their cell values)
  - the nearest statement-title line above each evidence line kept (ALL-CAPS or
    Title-Case heading; "CONSOLIDATED STATEMENTS OF..." are labels, not furniture)
  - nearest period-header-ish line above (>=2 period tokens) kept
  - row-label spans and period tokens inside a +/-HALO-line halo kept
  - numbers inside the halo kept (soft: only in halo, not globally — the model
    must LEARN scoped number-keeping, unlike blanket safenum)
  - a graded context floor for qualitative/multistep questions (paragraph around
    evidence), giving per-question retention variance:
      metrics-extraction   target ~3-10%
      multistep-numerical  target ~8-20%
      domain-qualitative   target ~15-35%

Output: v8data/masks.jsonl — {qa_id, file, question, qtype, keep_lines: [int],
keep_spans: {line: [[w0,w1], ...]} (word-index spans for partial lines),
n_lines, mask_token_frac}
Full-line keeps are expressed in keep_lines; partial keeps in keep_spans.

Usage: .venv/bin/python v8_build_masks.py
"""

import json
import os
import re
from collections import defaultdict

BASE = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-"
        "b409-cb03154f2355-financebench-check/838fe3de-3a8f-4846-ba75-9e663316210c/"
        "scratchpad/v8data")
HALO = 6            # lines around evidence for scoped number/label keeping
QUAL_PARA = 14      # extra full-context lines around evidence for qualitative

num_re = re.compile(r"\d")
period_re = re.compile(
    r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
    r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)


def is_numeric_word(w):
    return bool(num_re.search(w)) or w in ("$", "%", "€", "£")


def is_title_line(words):
    if not (1 <= len(words) <= 12):
        return False
    if any(is_numeric_word(w) for w in words):
        return False
    alpha = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha:
        return False
    caps = sum(1 for w in alpha if w.isupper() or w.istitle())
    return caps / len(alpha) >= 0.7


def is_period_header(words):
    n_per = sum(1 for w in words if period_re.match(w))
    return n_per >= 2 and n_per / max(1, len(words)) >= 0.4


def is_table_line(words):
    return sum(1 for w in words if is_numeric_word(w)) >= 3


def row_label_span(words):
    """word indices [0, first_numeric) capped at 12 — the row label."""
    for i, w in enumerate(words):
        if is_numeric_word(w):
            return (0, min(i, 12))
    return None


def build_mask(lines_words, evidence, qtype):
    """-> (keep_lines set, keep_spans dict line->list[(w0,w1)])"""
    n = len(lines_words)
    keep_lines, keep_spans = set(), defaultdict(list)

    for ev in evidence:
        if not (0 <= ev < n):
            continue
        keep_lines.add(ev)
        # nearest statement title + period header above
        found_title = found_period = False
        for k in range(ev - 1, max(-1, ev - 40), -1):
            w = lines_words[k]
            if not w:
                continue
            if not found_title and is_title_line(w):
                keep_lines.add(k)
                found_title = True
            if not found_period and is_period_header(w):
                keep_lines.add(k)
                found_period = True
            if found_title and found_period:
                break
        # halo: scoped numbers + row labels + periods
        for k in range(max(0, ev - HALO), min(n, ev + HALO + 1)):
            w = lines_words[k]
            if not w or k in keep_lines:
                continue
            spans = []
            if is_table_line(w):
                rl = row_label_span(w)
                if rl and rl[1] > rl[0]:
                    spans.append(rl)
            for wi, word in enumerate(w):
                if is_numeric_word(word) or period_re.match(word):
                    spans.append((wi, wi + 1))
            if spans:
                keep_spans[k].extend(spans)
        # qualitative/multistep: paragraph context as full lines
        if qtype != "metrics-extraction":
            pad = QUAL_PARA if qtype == "domain-qualitative" else QUAL_PARA // 2
            for k in range(max(0, ev - pad), min(n, ev + pad + 1)):
                if lines_words[k]:
                    keep_lines.add(k)

    for k in list(keep_spans):
        if k in keep_lines:
            del keep_spans[k]
    return keep_lines, keep_spans


def main():
    qa_path = os.path.join(BASE, "synthetic_qa.jsonl")
    out_path = os.path.join(BASE, "masks.jsonl")
    file_cache = {}
    n_out = 0
    fracs = []
    with open(qa_path) as f, open(out_path, "w") as out:
        for line in f:
            qa = json.loads(line)
            fp = qa["file"]
            if fp not in file_cache:
                text = open(os.path.join(BASE, "filings", fp)).read()
                file_cache[fp] = [l.split() for l in text.split("\n")]
                if len(file_cache) > 40:
                    file_cache.pop(next(iter(file_cache)))
            lw = file_cache[fp]
            keep_lines, keep_spans = build_mask(lw, qa["evidence_abs_lines"], qa["qtype"])
            total_w = sum(len(w) for w in lw) or 1
            kept_w = (sum(len(lw[k]) for k in keep_lines)
                      + sum(b - a for sp in keep_spans.values() for a, b in sp))
            frac = kept_w / total_w
            fracs.append(frac)
            out.write(json.dumps({
                "qa_id": qa["qa_id"], "file": fp, "question": qa["question"],
                "qtype": qa["qtype"], "keep_lines": sorted(keep_lines),
                "keep_spans": {str(k): sorted(set(map(tuple, v))) for k, v in keep_spans.items()},
                "n_lines": len(lw), "mask_token_frac": round(frac, 4),
            }) + "\n")
            n_out += 1
    fracs.sort()
    print(f"wrote {n_out} masks to {out_path}")
    if fracs:
        n = len(fracs)
        print("mask_token_frac p10/p50/p90:",
              f"{fracs[n//10]:.3f}/{fracs[n//2]:.3f}/{fracs[9*n//10]:.3f}")


if __name__ == "__main__":
    main()
