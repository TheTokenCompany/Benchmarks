#!/usr/bin/env python3
"""E31: labeled-render post-process on an existing fd- compression cache.

E30 showed the reader computes ratios only when figures carry their statement +
period + units labels. This applies that as a deterministic, model-free render
rule: group kept table lines under compact banner lines

    ### <statement title> | <units> | <period header>

resolved from the ORIGINAL document (nearest title/units/period line above the
kept line's source position). No content is added beyond these banners, which
themselves are verbatim doc lines. Token counts scaled by char ratio to stay
comparable with the source cache.

Usage: SCRATCH=... python e31_label_render.py <src_cache.json> <dst_cache.json>
"""

import json
import os
import re
import sys

SRC, DST = sys.argv[1], sys.argv[2]
SCRATCH = os.environ["SCRATCH"]

fd = json.load(open(os.path.join(SCRATCH, "fulldoc_contexts.json")))
src = json.load(open(SRC))

num_re = re.compile(r"\d")
units_re = re.compile(r"\(?\s*(in|dollars in)\s+(millions|thousands|billions)\b[^)\n]*\)?", re.IGNORECASE)
period_tok = re.compile(
    r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
    r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)


def is_num_word(w):
    return bool(num_re.search(w)) or w in ("$", "%")


def is_title(words):
    if not (1 <= len(words) <= 12) or any(is_num_word(w) for w in words):
        return False
    alpha = [w for w in words if any(c.isalpha() for c in w)]
    return bool(alpha) and sum(1 for w in alpha if w.isupper() or w.istitle()) / len(alpha) >= 0.7


def is_period_hdr(words):
    n = sum(1 for w in words if period_tok.match(w))
    return n >= 2 and n / max(1, len(words)) >= 0.4


def is_table(words):
    return sum(1 for w in words if is_num_word(w)) >= 3


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


out = {}
for qid, rec in src.items():
    doc_lines = fd[qid].split("\n")
    dwords = [l.split() for l in doc_lines]
    # per doc line: nearest title / period-header / units line at-or-above
    n = len(doc_lines)
    near_t, near_p, near_u = [None] * n, [None] * n, [None] * n
    t = p = u = None
    for i in range(n):
        w = dwords[i]
        if w and is_title(w):
            t = doc_lines[i].strip()
        if w and is_period_hdr(w):
            p = doc_lines[i].strip()
        if units_re.search(doc_lines[i]):
            u = units_re.search(doc_lines[i]).group(0).strip()
        near_t[i], near_p[i], near_u[i] = t, p, u
    # map normalized doc line -> first index
    idx = {}
    for i, l in enumerate(doc_lines):
        k = norm(l)
        if k and k not in idx:
            idx[k] = i

    out_lines, last_banner = [], None
    for line in rec["compressed_text"].split("\n"):
        w = line.split()
        i = idx.get(norm(line))
        if w and is_table(w) and i is not None:
            banner = " | ".join(x for x in (near_t[i], near_u[i], near_p[i]) if x)
            if banner and banner != last_banner:
                out_lines.append(f"### {banner}")
                last_banner = banner
        elif w and is_title(w):
            last_banner = None
        out_lines.append(line)
    new_text = "\n".join(out_lines)
    scale = len(new_text) / max(1, len(rec["compressed_text"]))
    out[qid] = {"compressed_text": new_text,
                "original_tokens": rec["original_tokens"],
                "compressed_tokens": min(int(rec["compressed_tokens"] * scale),
                                         rec["original_tokens"])}

json.dump(out, open(DST, "w"))
rats = [v["compressed_tokens"] / v["original_tokens"] for v in out.values()]
print(f"{DST}: {len(out)} docs, mean retention {sum(rats)/len(rats):.3f}")
