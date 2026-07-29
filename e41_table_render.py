#!/usr/bin/env python3
"""E41: structured table render — real column-period parsing (the E31 fix).

E30 proved per-figure labels (statement + period + units) unlock reader
arithmetic; E31 proved *misattached* labels are worse than none. This does the
attribution properly: parse each table region from the ORIGINAL document —
period-header line defines K columns; a data row is renderable only when its
value count equals K — and rewrite only those unambiguous rows in the
compressed cache. Everything else passes through raw.

Row render:  Cash and cash equivalents: Jun 30, 2023 = $4,258 | Dec 31, 2022 = $3,655
Region head: ### 3M ... Balance Sheet (Dollars in millions) — cols: Jun 30, 2023 | Dec 31, 2022

Modes: --mode rows (per-row rewrite, E30-style), --mode schema (region header
line only, rows left raw), --mode both.

Usage:
  SCRATCH=... python e41_table_render.py <src_cache> <dst_cache> --mode rows
"""

import argparse
import json
import os
import re

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
month_alt = "|".join(MONTHS)
# "June 30, 2023" / "December 31,2022" / bare years as fallback columns
date_re = re.compile(rf"((?:{month_alt})\s+\d{{1,2}},?\s*\d{{4}})")
year_re = re.compile(r"\b(19\d\d|20\d\d)\b")
units_re = re.compile(r"\((?:dollars |amounts )?in (millions|thousands|billions)[^)]*\)", re.I)
# a value: optional $, number with commas/decimals, optionally parenthesized (negative), or em/en dash (nil)
value_re = re.compile(r"\$?\s*\(?\d[\d,]*(?:\.\d+)?\)?%?|—|–")
num_re = re.compile(r"\d")


def parse_periods(line):
    """Extract ordered period column labels from a candidate header line."""
    dates = date_re.findall(line)
    if len(dates) >= 2:
        return dates
    # fiscal-year style: ">= 2 bare years, low other content"
    years = year_re.findall(line)
    if len(years) >= 2:
        words = line.split()
        if sum(1 for w in words if year_re.fullmatch(w.strip(".,:;"))) / max(1, len(words)) >= 0.4:
            return years
    return None


def split_row(line):
    """-> (label, [values]) using the first value token as the boundary.
    Returns None for rows without values (section labels)."""
    m = None
    for m0 in value_re.finditer(line):
        # ignore leading footnote-style small ints glued in labels like "Note 9"
        prefix = line[:m0.start()].rstrip()
        if prefix.endswith(("Note", "note")):
            continue
        m = m0
        break
    if m is None or not num_re.search(line[m.start():]) and "—" not in line[m.start():]:
        return None
    label = line[:m.start()].strip()
    if not label or num_re.search(label.split()[0] if label.split() else ""):
        return None
    vals = [v.strip() for v in value_re.findall(line[m.start():])]
    # merge "$ 4,258" style: drop bare $ tokens, attach to following value
    merged, pending_dollar = [], False
    for v in vals:
        if v == "$":
            pending_dollar = True
            continue
        merged.append(("$" + v.lstrip("$ ")) if (pending_dollar or v.startswith("$")) else v)
        pending_dollar = False
    return label, merged


def parse_document(text):
    """-> {normalized_row_line: rendered_row}, {region_first_line_norm: schema_line}"""
    lines = text.split("\n")
    row_map, schema_map = {}, {}
    cur_periods, cur_units, cur_title, region_rows = None, "", "", 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        periods = parse_periods(s)
        um = units_re.search(s)
        if periods:
            cur_periods = periods
            if um:
                cur_units = um.group(0)
            # nearest plausible title above
            cur_title = ""
            for k in range(i - 1, max(-1, i - 6), -1):
                t = lines[k].strip()
                if t and not num_re.search(t) and len(t.split()) <= 14:
                    cur_title = t
                    break
                if "Statement" in t or "Balance Sheet" in t or "Cash Flow" in t:
                    cur_title = t
                    break
            region_rows = 0
            continue
        if um and not periods:
            cur_units = um.group(0)
        if cur_periods is None:
            continue
        # region expiry: a long prose line without numbers ends the table
        if len(s.split()) > 30 and not num_re.search(s):
            cur_periods = None
            continue
        parsed = split_row(s)
        if not parsed:
            continue
        label, vals = parsed
        if len(vals) != len(cur_periods):
            continue  # ambiguous — leave raw (THE E31 fix)
        rendered = f"{label}: " + " | ".join(
            f"{p} = {v}" for p, v in zip(cur_periods, vals))
        key = norm(s)
        if key not in row_map:
            row_map[key] = rendered
        if region_rows == 0:
            head = f"### {cur_title} {cur_units} — cols: " + " | ".join(cur_periods)
            schema_map[key] = head.strip()
        region_rows += 1
    return row_map, schema_map


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=["rows", "schema", "both"], default="rows")
    args = ap.parse_args()

    scratch = os.environ["SCRATCH"]
    fd = json.load(open(os.path.join(scratch, "fulldoc_contexts.json")))
    src = json.load(open(args.src))
    out = {}
    n_rewritten = n_lines = 0
    for qid, rec in src.items():
        row_map, schema_map = parse_document(fd[qid])
        out_lines = []
        for line in rec["compressed_text"].split("\n"):
            key = norm(line)
            n_lines += 1
            if args.mode in ("schema", "both") and key in schema_map:
                out_lines.append(schema_map[key])
            if args.mode in ("rows", "both") and key in row_map:
                out_lines.append(row_map[key])
                n_rewritten += 1
            else:
                if args.mode == "schema" or key not in row_map:
                    out_lines.append(line)
        new_text = "\n".join(out_lines)
        scale = len(new_text) / max(1, len(rec["compressed_text"]))
        out[qid] = {"compressed_text": new_text,
                    "original_tokens": rec["original_tokens"],
                    "compressed_tokens": min(int(rec["compressed_tokens"] * scale),
                                             rec["original_tokens"])}
    json.dump(out, open(args.dst, "w"))
    rats = [v["compressed_tokens"] / v["original_tokens"] for v in out.values()]
    print(f"{args.dst}: rewrote {n_rewritten}/{n_lines} lines "
          f"({args.mode}), mean retention {sum(rats)/len(rats):.3f}")


if __name__ == "__main__":
    main()
