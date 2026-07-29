#!/usr/bin/env python3
"""v12 verification: prove the corpus is clean before anything trains on it.

Four checks, in order of how much damage they prevent:

  1. FINANCEBENCH OVERLAP -- HARD FAIL. Re-derives the exclusion set from the
     live SEC ticker file (not from the exclusions.json the fetcher wrote, which
     would just be marking its own homework) and asserts no manifest CIK and no
     manifest company name matches it. If this fails, the corpus is unusable and
     the script exits non-zero: a leaked FinanceBench company turns every
     downstream eval number into a claim about memorization.
  2. FACT-ON-CITED-LINE RATE -- recomputed independently of v12_gen_qa's own
     validator, straight from the stored .txt files. v12_gen_qa drops failures
     at write time, so a healthy corpus reads ~100% here; anything lower means
     the docs on disk have drifted from the ones the QA was written against.
  3. CORPUS STATS -- form mix, token/line distribution, sector spread, per-filer
     concentration. The number to watch is filer concentration: a corpus where
     20 companies supply half the documents teaches those 20 companies.
  4. EYEBALL SAMPLE -- 50 random QA printed with their cited lines, because no
     automated check catches "the question is technically answerable but dumb".

Run:
    .venv/bin/python v12_verify.py --selftest
    .venv/bin/python v12_verify.py --corpus $SCRATCH/v12corpus
    .venv/bin/python v12_verify.py --corpus $SCRATCH/v12corpus --sample 50
    .venv/bin/python v12_verify.py --corpus $SCRATCH/v12corpus --upload
"""

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from v12_fetch_corpus import (
    DEFAULT_OUT, TICKERS_URL, Fetcher, Throttle, MAX_RPS, norm_name,
    resolve_exclusions, FB_NAME_FRAGMENTS,
)
from v12_gen_qa import QTYPES, content_words, figures

DEFAULT_CORPUS = DEFAULT_OUT


def pct(a, b):
    return f"{a}/{b} ({a / max(1, b):.1%})"


def hist(values, buckets):
    c = Counter()
    for v in values:
        label = f">={buckets[-1]:,}"
        for lo, hi in zip([0] + buckets, buckets + [float("inf")]):
            if lo <= v < hi:
                label = f"{lo:,}-{hi:,}" if hi != float("inf") else f">={lo:,}"
                break
        c[label] += 1
    return c


# --------------------------------------------------------------------------- #
def check_overlap(man, offline_ciks=None):
    """-> (n_violations, detail lines). Independent re-derivation of the
    blocklist; `offline_ciks` is only for the selftest."""
    if offline_ciks is None:
        f = Fetcher(Throttle(MAX_RPS))
        raw = f.get(TICKERS_URL, as_json=True)
        filers = [dict(zip(raw["fields"], row)) for row in raw["data"]]
        excl, _ = resolve_exclusions(filers)
    else:
        excl = offline_ciks

    bad = []
    for r in man:
        if int(r["cik"]) in excl:
            bad.append(f"CIK {r['cik']} ({r['name']}) via {r['file']}")
            continue
        n = norm_name(r["name"])
        hitf = next((fr for fr in FB_NAME_FRAGMENTS if fr in n), None)
        if hitf:
            bad.append(f"NAME '{r['name']}' matches '{hitf}' via {r['file']}")
    return excl, bad


def check_facts(corpus, qa):
    """Recompute the fact-on-cited-line rate from the .txt files on disk."""
    cache, ok, why = {}, 0, Counter()
    for r in qa:
        p = corpus / "docs" / r["file"]
        if r["file"] not in cache:
            cache[r["file"]] = p.read_text().split("\n") if p.exists() else None
        lines = cache[r["file"]]
        if lines is None:
            why["missing_doc"] += 1
            continue
        ev = r["evidence_abs_lines"]
        if any(e >= len(lines) for e in ev):
            why["line_out_of_range"] += 1
            continue
        cited = " ".join(lines[e] for e in ev)
        want = figures(r["gold_answer"])
        if want:
            if want & figures(cited):
                ok += 1
            else:
                why["no_figure_on_cited_lines"] += 1
        else:
            if len(content_words(r["gold_answer"]) & content_words(cited)) >= 2:
                ok += 1
            else:
                why["no_content_overlap"] += 1
    return ok, why


# --------------------------------------------------------------------------- #
def upload(corpus, volume, subdir):
    """Push the corpus to a Modal volume so a scratchpad wipe cannot lose it again.

    docs/ is shipped as a single tar.gz rather than ~2000 individual objects:
    `modal volume put` on a directory uploads file by file, which for a corpus of
    this shape is dominated by per-object round trips."""
    tar = corpus / "docs.tar.gz"
    if not tar.exists() or tar.stat().st_mtime < (corpus / "manifest.jsonl").stat().st_mtime:
        print(f"\npacking docs -> {tar.name}")
        subprocess.run(["tar", "-czf", str(tar), "-C", str(corpus), "docs"], check=True)
    print(f"  docs.tar.gz {tar.stat().st_size/1e6:.0f} MB")

    for name in ("docs.tar.gz", "manifest.jsonl", "qa.jsonl", "exclusions.json",
                 "verify_report.json"):
        p = corpus / name
        if not p.exists():
            print(f"  skip {name} (absent)")
            continue
        cmd = ["modal", "volume", "put", "--force", volume, str(p),
               f"{subdir}/{name}"]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    print(f"uploaded -> {volume}:/{subdir}")


def selftest():
    man = [{"cik": 111, "name": "CLEAN WIDGETS INC", "file": "a.txt"},
           {"cik": 789019, "name": "MICROSOFT CORP", "file": "b.txt"}]
    _, bad = check_overlap(man[:1], offline_ciks={789019})
    assert bad == [], bad
    _, bad = check_overlap(man, offline_ciks={789019})
    assert len(bad) == 1 and "789019" in bad[0], bad
    # the name net alone must catch a filer whose CIK is not in the blocklist
    _, bad = check_overlap([{"cik": 4242, "name": "Pfizer Ireland Pharmaceuticals",
                             "file": "c.txt"}], offline_ciks=set())
    assert len(bad) == 1 and "PFIZER" in bad[0], bad

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        c = Path(td)
        (c / "docs").mkdir()
        (c / "docs" / "d.txt").write_text(
            "HEADER\nNet revenues $ 12,345 $ 10,987\nprose about pricing and volume\n")
        good = {"file": "d.txt", "gold_answer": "$12,345", "evidence_abs_lines": [1]}
        bad_fig = {"file": "d.txt", "gold_answer": "$99,999",
                   "evidence_abs_lines": [1]}
        oob = {"file": "d.txt", "gold_answer": "$12,345", "evidence_abs_lines": [99]}
        ok, why = check_facts(c, [good, bad_fig, oob])
        assert ok == 1, (ok, why)
        assert why["no_figure_on_cited_lines"] == 1 and why["line_out_of_range"] == 1, why

    assert hist([5, 5000, 90000], [1000, 20000])["0-1,000"] == 1
    print("selftest OK: overlap by CIK + by name, fact-on-line recompute, histogram")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--upload", action="store_true",
                    help="on PASS only, modal volume put -> otso-v8-data:/v12-corpus")
    ap.add_argument("--volume", default="otso-v8-data")
    ap.add_argument("--volume-subdir", default="v12-corpus")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    corpus = Path(args.corpus)
    man = [json.loads(l) for l in (corpus / "manifest.jsonl").read_text().splitlines()
           if l.strip()]
    qa_path = corpus / "qa.jsonl"
    qa = ([json.loads(l) for l in qa_path.read_text().splitlines() if l.strip()]
          if qa_path.exists() else [])

    print("=" * 78)
    print(f"v12 VERIFY  {corpus}")
    print("=" * 78)

    # --- 1. overlap (hard gate) ------------------------------------------- #
    print("\n[1] FINANCEBENCH OVERLAP (hard fail)")
    excl, bad = check_overlap(man)
    print(f"    blocklist re-derived from SEC: {len(excl)} CIKs")
    print(f"    manifest: {len(man)} docs / {len({r['cik'] for r in man})} filers")
    if bad:
        print(f"    !! {len(bad)} VIOLATIONS:")
        for b in bad[:20]:
            print(f"       {b}")
        print("\nCORPUS IS CONTAMINATED -- do not train on it.")
        sys.exit(1)
    print("    PASS: no excluded CIK and no excluded company name in the corpus")

    # --- 2. corpus stats --------------------------------------------------- #
    print("\n[2] CORPUS STATS")
    forms = Counter(r["family"] for r in man)
    print(f"    forms: {dict(forms.most_common())}")
    words = sorted(r["words"] for r in man)
    print(f"    words/doc: total={sum(words):,} median={words[len(words)//2]:,} "
          f"p10={words[len(words)//10]:,} p90={words[len(words)*9//10]:,} "
          f"max={words[-1]:,}")
    for k, v in sorted(hist(words, [2000, 10000, 40000, 100000]).items()):
        print(f"        {k:>18} words: {v}")
    yrs = Counter(r["filing_date"][:4] for r in man)
    print(f"    filing years: {dict(sorted(yrs.items()))}")
    sect = Counter((r.get("sic") or "?").split(" - ")[0][:44] for r in man)
    print(f"    sectors ({len(sect)} distinct SIC), top 10:")
    for s, n in sect.most_common(10):
        print(f"        {n:>5}  {s}")
    per_filer = Counter(r["cik"] for r in man)
    top20 = sum(n for _, n in per_filer.most_common(20))
    print(f"    filer concentration: {len(per_filer)} filers, "
          f"max {max(per_filer.values())} docs, "
          f"top-20 hold {pct(top20, len(man))}")

    if not qa:
        print("\n[3] no qa.jsonl yet -- run v12_gen_qa.py")
        return

    # --- 3. QA quality ----------------------------------------------------- #
    print(f"\n[3] QA QUALITY  ({len(qa)} items)")
    print(f"    qtypes: {dict(Counter(r['qtype'] for r in qa).most_common())}")
    ev = [len(r["evidence_abs_lines"]) for r in qa]
    print(f"    evidence lines/item: mean={sum(ev)/len(ev):.2f} "
          f"max={max(ev)} single-line={pct(sum(1 for e in ev if e == 1), len(qa))}")
    docs_with = len({r["file"] for r in qa})
    print(f"    coverage: {pct(docs_with, len(man))} of docs carry >=1 QA "
          f"({len(qa)/max(1,docs_with):.1f} QA per covered doc)")
    qlen = sorted(len(r["question"].split()) for r in qa)
    alen = sorted(len(r["gold_answer"].split()) for r in qa)
    print(f"    question words: median={qlen[len(qlen)//2]}  "
          f"answer words: median={alen[len(alen)//2]}")
    dupes = len(qa) - len({r["question"].strip().lower() for r in qa})
    print(f"    duplicate questions: {pct(dupes, len(qa))}")

    ok, why = check_facts(corpus, qa)
    print(f"    fact-on-cited-line (recomputed from disk): {pct(ok, len(qa))}")
    for k, v in why.most_common():
        print(f"        fail {k}: {v}")
    if ok / max(1, len(qa)) < 0.98:
        print("    !! WARNING: the stored docs no longer agree with the QA that was "
              "written against them")

    # --- 4. eyeball sample ------------------------------------------------- #
    print(f"\n[4] RANDOM SAMPLE ({args.sample} QA, seed {args.seed})")
    rng = random.Random(args.seed)
    by_t = defaultdict(list)
    for r in qa:
        by_t[r["qtype"]].append(r)
    picks = []
    per = max(1, args.sample // max(1, len(by_t)))
    for t in QTYPES:
        picks += rng.sample(by_t[t], min(per, len(by_t[t]))) if by_t[t] else []
    picks += rng.sample(qa, max(0, args.sample - len(picks)))
    cache = {}
    for i, r in enumerate(picks[:args.sample], 1):
        if r["file"] not in cache:
            cache[r["file"]] = (corpus / "docs" / r["file"]).read_text().split("\n")
        lines = cache[r["file"]]
        print(f"\n--- {i}. [{r['qtype']}] {r['file']} ({r.get('ticker','?')})")
        print(f"    Q: {r['question']}")
        print(f"    A: {r['gold_answer']}")
        for e in r["evidence_abs_lines"]:
            body = lines[e] if e < len(lines) else "<OUT OF RANGE>"
            print(f"    L{e}| {body[:150]}")

    print("\n" + "=" * 78)
    print(f"VERDICT: {len(man)} docs, {len(qa)} QA, "
          f"fact-on-line {ok/max(1,len(qa)):.1%}, FinanceBench overlap 0")
    print("=" * 78)

    report = {
        "corpus": str(corpus), "n_docs": len(man), "n_filers": len(per_filer),
        "n_qa": len(qa), "forms": dict(forms), "filing_years": dict(sorted(yrs.items())),
        "words_total": sum(words), "words_median": words[len(words) // 2],
        "n_sic": len(sect), "top20_filer_share": round(top20 / len(man), 4),
        "qtypes": dict(Counter(r["qtype"] for r in qa)),
        "fact_on_cited_line": round(ok / max(1, len(qa)), 4),
        "fact_failures": dict(why),
        "financebench_overlap": 0, "blocklist_ciks": len(excl),
    }
    (corpus / "verify_report.json").write_text(json.dumps(report, indent=1))

    # v9_rl_prep.py reads <data-dir>/synthetic_qa.jsonl and <data-dir>/filings/.
    # Two symlinks make the v12 corpus a drop-in --data-dir for the existing
    # pipeline, so nothing downstream needs a code change to consume it.
    for link, target in (("filings", "docs"), ("synthetic_qa.jsonl", "qa.jsonl")):
        p = corpus / link
        if not p.exists():
            p.symlink_to(target)
    print(f"\nv9_rl_prep compatibility: {corpus}/filings -> docs, "
          f"synthetic_qa.jsonl -> qa.jsonl")

    if args.upload:
        upload(corpus, args.volume, args.volume_subdir)


if __name__ == "__main__":
    main()
