#!/usr/bin/env python3
"""E54: discovered labels — per-item minimal-sufficient keep-sets via reader search.

Replaces the hand-designed v10 label policy with MEASURED necessity: start from the
v10 keep-set (fall back to the full chunk when v10's set doesn't answer), then
greedy group-testing backward elimination over kept LINES — remove a group, ask the
reader, commit the removal if the answer still survives, binary-split the group when
it doesn't. The output is, per item, the minimal line set whose text still lets the
reader answer, plus the list of verified BREAKER lines (removing that one line flips
the answer) — hard targets and a necessity signal for loss weighting in one pass.

Search unit is the LINE (not the word): serving keeps/drops whole table rows in
effect, v10 labels are line-graded, and line-level search costs ~10-25 reader calls
per item instead of hundreds.

Statuses: ok (searched), v10_insufficient_full_ok (searched from full chunk),
unanswerable (even the full chunk fails the reader -> no label emitted, item keeps
its v10 labels downstream), error.

Run (needs GEMINI_API_KEY):
    .venv/bin/python e54_label_discovery.py --limit 12          # smoke, prints detail
    .venv/bin/python e54_label_discovery.py                     # full train split
"""

import argparse
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from v9_rl_prep import render_mask
from v9_rwr_train import READER_SYSTEM, answer_hit, ask_reader, make_reader_client
from v10_build_targets import build_labels, word_lines

SCRATCH = Path(os.environ.get("SCRATCH", "."))
META = SCRATCH / "v9rl-meta" / "train_meta.jsonl"
OUT_DIR = SCRATCH / "e54"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
READER_MODEL = os.environ.get("READER_MODEL", "gemini-3.5-flash-lite")


class Searcher:
    def __init__(self, client, cache, cache_lock, calls_cap, order_seed=0):
        self.client = client
        self.cache = cache
        self.cache_lock = cache_lock
        self.calls_cap = calls_cap
        self.order_seed = order_seed
        self.rng = random.Random(0)

    def survives(self, item, keep_lines, line_words):
        """Render the union of `keep_lines` and ask the reader. Cached by text hash."""
        keep = [False] * item["n_words"]
        for l in keep_lines:
            for w in line_words[l]:
                keep[w] = True
        rendered = render_mask(item["words"], item["nl_after"], keep)
        key = f"{item['qa_id']}:{hashlib.md5(rendered.encode()).hexdigest()}"
        with self.cache_lock:
            if key in self.cache:
                return self.cache[key], 0
        reply = ask_reader(self.client, READER_MODEL, item["question"], rendered,
                           256, 5, self.rng)
        ok = bool(answer_hit(item["answer"], reply))
        with self.cache_lock:
            self.cache[key] = ok
        return ok, 1

    def search(self, item, init_lines=None, init_status="ok"):
        words, nl_after = item["words"], item["nl_after"]
        line_of_word, line_words = word_lines(words, nl_after)
        v10_keep, _info = build_labels(words, nl_after, item["gold_fact_words"],
                                       item["qtype"])
        v10_lines = sorted({line_of_word[i] for i, k in enumerate(v10_keep) if k})
        all_lines = sorted(line_words)
        calls = 0

        if init_lines is not None:          # resume from a prior pass's kept set
            kept = [l for l in init_lines if l in line_words]
            status = init_status
        else:
            ok, c = self.survives(item, v10_lines, line_words)
            calls += c
            status = "ok"
            if ok:
                kept = list(v10_lines)
            else:
                ok_full, c = self.survives(item, all_lines, line_words)
                calls += c
                if not ok_full:
                    return {"qa_id": item["qa_id"], "status": "unanswerable",
                            "calls": calls}
                status = "v10_insufficient_full_ok"
                kept = list(all_lines)

        # Removal priority: least-likely-needed first. Lines with no gold-fact hit
        # go before fact-hitting lines; within a class, farthest from the nearest
        # fact line first. Fact lines are still ELIGIBLE — a "gold fact" like a
        # year can hit lines the answer never needed.
        fact_lines = {l for l in kept
                      if any(any(f.lower() in words[w].lower()
                                 for f in item["gold_fact_words"])
                             for w in line_words[l])}
        anchor = sorted(fact_lines) or kept

        def prio(l):
            d = min(abs(l - a) for a in anchor)
            return (l in fact_lines, -d)

        order = sorted(kept, key=prio)          # non-fact far lines first
        if self.order_seed:                     # alternate-minima pass: shuffled
            random.Random(f"{self.order_seed}:{item['qa_id']}").shuffle(order)
        breakers = []

        # Group-testing backward elimination: try dropping a prefix group of the
        # remaining candidates; on failure split the group. Singleton failures are
        # verified breakers.
        pending = list(order)
        group = max(1, len(pending) // 4)
        while pending and calls < self.calls_cap:
            g = pending[:group]
            trial = [l for l in kept if l not in g]
            if not trial:                        # never emit an all-drop label
                break
            ok, c = self.survives(item, trial, line_words)
            calls += c
            if ok:
                kept = trial
                pending = pending[len(g):]
                group = max(1, min(len(pending), group * 2))
            elif len(g) == 1:
                breakers.append(g[0])
                pending = pending[1:]
                group = max(1, len(pending) // 4)
            else:
                group = max(1, len(g) // 2)

        keep_words = sorted(w for l in kept for w in line_words[l])
        n = item["n_words"]
        return {"qa_id": item["qa_id"], "status": status, "calls": calls,
                "kept_lines": kept, "breaker_lines": sorted(breakers),
                "keep_words": keep_words,
                "keep_frac_v10": round(sum(v10_keep) / n, 4),
                "keep_frac_final": round(len(keep_words) / n, 4),
                "search_exhausted": bool(pending)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--calls-cap", type=int, default=25)
    ap.add_argument("--meta", default=str(META))
    ap.add_argument("--deepen", action="store_true",
                    help="re-search items whose pass-1 search exhausted its call "
                         "cap, resuming from their stored kept set; results go to "
                         "labels_deep.jsonl (merge: deep overrides pass-1)")
    ap.add_argument("--order-seed", type=int, default=0,
                    help="nonzero: shuffle the removal order per item to find an "
                         "ALTERNATE minimal set; results go to labels_o<seed>.jsonl. "
                         "Per-line keep frequency across passes = soft necessity.")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / ("labels_deep.jsonl" if args.deepen
                          else f"labels_o{args.order_seed}.jsonl" if args.order_seed
                          else "labels.jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(l)["qa_id"] for l in open(out_path)}
    items = [json.loads(l) for l in open(args.meta)]
    if args.limit:
        items = items[:args.limit]
    prior = {}
    if args.deepen:
        prior = {r["qa_id"]: r
                 for l in open(OUT_DIR / "labels.jsonl") for r in [json.loads(l)]
                 if r.get("search_exhausted")}
        items = [it for it in items if it["qa_id"] in prior]
    todo = [it for it in items if it["qa_id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to search")

    client = make_reader_client(GEMINI_BASE, 120)
    cache, cache_lock, write_lock = {}, threading.Lock(), threading.Lock()
    searcher = Searcher(client, cache, cache_lock, args.calls_cap, args.order_seed)
    fh = open(out_path, "a")
    stats = {"n": 0, "calls": 0}

    def run(item):
        try:
            p = prior.get(item["qa_id"])
            rec = (searcher.search(item, init_lines=p["kept_lines"],
                                   init_status=p["status"]) if p
                   else searcher.search(item))
        except Exception as e:
            rec = {"qa_id": item["qa_id"], "status": "error", "error": str(e)[:200]}
        with write_lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["n"] += 1
            stats["calls"] += rec.get("calls", 0)
            if stats["n"] % 50 == 0:
                print(f"  {stats['n']}/{len(todo)}  avg calls "
                      f"{stats['calls']/stats['n']:.1f}")
        return rec

    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(run, todo))

    fh.close()
    oks = [r for r in recs if r["status"].startswith(("ok", "v10_insufficient"))]
    if oks:
        import statistics
        before = statistics.mean(r["keep_frac_v10"] for r in oks)
        after = statistics.mean(r["keep_frac_final"] for r in oks)
        nbrk = statistics.mean(len(r["breaker_lines"]) for r in oks)
        print(f"searched {len(oks)}/{len(recs)}  keep_frac v10 {before:.3f} -> "
              f"discovered {after:.3f}   breakers/item {nbrk:.1f}")
    by = {}
    for r in recs:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print("statuses:", by)


if __name__ == "__main__":
    main()
