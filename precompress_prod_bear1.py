#!/usr/bin/env python3
"""Full-filing compression caches via the PROD bear API (reference series).

Splits oversized documents client-side (~300k chars) — the API chunks at 512
internally, so client-side splits don't change semantics. Gentle concurrency
so prod latency is unaffected. Resumable per config file.

Env: BEAR_API_KEY, SCRATCH (fulldoc_contexts.json location), PROD_MODEL.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compress

compress._BEAR_API_KEY = os.environ["BEAR_API_KEY"]

MODEL = os.getenv("PROD_MODEL", "bear-1.2")
LEVELS = [float(x) for x in os.getenv("LEVELS", "0.3,0.5,0.7").split(",")]
SPLIT = 300_000
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "financebench", "compression_cache")

fd = json.load(open(os.environ["SCRATCH"] + "/fulldoc_contexts.json"))
lock = threading.Lock()

for aggr in LEVELS:
    path = os.path.join(CACHE, f"fd-{MODEL}--{aggr}.json")
    entry = json.load(open(path)) if os.path.exists(path) else {}

    def work(item):
        qid, text = item
        if qid in entry:
            return
        pieces = [text[i:i + SPLIT] for i in range(0, len(text), SPLIT)]
        comp_texts, orig_n, comp_n = [], 0, 0
        for p in pieces:
            r = compress.compress_text(p, aggr, MODEL)
            comp_texts.append(r["compressed_text"])
            orig_n += r["original_tokens"]
            comp_n += r["compressed_tokens"]
        with lock:
            entry[qid] = {"compressed_text": "\n".join(comp_texts),
                          "original_tokens": orig_n,
                          "compressed_tokens": min(comp_n, orig_n)}
            if len(entry) % 15 == 0:
                json.dump(entry, open(path, "w"))

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, sorted(fd.items())))
    json.dump(entry, open(path, "w"))
    rats = [v["compressed_tokens"] / v["original_tokens"]
            for v in entry.values() if v["original_tokens"]]
    print(f"fd-{MODEL}--{aggr}: {len(entry)} docs, mean retention {sum(rats)/len(rats):.3f}")
