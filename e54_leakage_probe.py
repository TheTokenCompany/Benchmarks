#!/usr/bin/env python3
"""E54 leakage probe: can the reader answer WITHOUT any context?

If yes, every keep-set "suffices" for that item and its discovered labels carry no
selection signal — the search was measuring the reader's prior, not the text. Such
items get flagged; the tensor build down-weights or drops them. Uses a
knowledge-permissive system prompt (the training READER_SYSTEM instructs NOT FOUND
without an excerpt, which would mask leakage, so it is not reusable here).
"""

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from v9_rwr_train import answer_hit, make_reader_client

SCRATCH = Path(os.environ.get("SCRATCH", "."))
META = SCRATCH / "v9rl-meta" / "train_meta.jsonl"
OUT = SCRATCH / "e54" / "leakage.jsonl"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-3.5-flash-lite"
SYSTEM = ("Answer the question from your own knowledge, as concisely as possible: "
          "a number, a phrase, or a short list. If you do not know, say so.")


def main():
    items = [json.loads(l) for l in open(META)]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["qa_id"] for l in open(OUT)}
    todo = [it for it in items if it["qa_id"] not in done]
    print(f"{len(todo)} items to probe")
    client = make_reader_client(GEMINI_BASE, 60)
    lock = threading.Lock()
    fh = open(OUT, "a")
    rng = random.Random(0)

    def run(item):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=256,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Question: {item['question']}"}])
            reply = (r.choices[0].message.content or "").strip()
            leaked = bool(answer_hit(item["answer"], reply))
        except Exception as e:
            reply, leaked = f"ERROR: {e}"[:150], None
        with lock:
            fh.write(json.dumps({"qa_id": item["qa_id"], "leaked": leaked,
                                 "reply": reply[:200]}) + "\n")
            fh.flush()

    with ThreadPoolExecutor(24) as ex:
        list(ex.map(run, todo))
    fh.close()
    recs = [json.loads(l) for l in open(OUT)]
    n_leak = sum(1 for r in recs if r["leaked"])
    n_err = sum(1 for r in recs if r["leaked"] is None)
    print(f"leaked {n_leak}/{len(recs)} ({100*n_leak/len(recs):.1f}%)  errors {n_err}")


if __name__ == "__main__":
    main()
