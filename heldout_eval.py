#!/usr/bin/env python3
"""E50: held-out generalization eval on the v12 corpus.

The benchmark-informed-design caveat on v11b's 81.1% needs an out-of-selection
number: fresh filings, fresh Claude-written questions, companies disjoint from
the v11 TRAINING set (and, by construction of v12, from FinanceBench). Three
arms at budget33 serving, same reader + judge as the campaign.

Phases (resumable): select -> compress (Modal, both models, one pass each) ->
answer+judge (Gemini + gpt-5-mini) -> report.

Env: SCRATCH, GEMINI_API_KEY, OPENAI_API_KEY.
Usage: .venv/bin/python heldout_eval.py [--n 300] [--phase all|select|answer]
"""

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRATCH = Path(os.environ["SCRATCH"])
CORPUS = SCRATCH / "v12corpus"
OUT = SCRATCH / "heldout"
OUT.mkdir(exist_ok=True)

TRAIN_TICKERS_SRC = SCRATCH / "v9rl-meta" / "train_meta.jsonl"

ARMS = {
    "v11b": "exp-20260728-011924-v11b-warm-w2140s2",
    "v9rl": "exp-20260726-023910-v9-rwr-s2ep4-kl03",
}


def select(n):
    train_cos = set()
    for l in open(TRAIN_TICKERS_SRC):
        train_cos.add(json.loads(l)["qa_id"].split("_")[0].upper())
    qa_by_type = {}
    for l in open(CORPUS / "qa.jsonl"):
        qa = json.loads(l)
        tick = (qa.get("ticker") or "").upper()
        if not tick or tick in train_cos:
            continue
        if not (CORPUS / "docs" / qa["file"]).exists():
            continue
        qa_by_type.setdefault(qa["qtype"], []).append(qa)
    rng = random.Random(42)
    picked = []
    per = n // 3
    for qt, items in sorted(qa_by_type.items()):
        rng.shuffle(items)
        picked.extend(items[:per])
    with open(OUT / "items.jsonl", "w") as f:
        for i, qa in enumerate(picked):
            qa["hid"] = f"h{i:03d}"
            f.write(json.dumps(qa) + "\n")
    if not picked:
        raise SystemExit("select() produced 0 items — ticker filter broken, refusing to continue")
    tickers = {q.get("ticker") for q in picked}
    print(f"selected {len(picked)} held-out QA over {len(tickers)} disjoint tickers "
          f"({ {qt: len(v) for qt, v in qa_by_type.items()} } available)")


def compress():
    import v8_eval_precompress as ev
    items = [json.loads(l) for l in open(OUT / "items.jsonl")]
    texts, questions, ids = [], [], []
    doc_cache = {}
    for qa in items:
        if qa["file"] not in doc_cache:
            doc_cache[qa["file"]] = (CORPUS / "docs" / qa["file"]).read_text(errors="replace")
        texts.append(doc_cache[qa["file"]])
        questions.append(qa["question"])
        ids.append(qa["hid"])
    with ev.app.run():
        for alias, path in ARMS.items():
            out_path = OUT / f"cache_{alias}.json"
            if out_path.exists():
                continue
            result = ev.compress_all.remote(texts, questions, ids, path, alias, 2048,
                                            ["budget33"], False, 1000, False, None, [0.5])
            key = next(k for k in result if "budget33" in k)
            json.dump(result[key], open(out_path, "w"))
            print(f"{alias}: cached {len(result[key])} compressions")


def answer_and_judge():
    import sys
    sys.path.insert(0, "financebench")
    import config as fb_config
    from evaluate import judge_answer
    from openai import OpenAI
    gclient = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    items = [json.loads(l) for l in open(OUT / "items.jsonl")]
    caches = {a: json.load(open(OUT / f"cache_{a}.json")) for a in ARMS}
    doc_cache = {}
    lock = threading.Lock()
    res_path = OUT / "results.json"
    results = json.load(open(res_path)) if res_path.exists() else {}

    def run_one(arm, qa):
        key = f"{arm}:{qa['hid']}"
        if key in results:
            return
        if arm == "raw":
            if qa["file"] not in doc_cache:
                doc_cache[qa["file"]] = (CORPUS / "docs" / qa["file"]).read_text(errors="replace")
            ctx = doc_cache[qa["file"]]
        else:
            ctx = caches[arm][qa["hid"]]["compressed_text"]
        msgs = [{"role": "system", "content": fb_config.SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {qa['question']}"}]
        try:
            r = gclient.chat.completions.create(model="gemini-3.5-flash-lite",
                                                messages=msgs, max_completion_tokens=8000)
            ans = (r.choices[0].message.content or "").strip()
        except Exception as e:
            ans = f"ERROR: {e}"
        verdict = (judge_answer(qa["question"], qa["gold_answer"], ans)
                   if ans and not ans.startswith("ERROR") else {"correct": False, "explanation": "no answer"})
        with lock:
            results[key] = {"correct": bool(verdict["correct"]), "answer": ans[:400],
                            "qtype": qa["qtype"]}
            if len(results) % 50 == 0:
                json.dump(results, open(res_path, "w"))

    ex = ThreadPoolExecutor(10)
    futs = [ex.submit(run_one, arm, qa) for arm in ["raw", *ARMS] for qa in items]
    for f in futs:
        f.result()
    json.dump(results, open(res_path, "w"))

    for arm in ["raw", *ARMS]:
        rows = {q["hid"]: results[f"{arm}:{q['hid']}"] for q in items if f"{arm}:{q['hid']}" in results}
        acc = 100 * sum(1 for r in rows.values() if r["correct"]) / len(rows)
        by_t = {}
        for r in rows.values():
            by_t.setdefault(r["qtype"], []).append(r["correct"])
        detail = " ".join(f"{t}:{100*sum(v)/len(v):.0f}" for t, v in sorted(by_t.items()))
        print(f"{arm:5s}: {acc:5.1f}%  n={len(rows)}   [{detail}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--phase", default="all")
    a = ap.parse_args()
    if a.phase in ("all", "select"):
        select(a.n)
    if a.phase in ("all", "compress"):
        compress()
    if a.phase in ("all", "answer"):
        answer_and_judge()
