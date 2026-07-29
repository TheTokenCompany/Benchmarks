#!/usr/bin/env python3
"""Fleet a compression cache against the Gemini reader with the key kept in Modal.

Same experiment as `run_benchmark.py` with the documented Gemini env, but the
answering step runs inside a Modal container with `gemini-api-secret` attached,
because GEMINI_API_KEY is taaha's and that secret is the only place it lives (see
v9_rwr_train.py:112) — the Modal CLI cannot read a secret value back, and pulling
it onto a laptop to pass as ANSWER_API_KEY would copy someone else's credential
into local shell history and a mirrored transcript.

Split of work:
  answering  -> Modal container, needs GEMINI_API_KEY
  judging    -> local, reuses evaluate.judge_answer and the OPENAI_API_KEY in .env

Output is written to RESULTS_DIR/<config>.json in the exact schema
run_benchmark.py writes, so results land next to the v1 arms and every existing
reader of that directory keeps working. Resumable: questions already present in
the results file are skipped.

Usage:
    cd financebench
    RESULTS_SUBDIR=results_gemini35fl_fulldoc \
      ../.venv/bin/modal run fleet_gemini_modal.py --configs fd-v9kl03-budget22-v2--0.5
"""

import json
import os
import sys

import modal

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "compression_cache")

READER_MODEL = os.getenv("OPENAI_MODEL", "gemini-3.5-flash-lite")
READER_BASE_URL = os.getenv(
    "ANSWER_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

app = modal.App("financebench-fleet-gemini")
image = modal.Image.debian_slim(python_version="3.11").pip_install("openai")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gemini-api-secret")],
    timeout=5400,
    cpu=4,
)
def answer_batch(jobs: list[dict], model: str, base_url: str, system_prompt: str,
                 concurrency: int = 6) -> dict:
    """[{qid, question, context}] -> {qid: answer}. Runs the reader only."""
    import os as _os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from openai import OpenAI

    key = _os.environ.get("GEMINI_API_KEY") or _os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set -- the Modal secret "
                           "'gemini-api-secret' did not attach.")
    client = OpenAI(api_key=key, base_url=base_url)

    def one(job):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"Context:\n{job['context']}\n\nQuestion: {job['question']}"},
        ]
        last = None
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=model, messages=messages, max_completion_tokens=8000)
                return job["qid"], (r.choices[0].message.content or "").strip()
            except Exception as e:                        # noqa: BLE001
                last = e
                time.sleep(2 * (2 ** attempt))
        print(f"  [{job['qid']}] reader failed: {last}")
        return job["qid"], ""

    out = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            qid, ans = f.result()
            out[qid] = ans
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} answered", flush=True)
    return out


@app.local_entrypoint()
def main(configs: str = "", concurrency: int = 6, judge_concurrency: int = 8,
         limit: int = 0):
    """`configs` is a comma-separated list of cache names, e.g.
    'fd-v9kl03-budget22-v2--0.5,fd-v9kl03-budget24-v2--0.5'."""
    sys.path.insert(0, ROOT)
    from concurrent.futures import ThreadPoolExecutor

    from datasets import load_dataset

    import config
    from evaluate import judge_answer

    names = [c.strip() for c in configs.split(",") if c.strip()]
    if not names:
        print("nothing to do: pass --configs <cache-name>[,<cache-name>...]")
        return

    items = list(load_dataset(config.DATASET_NAME, split="train"))
    if config.NUM_QUESTIONS_LIMIT:
        items = items[:config.NUM_QUESTIONS_LIMIT]
    if limit:
        items = items[:limit]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    for name in names:
        cache_path = os.path.join(CACHE_DIR, f"{name}.json")
        if not os.path.exists(cache_path):
            print(f"SKIP {name}: no cache at {cache_path}")
            continue
        with open(cache_path) as f:
            cache = json.load(f)

        results_path = os.path.join(config.RESULTS_DIR, f"{name}.json")
        results = json.load(open(results_path)) if os.path.exists(results_path) else []
        done = {r["question_id"] for r in results if "question_id" in r}

        jobs, meta = [], {}
        for i, it in enumerate(items):
            qid = it.get("question_id", str(i))
            if qid in done or qid not in cache:
                continue
            jobs.append({"qid": qid, "question": it["question"],
                         "context": cache[qid]["compressed_text"]})
            meta[qid] = it
        if not jobs:
            print(f"[{name}] all {len(items)} questions already done")
            _summarize(name, results)
            continue

        print(f"[{name}] {len(results)} done, {len(jobs)} to answer "
              f"(reader={READER_MODEL})")
        answers = answer_batch.remote(jobs, READER_MODEL, READER_BASE_URL,
                                      config.SYSTEM_PROMPT, concurrency)

        print(f"[{name}] judging {len(answers)} answers locally "
              f"(judge={config.JUDGE_MODEL}, {judge_concurrency} workers)")

        def _judge(item):
            qid, ans = item
            it = meta[qid]
            try:
                return qid, ans, judge_answer(it["question"], it["answer"], ans)
            except RuntimeError as e:
                return qid, ans, {"correct": None, "explanation": f"ERROR: {e}"}

        # run_benchmark.py already judges from a 10-worker pool, so the OpenAI
        # client in evaluate.py is exercised concurrently in production today.
        judged = []
        with ThreadPoolExecutor(max_workers=judge_concurrency) as ex:
            for n, out in enumerate(ex.map(_judge, list(answers.items())), 1):
                judged.append(out)
                if n % 25 == 0:
                    print(f"  {n}/{len(answers)} judged", flush=True)

        for qid, ans, ev in judged:
            it = meta[qid]
            c = cache[qid]
            results.append({
                "question_id": qid,
                "question": it["question"],
                "question_type": it.get("question_type", ""),
                "question_reasoning": it.get("question_reasoning", ""),
                "gold_answer": it["answer"],
                "model_answer": ans,
                "correct": ev["correct"],
                "judge_explanation": ev["explanation"],
                "config": name,
                "compressed": True,
                "aggressiveness": float(name.rsplit("--", 1)[-1]),
                "original_tokens": c["original_tokens"],
                "compressed_tokens": c["compressed_tokens"],
                "compression_ratio": (c["compressed_tokens"] / c["original_tokens"]
                                      if c["original_tokens"] else 1.0),
            })
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        _summarize(name, results)


def _summarize(name, results):
    ev = [r for r in results if r.get("correct") is not None]
    if not ev:
        print(f"  {name}: no judged results")
        return
    n_correct = sum(1 for r in ev if r["correct"])
    ot = sum(r.get("original_tokens", 0) for r in results)
    ct = sum(r.get("compressed_tokens", 0) for r in results)
    line = f"  {name}: accuracy {n_correct / len(ev):.1%} ({n_correct}/{len(ev)})"
    if ot:
        line += f"   retention {ct / ot:.3f}"
    print(line)
