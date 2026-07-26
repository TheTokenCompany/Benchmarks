#!/usr/bin/env python3
"""E29: Fable-5-as-ABSTRACTIVE-compressor on the 17 never-answered questions.

Fable 5 (max effort) sees question + full filing (NOT the gold) and must emit a
purely extractive compressed context (verbatim lines only). Gemini 3.5
Flash-Lite answers from that extract; gpt-5-mini judges — same prompts as the
harness. If Gemini flips a question, ideal extraction beats every config we
have, i.e. compression still has headroom on that question.

Phases (resumable via JSON checkpoints in SCRATCH/fable5_extract/):
  A extract   B validate extractiveness   C answer+judge

Env: ANTHROPIC_API_KEY (fable), GEMINI_API_KEY, OPENAI_API_KEY (judge),
     SCRATCH (fulldoc_contexts.json location)
"""

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import tiktoken
from openai import OpenAI

SCRATCH = os.environ["SCRATCH"]
OUT = os.path.join(SCRATCH, "fable5_abstract")
os.makedirs(OUT, exist_ok=True)

QIDS = {
    "2": "gold-ambiguous", "17": "gold-ambiguous", "23": "gold-ambiguous",
    "55": "gold-ambiguous", "66": "gold-ambiguous", "73": "gold-ambiguous",
    "107": "gold-ambiguous", "118": "gold-ambiguous", "135": "gold-ambiguous",
    "141": "gold-ambiguous",
    "5": "needs-computation", "14": "needs-computation", "31": "needs-computation",
    "133": "needs-computation", "144": "needs-computation",
    "76": "judge-artifact", "145": "judge-artifact",
}

fd = json.load(open(os.path.join(SCRATCH, "fulldoc_contexts.json")))
control = json.load(open("financebench/results_gemini35fl_fulldoc/control.json"))
meta = {str(r["question_id"]): r for r in control}

enc = tiktoken.get_encoding("o200k_base")

EXTRACT_SYSTEM = """You are an expert financial-document compressor. Given a question and a full SEC filing (plain text), write a compact ABSTRACTIVE brief that would let a small reader model answer the question.

You may (and should):
- Reorganize and rewrite freely: normalize tables into clearly labeled lines ("Consolidated Balance Sheet, June 30 2023, in $ millions: Total current assets = 15,754"), spell out units, periods, and segment names.
- Include every figure relevant to the question, and for questions that require arithmetic, place all component figures adjacent with unambiguous labels.
- Note relevant qualitative statements (paraphrased is fine) with their section of origin.

You MUST NOT:
- State, imply, or hint at the final answer to the question.
- Perform any arithmetic (no ratios, differences, growth rates, percentages you computed yourself) — present raw reported figures only.
- Editorialize or recommend.

Budget: aim for under 1500 tokens. Output ONLY the brief. Nothing else."""


def extract_one(qid):
    path = os.path.join(OUT, f"extract_{qid}.json")
    if os.path.exists(path):
        return json.load(open(path))
    client = anthropic.Anthropic()
    doc = fd[qid]
    q = meta[qid]["question"]
    user = f"Question: {q}\n\nDocument:\n{doc}"
    for attempt in range(5):
        try:
            with client.messages.stream(
                model="claude-fable-5",
                max_tokens=16000,
                output_config={"effort": "max"},
                system=EXTRACT_SYSTEM,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                resp = stream.get_final_message()
            if resp.stop_reason == "refusal":
                raise RuntimeError("refusal")
            text = next(b.text for b in resp.content if b.type == "text")
            rec = {"qid": qid, "extract": text,
                   "usage": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}}
            json.dump(rec, open(path, "w"))
            return rec
        except anthropic.RateLimitError:
            time.sleep(30 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(15 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"extract failed for {qid}")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def validate(qid, extract):
    doc_norm = norm(fd[qid])
    lines = [l for l in extract.split("\n") if l.strip() and l.strip() != "..."]
    bad = [l for l in lines if norm(l) not in doc_norm]
    return {"n_lines": len(lines), "n_nonverbatim": len(bad),
            "bad_lines": bad[:5],
            "purity": 1 - len(bad) / max(1, len(lines))}


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "financebench"))
import config as fb_config  # noqa: E402
from evaluate import JUDGE_SYSTEM_PROMPT  # noqa: E402


def answer_and_judge(qid, extract):
    gclient = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    q = meta[qid]["question"]
    gold = meta[qid]["gold_answer"]
    msgs = [{"role": "system", "content": fb_config.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{extract}\n\nQuestion: {q}"}]
    ans = None
    for attempt in range(5):
        try:
            r = gclient.chat.completions.create(model="gemini-3.5-flash-lite",
                                                messages=msgs, max_completion_tokens=8000)
            ans = (r.choices[0].message.content or "").strip()
            if ans:
                break
        except Exception:
            time.sleep(10 * (attempt + 1))
    if not ans:
        return {"model_answer": "", "correct": False, "judge": "NO ANSWER"}
    jclient = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    jm = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
          {"role": "user", "content": f"Question: {q}\n\nGold Answer: {gold}\n\nModel Answer: {ans}"}]
    jr = jclient.chat.completions.create(model="gpt-5-mini", messages=jm,
                                         max_completion_tokens=2000)
    jt = jr.choices[0].message.content.strip()
    return {"model_answer": ans, "correct": jt.upper().startswith("CORRECT"), "judge": jt}


def main():
    lock = threading.Lock()
    results = {}

    def work(qid):
        rec = extract_one(qid)
        val = validate(qid, rec["extract"])
        aj = answer_and_judge(qid, rec["extract"])
        row = {
            "qid": qid, "class": QIDS[qid],
            "question": meta[qid]["question"], "gold": meta[qid]["gold_answer"],
            "extract_tokens": len(enc.encode(rec["extract"])),
            "doc_tokens": len(enc.encode(fd[qid])),
            **val, **aj,
        }
        row["retention"] = row["extract_tokens"] / row["doc_tokens"]
        with lock:
            results[qid] = row
            json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
        print(f"[{qid}] {QIDS[qid]:17s} purity={val['purity']:.2f} "
              f"ret={row['retention']:.3f} correct={aj['correct']}", flush=True)

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(work, sorted(QIDS, key=int)))

    flips = [q for q, r in results.items() if r["correct"]]
    print(f"\nFLIPPED {len(flips)}/17: {sorted(flips, key=int)}")
    by = {}
    for q, r in results.items():
        by.setdefault(r["class"], []).append(r["correct"])
    for c, v in sorted(by.items()):
        print(f"  {c}: {sum(v)}/{len(v)}")


if __name__ == "__main__":
    main()
