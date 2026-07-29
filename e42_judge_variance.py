#!/usr/bin/env python3
"""E42: variance decomposition — judge noise vs reader noise.

Re-judges SAVED answers (no new reader calls) to split the observed 6-11%
run-to-run verdict flips into components:
  judge-draw variance : same answer, same judge model, 3 samples
  judge-model bias    : same answer, different judge models
  reader variance     : rep-to-rep flips AFTER majority-vote judging

Outputs per config: single-judge accuracies as recorded, majority-vote(3x
gpt-5-mini) accuracies per rep, cross-judge accuracies, and the decomposition.

Usage: .venv/bin/python e42_judge_variance.py   (keys: OPENAI_API_KEY env,
ANTHROPIC_API_KEY env for the sonnet judge)
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

JUDGE_SYSTEM = open("financebench/evaluate.py").read().split('JUDGE_SYSTEM_PROMPT = """\\\n')[1].split('"""')[0]

CONFIGS = {
    "control": "control.json",
    "dscale-safetab@0.3": "fd-dscale-financial-0.5M-s1-safetab--0.3.json",
    "fincoldfull-safetab@0.5": "fd-fincoldfull-safetab--0.5.json",
}
REPS = ["results_gemini35fl_fulldoc", "results_gemini35fl_fulldoc_r2", "results_gemini35fl_fulldoc_r3"]

oai = OpenAI()
lock = threading.Lock()
CACHE_PATH = "financebench/e42_judge_cache.json"
cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def judge_openai(model, question, gold, answer, draw):
    key = f"{model}|{draw}|{hash((question, gold, answer)) & 0xffffffffffff:x}"
    if key in cache:
        return cache[key]
    msg = f"Question: {question}\n\nGold Answer: {gold}\n\nModel Answer: {answer}"
    r = oai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": msg}],
        max_completion_tokens=2000,
    )
    v = r.choices[0].message.content.strip().upper().startswith("CORRECT")
    with lock:
        cache[key] = v
        if len(cache) % 200 == 0:
            json.dump(cache, open(CACHE_PATH, "w"))
    return v


def judge_anthropic(question, gold, answer):
    import anthropic
    key = f"sonnet5|0|{hash((question, gold, answer)) & 0xffffffffffff:x}"
    if key in cache:
        return cache[key]
    ac = anthropic.Anthropic()
    msg = f"Question: {question}\n\nGold Answer: {gold}\n\nModel Answer: {answer}"
    resp = ac.messages.create(model="claude-sonnet-5", max_tokens=2000,
                              system=JUDGE_SYSTEM,
                              messages=[{"role": "user", "content": msg}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    v = text.strip().upper().startswith("CORRECT")
    with lock:
        cache[key] = v
        if len(cache) % 200 == 0:
            json.dump(cache, open(CACHE_PATH, "w"))
    return v


def load(rep, fname):
    return {str(r["question_id"]): r for r in json.load(open(f"financebench/{rep}/{fname}"))}


def main():
    ex = ThreadPoolExecutor(16)
    for cname, fname in CONFIGS.items():
        reps = []
        for rep in REPS:
            try:
                reps.append(load(rep, fname))
            except FileNotFoundError:
                pass
        if len(reps) < 3:
            print(f"{cname}: only {len(reps)} reps on disk, skipping decomposition rows that need 3")
        qids = sorted(set(reps[0]) & set(reps[-1]), key=int) if reps else []

        # judge-draw variance + majority vote, per rep
        maj_acc, draw_flip_rates = [], []
        for ri, rep in enumerate(reps):
            futs = {}
            for q in qids:
                r = rep[q]
                for d in range(3):
                    futs[(q, d)] = ex.submit(judge_openai, "gpt-5-mini",
                                             r["question"], r["gold_answer"],
                                             r["model_answer"] or "", d)
            votes = {q: [futs[(q, d)].result() for d in range(3)] for q in qids}
            n_unstable = sum(1 for q in qids if len(set(votes[q])) > 1)
            draw_flip_rates.append(n_unstable / len(qids))
            maj = {q: sum(votes[q]) >= 2 for q in qids}
            maj_acc.append(100 * sum(maj.values()) / len(qids))
            if ri == 0:
                maj0 = maj
                # cross-judge on rep 0
                g52 = {q: ex.submit(judge_openai, "gpt-5.2", rep[q]["question"],
                                    rep[q]["gold_answer"], rep[q]["model_answer"] or "", 0)
                       for q in qids}
                son = {q: ex.submit(judge_anthropic, rep[q]["question"],
                                    rep[q]["gold_answer"], rep[q]["model_answer"] or "")
                       for q in qids}
                g52v = {q: f.result() for q, f in g52.items()}
                sonv = {q: f.result() for q, f in son.items()}
            if ri > 0:
                pass
        # reader variance: rep-to-rep flips under majority judging
        maj_all = []
        for rep in reps:
            votes = {q: [judge_openai("gpt-5-mini", rep[q]["question"], rep[q]["gold_answer"],
                                      rep[q]["model_answer"] or "", d) for d in range(3)]
                     for q in qids}
            maj_all.append({q: sum(votes[q]) >= 2 for q in qids})
        reader_flips = sum(1 for q in qids
                           if len(set(m[q] for m in maj_all)) > 1) / len(qids)
        recorded = [100 * sum(1 for q in qids if reps[i][q].get("correct") is True) / len(qids)
                    for i in range(len(reps))]
        print(f"\n=== {cname} (n={len(qids)}) ===")
        print(f"recorded single-judge accs : {[round(a,1) for a in recorded]}")
        print(f"majority-3 accs            : {[round(a,1) for a in maj_acc]}")
        print(f"judge-DRAW instability     : {[round(100*r,1) for r in draw_flip_rates]}% of questions (same answer, 3 draws)")
        print(f"judge gpt-5.2 acc (rep1)   : {100*sum(g52v.values())/len(qids):.1f}  agree with 5-mini-maj: {100*sum(1 for q in qids if g52v[q]==maj0[q])/len(qids):.1f}%")
        print(f"judge sonnet-5 acc (rep1)  : {100*sum(sonv.values())/len(qids):.1f}  agree with 5-mini-maj: {100*sum(1 for q in qids if sonv[q]==maj0[q])/len(qids):.1f}%")
        print(f"READER variance (maj-judged rep flips): {100*reader_flips:.1f}% of questions")
    json.dump(cache, open(CACHE_PATH, "w"))


if __name__ == "__main__":
    main()
