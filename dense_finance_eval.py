#!/usr/bin/env python3
"""Dense-context finance eval: TAT-QA + FinQA dev, raw vs bear-compressed.

FinanceBench contexts are long filing pages where a compressor can drop a lot of
prose before it touches the evidence. TAT-QA and FinQA are the opposite regime:
one table plus a few sentences, ~1-2k tokens, almost every line load-bearing.
This harness measures the same compressors there.

  ./.venv/bin/python dense_finance_eval.py \
      --model-vol-path exp-20260728-011924-v11b-warm-w2140s2 --alias v11b

Compression runs on Modal via v8_eval_precompress.compress_all; answers come
from an OpenAI-protocol reader (gemini-3.5-flash-lite by default).
"""

import argparse
import json
import os
import re
import random
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import v8_eval_precompress

SCRATCH = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-"
           "8144834a-c76f-41f6-b409-cb03154f2355-financebench-check/"
           "3224cbfc-fe21-4c45-9714-a9878f8f7d10/scratchpad/dense_eval")

SOURCES = {
    "tatqa": ("tatqa_dataset_dev.json",
              "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/"
              "dataset_raw/tatqa_dataset_dev.json"),
    "finqa": ("finqa_dev.json",
              "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json"),
}

SYSTEM_PROMPT = ("You answer questions about financial documents. Use ONLY the "
                 "provided context. Reply with the answer alone — a number or "
                 "short phrase.")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def fetch(dataset):
    fname, url = SOURCES[dataset]
    path = os.path.join(SCRATCH, fname)
    if not os.path.exists(path):
        os.makedirs(SCRATCH, exist_ok=True)
        print(f"downloading {dataset} -> {path}")
        with urllib.request.urlopen(url) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path) as f:
        return json.load(f)


def render_table(rows):
    """Row per line, cells space-separated in column order.

    Empty cells collapse away rather than emitting runs of spaces: the column
    padding buys nothing once the text is linearized, and the compressor's
    is_table heuristic counts numeric words per line either way.
    """
    out = []
    for row in rows:
        line = re.sub(r"\s+", " ", " ".join(str(c) for c in row)).strip()
        if line:
            out.append(line)
    return out


def as_number(x):
    """float(x) for the many shapes a gold answer takes, else None."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def load_tatqa(limit_pool=None):
    """One item per numeric-answer question; context = paragraphs then table."""
    items = []
    for doc in fetch("tatqa"):
        paras = [p["text"].strip()
                 for p in sorted(doc["paragraphs"], key=lambda p: p["order"])
                 if p["text"].strip()]
        ctx = "\n".join(paras + [""] + render_table(doc["table"]["table"]))
        for q in doc["questions"]:
            if q["answer_type"] not in ("arithmetic", "count", "span"):
                continue
            ans = q["answer"]
            if isinstance(ans, list):
                if len(ans) != 1:
                    continue
                ans = ans[0]
            if as_number(ans) is None:
                continue
            items.append({
                "id": q["uid"], "question": q["question"].strip(),
                "gold": str(ans), "context": ctx, "scale": q.get("scale", ""),
            })
    return items


def load_finqa(limit_pool=None):
    """One item per question; context = pre_text, table, post_text (doc order)."""
    items = []
    for it in fetch("finqa"):
        qa = it["qa"]
        gold = as_number(qa.get("exe_ans"))
        if gold is None:
            continue
        lines = [t.strip() for t in it.get("pre_text", []) if t.strip()]
        lines += render_table(it.get("table") or [])
        lines += [t.strip() for t in it.get("post_text", []) if t.strip()]
        items.append({
            "id": it["id"], "question": qa["question"].strip(),
            "gold": repr(gold), "context": "\n".join(lines), "scale": "",
        })
    return items


def sample(items, n, seed=42):
    """Deterministic subsample; stable order so resume files stay aligned."""
    items = sorted(items, key=lambda x: x["id"])
    if n is None or n >= len(items):
        return items
    idx = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in idx]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
NUM_IN_TEXT = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")

# A gold in millions answered in units, a FinQA ratio answered as a percent, and
# so on. Tolerating the scale is the point: we are measuring whether the
# evidence survived compression, not whether the reader picked our unit.
MULTIPLIERS = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100.0, 0.01)
REL_TOL = 0.02


def numbers_in(text):
    out = []
    for m in NUM_IN_TEXT.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def numeric_match(gold, reply):
    """True if any number in `reply` matches `gold` within 2% at some scale.

    Sign-insensitive: readers phrase a negative as "a decrease of 22.22%" about
    as often as "-22.22", and the leniency applies to every arm equally.
    """
    cands = numbers_in(reply)
    if not cands:
        return False
    for m in MULTIPLIERS:
        target = abs(gold * m)
        for c in cands:
            c = abs(c)
            if target == 0.0:
                if c == 0.0:
                    return True
            elif abs(c - target) <= REL_TOL * target:
                return True
    return False


def score(gold_str, reply):
    gold = as_number(gold_str)
    if gold is not None:
        return numeric_match(gold, reply)
    g = re.sub(r"\s+", " ", gold_str.lower()).strip()
    return bool(g) and g in re.sub(r"\s+", " ", (reply or "").lower())


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #
def expected_configs(alias, variants, aggr_levels):
    """Config names compress_all will emit, mirroring its own naming.

    Budget/linebudget/dose variants only run at aggr 0.5 -- everything else is
    skipped inside the Modal function, so asking for them at other levels is
    not an error, it just yields nothing.
    """
    names = []
    for aggr in aggr_levels:
        for v in variants:
            gated = v.startswith("budget") or v.startswith("linebudget")
            if gated and aggr != 0.5:
                continue
            suffix = "" if v == "plain" else f"-{v}"
            names.append(f"{alias}{suffix}--{aggr}")
    return names


def compress(dataset, items, args, cache_path):
    """{config_name: {id: entry}} for `items`, filling the cache on miss."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    want = expected_configs(args.alias, args.variants, args.aggr_levels)
    ids = [it["id"] for it in items]
    missing = [c for c in want
               if c not in cache or any(i not in cache[c] for i in ids)]
    if not missing:
        print(f"[{dataset}] compression cache hit ({len(want)} configs, {len(ids)} items)")
        return {c: cache[c] for c in want}

    print(f"[{dataset}] compressing {len(ids)} items on Modal "
          f"(missing {len(missing)}/{len(want)} configs)...")
    # Everything the Modal container needs travels as an argument: env vars set
    # here are read at import time locally and never reach the container.
    with v8_eval_precompress.app.run():
        result = v8_eval_precompress.compress_all.remote(
            [it["context"] for it in items],
            [it["question"] for it in items],
            ids,
            args.model_vol_path,
            args.alias,
            2048,
            args.variants,
            False,
            1000,
            False,
            None,
            args.aggr_levels,
        )
    result.pop("__probs_hist__", None)
    result.pop("__evidence_ranks__", None)

    for cfg, entry in result.items():
        cache.setdefault(cfg, {}).update(entry)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"[{dataset}] cache -> {cache_path}")
    return {c: cache[c] for c in want if c in cache}


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
def make_client(args):
    from openai import OpenAI
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set")
    return OpenAI(api_key=key, base_url=args.base_url)


def ask(client, args, ctx, question):
    last = None
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
                ],
                temperature=0,
                max_completion_tokens=4000,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - retry anything the endpoint throws
            last = e
            if attempt < 3:
                import time
                time.sleep(2 ** attempt)
    return f"ERROR: {last}"


def run_arm(dataset, arm, items, comp, client, args):
    """Answer + score one arm, resuming from disk."""
    path = os.path.join(SCRATCH,
                        f"results_{args.alias}_{dataset}_{arm.replace('/', '_')}.json")
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    done = {r["id"] for r in results}

    entries = comp.get(arm) if arm != "raw" else None
    todo = []
    for it in items:
        if it["id"] in done:
            continue
        if entries is not None:
            e = entries.get(it["id"])
            if e is None:
                continue
            ctx, info = e["compressed_text"], e
        else:
            ctx, info = it["context"], None
        todo.append((it, ctx, info))

    if todo:
        lock = threading.Lock()

        def work(it, ctx, info):
            reply = ask(client, args, ctx, it["question"])
            row = {"id": it["id"], "question": it["question"], "gold": it["gold"],
                   "reply": reply, "correct": score(it["gold"], reply)}
            if info is not None:
                row["original_tokens"] = info["original_tokens"]
                row["compressed_tokens"] = info["compressed_tokens"]
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, *t) for t in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                with lock:
                    results.append(row)
                    if i % 20 == 0 or i == len(futs):
                        with open(path, "w") as f:
                            json.dump(results, f, indent=2)
                        print(f"  [{dataset}/{arm}] {i}/{len(futs)}")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    keep = {it["id"] for it in items}
    results = [r for r in results if r["id"] in keep]
    n = len(results)
    acc = sum(1 for r in results if r["correct"]) / n if n else 0.0
    rets = [r["compressed_tokens"] / r["original_tokens"]
            for r in results if r.get("original_tokens")]
    ret = sum(rets) / len(rets) if rets else 1.0
    return {"arm": arm, "n": n, "acc": acc, "ret": ret}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-vol-path", required=True,
                    help="checkpoint dir on the otso-v8-training Modal volume")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--variants", default="plain,budget33,budget66")
    ap.add_argument("--aggr-levels", default="0.3,0.5,0.7")
    ap.add_argument("--datasets", default="tatqa,finqa")
    ap.add_argument("--n", type=int, default=150, help="QA per dataset")
    ap.add_argument("--limit", type=int, default=None,
                    help="override --n, for smoke runs")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--base-url",
                    default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args()

    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.aggr_levels = [float(a) for a in args.aggr_levels.split(",") if a.strip()]
    n = args.limit if args.limit is not None else args.n
    os.makedirs(SCRATCH, exist_ok=True)

    client = make_client(args)
    loaders = {"tatqa": load_tatqa, "finqa": load_finqa}
    summary = {}

    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        items = sample(loaders[dataset](), n)
        chars = sum(len(it["context"]) for it in items) / max(len(items), 1)
        print(f"\n=== {dataset}: {len(items)} items, mean context {chars:.0f} chars ===")

        comp = compress(dataset, items, args,
                        os.path.join(SCRATCH, f"cache_{args.alias}_{dataset}.json"))
        arms = ([] if args.skip_raw else ["raw"]) + sorted(comp)
        summary[dataset] = [run_arm(dataset, a, items, comp, client, args)
                            for a in arms]

    print(f"\n{'=' * 68}")
    print(f"  Dense finance eval — {args.alias} @ {args.model}")
    print(f"{'=' * 68}")
    for dataset, rows in summary.items():
        print(f"\n{dataset}")
        print(f"  {'arm':<28} {'acc':>7} {'retention':>10} {'n':>5}")
        for r in rows:
            print(f"  {r['arm']:<28} {r['acc']:>6.1%} {r['ret']:>10.3f} {r['n']:>5}")
    print()


if __name__ == "__main__":
    main()
