#!/usr/bin/env python3
"""E55: merge the E54 search passes into the final label artifact.

Inputs (SCRATCH/e54/): labels.jsonl + labels_deep.jsonl (primary pass, deep-resumed),
labels_o2.jsonl / labels_o3.jsonl (alternate removal orders -> alternate minimal
sets), leakage.jsonl (reader answers with no context at all).

Output labels_final.jsonl, per item:
  keep_words     primary minimal set (deep pass)
  alt_keep_words words kept by an alternate pass but not primary — interchangeable
                 evidence; labeled 0.6 (kept at natural tau 0.5, ranked below core)
  breaker_lines  union of verified breakers across passes
  leaked         reader answered from prior knowledge -> loss down-weighted
  status         from the primary pass
"""

import json
import os
from pathlib import Path

SCRATCH = Path(os.environ.get("SCRATCH", "."))
D = SCRATCH / "e54"


def load(name):
    p = D / name
    if not p.exists():
        return {}
    return {r["qa_id"]: r for l in open(p) for r in [json.loads(l)]}


def main():
    base = load("labels.jsonl")
    base.update(load("labels_deep.jsonl"))          # deep pass overrides
    alts = [load("labels_o2.jsonl"), load("labels_o3.jsonl")]
    leak = load("leakage.jsonl")

    out = D / "labels_final.jsonl"
    n_ok = n_alt_words = n_leak = 0
    with open(out, "w") as f:
        for qa_id, r in base.items():
            rec = {"qa_id": qa_id, "status": r["status"],
                   "leaked": bool(leak.get(qa_id, {}).get("leaked"))}
            if r["status"].startswith(("ok", "v10_insufficient")):
                keep = set(r["keep_words"])
                breakers = set(r.get("breaker_lines", []))
                alt = set()
                for a in alts:
                    ar = a.get(qa_id)
                    if ar and ar["status"].startswith(("ok", "v10_insufficient")):
                        alt |= set(ar["keep_words"]) - keep
                        breakers |= set(ar.get("breaker_lines", []))
                rec.update(keep_words=sorted(keep), alt_keep_words=sorted(alt),
                           breaker_lines=sorted(breakers),
                           keep_frac_v10=r["keep_frac_v10"],
                           keep_frac_final=r["keep_frac_final"])
                n_ok += 1
                n_alt_words += len(alt)
                n_leak += rec["leaked"]
            f.write(json.dumps(rec) + "\n")
    print(f"labels_final.jsonl: {len(base)} items, {n_ok} labeled, "
          f"{n_alt_words/max(n_ok,1):.1f} alt words/item, {n_leak} leaked")


if __name__ == "__main__":
    main()
