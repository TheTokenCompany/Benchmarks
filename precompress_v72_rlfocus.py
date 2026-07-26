#!/usr/bin/env python3
"""bear-v7.2-rl-focus on FinanceBench: standalone and OR-fused with v7.1.

Per Otso's spec: no serving harness around the model — input is
[CLS] query [SEP] chunk [SEP], keep-probs are the raw token logits
(word min-pooled like every other bear), thresholded directly. No
sentence-based scaffolding.

Outputs (aggr 0.3/0.5/0.7 each):
  bear-v7.2-rlf--{a}              standalone, plain
  bear-v7.2-rlf-safetab--{a}      standalone + number/period/row-label rules
  bear-v71x72rlf--{a}             keep if v7.1 >= a OR v7.2-rl-focus >= a
  bear-v71x72rlf-safetab--{a}     fused + rules

Both checkpoints must share a tokenizer (asserted at runtime), so both
models score the same word structure directly.

Usage:
  modal run precompress_v72_rlfocus.py
"""

import json
import os

import modal

V71 = "bear-v7.1-rl-max-v2"
V72F = "bear-v7.2-rl-focus"
AGGRESSIVENESS_LEVELS = [0.3, 0.5, 0.7]

app = modal.App("financebench-precompress-v72rlf")
volume = modal.Volume.from_name("compression-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "safetensors",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, "financebench", "compression_cache")


@app.function(
    image=image,
    gpu="B200",
    volumes={"/models": volume},
    timeout=3600,
    memory=49152,
)
def compress_all(texts: list[str], questions: list[str], text_ids: list[str],
                 prefix: str = "") -> dict:
    import re
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(f"/models/{V71}")
    tok72 = AutoTokenizer.from_pretrained(f"/models/{V72F}")
    assert tok.get_vocab() == tok72.get_vocab(), "tokenizers differ; word alignment invalid"
    m71 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V71}", attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).eval().to(device)
    m72 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V72F}", attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).eval().to(device)

    SEQ71 = 512   # v7.1 trained at 512
    SEQ72 = 2048  # the RL focus models are 2048-ctx
    num_re = re.compile(r"\d")
    sym_set = {"$", "%", "€", "£"}
    period_re = re.compile(
        r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
        r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
        r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)

    def word_str(toks):
        return tok.convert_tokens_to_string(toks).strip()

    def is_protected(s):
        return bool(num_re.search(s)) or s in sym_set

    def batch_probs(model, chunk_lists, chunk_maps, prefix_ids):
        """min-pooled word probs for chunks, each wrapped [CLS](+prefix)[...][SEP]."""
        probs_out = {}
        with torch.no_grad():
            for b0 in range(0, len(chunk_lists), 96):
                batch = chunk_lists[b0:b0 + 96]
                maxlen = max(len(c) for c in batch) + len(prefix_ids) + 2
                seqs, attn = [], []
                for c in batch:
                    s = ([tok.cls_token_id] + prefix_ids
                         + tok.convert_tokens_to_ids(c) + [tok.sep_token_id])
                    seqs.append(s + [tok.pad_token_id] * (maxlen - len(s)))
                    attn.append([1] * len(s) + [0] * (maxlen - len(s)))
                seqs = torch.tensor(seqs, device=device)
                attn = torch.tensor(attn, device=device)
                logits = model(input_ids=seqs, attention_mask=attn).logits
                pr = F.softmax(logits.float(), dim=-1)[:, :, 1]
                off = 1 + len(prefix_ids)
                for j, c in enumerate(batch):
                    p = pr[j, off:off + len(c)].tolist()
                    for pk, key in zip(p, chunk_maps[b0 + j]):
                        probs_out[key] = min(probs_out.get(key, 1.0), pk)
        return probs_out

    t0 = time.time()
    out = {}
    for qid, question, text in zip(text_ids, questions, texts):
        # word structure per line (shared tokenizer)
        text_lines = text.split("\n")
        line_words = []
        for li, line in enumerate(text_lines):
            payload = line + ("\n" if li < len(text_lines) - 1 else "")
            ids = tok(payload, add_special_tokens=False)["input_ids"]
            toks = tok.convert_ids_to_tokens(ids) if ids else []
            words, cur = [], []
            for tk in toks:
                if (tk.startswith("▁") or tk.startswith("Ġ") or not cur) and cur:
                    words.append(cur)
                    cur = []
                cur.append(tk)
            if cur:
                words.append(cur)
            line_words.append(words)

        def pack(budget):
            chunks, maps = [], []
            cur_t, cur_m = [], []
            for li, words in enumerate(line_words):
                for wi, w in enumerate(words):
                    if len(cur_t) + len(w) > budget and cur_t:
                        chunks.append(cur_t); maps.append(cur_m)
                        cur_t, cur_m = [], []
                    cur_t.extend(w)
                    cur_m.extend([(li, wi)] * len(w))
            if cur_t:
                chunks.append(cur_t); maps.append(cur_m)
            return chunks, maps

        # v7.1: no prefix, budget 510
        c71, m71map = pack(SEQ71 - 2)
        p71 = batch_probs(m71, c71, m71map, [])
        # v7.2-rl-focus: query prefix, 2048-token windows
        q_ids = tok(question, add_special_tokens=False)["input_ids"][:128]
        c72, m72map = pack(SEQ72 - len(q_ids) - 3)
        p72 = batch_probs(m72, c72, m72map, q_ids + [tok.sep_token_id])

        for aggr in AGGRESSIVENESS_LEVELS:
            for mode in ("solo", "fused"):
                for rules in (False, True):
                    name = ("bear-v7.2-rlf" if mode == "solo" else "bear-v71x72rlf") \
                        + ("-safetab" if rules else "") + f"--{aggr}"
                    kept_line_texts = []
                    orig_n = comp_n = 0
                    for li, words in enumerate(line_words):
                        strs = [word_str(w) for w in words]
                        numeric = [is_protected(s) for s in strs]
                        keep = []
                        for wi in range(len(words)):
                            k72 = p72.get((li, wi), 0.0) >= aggr
                            k71 = p71.get((li, wi), 0.0) >= aggr
                            keep.append(k72 if mode == "solo" else (k71 or k72))
                        if rules:
                            for wi, s in enumerate(strs):
                                if numeric[wi] or period_re.match(s):
                                    keep[wi] = True
                            if sum(numeric) >= 3 and any(
                                    keep[wi] for wi, n in enumerate(numeric) if n):
                                first_num = numeric.index(True)
                                for k in range(0, min(first_num, 12)):
                                    keep[k] = True
                        orig_n += sum(len(w) for w in words)
                        kept_w = [w for w, k in zip(words, keep) if k]
                        comp_n += sum(len(w) for w in kept_w)
                        if kept_w:
                            kept_line_texts.append(tok.convert_tokens_to_string(
                                [t for w in kept_w for t in w]))
                    out.setdefault(prefix + name, {})[qid] = {
                        "compressed_text": "\n".join(kept_line_texts),
                        "original_tokens": max(1, orig_n),
                        "compressed_tokens": min(comp_n, max(1, orig_n)),
                    }

    print(f"Dual-model inference done in {time.time() - t0:.1f}s on "
          f"{torch.cuda.get_device_name()}")
    return out


@app.local_entrypoint()
def main():
    from datasets import load_dataset

    dataset = load_dataset("PatronusAI/financebench", split="train")
    items = list(dataset)[:150]

    texts, questions, text_ids = [], [], []
    for i, item in enumerate(items):
        pages = [
            ev.get("evidence_text_full_page", "").strip()
            for ev in item["evidence"]
            if ev.get("evidence_text_full_page")
        ]
        texts.append("\n\n---\n\n".join(pages))
        questions.append(item["question"])
        text_ids.append(item.get("question_id", str(i)))

    prefix = os.getenv("ALIAS_PREFIX", "")
    if os.getenv("FULLDOC_JSON"):
        fulldoc = json.load(open(os.environ["FULLDOC_JSON"]))
        texts = [fulldoc[q] for q in text_ids]
        print(f"Full-document mode ({len(texts)} filings)")

    print(f"{V72F}: standalone + fused with {V71}, raw word-level logits...")
    result = compress_all.remote(texts, questions, text_ids, prefix)

    os.makedirs(CACHE_DIR, exist_ok=True)
    for config_name, entry in result.items():
        path = os.path.join(CACHE_DIR, f"{config_name}.json")
        with open(path, "w") as f:
            json.dump(entry, f)
        ratios = [
            v["compressed_tokens"] / v["original_tokens"]
            for v in entry.values() if v["original_tokens"]
        ]
        print(f"  {config_name}: mean retention {sum(ratios)/len(ratios):.3f}")
