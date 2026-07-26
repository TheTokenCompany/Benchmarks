#!/usr/bin/env python3
"""FinanceBench pre-compression with bear-4.0-focus-1-ctx2048 on Modal B200.

Faithful port of the standalone FocusExtractor runner (per-token p_keep ->
sentence-mean -> tau threshold), tiled into 2048-token windows with the query
re-prefixed per window. The token budget cap is disabled (their default 150 is
sized for RAG chunks, not filing pages) so tau is the only operating knob;
tau is recorded as the config's "aggressiveness" for harness compatibility.

Usage:
  modal run precompress_v40_focus.py
"""

import json
import os

import modal

MODEL_NAME = "bear-4.0-focus-1-ctx2048"
MODEL_ALIAS = "bear-4.0-focus-1"
TAUS = [0.01, 0.03, 0.05, 0.1, 0.15]

app = modal.App("financebench-precompress-v40focus")
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
    memory=32768,
)
def compress_all(texts: list[str], questions: list[str], text_ids: list[str],
                 prefix: str = "") -> dict:
    import re
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    MAX_LEN, FOCUS_BUDGET, KEEP_LABEL = 2048, 64, 1
    _SENT = re.compile(r'[.!?]+["\')\]]*(?=\s|$)|\n{2,}')

    def _sentences(text):
        out, start = [], 0
        for m in _SENT.finditer(text):
            if text[start:m.end()].strip():
                out.append((start, m.end(), text[start:m.end()]))
            start = m.end()
        if start < len(text) and text[start:].strip():
            out.append((start, len(text), text[start:]))
        return out

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(f"/models/{MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        f"/models/{MODEL_NAME}", num_labels=2).eval().to(device)

    @torch.no_grad()
    def token_probs(query, context):
        fids = tok(query, add_special_tokens=False)["input_ids"][:FOCUS_BUDGET]
        enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if not ids or not fids:
            return [], []
        cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
        budget = MAX_LEN - len(fids) - 3
        windows = [ids[w0:w0 + budget] for w0 in range(0, len(ids), budget)]
        probs = []
        n_pre = 2 + len(fids)
        for b0 in range(0, len(windows), 32):
            batch = windows[b0:b0 + 32]
            maxlen = max(len(w) for w in batch) + len(fids) + 3
            seqs, attn = [], []
            for w in batch:
                s = [cls] + fids + [sep] + w + [sep]
                seqs.append(s + [pad] * (maxlen - len(s)))
                attn.append([1] * len(s) + [0] * (maxlen - len(s)))
            seqs = torch.tensor(seqs, device=device)
            attn = torch.tensor(attn, device=device)
            logits = model(input_ids=seqs, attention_mask=attn).logits.float()
            p = F.softmax(logits, dim=-1)[:, :, KEEP_LABEL]
            for j, w in enumerate(batch):
                probs.extend(p[j, n_pre:n_pre + len(w)].cpu().tolist())
        return offsets, probs

    t0 = time.time()
    # score once per (question, context); derive all taus from the same scores
    per_text = []  # (sents, scores, cnts, n_content_tok)
    for question, context in zip(questions, texts):
        offsets, probs = token_probs(question, context)
        sents = _sentences(context)
        sums, cnts = [0.0] * len(sents), [0] * len(sents)
        for (ts, te), p in zip(offsets, probs):
            if te <= ts:
                continue
            mid = (ts + te) / 2
            for si, (s, e, _) in enumerate(sents):
                if s <= mid < e:
                    sums[si] += p
                    cnts[si] += 1
                    break
        scores = [sums[i] / cnts[i] if cnts[i] else 0.0 for i in range(len(sents))]
        per_text.append((sents, scores, cnts, len(probs)))
    print(f"Inference done in {time.time() - t0:.1f}s on {torch.cuda.get_device_name()}")

    out = {}
    for tau in TAUS:
        config_name = f"{prefix}{MODEL_ALIAS}--{tau}"
        entry = {}
        for qid, (sents, scores, cnts, n_tok) in zip(text_ids, per_text):
            kept = sorted(i for i, sc in enumerate(scores) if sc >= tau)
            used = sum(cnts[i] for i in kept)
            snippet = " … ".join(" ".join(sents[i][2].split()) for i in kept)
            entry[qid] = {
                "compressed_text": snippet,
                "original_tokens": max(1, n_tok),
                "compressed_tokens": min(used, max(1, n_tok)),
            }
        out[config_name] = entry
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

    print(f"Query-aware sentence extraction with {MODEL_NAME}, tau sweep {TAUS}...")
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
