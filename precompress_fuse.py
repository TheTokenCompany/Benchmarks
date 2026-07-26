#!/usr/bin/env python3
"""OR-fusion of bear-v7.1-rl-max-v2 (query-blind RL) and bear-4.0-focus-1-ctx2048
(query-aware) on Modal B200.

Keep a word if EITHER model keeps it at its own operating point:
  v7.1 word keep-prob >= aggressiveness   OR   focus token prob >= TAU (0.01)

Decision-OR rather than score-max because the two models' probability scales
are not calibrated against each other (focus scores are collapsed near zero).
The two checkpoints use different tokenizers, so fusion aligns on character
spans: each v7.1 word carries its char span; a word's focus score is the max
prob of focus tokens whose midpoint falls inside that span.

Variants: fuse (pure OR), fuse-safetab (OR + number/period/row-label rules).
Supports FULLDOC_JSON + ALIAS_PREFIX like the other precompress scripts.

Usage:
  modal run precompress_fuse.py
"""

import json
import os

import modal

V71 = "bear-v7.1-rl-max-v2"
V40 = "bear-4.0-focus-1-ctx2048"
FUSE_ALIAS = "bear-v71xfocus"
AGGRESSIVENESS_LEVELS = [0.3, 0.5, 0.7]
TAU = 0.01
VARIANTS = ["fuse", "fuse-safetab"]

app = modal.App("financebench-precompress-fuse")
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
    tok71 = AutoTokenizer.from_pretrained(f"/models/{V71}")
    m71 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V71}", attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).eval().to(device)
    tok40 = AutoTokenizer.from_pretrained(f"/models/{V40}")
    m40 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V40}", num_labels=2).eval().to(device)

    SEQ71, MAX40, FOCUS_BUDGET = 512, 2048, 64
    num_re = re.compile(r"\d")
    sym_set = {"$", "%", "€", "£"}
    period_re = re.compile(
        r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
        r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
        r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)

    def word_str(toks):
        return tok71.convert_tokens_to_string(toks).strip()

    def is_protected(s):
        return bool(num_re.search(s)) or s in sym_set

    @torch.no_grad()
    def focus_token_probs(query, context):
        """[(char_mid, prob)] over the whole context, batched windows."""
        fids = tok40(query, add_special_tokens=False)["input_ids"][:FOCUS_BUDGET]
        enc = tok40(context, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if not ids or not fids:
            return []
        cls, sep, pad = tok40.cls_token_id, tok40.sep_token_id, tok40.pad_token_id
        budget = MAX40 - len(fids) - 3
        windows = [ids[w0:w0 + budget] for w0 in range(0, len(ids), budget)]
        out = []
        BW = 32
        for b0 in range(0, len(windows), BW):
            batch = windows[b0:b0 + BW]
            maxlen = max(len(w) for w in batch) + len(fids) + 3
            seqs, attn = [], []
            for w in batch:
                s = [cls] + fids + [sep] + w + [sep]
                seqs.append(s + [pad] * (maxlen - len(s)))
                attn.append([1] * len(s) + [0] * (maxlen - len(s)))
            seqs = torch.tensor(seqs, device=device)
            attn = torch.tensor(attn, device=device)
            logits = m40(input_ids=seqs, attention_mask=attn).logits.float()
            probs = F.softmax(logits, dim=-1)[:, :, 1]
            n_pre = 2 + len(fids)
            for j, w in enumerate(batch):
                p = probs[j, n_pre:n_pre + len(w)].cpu().tolist()
                base = (b0 + j) * budget
                for k, pk in enumerate(p):
                    ts, te = offsets[base + k]
                    if te > ts:
                        out.append(((ts + te) / 2, pk))
        return out

    t0 = time.time()
    results_per_text = []  # (line_data, focus_probs)
    for question, text in zip(questions, texts):
        # ---- v7.1 per-line words with char spans ----
        line_data = []  # (words:[toklist], probs:[float], spans:[(s,e)], meta)
        text_lines = text.split("\n")
        char_pos = 0
        chunks, chunk_map = [], []
        cur_toks, cur_map = [], []
        all_words = []  # global list: (line_idx, word_idx)
        for li, line in enumerate(text_lines):
            payload = line + ("\n" if li < len(text_lines) - 1 else "")
            enc = tok71(payload, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            toks = tok71.convert_ids_to_tokens(ids) if ids else []
            words, spans, cur, cur_off = [], [], [], []
            for tk, (ts, te) in zip(toks, offs):
                if (tk.startswith("▁") or tk.startswith("Ġ") or not cur) and cur:
                    words.append(cur)
                    spans.append((char_pos + min(o[0] for o in cur_off),
                                  char_pos + max(o[1] for o in cur_off)))
                    cur, cur_off = [], []
                cur.append(tk)
                cur_off.append((ts, te))
            if cur:
                words.append(cur)
                spans.append((char_pos + min(o[0] for o in cur_off),
                              char_pos + max(o[1] for o in cur_off)))
            strs = [word_str(w) for w in words]
            numeric = [is_protected(s) for s in strs]
            line_data.append({
                "words": words, "spans": spans, "strs": strs, "numeric": numeric,
                "first_num": numeric.index(True) if any(numeric) else None,
                "is_table": sum(numeric) >= 3,
                "probs": [1.0] * len(words),
            })
            for wi, w in enumerate(words):
                if len(cur_toks) + len(w) > SEQ71 - 2 and cur_toks:
                    chunks.append(cur_toks); chunk_map.append(cur_map)
                    cur_toks, cur_map = [], []
                cur_toks.extend(w)
                cur_map.extend([(li, wi)] * len(w))
            char_pos += len(payload)
        if cur_toks:
            chunks.append(cur_toks); chunk_map.append(cur_map)

        with torch.no_grad():
            for b0 in range(0, len(chunks), 128):
                batch = chunks[b0:b0 + 128]
                maxlen = max(len(c) for c in batch) + 2
                seqs, attn = [], []
                for c in batch:
                    s = ([tok71.cls_token_id] + tok71.convert_tokens_to_ids(c)
                         + [tok71.sep_token_id])
                    seqs.append(s + [tok71.pad_token_id] * (maxlen - len(s)))
                    attn.append([1] * len(s) + [0] * (maxlen - len(s)))
                seqs = torch.tensor(seqs, device=device)
                attn = torch.tensor(attn, device=device)
                logits = m71(input_ids=seqs, attention_mask=attn).logits
                probs = F.softmax(logits.float(), dim=-1)[:, :, 1]
                for j, c in enumerate(batch):
                    p = probs[j, 1:1 + len(c)].tolist()
                    for pk, (li, wi) in zip(p, chunk_map[b0 + j]):
                        ld = line_data[li]
                        ld["probs"][wi] = min(ld["probs"][wi], pk)

        # ---- focus probs -> per-word max via char spans (pointer sweep) ----
        fprobs = sorted(focus_token_probs(question, text))
        flat = []
        for li, ld in enumerate(line_data):
            for wi in range(len(ld["words"])):
                flat.append((ld["spans"][wi], li, wi))
        flat.sort()
        fscores = {}
        fp = 0
        for (ws, we), li, wi in flat:
            while fp < len(fprobs) and fprobs[fp][0] < ws:
                fp += 1
            best, k = 0.0, fp
            while k < len(fprobs) and fprobs[k][0] < we:
                best = max(best, fprobs[k][1])
                k += 1
            fscores[(li, wi)] = best
        for li, ld in enumerate(line_data):
            ld["fscore"] = [fscores.get((li, wi), 0.0) for wi in range(len(ld["words"]))]
        results_per_text.append(line_data)

    print(f"Dual-model inference done in {time.time() - t0:.1f}s "
          f"on {torch.cuda.get_device_name()}")

    out = {}
    for aggr in AGGRESSIVENESS_LEVELS:
        for variant in VARIANTS:
            config_name = (f"{prefix}{FUSE_ALIAS}--{aggr}" if variant == "fuse"
                           else f"{prefix}{FUSE_ALIAS}-safetab--{aggr}")
            entry = {}
            for qid, line_data in zip(text_ids, results_per_text):
                kept_line_texts = []
                orig_n = comp_n = 0
                for ld in line_data:
                    n = len(ld["words"])
                    keep = [ld["probs"][i] >= aggr or ld["fscore"][i] >= TAU
                            for i in range(n)]
                    if variant == "fuse-safetab":
                        for wi, s in enumerate(ld["strs"]):
                            if is_protected(s) or period_re.match(s):
                                keep[wi] = True
                        if ld["is_table"] and ld["first_num"] is not None and any(
                                keep[wi] for wi, x in enumerate(ld["numeric"]) if x):
                            for k in range(0, min(ld["first_num"], 12)):
                                keep[k] = True
                    orig_n += sum(len(t) for t in ld["words"])
                    kept = [t for t, k in zip(ld["words"], keep) if k]
                    comp_n += sum(len(t) for t in kept)
                    if kept:
                        kept_line_texts.append(tok71.convert_tokens_to_string(
                            [tok for w in kept for tok in w]))
                entry[qid] = {
                    "compressed_text": "\n".join(kept_line_texts),
                    "original_tokens": orig_n,
                    "compressed_tokens": min(comp_n, orig_n),
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

    print(f"OR-fusion {V71} + {V40} (tau={TAU})...")
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
