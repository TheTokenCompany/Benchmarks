#!/usr/bin/env python3
"""Query-aware pre-compression with taaha's v7.2-focus-rl model on Modal B200.

Input layout per taaha's focus research: [CLS] question [SEP] chunk [SEP],
keep-probs read on chunk tokens only. Emits plain / safenum / safetab variants
(same rules as precompress_v71.py). Includes a focus-swap probe: mean |delta|
of keep-probs when the question is swapped — near-zero means the model is NOT
actually query-conditioned (wrong layout or collapsed head).

Usage:
  modal run precompress_v72_focus.py
"""

import json
import os

import modal

MODEL_VOL = os.getenv("MODEL_VOL", "otso-v8-training")
MODEL_VOL_PATH = os.getenv("MODEL_VOL_PATH", "")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "v8")
AGGRESSIVENESS_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
VARIANTS = ["plain", "safetab", "budget22", "budget22-safetab", "budget33", "budget33-safetab"]

app = modal.App("financebench-precompress-v8eval")
train_volume = modal.Volume.from_name(os.getenv("MODEL_VOL", "otso-v8-training"))

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
    volumes={"/train": train_volume},
    timeout=1800,
    memory=32768,
)
def compress_all(texts: list[str], questions: list[str], text_ids: list[str],
                 model_vol_path: str = None, model_alias: str = None,
                 seq_len_arg: int = 2048) -> dict:
    import re
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    device = "cuda"
    model_vol_path = model_vol_path or MODEL_VOL_PATH
    model_alias = model_alias or MODEL_ALIAS
    model_path = f"/train/{model_vol_path}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    seq_len = seq_len_arg
    num_re = re.compile(r"\d")
    sym_set = {"$", "%", "€", "£"}
    period_re = re.compile(
        r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
        r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
        r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)

    def word_str(toks):
        return tokenizer.convert_tokens_to_string(toks).strip()

    def is_protected(s):
        return bool(num_re.search(s)) or s in sym_set

    def group_words(ids):
        toks = tokenizer.convert_ids_to_tokens(ids)
        words, cur = [], []
        for tok in toks:
            if (tok.startswith("▁") or tok.startswith("Ġ") or not cur) and cur:
                words.append(cur)
                cur = []
            cur.append(tok)
        if cur:
            words.append(cur)
        return words

    def infer_probs(question, line_words_per_text):
        """Given [(ref, words)] for one text, return {(ref, wi): prob} conditioned on question."""
        q_ids = tokenizer(question, add_special_tokens=False)["input_ids"][:96]
        budget = seq_len - len(q_ids) - 3  # CLS + q + SEP + chunk + SEP
        chunks, chunk_map = [], []
        cur_toks, cur_map = [], []
        for ref, words in line_words_per_text:
            for wi, toks in enumerate(words):
                if len(cur_toks) + len(toks) > budget and cur_toks:
                    chunks.append(cur_toks); chunk_map.append(cur_map)
                    cur_toks, cur_map = [], []
                cur_toks.extend(toks)
                cur_map.extend([(ref, wi)] * len(toks))
        if cur_toks:
            chunks.append(cur_toks); chunk_map.append(cur_map)

        probs_out = {}
        for bs in range(0, len(chunks), 64):
            batch = chunks[bs:bs + 64]
            maxlen = max(len(c) for c in batch) + len(q_ids) + 3
            input_ids, attn = [], []
            for c in batch:
                ids = ([tokenizer.cls_token_id] + q_ids + [tokenizer.sep_token_id]
                       + tokenizer.convert_tokens_to_ids(c) + [tokenizer.sep_token_id])
                pad = maxlen - len(ids)
                input_ids.append(ids + [tokenizer.pad_token_id] * pad)
                attn.append([1] * len(ids) + [0] * pad)
            input_ids = torch.tensor(input_ids, device=device)
            attn = torch.tensor(attn, device=device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                probs = F.softmax(logits.float(), dim=-1)[:, :, 1]
            off = len(q_ids) + 2  # position of first chunk token
            for j, c in enumerate(batch):
                p = probs[j, off:off + len(c)].tolist()
                for tk, key in zip(p, chunk_map[bs + j]):
                    probs_out[key] = min(probs_out.get(key, 1.0), tk)
        return probs_out

    t0 = time.time()
    # tokenize lines once per text
    per_text_lines = []  # per text: [(ref, words)]
    line_meta_all = []   # global ref -> meta
    for ti, text in enumerate(texts):
        lines = []
        text_lines = text.split("\n")
        for li, line in enumerate(text_lines):
            payload = line + ("\n" if li < len(text_lines) - 1 else "")
            ids = tokenizer(payload, add_special_tokens=False)["input_ids"]
            words = group_words(ids) if ids else []
            ref = len(line_meta_all)
            strs = [word_str(t) for t in words]
            numeric = [is_protected(s) for s in strs]
            n_per = sum(1 for s in strs if period_re.match(s))
            line_meta_all.append({
                "strs": strs, "numeric": numeric,
                "first_num": numeric.index(True) if any(numeric) else None,
                "is_table": sum(numeric) >= 3, "n_per": n_per,
            })
            lines.append((ref, words))
        per_text_lines.append(lines)

    # per-question conditioned probs
    all_probs = []
    for ti, (question, lines) in enumerate(zip(questions, per_text_lines)):
        all_probs.append(infer_probs(question, lines))

    # focus-swap probe on first 5 texts: probs with the WRONG question
    swap_deltas = []
    for ti in range(min(5, len(texts))):
        wrong_q = questions[(ti + 7) % len(questions)]
        p_wrong = infer_probs(wrong_q, per_text_lines[ti])
        p_right = all_probs[ti]
        common = set(p_right) & set(p_wrong)
        if common:
            swap_deltas.append(sum(abs(p_right[k] - p_wrong[k]) for k in common) / len(common))
    print(f"FOCUS-SWAP PROBE mean|dP| = {sum(swap_deltas)/len(swap_deltas):.4f} "
          f"(near 0 => model ignores the question)")
    print(f"Inference done in {time.time() - t0:.1f}s on {torch.cuda.get_device_name()}")

    out = {}
    for aggr in AGGRESSIVENESS_LEVELS:
        for variant in VARIANTS:
            if variant.startswith("budget") and aggr != 0.5:
                continue
            suffix = "" if variant == "plain" else f"-{variant}"
            config_name = f"{model_alias}{suffix}--{aggr}"
            entry = {}
            for ti, qid in enumerate(text_ids):
                probs = all_probs[ti]
                budget_keep = set()
                if variant.startswith("budget"):
                    bfrac = 0.33 if variant.startswith("budget33") else 0.22
                    allw = sorted(((p, k) for k, p in probs.items()), reverse=True)
                    total_toks = sum(len(w) for _, ws in per_text_lines[ti] for w in ws)
                    budget = int(bfrac * total_toks)
                    used = 0
                    wlen = {}
                    for ref, ws in per_text_lines[ti]:
                        for wi, w in enumerate(ws):
                            wlen[(ref, wi)] = len(w)
                    for p, k in allw:
                        n = wlen.get(k, 1)
                        if used + n > budget and budget_keep:
                            continue
                        budget_keep.add(k)
                        used += n
                        if used >= budget:
                            break
                kept_line_texts = []
                orig_n = comp_n = 0
                for ref, words in per_text_lines[ti]:
                    meta = line_meta_all[ref]
                    strs = meta["strs"]
                    if variant.startswith("budget"):
                        keep = [(ref, wi) in budget_keep for wi in range(len(words))]
                    else:
                        keep = [probs.get((ref, wi), 0.0) >= aggr for wi in range(len(words))]
                    if "safenum" in variant or "safetab" in variant:
                        for wi, s in enumerate(strs):
                            if is_protected(s):
                                keep[wi] = True
                    if "safetab" in variant:
                        for wi, s in enumerate(strs):
                            if period_re.match(s):
                                keep[wi] = True
                        if meta["is_table"] and meta["first_num"] is not None and any(
                                keep[wi] for wi, n in enumerate(meta["numeric"]) if n):
                            for k in range(0, min(meta["first_num"], 12)):
                                keep[k] = True
                    orig_n += sum(len(t) for t in words)
                    kept = [t for t, k in zip(words, keep) if k]
                    comp_n += sum(len(t) for t in kept)
                    if kept:
                        kept_line_texts.append(
                            tokenizer.convert_tokens_to_string([tok for w in kept for tok in w]))
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

    alias = os.getenv("ALIAS_PREFIX", "") + MODEL_ALIAS
    if os.getenv("FULLDOC_JSON"):
        fulldoc = json.load(open(os.environ["FULLDOC_JSON"]))
        texts = [fulldoc[q] for q in text_ids]
        print(f"Full-document mode ({len(texts)} filings)")
    print(f"Query-aware compression of {len(texts)} contexts with {alias}...")
    result = compress_all.remote(texts, questions, text_ids, MODEL_VOL_PATH, alias, int(os.getenv("SEQ_LEN", "2048")))

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
