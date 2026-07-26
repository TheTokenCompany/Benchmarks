#!/usr/bin/env python3
"""Pre-compress FinanceBench contexts with bear-v7.1-rl-max-v2 on Modal B200.

Computes keep-probs once per text (per line, so table structure is available),
then derives every aggressiveness level and all keep-rule variants from the
same probs:

  plain     model threshold only
  safenum   force-keep words containing digits or currency/percent symbols
            (offline equivalent of <ttc_safe> wrapping)
  safectx2  safenum + force-keep 2 words before/after each protected word
  saferow   safenum + row-anchor: in a table line (>=3 numeric words) whose
            numbers survive, force-keep the leading row-label span
  safehdr   safenum + column-header propagation: force-keep period-header
            lines when any table line in the block below survives
  safeline  safenum + keep entire table lines that have any surviving number
  safeper   safenum + force-keep fiscal-period words (months, quarters, years)
  safetab   safenum + saferow + safehdr + safeper combined
  weld      no forcing; merge value spans ($ + number + scale, "label: number")
            and keep the whole span if any member passes the threshold

Usage:
  modal run precompress_v71.py
"""

import json
import os

import modal

MODEL_NAME = os.getenv("MODEL_NAME", "bear-v7.1-rl-max-v2")
ALIAS_PREFIX = os.getenv("ALIAS_PREFIX", "")  # e.g. "fd-" for full-document caches
AGGRESSIVENESS_LEVELS = [0.3, 0.5, 0.7]
VARIANTS = (os.environ["VARIANTS"].split(",") if os.getenv("VARIANTS")
            else ["plain", "safenum", "safectx2", "saferow", "safehdr",
                  "safeline", "safeper", "safetab", "weld"])

app = modal.App("financebench-precompress-v71")
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
    timeout=1800,
    memory=32768,
)
def compress_all(texts: list[str], text_ids: list[str],
                 variants: list[str] = None, prefix: str = "",
                 model_name: str = "bear-v7.1-rl-max-v2") -> dict:
    """Returns {config_name: {qid: {compressed_text, original_tokens, compressed_tokens}}}."""
    import re
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    device = "cuda"
    model_path = f"/models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    # Checkpoint was trained at 512 ctx (config.json's 8192 is the ModernBert default)
    seq_len = 512
    max_chunk_tokens = seq_len - 2
    specials = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}

    num_re = re.compile(r"\d")
    sym_set = {"$", "%", "€", "£"}
    period_re = re.compile(
        r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
        r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
        r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)
    scale_re = re.compile(r"^(million|billion|thousand|millions|billions|thousands|bn|mn)[.,:;)]?$", re.IGNORECASE)

    def word_str(toks):
        return tokenizer.convert_tokens_to_string(toks).strip()

    def is_protected(s):
        return bool(num_re.search(s)) or s in sym_set

    # ---- Tokenize per line, group into words, batch-infer probs ----
    t0 = time.time()
    all_lines = []   # (text_idx, line_idx, [word_token_lists])
    for ti, text in enumerate(texts):
        text_lines = text.split("\n")
        for li, line in enumerate(text_lines):
            # keep the newline in the model input so it sees real line structure
            payload = line + ("\n" if li < len(text_lines) - 1 else "")
            ids = tokenizer(payload, add_special_tokens=False)["input_ids"]
            if not ids:
                all_lines.append((ti, li, []))
                continue
            words = []
            toks = tokenizer.convert_ids_to_tokens(ids)
            cur = []
            for tok in toks:
                if (tok.startswith("▁") or tok.startswith("Ġ") or not cur) and cur:
                    words.append(cur)
                    cur = []
                cur.append(tok)
            if cur:
                words.append(cur)
            all_lines.append((ti, li, words))

    # Pack lines into chunks of <=510 tokens for inference (split long lines)
    chunks = []       # list of flat token lists
    chunk_map = []    # per chunk: list of (line_ref, word_idx) per word
    cur_toks, cur_map = [], []
    for ref, (ti, li, words) in enumerate(all_lines):
        for wi, toks in enumerate(words):
            if len(cur_toks) + len(toks) > max_chunk_tokens and cur_toks:
                chunks.append(cur_toks); chunk_map.append(cur_map)
                cur_toks, cur_map = [], []
            cur_toks.extend(toks)
            cur_map.extend([(ref, wi)] * len(toks))
    if cur_toks:
        chunks.append(cur_toks); chunk_map.append(cur_map)

    # Batch inference; min-pool token probs into word probs
    word_probs = {}   # (line_ref, word_idx) -> min prob
    BATCH = 128
    for bs in range(0, len(chunks), BATCH):
        batch = chunks[bs:bs + BATCH]
        maxlen = max(len(c) for c in batch) + 2
        input_ids, attn = [], []
        for c in batch:
            ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(c) + [tokenizer.sep_token_id]
            pad = maxlen - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        input_ids = torch.tensor(input_ids, device=device)
        attn = torch.tensor(attn, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = F.softmax(logits.float(), dim=-1)[:, :, 1]
        for j, c in enumerate(batch):
            p = probs[j, 1:1 + len(c)].tolist()
            for tk, (key) in zip(p, chunk_map[bs + j]):
                if key in word_probs:
                    word_probs[key] = min(word_probs[key], tk)
                else:
                    word_probs[key] = tk

    print(f"Inference done in {time.time() - t0:.1f}s on {torch.cuda.get_device_name()} "
          f"({len(chunks)} chunks)")

    # ---- Per-line metadata for structural rules ----
    line_meta = []  # per line_ref: dict
    for ref, (ti, li, words) in enumerate(all_lines):
        strs = [word_str(t) for t in words]
        numeric = [is_protected(s) for s in strs]
        n_num = sum(numeric)
        n_per = sum(1 for s in strs if period_re.match(s))
        first_num = numeric.index(True) if any(numeric) else None
        # header line: dominated by period tokens (years/quarters/months), no money
        has_money = any(("$" in s or "%" in s or ("," in s and num_re.search(s))) for s in strs)
        is_header = (n_per >= 2 and len(strs) > 0 and n_per / len(strs) >= 0.5
                     and not has_money)
        is_table = n_num >= 3 and not is_header
        line_meta.append({
            "ti": ti, "strs": strs, "numeric": numeric, "first_num": first_num,
            "is_table": is_table, "is_header": is_header, "n_words": len(strs),
        })

    # weld spans per line: [(start, end_inclusive)]
    weld_spans = {}
    for ref, meta in enumerate(line_meta):
        strs = meta["strs"]
        spans = []
        i = 0
        while i < len(strs):
            j = i
            if strs[i] in ("$", "€", "£") and i + 1 < len(strs) and num_re.search(strs[i + 1]):
                j = i + 1
            elif num_re.search(strs[i]):
                j = i
            elif strs[i].endswith(":") and i + 1 < len(strs) and num_re.search(strs[i + 1]):
                j = i + 1
            else:
                i += 1
                continue
            if j + 1 < len(strs) and scale_re.match(strs[j + 1]):
                j += 1
            if j > i:
                spans.append((i, j))
            i = j + 1
        if spans:
            weld_spans[ref] = spans

    # ---- Derive all variants ----
    out = {}
    refs_by_text = {}
    for ref, meta in enumerate(line_meta):
        refs_by_text.setdefault(meta["ti"], []).append(ref)

    for aggr in AGGRESSIVENESS_LEVELS:
        base_keep = {}  # ref -> [bool] model-threshold keeps
        for ref, (ti, li, words) in enumerate(all_lines):
            base_keep[ref] = [word_probs.get((ref, wi), 0.0) >= aggr for wi in range(len(words))]

        for variant in (variants or VARIANTS):
            suffix = "" if variant == "plain" else f"-{variant}"
            config_name = f"{prefix}{model_name}{suffix}--{aggr}"
            entry = {}
            for ti, qid in enumerate(text_ids):
                kept_line_texts = []
                orig_n = 0
                comp_n = 0
                # first pass: word-level keep mask per line
                masks = {}
                for ref in refs_by_text.get(ti, []):
                    meta = line_meta[ref]
                    words = all_lines[ref][2]
                    keep = list(base_keep[ref])
                    strs = meta["strs"]
                    if variant != "plain" and variant != "weld":
                        for wi, s in enumerate(strs):
                            if is_protected(s):
                                keep[wi] = True
                    if variant == "safectx2":
                        prot = [is_protected(s) for s in strs]
                        for wi, p in enumerate(prot):
                            if p:
                                for k in range(max(0, wi - 2), min(len(strs), wi + 3)):
                                    keep[k] = True
                    if variant in ("safeper", "safetab"):
                        for wi, s in enumerate(strs):
                            if period_re.match(s):
                                keep[wi] = True
                    if variant in ("saferow", "safetab"):
                        if meta["is_table"] and meta["first_num"] is not None and any(
                                keep[wi] for wi, n in enumerate(meta["numeric"]) if n):
                            for k in range(0, min(meta["first_num"], 12)):
                                keep[k] = True
                    if variant == "safeline":
                        if meta["is_table"] and any(
                                keep[wi] for wi, n in enumerate(meta["numeric"]) if n):
                            keep = [True] * len(keep)
                    if variant == "weld":
                        for (a, b) in weld_spans.get(ref, []):
                            if any(keep[a:b + 1]):
                                for k in range(a, b + 1):
                                    keep[k] = True
                    masks[ref] = keep
                # second pass: header propagation needs to know surviving table lines
                if variant in ("safehdr", "safetab"):
                    refs = refs_by_text.get(ti, [])
                    for idx, ref in enumerate(refs):
                        if not line_meta[ref]["is_header"]:
                            continue
                        # look at the contiguous block of table lines below (skip blanks)
                        block_survives = False
                        for ref2 in refs[idx + 1: idx + 8]:
                            m2 = line_meta[ref2]
                            if m2["is_table"] and any(masks[ref2]):
                                block_survives = True
                                break
                            if m2["n_words"] > 0 and not m2["is_table"]:
                                break
                        if block_survives:
                            masks[ref] = [True] * len(masks[ref])
                # reconstruct
                for ref in refs_by_text.get(ti, []):
                    words = all_lines[ref][2]
                    orig_n += sum(len(t) for t in words)
                    kept = [t for t, k in zip(words, masks[ref]) if k]
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


def _write_caches(result):
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


@app.local_entrypoint()
def main():
    from datasets import load_dataset

    if os.getenv("FULLDOC_JSON"):
        fulldoc = json.load(open(os.environ["FULLDOC_JSON"]))
        text_ids = sorted(fulldoc, key=lambda q: int(q))
        texts = [fulldoc[q] for q in text_ids]
        print(f"Full-document mode: {len(texts)} contexts, "
              f"variants={VARIANTS}, prefix={ALIAS_PREFIX!r}")
        result = compress_all.remote(texts, text_ids, VARIANTS, ALIAS_PREFIX, MODEL_NAME)
        _write_caches(result)
        return

    dataset = load_dataset("PatronusAI/financebench", split="train")
    items = list(dataset)[:150]

    texts, text_ids = [], []
    for i, item in enumerate(items):
        pages = [
            ev.get("evidence_text_full_page", "").strip()
            for ev in item["evidence"]
            if ev.get("evidence_text_full_page")
        ]
        texts.append("\n\n---\n\n".join(pages))
        text_ids.append(item.get("question_id", str(i)))

    print(f"Compressing {len(texts)} contexts with {MODEL_NAME} on B200 "
          f"({len(VARIANTS)} variants x {len(AGGRESSIVENESS_LEVELS)} levels)...")
    result = compress_all.remote(texts, text_ids, VARIANTS, ALIAS_PREFIX, MODEL_NAME)
    _write_caches(result)
