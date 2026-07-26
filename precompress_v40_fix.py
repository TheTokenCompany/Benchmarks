#!/usr/bin/env python3
"""Fixed serving pipelines for bear-4.0-focus-1-ctx2048 — 7 experiments.

Diagnosed failures of the stock runner (16% gold-figure recall):
  (a) punctuation sentence splitter merges whole tables into one "sentence"
  (b) mean pooling dilutes answer-row spikes across hundreds of cells
  (c) tau thresholding has no usable operating point (scores collapsed near 0)
  (d) directive query phrasing is out-of-distribution

Experiments (config name -> fix under test), all from two B200 prob passes:
  bear-focusfix-sentmax--0.1   original splitter, MAX pooling, top-ranked to 10% budget   (b)
  bear-focusfix-linemax--0.1   LINE splitter, max pooling, 10% budget                     (a+b)
  bear-focusfix-linemean--0.1  line splitter, mean pooling, 10% budget                    (a only)
  bear-focusfix-wordtop--0.1   word-level top-k by prob + row-label anchor, 10% budget    (a+b, finer)
  bear-focusfix-linemax--0.3   line/max at 30% budget                                     (c: rank-based selection)
  bear-focusfix-rwlinemax--0.1 line/max 10% with directive boilerplate stripped from query (d)
  bear-fuse2-safetab--0.5      v7.1@0.5+safetab OR top-10% focus lines (improved needle)

Budget = fraction of the context's focus-tokenizer tokens; selection is by
rank, so calibration of absolute scores no longer matters.

Usage:
  modal run precompress_v40_fix.py
"""

import json
import os

import modal

V40 = "bear-4.0-focus-1-ctx2048"
V71 = "bear-v7.1-rl-max-v2"

app = modal.App("financebench-precompress-v40fix")
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


def rewrite_query(q: str) -> str:
    """Strip directive boilerplate; keep the actual question sentence(s)."""
    import re
    cut = re.split(
        r"(?i)\b(give a response|answer the question by|by relying|please base|"
        r"respond to the question|rely(?:ing)? on the details)\b", q)[0].strip()
    return cut if len(cut) >= 15 else q


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
    MAX40, FOCUS_BUDGET = 2048, 64
    tok40 = AutoTokenizer.from_pretrained(f"/models/{V40}")
    m40 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V40}", num_labels=2).eval().to(device)
    tok71 = AutoTokenizer.from_pretrained(f"/models/{V71}")
    m71 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V71}", attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).eval().to(device)

    num_re = re.compile(r"\d")
    sym_set = {"$", "%", "€", "£"}
    period_re = re.compile(
        r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
        r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
        r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)
    _SENT = re.compile(r'[.!?]+["\')\]]*(?=\s|$)|\n{2,}')

    @torch.no_grad()
    def focus_probs(query, context):
        """[(ts, te, prob)] batched windows."""
        fids = tok40(query, add_special_tokens=False)["input_ids"][:FOCUS_BUDGET]
        enc = tok40(context, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if not ids or not fids:
            return []
        cls, sep, pad = tok40.cls_token_id, tok40.sep_token_id, tok40.pad_token_id
        budget = MAX40 - len(fids) - 3
        windows = [ids[w0:w0 + budget] for w0 in range(0, len(ids), budget)]
        out = []
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
            logits = m40(input_ids=seqs, attention_mask=attn).logits.float()
            probs = F.softmax(logits, dim=-1)[:, :, 1]
            for j, w in enumerate(batch):
                p = probs[j, n_pre:n_pre + len(w)].cpu().tolist()
                base = (b0 + j) * budget
                for k, pk in enumerate(p):
                    ts, te = offsets[base + k]
                    if te > ts:
                        out.append((ts, te, pk))
        return out

    def units_from(context, mode):
        """Split context into units [(start, end, text)]. mode: 'sent'|'line'|'word'."""
        if mode == "sent":
            out, start = [], 0
            for m in _SENT.finditer(context):
                if context[start:m.end()].strip():
                    out.append((start, m.end(), context[start:m.end()]))
                start = m.end()
            if start < len(context) and context[start:].strip():
                out.append((start, len(context), context[start:]))
            return out
        if mode == "line":
            out, pos = [], 0
            for line in context.split("\n"):
                if line.strip():
                    out.append((pos, pos + len(line), line))
                pos += len(line) + 1
            return out
        out = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", context)]
        return out

    def score_units(units, fprobs, pool):
        """Per-unit (score, tok_count) via char-midpoint assignment."""
        fprobs = sorted(fprobs)
        scores, counts = [], []
        fp = 0
        for (us, ue, _) in units:
            while fp < len(fprobs) and (fprobs[fp][0] + fprobs[fp][1]) / 2 < us:
                fp += 1
            k, best, tot, cnt = fp, 0.0, 0.0, 0
            while k < len(fprobs) and (fprobs[k][0] + fprobs[k][1]) / 2 < ue:
                best = max(best, fprobs[k][2])
                tot += fprobs[k][2]
                cnt += 1
                k += 1
            scores.append(best if pool == "max" else (tot / cnt if cnt else 0.0))
            counts.append(cnt)
        return scores, counts

    def select_budget(units, scores, counts, frac, total_tokens):
        budget = max(1, int(frac * total_tokens))
        order = sorted(range(len(units)), key=lambda i: -scores[i])
        kept, used = set(), 0
        for i in order:
            if counts[i] == 0:
                continue
            if used + counts[i] > budget and kept:
                continue
            kept.add(i)
            used += counts[i]
            if used >= budget:
                break
        return kept, used

    def assemble(units, kept, joiner):
        idx = sorted(kept)
        return joiner.join(" ".join(units[i][2].split()) for i in idx)

    t0 = time.time()
    out_cfg = {}

    def put(cfg, qid, text_out, used, total):
        out_cfg.setdefault(prefix + cfg, {})[qid] = {
            "compressed_text": text_out,
            "original_tokens": max(1, total),
            "compressed_tokens": min(used, max(1, total)),
        }

    for qid, question, context in zip(text_ids, questions, texts):
        fp_orig = focus_probs(question, context)
        total40 = len(fp_orig)

        # sentence & line & word units
        for mode, pool, frac, cfg in [
            ("sent", "max", 0.10, "bear-focusfix-sentmax--0.1"),
            ("line", "max", 0.10, "bear-focusfix-linemax--0.1"),
            ("line", "mean", 0.10, "bear-focusfix-linemean--0.1"),
            ("line", "max", 0.30, "bear-focusfix-linemax--0.3"),
        ]:
            units = units_from(context, mode)
            scores, counts = score_units(units, fp_orig, pool)
            kept, used = select_budget(units, scores, counts, frac, total40)
            put(cfg, qid, assemble(units, kept, "\n" if mode == "line" else " … "), used, total40)

        # word-level top-k + row-label anchor
        wunits = units_from(context, "word")
        wscores, wcounts = score_units(wunits, fp_orig, "max")
        kept, used = select_budget(wunits, wscores, wcounts, 0.10, total40)
        # row anchor: group words into lines; for table lines with a kept digit word,
        # keep the leading label span (words before first digit word, cap 12)
        line_of, lines = {}, []
        pos = 0
        for li, line in enumerate(context.split("\n")):
            lines.append((pos, pos + len(line)))
            pos += len(line) + 1
        wl = []
        li = 0
        for wi, (ws, we, _) in enumerate(wunits):
            while li < len(lines) - 1 and ws > lines[li][1]:
                li += 1
            wl.append(li)
        from collections import defaultdict
        by_line = defaultdict(list)
        for wi in range(len(wunits)):
            by_line[wl[wi]].append(wi)
        for li2, wis in by_line.items():
            digit_wis = [wi for wi in wis if num_re.search(wunits[wi][2])]
            if len(digit_wis) >= 3 and any(wi in kept for wi in digit_wis):
                first_d = digit_wis[0]
                for wi in wis:
                    if wi >= first_d or wi - wis[0] >= 12:
                        break
                    if wi not in kept:
                        kept.add(wi)
                        used += max(1, wcounts[wi])
        put("bear-focusfix-wordtop--0.1", qid,
            assemble(wunits, kept, " "), used, total40)

        # rewritten query, line/max, 10%
        rq = rewrite_query(question)
        fp_rw = focus_probs(rq, context) if rq != question else fp_orig
        units = units_from(context, "line")
        scores, counts = score_units(units, fp_rw, "max")
        kept, used = select_budget(units, scores, counts, 0.10, max(1, len(fp_rw)))
        put("bear-focusfix-rwlinemax--0.1", qid,
            assemble(units, kept, "\n"), used, max(1, len(fp_rw)))

        # fuse2 grid: v7.1@{aggr} + safetab, OR words inside top-{frac} focus lines
        lunits = units_from(context, "line")
        lscores, lcounts = score_units(lunits, fp_orig, "max")
        focus_spans_by_frac = {}
        for frac in (0.10, 0.30, 0.50, 0.70, 0.90):
            lkept, _ = select_budget(lunits, lscores, lcounts, frac, total40)
            focus_spans_by_frac[frac] = [(lunits[i][0], lunits[i][1]) for i in sorted(lkept)]

        text_lines = context.split("\n")
        char_pos = 0
        line_words = []
        chunks, chunk_map = [], []
        cur_toks, cur_map = [], []
        for li3, line in enumerate(text_lines):
            payload = line + ("\n" if li3 < len(text_lines) - 1 else "")
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
            line_words.append((words, spans))
            for wi, w in enumerate(words):
                if len(cur_toks) + len(w) > 510 and cur_toks:
                    chunks.append(cur_toks); chunk_map.append(cur_map)
                    cur_toks, cur_map = [], []
                cur_toks.extend(w)
                cur_map.extend([(li3, wi)] * len(w))
            char_pos += len(payload)
        if cur_toks:
            chunks.append(cur_toks); chunk_map.append(cur_map)
        probs71 = {}
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
                pr = F.softmax(logits.float(), dim=-1)[:, :, 1]
                for j, c in enumerate(batch):
                    p = pr[j, 1:1 + len(c)].tolist()
                    for pk, key in zip(p, chunk_map[b0 + j]):
                        probs71[key] = min(probs71.get(key, 1.0), pk)
        for v71_aggr, frac, cfg in [
            (0.5, 0.10, "bear-fuse2-safetab--0.5"),
            (0.7, 0.10, "bear-fuse2-safetab--0.7"),
            (0.5, 0.30, "bear-fuse2f30-safetab--0.5"),
            (0.7, 0.30, "bear-fuse2f30-safetab--0.7"),
            (0.7, 0.50, "bear-fuse2f50-safetab--0.7"),
            (0.7, 0.70, "bear-fuse2f70-safetab--0.7"),
            (0.7, 0.90, "bear-fuse2f90-safetab--0.7"),
        ]:
            focus_spans = focus_spans_by_frac[frac]
            kept_line_texts = []
            orig_n = comp_n = 0
            for li3, (words, spans) in enumerate(line_words):
                strs = [tok71.convert_tokens_to_string(w).strip() for w in words]
                numeric = [bool(num_re.search(s)) or s in sym_set for s in strs]
                keep = []
                for wi in range(len(words)):
                    v71keep = probs71.get((li3, wi), 0.0) >= v71_aggr
                    ws, we = spans[wi]
                    mid = (ws + we) / 2
                    infocus = any(fs <= mid < fe for fs, fe in focus_spans)
                    keep.append(v71keep or infocus or numeric[wi] or bool(period_re.match(strs[wi])))
                if sum(numeric) >= 3 and any(
                        keep[wi] for wi, n in enumerate(numeric) if n):
                    first_num = numeric.index(True)
                    for k in range(0, min(first_num, 12)):
                        keep[k] = True
                orig_n += sum(len(w) for w in words)
                kept_w = [w for w, k in zip(words, keep) if k]
                comp_n += sum(len(w) for w in kept_w)
                if kept_w:
                    kept_line_texts.append(tok71.convert_tokens_to_string(
                        [tok for w in kept_w for tok in w]))
            put(cfg, qid, "\n".join(kept_line_texts), comp_n, orig_n)

    print(f"All experiments done in {time.time() - t0:.1f}s on {torch.cuda.get_device_name()}")
    return out_cfg


@app.local_entrypoint()
def main():
    import re
    from datasets import load_dataset

    dataset = load_dataset("PatronusAI/financebench", split="train")
    items = list(dataset)[:150]

    texts, questions, text_ids, golds = [], [], [], {}
    for i, item in enumerate(items):
        pages = [
            ev.get("evidence_text_full_page", "").strip()
            for ev in item["evidence"]
            if ev.get("evidence_text_full_page")
        ]
        texts.append("\n\n---\n\n".join(pages))
        questions.append(item["question"])
        qid = item.get("question_id", str(i))
        text_ids.append(qid)
        golds[qid] = item["answer"]

    prefix = os.getenv("ALIAS_PREFIX", "")
    if os.getenv("FULLDOC_JSON"):
        fulldoc = json.load(open(os.environ["FULLDOC_JSON"]))
        texts = [fulldoc[q] for q in text_ids]
        print(f"Full-document mode ({len(texts)} filings)")

    print("Running fixed-pipeline experiments on B200...")
    result = compress_all.remote(texts, questions, text_ids, prefix)

    def nums(s):
        return set(re.sub(r"[,$%]", "", m) for m in re.findall(r"\$?[\d,]+\.?\d*%?", s or "")
                   if len(re.sub(r"[,$%.]", "", m)) >= 3)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\n{'config':40s} {'retention':>9s} {'gold-figure recall':>18s}")
    for config_name, entry in result.items():
        path = os.path.join(CACHE_DIR, f"{config_name}.json")
        with open(path, "w") as f:
            json.dump(entry, f)
        ratios = [v["compressed_tokens"] / v["original_tokens"]
                  for v in entry.values() if v["original_tokens"]]
        hit = tot = 0
        for qid, g in golds.items():
            gn = nums(g)
            if not gn:
                continue
            tot += 1
            snip = re.sub(r"[,$%]", "", entry[qid]["compressed_text"])
            if any(n in snip for n in gn):
                hit += 1
        print(f"{config_name:40s} {sum(ratios)/len(ratios):9.3f} {hit}/{tot} = {hit/tot:.0%}")
