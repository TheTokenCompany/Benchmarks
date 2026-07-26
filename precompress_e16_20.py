#!/usr/bin/env python3
"""E16-E20: inference-time quality/retention experiments (no training).

  E16 focused-safetab   safetab forcing (numbers+periods+row labels) applied ONLY
                        inside the top-K% focus-ranked lines; v7.1 handles the rest
                        at a harsh threshold. Kills safenum's keep-every-number waste.
  E17 rank-budget v7.1  keep top-B% of words per doc by v7.1 score (rank, not
                        threshold) — equalizes retention across docs, kills the
                        over-compressed tail.
  E18 forced-drop       safetab minus page furniture (force-DROP) and duplicate lines.
  E19 lexical anchor    force-keep lines sharing >=2 content words with the question
                        (plus row labels). No model — the question text is the signal.
  E20 compose           E17 budget 30% ∪ E16 top-30 forcing ∪ E19 lexical lines,
                        minus E18 furniture/dupes.

Configs (aggr slot encodes the sweep knob per experiment):
  bear-e16-fsafetab--0.1|0.3    (v7.1@0.7 + safetab in top-10/30% focus lines)
  bear-e16b-fsafetab--0.3       (same, v7.1@0.5)
  bear-e17-v71budget--0.3|0.4|0.5
  bear-e17b-v71budget-safetab--0.4
  bear-e18-dropfurn-safetab--0.3|0.5
  bear-e19-lexanchor--0.7
  bear-e20-compose--0.3

Usage:  modal run precompress_e16_20.py
"""

import json
import os

import modal

V71 = "bear-v7.1-rl-max-v2"
V40 = "bear-4.0-focus-1-ctx2048"

app = modal.App("financebench-precompress-e16-20")
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

STOPWORDS = {
    "the", "of", "and", "to", "in", "a", "for", "on", "as", "by", "with", "is",
    "are", "was", "were", "that", "this", "its", "or", "at", "from", "what",
    "which", "how", "does", "did", "based", "answer", "question", "response",
    "details", "shown", "relying", "give", "amount", "value", "fiscal", "year",
}


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
    m71 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V71}", attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).eval().to(device)
    tok40 = AutoTokenizer.from_pretrained(f"/models/{V40}")
    m40 = AutoModelForTokenClassification.from_pretrained(
        f"/models/{V40}", num_labels=2).eval().to(device)

    num_re = re.compile(r"\d")
    sym_set = {"$", "%", "€", "£"}
    period_re = re.compile(
        r"^(Q[1-4]|FY\d{2,4}|19\d\d|20\d\d|January|February|March|April|May|June|"
        r"July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
        r"Aug|Sep|Sept|Oct|Nov|Dec)[.,:;)]?$", re.IGNORECASE)
    furn_re = re.compile(
        r"(table of contents|form 10-[kq]|annual report|quarterly report|"
        r"accompanying notes|see accompanying|^\s*page \d+|^\s*\d+\s*$|"
        r"^\s*item \d+[a-z]?\.?\s*$|^_+$|^-+$)", re.IGNORECASE)

    def word_str(toks):
        return tok.convert_tokens_to_string(toks).strip()

    def is_protected(s):
        return bool(num_re.search(s)) or s in sym_set

    @torch.no_grad()
    def focus_line_scores(query, context, line_spans, needle_tau=0.01):
        """max token prob per line + needle char-midpoints with prob >= needle_tau."""
        fids = tok40(query, add_special_tokens=False)["input_ids"][:64]
        enc = tok40(context, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if not ids or not fids:
            return [0.0] * len(line_spans), [0] * len(line_spans), []
        cls, sep, pad = tok40.cls_token_id, tok40.sep_token_id, tok40.pad_token_id
        budget = 2048 - len(fids) - 3
        windows = [ids[w0:w0 + budget] for w0 in range(0, len(ids), budget)]
        tokp = []
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
                        tokp.append(((ts + te) / 2, pk))
        tokp.sort()
        scores, counts = [], []
        fp = 0
        for (ls, le) in line_spans:
            while fp < len(tokp) and tokp[fp][0] < ls:
                fp += 1
            k, best, cnt = fp, 0.0, 0
            while k < len(tokp) and tokp[k][0] < le:
                best = max(best, tokp[k][1])
                cnt += 1
                k += 1
            scores.append(best)
            counts.append(cnt)
        needles = [m for m, pk in tokp if pk >= needle_tau]
        return scores, counts, needles

    t0 = time.time()
    out = {}

    def put(cfg, qid, lines_kept_tokens, orig_n):
        # lines_kept_tokens: list of list-of-words (token lists)
        texts_out, cn = [], 0
        for lw in lines_kept_tokens:
            if lw:
                cn += sum(len(w) for w in lw)
                texts_out.append(tok.convert_tokens_to_string([t for w in lw for t in w]))
        out.setdefault(prefix + cfg, {})[qid] = {
            "compressed_text": "\n".join(texts_out),
            "original_tokens": max(1, orig_n),
            "compressed_tokens": min(cn, max(1, orig_n)),
        }

    for qid, question, text in zip(text_ids, questions, texts):
        # word structure + line metadata
        text_lines = text.split("\n")
        char_pos = 0
        lines = []   # dict per line
        chunks, chunk_map = [], []
        cur_t, cur_m = [], []
        seen_norm = {}
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
            strs = [word_str(w) for w in words]
            numeric = [is_protected(s) for s in strs]
            norm = " ".join(line.split()).lower()
            is_dup = bool(norm) and not any(numeric) and seen_norm.get(norm, 0) > 0
            if norm:
                seen_norm[norm] = seen_norm.get(norm, 0) + 1
            caps_words = [s for s in strs if s.isupper() and len(s) > 1]
            is_furn = bool(furn_re.search(line)) or (
                0 < len(strs) <= 6 and len(caps_words) >= max(1, len(strs) - 1)
                and not any(numeric))
            lines.append({
                "span": (char_pos, char_pos + len(line)),
                "words": words, "strs": strs, "numeric": numeric,
                "first_num": numeric.index(True) if any(numeric) else None,
                "is_table": sum(numeric) >= 3,
                "is_furn": is_furn, "is_dup": is_dup,
            })
            for wi, w in enumerate(words):
                if len(cur_t) + len(w) > 510 and cur_t:
                    chunks.append(cur_t); chunk_map.append(cur_m)
                    cur_t, cur_m = [], []
                cur_t.extend(w)
                cur_m.extend([(li, wi)] * len(w))
            char_pos += len(payload)
        if cur_t:
            chunks.append(cur_t); chunk_map.append(cur_m)

        # v7.1 word probs
        p71 = {}
        with torch.no_grad():
            for b0 in range(0, len(chunks), 128):
                batch = chunks[b0:b0 + 128]
                maxlen = max(len(c) for c in batch) + 2
                seqs, attn = [], []
                for c in batch:
                    s = [tok.cls_token_id] + tok.convert_tokens_to_ids(c) + [tok.sep_token_id]
                    seqs.append(s + [tok.pad_token_id] * (maxlen - len(s)))
                    attn.append([1] * len(s) + [0] * (maxlen - len(s)))
                seqs = torch.tensor(seqs, device=device)
                attn = torch.tensor(attn, device=device)
                logits = m71(input_ids=seqs, attention_mask=attn).logits
                pr = F.softmax(logits.float(), dim=-1)[:, :, 1]
                for j, c in enumerate(batch):
                    p = pr[j, 1:1 + len(c)].tolist()
                    for pk, key in zip(p, chunk_map[b0 + j]):
                        p71[key] = min(p71.get(key, 1.0), pk)

        # focus line scores + top-K line sets
        fscores, fcounts, needles = focus_line_scores(question, text, [l["span"] for l in lines])
        total40 = sum(fcounts)
        def top_lines(frac):
            order = sorted(range(len(lines)), key=lambda i: -fscores[i])
            kept, used = set(), 0
            budget = max(1, int(frac * total40))
            for i in order:
                if fcounts[i] == 0:
                    continue
                if used + fcounts[i] > budget and kept:
                    continue
                kept.add(i)
                used += fcounts[i]
                if used >= budget:
                    break
            return kept
        top10, top30 = top_lines(0.10), top_lines(0.30)
        top50, top70 = top_lines(0.50), top_lines(0.70)

        # lexical anchor lines
        qwords = {w for w in re.findall(r"[a-z]{4,}", question.lower())} - STOPWORDS
        lex_lines = set()
        for li, L in enumerate(lines):
            lw = {w for w in re.findall(r"[a-z]{4,}", " ".join(L["strs"]).lower())}
            if len(qwords & lw) >= 2:
                lex_lines.add(li)

        orig_n = sum(len(w) for L in lines for w in L["words"])

        def rules_keep(L, keep):
            """apply safetab forcing in-place on keep[] for line L"""
            for wi, s in enumerate(L["strs"]):
                if L["numeric"][wi] or period_re.match(s):
                    keep[wi] = True
            if L["is_table"] and L["first_num"] is not None and any(
                    keep[wi] for wi, n in enumerate(L["numeric"]) if n):
                for k in range(0, min(L["first_num"], 12)):
                    keep[k] = True

        def assemble(cfg, keepers):
            put(cfg, qid, keepers, orig_n)

        def build(v71_thr=None, budget_frac=None, force_lines=None,
                  rules_scope=None, drop_furn=False, lex=False, keep_lines=None,
                  needle=False):
            """returns per-line list of kept word-token-lists"""
            # budget mode: global rank of words by p71
            budget_keep = None
            if budget_frac is not None:
                allw = [(p71.get((li, wi), 0.0), li, wi)
                        for li, L in enumerate(lines) for wi in range(len(L["words"]))]
                allw.sort(reverse=True)
                budget = int(budget_frac * orig_n)
                budget_keep = set()
                used = 0
                for p, li, wi in allw:
                    n = len(lines[li]["words"][wi])
                    if used + n > budget and budget_keep:
                        continue
                    budget_keep.add((li, wi))
                    used += n
                    if used >= budget:
                        break
            keepers = []
            for li, L in enumerate(lines):
                if drop_furn and (L["is_furn"] or L["is_dup"]):
                    keepers.append([])
                    continue
                n = len(L["words"])
                if budget_keep is not None:
                    keep = [(li, wi) in budget_keep for wi in range(n)]
                else:
                    keep = [p71.get((li, wi), 0.0) >= v71_thr for wi in range(n)]
                if needle and needles:
                    # word char spans within this line: approximate via line span + word order
                    ls, le = L["span"]
                    import bisect
                    lo = bisect.bisect_left(needles, ls)
                    hi = bisect.bisect_right(needles, le)
                    if hi > lo:
                        # distribute needle hits to words by relative position
                        wlens = [max(1, len(s2) + 1) for s2 in L["strs"]]
                        starts, acc = [], ls
                        for wl in wlens:
                            starts.append(acc); acc += wl
                        for m in needles[lo:hi]:
                            wi2 = min(len(starts) - 1, max(0, bisect.bisect_right(starts, m) - 1))
                            keep[wi2] = True
                if keep_lines is not None and li in keep_lines:
                    for wi in range(n):
                        keep[wi] = True
                if rules_scope == "all" or (
                        rules_scope == "lines" and force_lines is not None and li in force_lines):
                    rules_keep(L, keep)
                if lex and li in lex_lines:
                    for wi in range(n):
                        keep[wi] = True
                keepers.append([w for w, k in zip(L["words"], keep) if k])
            return keepers

        # E16: v71@0.7 + safetab only in top-K focus lines
        assemble("bear-e16-fsafetab--0.1", build(v71_thr=0.7, force_lines=top10, rules_scope="lines"))
        assemble("bear-e16-fsafetab--0.3", build(v71_thr=0.7, force_lines=top30, rules_scope="lines"))
        assemble("bear-e16b-fsafetab--0.3", build(v71_thr=0.5, force_lines=top30, rules_scope="lines"))
        # E17: rank-budget
        assemble("bear-e17-v71budget--0.3", build(budget_frac=0.30))
        assemble("bear-e17-v71budget--0.4", build(budget_frac=0.40))
        assemble("bear-e17-v71budget--0.5", build(budget_frac=0.50))
        assemble("bear-e17b-v71budget-safetab--0.4", build(budget_frac=0.40, rules_scope="all"))
        # E18: safetab minus furniture/dupes
        assemble("bear-e18-dropfurn-safetab--0.3", build(v71_thr=0.3, rules_scope="all", drop_furn=True))
        assemble("bear-e18-dropfurn-safetab--0.5", build(v71_thr=0.5, rules_scope="all", drop_furn=True))
        # E19: lexical anchor
        assemble("bear-e19-lexanchor--0.7", build(v71_thr=0.7, lex=True))
        # E20: compose
        assemble("bear-e20-compose--0.3", build(budget_frac=0.30, force_lines=top30,
                                                rules_scope="lines", drop_furn=True, lex=True))
        # E23: validated-parts compose — focused-safetab + tau-needle + lexical
        for thr in (0.3, 0.5, 0.7):
            assemble(f"bear-e23-composeval--{thr}",
                     build(v71_thr=thr, force_lines=top30, rules_scope="lines",
                           needle=True, lex=True))
        # fuse2 sweep (v7.1@0.7 OR whole top-K focus lines, + safetab everywhere)
        assemble("bear-fuse2f30-safetab--0.7", build(v71_thr=0.7, keep_lines=top30, rules_scope="all"))
        assemble("bear-fuse2f50-safetab--0.7", build(v71_thr=0.7, keep_lines=top50, rules_scope="all"))
        assemble("bear-fuse2f70-safetab--0.7", build(v71_thr=0.7, keep_lines=top70, rules_scope="all"))
        # focus-only line-max @30% budget
        assemble("bear-focusfix-linemax--0.3", build(v71_thr=1.1, keep_lines=top30))

    print(f"E16-E20 caches built in {time.time() - t0:.1f}s on {torch.cuda.get_device_name()}")
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

    print("E16-E20 inference-time experiments...")
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
