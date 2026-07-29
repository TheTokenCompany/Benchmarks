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
import re

import modal

MODEL_VOL = os.getenv("MODEL_VOL", "otso-v8-training")
MODEL_VOL_PATH = os.getenv("MODEL_VOL_PATH", "")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "v8")
AGGRESSIVENESS_LEVELS = [float(x) for x in os.getenv(
    "AGGR_LEVELS", "0.1,0.3,0.5,0.7,0.9").split(",")]
VARIANTS = os.getenv("VARIANTS", "plain,safetab,budget22,budget22-safetab,budget33,budget33-safetab").split(",")

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


# --------------------------------------------------------------------------- #
# eval-v2 word model (PARITY.md P4 / P5 / P6 / P7 / P8)
#
# v1 folds non-content tokens into content words: a run of column-padding '▁'
# pieces opens its own empty word (5.7% of all words on real filings, each
# holding a slot in the budget accounting), and a line's trailing '\n' lands in
# that line's LAST word, where min-pooling lets it cap the row's final number --
# in a table, the numeric column. The v1 render then double-emits that same '\n'
# against the "\n".join, so all-keep does not round-trip.
#
# v2 keeps every token in the MODEL INPUT -- the encoder sees the same token
# stream it saw before -- but marks whitespace/newline pieces non-content, which
# takes them out of the pool, out of the budget, and out of the render. Note the
# stream is identical while the CHUNKING is not quite: infer_probs packs whole
# words to a budget and v2 words carry their whitespace prefix, so the greedy
# cuts land 1-3 tokens off v1's. Same model, same tokens, windowing perturbed by
# a few positions -- near-identical probs, not bit-identical ones.
#
# A v2 word is (tokens, content_flags): `tokens` is the full run fed to the
# encoder including any non-content prefix, `content_flags[i]` says whether
# token i is a keep/drop-bearing piece. Module scope so parity_test.py exercises
# this exact code rather than a copy of it.
# --------------------------------------------------------------------------- #
def is_content_tok(tok):
    """False for the pieces that carry no text: bare '▁'/'Ġ' padding and '\\n'."""
    return tok.lstrip("▁").lstrip("Ġ").strip() != ""


def group_words_v2(toks, pending):
    """token strings -> ([(tokens, content_flags)], pending).

    `pending` carries non-content tokens across the line boundary, so a blank
    line's '\\n' still reaches the encoder attached to the next word instead of
    standing up as a word of its own."""
    words = []
    for tok in toks:
        if not is_content_tok(tok):
            pending.append(tok)
            continue
        # `pending` opens a word too: column padding tokenizes to a run of bare
        # '▁' pieces and the piece that follows carries no marker of its own, so
        # keying only on the marker would glue "3,605" onto the previous label.
        if tok.startswith("▁") or tok.startswith("Ġ") or pending or not words:
            words.append(([*pending, tok], [False] * len(pending) + [True]))
        else:
            w = words[-1]
            w[0].append(tok)
            w[1].append(True)
        pending = []
    return words, pending


def group_document_v2(lines_tokens):
    """[(line_idx, tokens)] -> [(line_idx, words)] for one document.

    Owns the pending buffer across line boundaries and flushes whatever is left
    into the final word, so every token the v1 path fed to the encoder still
    reaches it -- including a trailing newline with no line after it."""
    out, pending = [], []
    for li, toks in lines_tokens:
        words, pending = group_words_v2(toks, pending)
        out.append((li, words))
    if pending:
        for _, ws in reversed(out):
            if ws:
                ws[-1][0].extend(pending)
                ws[-1][1].extend([False] * len(pending))
                break
    return out


def render_v2(strs, nl_after, keep):
    """Canonical render (PARITY.md section 3), the same rule as
    v9_rl_prep.render_mask: kept words space-joined, newlines re-emitted for
    dropped words too so a table keeps its geometry, and no line-final newline
    hiding inside a word to double up against the join."""
    parts = []
    for s, nl, k in zip(strs, nl_after, keep):
        if k:
            parts.append(s)
            parts.append("\n" * nl if nl else " ")
        elif nl:
            parts.append("\n" * nl)
    out = "".join(parts)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


@app.function(
    image=image,
    gpu="B200",
    volumes={"/train": train_volume},
    timeout=1800,
    memory=32768,
)
def compress_all(texts: list[str], questions: list[str], text_ids: list[str],
                 model_vol_path: str = None, model_alias: str = None,
                 seq_len_arg: int = 2048, variants: list[str] = None,
                 dump_probs: bool = False, hist_bins: int = 1000,
                 eval_v2: bool = False, evidence_numbers: dict = None,
                 aggr_levels: list = None) -> dict:
    import math
    import re
    import time

    import numpy as np
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

    # eval-v2 word model: helpers live at module scope (see the block above
    # CACHE_DIR); these three adapt the word representation to the flag.
    v2 = bool(eval_v2)

    def w_all(w):
        """every token of the word, in encoder order"""
        return w[0] if v2 else w

    def w_content(w):
        return [t for t, f in zip(*w) if f] if v2 else w

    def w_len(w):
        """token cost of the word -- content only under v2"""
        return sum(w[1]) if v2 else len(w)

    def infer_probs(question, line_words_per_text):
        """Given [(ref, words)] for one text, return {(ref, wi): prob} conditioned on question."""
        q_ids = tokenizer(question, add_special_tokens=False)["input_ids"][:96]
        budget = seq_len - len(q_ids) - 3  # CLS + q + SEP + chunk + SEP
        chunks, chunk_map = [], []
        cur_toks, cur_map = [], []
        for ref, words in line_words_per_text:
            for wi, w in enumerate(words):
                toks = w_all(w)
                if len(cur_toks) + len(toks) > budget and cur_toks:
                    chunks.append(cur_toks); chunk_map.append(cur_map)
                    cur_toks, cur_map = [], []
                cur_toks.extend(toks)
                # None marks a token that must not reach the pool (v2 only)
                cur_map.extend([(ref, wi) if f else None for f in w[1]] if v2
                               else [(ref, wi)] * len(toks))
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
                    if key is None:
                        continue
                    probs_out[key] = min(probs_out.get(key, 1.0), tk)
        return probs_out

    t0 = time.time()
    # tokenize lines once per text
    per_text_lines = []  # per text: [(ref, words)]
    line_meta_all = []   # global ref -> meta
    line_of_ref = {}     # ref -> source line index, for v2 nl_after geometry
    for ti, text in enumerate(texts):
        text_lines = text.split("\n")
        per_line_ids = []
        for li, line in enumerate(text_lines):
            payload = line + ("\n" if li < len(text_lines) - 1 else "")
            per_line_ids.append((li, tokenizer(payload, add_special_tokens=False)["input_ids"]))
        if v2:
            grouped = group_document_v2(
                [(li, tokenizer.convert_ids_to_tokens(ids) if ids else [])
                 for li, ids in per_line_ids])
        else:
            grouped = [(li, group_words(ids) if ids else []) for li, ids in per_line_ids]

        lines = []
        for li, words in grouped:
            ref = len(line_meta_all)
            strs = [word_str(w_content(w)) for w in words]
            numeric = [is_protected(s) for s in strs]
            n_per = sum(1 for s in strs if period_re.match(s))
            line_meta_all.append({
                "strs": strs, "numeric": numeric,
                "first_num": numeric.index(True) if any(numeric) else None,
                "is_table": sum(numeric) >= 3, "n_per": n_per,
            })
            line_of_ref[ref] = li
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
    if swap_deltas:
        print(f"FOCUS-SWAP PROBE mean|dP| = {sum(swap_deltas)/len(swap_deltas):.4f} "
              f"(near 0 => model ignores the question)")
    # Kept-set probe: does the TOP-22% budget selection change with the question?
    # mean|dP| over all tokens washes out sensitivity concentrated on evidence rows;
    # serving only cares which words survive the budget.
    jaccs = []
    for ti in range(min(5, len(texts))):
        wrong_q = questions[(ti + 7) % len(questions)]
        p_wrong = infer_probs(wrong_q, per_text_lines[ti])
        p_right = all_probs[ti]
        common = sorted(set(p_right) & set(p_wrong))
        if len(common) < 100:
            continue
        k = max(1, int(0.22 * len(common)))
        top_r = set(sorted(common, key=lambda w: -p_right[w])[:k])
        top_w = set(sorted(common, key=lambda w: -p_wrong[w])[:k])
        jaccs.append(len(top_r & top_w) / len(top_r | top_w))
    if jaccs:
        print(f"KEPT-SET SWAP PROBE Jaccard@22% = {sum(jaccs)/len(jaccs):.3f} "
              f"(near 1 => same words kept regardless of question)")
    print(f"Inference done in {time.time() - t0:.1f}s on {torch.cuda.get_device_name()}")

    # ---- flat per-item arrays (word order == per_text_lines order) ----------
    # Used by the adaptive-dose (adose*) policies and the probs dump. Built once;
    # every dose variant is a cheap numpy pass over these.
    flat = []  # per text: dict of arrays + line index
    for ti in range(len(texts)):
        probs = all_probs[ti]
        p_l, nt_l, prot_l, per_l, lines_idx = [], [], [], [], []
        j = 0
        for ref, words in per_text_lines[ti]:
            meta = line_meta_all[ref]
            n = len(words)
            lines_idx.append((ref, j, n))
            for wi in range(n):
                p_l.append(probs.get((ref, wi), 0.0))
                nt_l.append(w_len(words[wi]))
                prot_l.append(meta["numeric"][wi])
                per_l.append(bool(period_re.match(meta["strs"][wi])))
            j += n
        p = np.asarray(p_l, dtype=np.float32)
        nt = np.asarray(nt_l, dtype=np.int32)
        flat.append({
            "p": p,
            "nt": nt,
            "prot": np.asarray(prot_l, dtype=bool) | np.asarray(per_l, dtype=bool),
            "lines": lines_idx,
            "total": int(nt.sum()),
            "order_desc": np.argsort(-p, kind="stable"),
        })

    # ---- evidence-rank diagnostic -------------------------------------------
    # For each gold-evidence number, find the word(s) carrying it and report the
    # retention at which the word-budget renderer would FIRST include it
    # ("r_star"). r_star just above the serving budget = near miss the ranker
    # can be sharpened into; r_star deep in the tail = the model never saw it.
    def evidence_ranks(ti, targets):
        f = flat[ti]
        if not f["total"] or not targets:
            return []
        order = f["order_desc"]
        cum = np.empty(f["p"].size, dtype=np.float64)
        cum[order] = np.cumsum(f["nt"][order]) / f["total"]
        strs, owner = [], []
        for ref, j, n in f["lines"]:
            meta = line_meta_all[ref]
            for wi in range(n):
                strs.append(meta["strs"][wi])
                owner.append(ref)
        # exact match on the stripped word: substring matching lets a 3-digit
        # target hit any longer number that contains it, which fakes recall
        norm = [re.sub(r"[,\s$%()\[\]:;]", "", s).rstrip(".").lstrip("(-") for s in strs]
        out = []
        for tgt in targets:
            best = None
            for k, s in enumerate(norm):
                if s == tgt:
                    span = [k]
                elif s and tgt.startswith(s) and len(s) >= 2:
                    # number split across consecutive words: greedily extend
                    acc, span = s, [k]
                    for k2 in range(k + 1, min(k + 4, len(norm))):
                        if not norm[k2] or owner[k2] != owner[k]:
                            break
                        acc += norm[k2]
                        span.append(k2)
                        if acc == tgt:
                            break
                    if acc != tgt:
                        continue
                else:
                    continue
                # all constituent words must survive, so the binding one is the worst
                r = float(max(cum[w] for w in span))
                if best is None or r < best[0]:
                    best = (r, float(min(f["p"][w] for w in span)), len(span))
            out.append({"num": tgt, "found": best is not None,
                        "r_star": best[0] if best else None,
                        "prob": best[1] if best else None,
                        "words": best[2] if best else 0})
        return out

    def dose_keep(ti, tau, floor, cap, safetab):
        """Confidence dosing: keep prob>=tau, optional per-item floor/cap on
        retention and safetab protection restricted to already-kept lines."""
        f = flat[ti]
        p, nt, total = f["p"], f["nt"], f["total"]
        keep = p >= tau
        if safetab:
            for ref, j, n in f["lines"]:
                if n == 0 or not keep[j:j + n].any():
                    continue
                meta = line_meta_all[ref]
                sl = keep[j:j + n]
                sl |= f["prot"][j:j + n]
                if meta["is_table"] and meta["first_num"] is not None:
                    sl[:min(meta["first_num"], 12)] = True
        if total == 0:
            return keep
        kept = int(nt[keep].sum())
        if floor and kept < floor * total:
            need = int(floor * total) - kept
            order = f["order_desc"]
            cand = order[~keep[order]]
            csum = np.cumsum(nt[cand])
            take = int(np.searchsorted(csum, need) + 1)
            keep[cand[:take]] = True
        elif cap and kept > cap * total:
            excess = kept - int(cap * total)
            # drop lowest-prob kept words, protected ones last
            order = f["order_desc"][::-1]
            koi = order[keep[order]]
            rank = f["prot"][koi].astype(np.int8)  # 0 = droppable first
            koi = np.concatenate([koi[rank == 0], koi[rank == 1]])
            csum = np.cumsum(nt[koi])
            drop = int(np.searchsorted(csum, excess) + 1)
            keep[koi[:drop]] = False
        return keep

    def dose_retention(tau, floor, cap, safetab):
        rs = []
        for ti in range(len(texts)):
            f = flat[ti]
            if not f["total"]:
                continue
            keep = dose_keep(ti, tau, floor, cap, safetab)
            rs.append(float(f["nt"][keep].sum()) / f["total"])
        return sum(rs) / len(rs)

    def solve_tau(target, floor, cap, safetab, iters=16):
        """Global tau whose MEAN per-item retention hits `target`."""
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = (lo + hi) / 2
            if dose_retention(mid, floor, cap, safetab) > target:
                lo = mid
            else:
                hi = mid
        tau = (lo + hi) / 2
        print(f"    solve_tau target={target} floor={floor} cap={cap} "
              f"safetab={safetab} -> tau={tau:.4f} "
              f"mean_ret={dose_retention(tau, floor, cap, safetab):.4f}")
        return tau

    # Adaptive-dose variants: "auto22" (solve a global tau for 22% MEAN
    # retention, per-item retention floats), "t45" (literal tau=0.45), plus
    # optional "-f8"/"-c50" per-item retention floor/cap and "-safetab"
    # (protection applied only to lines that already have a kept word).
    dose_re = re.compile(
        r"^(?:adose-?)?(?:auto(?P<target>\d+))?(?:-?t(?P<tau>\d+))?"
        r"(?:-f(?P<floor>\d+))?(?:-c(?P<cap>\d+))?(?P<safetab>-safetab)?$")
    dose_cfg = {}
    for variant in (variants or VARIANTS):
        m = dose_re.match(variant)
        if not m or not (m.group("target") or m.group("tau")):
            continue
        g = m.groupdict()
        floor = int(g["floor"]) / 100 if g["floor"] else 0.0
        cap = int(g["cap"]) / 100 if g["cap"] else 0.0
        safetab = bool(g["safetab"])
        if g["target"]:
            tau = solve_tau(int(g["target"]) / 100, floor, cap, safetab)
        else:
            tau = int(g["tau"]) / 100
        dose_cfg[variant] = (tau, floor, cap, safetab)

    # Budget-based adaptive dosing: keep the fixed-budget renderer (top-k words
    # by prob, which beats tau rendering) but make the per-item budget a
    # function of that item's confident mass instead of a constant.
    #   b_i = clip(alpha * s_i**gamma, floor, cap),  s_i = tokens(p >= ref)/total
    # alpha is solved so the MEAN budget equals the target, i.e. same mean
    # retention as the fixed-budget baseline it is compared against.
    abud_re = re.compile(
        r"^abud(?P<target>\d+)-r(?P<ref>\d+)(?:-g(?P<gamma>\d+))?"
        r"(?:-f(?P<floor>\d+))?(?:-c(?P<cap>\d+))?$")
    abud_budgets = {}
    for variant in (variants or VARIANTS):
        m = abud_re.match(variant)
        if not m:
            continue
        g = m.groupdict()
        target = int(g["target"]) / 100
        ref = int(g["ref"]) / 100
        gamma = int(g["gamma"]) / 100 if g["gamma"] else 1.0
        floor = int(g["floor"]) / 100 if g["floor"] else 0.0
        cap = int(g["cap"]) / 100 if g["cap"] else 1.0
        s = np.array([
            (float(f["nt"][f["p"] >= ref].sum()) / f["total"]) if f["total"] else 0.0
            for f in flat])
        base = np.power(np.maximum(s, 1e-9), gamma)
        lo_a, hi_a = 1e-9, 1e9
        for _ in range(60):
            mid = math.sqrt(lo_a * hi_a)
            if np.clip(mid * base, floor, cap).mean() < target:
                lo_a = mid
            else:
                hi_a = mid
        b = np.clip(math.sqrt(lo_a * hi_a) * base, floor, cap)
        print(f"    {variant}: ref={ref} gamma={gamma} mean_budget={b.mean():.4f} "
              f"min={b.min():.3f} p10={np.percentile(b,10):.3f} med={np.median(b):.3f} "
              f"p90={np.percentile(b,90):.3f} max={b.max():.3f}")
        abud_budgets[variant] = b

    def abud_keep(ti, budget_frac):
        f = flat[ti]
        keep = np.zeros(f["p"].size, dtype=bool)
        if not f["total"]:
            return keep
        order = f["order_desc"]
        csum = np.cumsum(f["nt"][order])
        k = int(np.searchsorted(csum, budget_frac * f["total"]) + 1)
        keep[order[:k]] = True
        return keep

    out = {}
    if dump_probs:
        edges = np.linspace(0.0, 1.0, hist_bins + 1)
        dump = {}
        for ti, qid in enumerate(text_ids):
            f = flat[ti]
            h_tok, _ = np.histogram(f["p"], bins=edges, weights=f["nt"])
            h_word, _ = np.histogram(f["p"], bins=edges)
            dump[qid] = {
                "hist_tokens": [int(x) for x in h_tok],
                "hist_words": [int(x) for x in h_word],
                "total_tokens": f["total"],
                "total_words": int(f["p"].size),
                "n_lines": len(f["lines"]),
                "prot_tokens": int(f["nt"][f["prot"]].sum()),
                "p_max": float(f["p"].max()) if f["p"].size else 0.0,
                "p_mean": float(f["p"].mean()) if f["p"].size else 0.0,
                "question": questions[ti],
            }
        out["__probs_hist__"] = {"bins": hist_bins, "items": dump}

    if evidence_numbers:
        ev_out = {}
        for ti, qid in enumerate(text_ids):
            tg = evidence_numbers.get(qid) or []
            ev_out[qid] = evidence_ranks(ti, tg)
        out["__evidence_ranks__"] = ev_out
        print(f"evidence-rank diagnostic: {len(ev_out)} questions")

    for aggr in (aggr_levels or AGGRESSIVENESS_LEVELS):
        for variant in (variants or VARIANTS):
            if (variant.startswith("budget") or variant.startswith("linebudget")
                    or variant in dose_cfg or variant in abud_budgets) and aggr != 0.5:
                continue
            suffix = "" if variant == "plain" else f"-{variant}"
            # v2 caches get their own name so a v1/v2 A/B can sit side by side
            config_name = f"{model_alias}{suffix}{'-v2' if v2 else ''}--{aggr}"
            entry = {}
            for ti, qid in enumerate(text_ids):
                probs = all_probs[ti]
                budget_keep = set()
                line_keep = None
                dose_arr = None
                if variant in dose_cfg:
                    dose_arr = dose_keep(ti, *dose_cfg[variant])
                elif variant in abud_budgets:
                    dose_arr = abud_keep(ti, float(abud_budgets[variant][ti]))
                if variant.startswith("linebudget"):
                    # LINE-atomic budget: score each line by its max word prob,
                    # spend the token budget on whole lines in score order.
                    # Tests whether structural coherence (intact rows/sentences)
                    # closes the deep-budget gap that word confetti opens.
                    bfrac = int(re.search(r"linebudget(\d+)", variant).group(1)) / 100
                    total_toks = sum(w_len(w) for _, ws in per_text_lines[ti] for w in ws)
                    budget = int(bfrac * total_toks)
                    scored = []
                    for ref, ws in per_text_lines[ti]:
                        if not ws:
                            continue
                        s = max(probs.get((ref, wi), 0.0) for wi in range(len(ws)))
                        scored.append((s, ref, sum(w_len(w) for w in ws)))
                    scored.sort(reverse=True)
                    line_keep, used = set(), 0
                    for s, ref, n in scored:
                        if used + n > budget and line_keep:
                            continue
                        line_keep.add(ref)
                        used += n
                        if used >= budget:
                            break
                if variant.startswith("budget"):
                    # budget<NN> -> NN%. Was hardcoded to {33, else 22}, which
                    # silently rounded any other budget down to 22.
                    bfrac = int(re.search(r"budget(\d+)", variant).group(1)) / 100
                    allw = sorted(((p, k) for k, p in probs.items()), reverse=True)
                    total_toks = sum(w_len(w) for _, ws in per_text_lines[ti] for w in ws)
                    budget = int(bfrac * total_toks)
                    used = 0
                    wlen = {}
                    for ref, ws in per_text_lines[ti]:
                        for wi, w in enumerate(ws):
                            wlen[(ref, wi)] = w_len(w)
                    for p, k in allw:
                        n = wlen.get(k, 1)
                        if used + n > budget and budget_keep:
                            continue
                        budget_keep.add(k)
                        used += n
                        if used >= budget:
                            break
                kept_line_texts = []
                v2_strs, v2_lines, v2_keep = [], [], []
                orig_n = comp_n = 0
                wcur = 0
                for ref, words in per_text_lines[ti]:
                    meta = line_meta_all[ref]
                    strs = meta["strs"]
                    if dose_arr is not None:
                        keep = dose_arr[wcur:wcur + len(words)].tolist()
                        wcur += len(words)
                    elif variant.startswith("linebudget"):
                        keep = [ref in line_keep] * len(words)
                    elif variant.startswith("budget"):
                        keep = [(ref, wi) in budget_keep for wi in range(len(words))]
                    else:
                        keep = [probs.get((ref, wi), 0.0) >= aggr for wi in range(len(words))]
                    if dose_arr is None and ("safenum" in variant or "safetab" in variant):
                        for wi, s in enumerate(strs):
                            if is_protected(s):
                                keep[wi] = True
                    if dose_arr is None and "safetab" in variant:
                        for wi, s in enumerate(strs):
                            if period_re.match(s):
                                keep[wi] = True
                        # safetab2: treat 2-numeric-cell rows as tables too. The >=3
                        # threshold blinds row-label protection to two-column tables
                        # (10-Q/10-K balance sheets), which the blind-spot diagnostic
                        # found 3.3x enriched on failures.
                        # safetab2a (anchored): >=2 numerics AND last word numeric —
                        # statement rows end in numbers, MD&A prose does not; captures
                        # ~all of naive-2's blind-spot recall at +0.8pp prose FP vs +7.2.
                        n_num = sum(meta["numeric"])
                        if "safetab2a" in variant:
                            tab_ok = meta["is_table"] or (
                                n_num >= 2 and bool(meta["numeric"]) and meta["numeric"][-1])
                        elif "safetab2" in variant:
                            tab_ok = meta["is_table"] or n_num >= 2
                        else:
                            tab_ok = meta["is_table"]
                        if tab_ok and meta["first_num"] is not None and any(
                                keep[wi] for wi, n in enumerate(meta["numeric"]) if n):
                            for k in range(0, min(meta["first_num"], 12)):
                                keep[k] = True
                    orig_n += sum(w_len(w) for w in words)
                    kept = [w for w, k in zip(words, keep) if k]
                    comp_n += sum(w_len(w) for w in kept)
                    if v2:
                        v2_strs.extend(strs)
                        v2_lines.extend([line_of_ref[ref]] * len(words))
                        v2_keep.extend(keep)
                    elif kept:
                        kept_line_texts.append(
                            tokenizer.convert_tokens_to_string([tok for w in kept for tok in w]))
                if v2:
                    # nl_after from source line geometry, not from '\n' tokens
                    nl_after = [(v2_lines[i + 1] - v2_lines[i]) if i + 1 < len(v2_lines) else 0
                                for i in range(len(v2_lines))]
                    text_out = render_v2(v2_strs, nl_after, v2_keep)
                else:
                    text_out = "\n".join(kept_line_texts)
                entry[qid] = {
                    "compressed_text": text_out,
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

    if os.getenv("MAX_ITEMS"):
        n = int(os.environ["MAX_ITEMS"])
        texts, questions, text_ids = texts[:n], questions[:n], text_ids[:n]

    alias = os.getenv("ALIAS_PREFIX", "") + MODEL_ALIAS
    if os.getenv("FULLDOC_JSON"):
        fulldoc = json.load(open(os.environ["FULLDOC_JSON"]))
        texts = [fulldoc[q] for q in text_ids]
        print(f"Full-document mode ({len(texts)} filings)")
    print(f"Query-aware compression of {len(texts)} contexts with {alias}...")
    dump_probs = bool(os.getenv("DUMP_PROBS"))
    eval_v2 = bool(os.getenv("EVAL_V2"))
    if eval_v2:
        print("eval-v2 word model ON (PARITY.md P4/P5/P6/P7/P8); caches get a -v2 suffix")
    evidence_numbers = None
    if os.getenv("EVIDENCE_NUMBERS_JSON"):
        evidence_numbers = json.load(open(os.environ["EVIDENCE_NUMBERS_JSON"]))
        print(f"evidence-rank diagnostic ON ({len(evidence_numbers)} questions)")
    result = compress_all.remote(texts, questions, text_ids, MODEL_VOL_PATH, alias,
                                 int(os.getenv("SEQ_LEN", "2048")), VARIANTS,
                                 dump_probs, int(os.getenv("HIST_BINS", "1000")),
                                 eval_v2, evidence_numbers, AGGRESSIVENESS_LEVELS)

    os.makedirs(CACHE_DIR, exist_ok=True)
    ev = result.pop("__evidence_ranks__", None)
    if ev is not None:
        path = os.getenv("EVIDENCE_OUT") or os.path.join(CACHE_DIR, f"{alias}--evrank.json")
        with open(path, "w") as f:
            json.dump(ev, f)
        print(f"  evidence-rank artifact -> {path}")
    hist = result.pop("__probs_hist__", None)
    if hist is not None:
        path = os.getenv("PROBS_OUT") or os.path.join(CACHE_DIR, f"{alias}--probs.json")
        with open(path, "w") as f:
            json.dump(hist, f)
        print(f"  probs histogram artifact -> {path}")
    for config_name, entry in result.items():
        path = os.path.join(CACHE_DIR, f"{config_name}.json")
        with open(path, "w") as f:
            json.dump(entry, f)
        ratios = [
            v["compressed_tokens"] / v["original_tokens"]
            for v in entry.values() if v["original_tokens"]
        ]
        print(f"  {config_name}: mean retention {sum(ratios)/len(ratios):.3f}")
