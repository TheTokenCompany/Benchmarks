#!/usr/bin/env python3
"""Golden train/serve parity tests for the bear compression stack.

Every incident this suite exists to catch was found AFTER a training run was paid
for: evidence-centered training windows vs sliding-window serving, min-pool in eval
vs max-pool in otsofier's merge_words (PR #714), digit-run atomicity, and a
line-atomic render applied to word-trained scores. The point of the file is that a
divergence has to show up here, on a fixture, in seconds -- not in a benchmark
delta three days later.

Two kinds of check:

  check(...)          a real invariant. Failing it is a regression, exit 1.
  check(..., xfail=)  a KNOWN divergence, documented in PARITY.md under that row
                      id. It is expected to fail. If it starts PASSING the suite
                      also fails -- the debt was paid and PARITY.md is now lying.

So the suite is green today, on purpose, while recording every place our three
implementations disagree.

The prod reference is READ-ONLY at
  packages/otsofier/otsofier-rs/crates/otsofier-core/src/postprocess.rs
and its `mod tests` is the source of the invariants asserted here (Qatar's stays
one unit, 3,605,357 is atomic, seam spaces survive, no placeholder leaks).

Run:
    HF_HUB_OFFLINE=1 .venv/bin/python parity_test.py
"""

import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from v9_rl_prep import (map_tokens_to_words, newlines_after, render_mask,
                        segment_words)

TOKENIZER = "jhu-clsp/mmBERT-small"
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
_fails, _xfails, _xpasses = [], [], []


def check(name, got, want, xfail=None):
    ok = got == want
    if xfail:
        if ok:
            _xpasses.append((name, xfail))
            print(f"  XPASS {name}  (PARITY.md {xfail} is stale -- debt was paid)")
        else:
            _xfails.append((name, xfail))
            print(f"  xfail {name}  [{xfail}]  got {got!r}")
        return
    if ok:
        print(f"  ok    {name}")
    else:
        _fails.append((name, got, want))
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def section(title):
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# mirrors of the two production-side python implementations
#
# These are copies, not imports: the originals live inside modal `@app.function`
# bodies that need a GPU and a model volume to even import. `guard_mirrors_match_source`
# below fails the suite if the originals drift away from these copies.
# --------------------------------------------------------------------------- #
EVAL_BOUNDARY_SRC = 'if (tok.startswith("▁") or tok.startswith("Ġ") or not cur) and cur:'
EVAL_POOL_SRC = "min(probs_out.get(key, 1.0), tk)"
EVAL_SOURCES = ["v8_eval_precompress.py", "precompress_v72_focus.py", "precompress_v71.py"]


def group_words_eval(tokens):
    """Mirror of group_words in v8_eval_precompress / precompress_v7x.

    A word starts only at a SentencePiece/BPE space marker. Punctuation never
    opens a word, so a trailing '.' or '%' and any '\\n' token are glued onto
    whatever came before them."""
    words, cur = [], []
    for tok in tokens:
        if (tok.startswith("▁") or tok.startswith("Ġ") or not cur) and cur:
            words.append(cur)
            cur = []
        cur.append(tok)
    if cur:
        words.append(cur)
    return words


def eval_words_for_text(text, tok):
    """The eval path's full word list for a document: tokenize PER LINE with the
    newline kept in the payload, then group. Returns [(line_idx, [tokens])]."""
    out = []
    lines = text.split("\n")
    for li, line in enumerate(lines):
        payload = line + ("\n" if li < len(lines) - 1 else "")
        ids = tok(payload, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        for w in group_words_eval(tok.convert_ids_to_tokens(ids)):
            out.append((li, w))
    return out


def eval_render(kept_words_per_line, tok):
    """Mirror of the eval reconstruct: detokenize each line's kept tokens, join
    the non-empty lines with '\\n'. Lines that keep nothing disappear."""
    parts = []
    for word_toks in kept_words_per_line:
        if word_toks:
            parts.append(tok.convert_tokens_to_string([t for w in word_toks for t in w]))
    return "\n".join(parts)


def eval_chunks(words, budget):
    """Mirror of the eval chunker (v8_eval_precompress.infer_probs): pack whole
    words up to `budget` content tokens. Returns [[(word_index, token)]]."""
    chunks, cur, cur_map = [], [], []
    for wi, toks in enumerate(words):
        if len(cur) + len(toks) > budget and cur:
            chunks.append(cur_map)
            cur, cur_map = [], []
        cur.extend(toks)
        cur_map.extend([wi] * len(toks))
    if cur_map:
        chunks.append(cur_map)
    return chunks


def v9_words_for_text(text, tok):
    """The training path's word list + token->word map for the same document."""
    spans = segment_words(text)
    words = [text[s:e] for s, e in spans]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    word_id = map_tokens_to_words(enc["offset_mapping"], spans)
    return words, newlines_after(text, spans), word_id, enc


def surface(word_toks, tok):
    return tok.convert_tokens_to_string(word_toks).strip()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
TABLE = (
    "CONSOLIDATED STATEMENTS OF INCOME\n"
    "(in thousands)\n"
    "\n"
    "                                         2024        2023\n"
    "Total revenue                       $  3,605,357  $ 3,101,204\n"
    "Cost of revenue                        1, 904, 220   1,755,001\n"
    "Operating margin                            12.5 %      11.8 %\n"
    "Interest expense                              25 bps      30 bps\n"
    "Net income attributable to Qatar's subsidiary   $ 15,000.\n"
)

PROSE = (
    "The Company isn't required to adopt ASU 2023-09 until fiscal 2026.\n"
    "Management's discussion follows on page 42.\n"
)

UNICODE = "naïve café 伾 test\n该航空公司的机队迅速扩大了。\n"

SEAM = (
    "The amendment was filed with the courts. The courts rejected it.\n"
    "Revenue rose to $1,234 in the period then ended.\n"
)


# --------------------------------------------------------------------------- #
# 1. number and contraction atomicity (mirrors prod postprocess.rs mod tests)
# --------------------------------------------------------------------------- #
def test_atomicity(tok):
    section("number / contraction atomicity -- prod postprocess.rs invariants")

    def eval_units(s):
        toks = tok.convert_ids_to_tokens(tok(s, add_special_tokens=False)["input_ids"])
        return [surface(w, tok) for w in group_words_eval(toks)]

    def v9_units(s):
        return [s[a:b] for a, b in segment_words(s)]

    # prod: merge_glues_thousands_separators
    check("eval: 3,605,357 is one unit", eval_units("Total 3,605,357 spectators"),
          ["Total", "3,605,357", "spectators"])
    check("v9:   3,605,357 is one unit", v9_units("Total 3,605,357 spectators"),
          ["Total", "3,605,357", "spectators"])

    # prod: merge_glues_possessive_contraction / merge_glues_negation_contraction
    check("eval: Qatar's is one unit", eval_units("Qatar's stadium"),
          ["Qatar's", "stadium"])
    check("v9:   Qatar's is one unit", v9_units("Qatar's stadium"),
          ["Qatar's", "stadium"])
    check("eval: isn't is one unit", eval_units("it isn't willing"),
          ["it", "isn't", "willing"])

    # prod: merge_glues_real_per_digit_tokenization -- "$15,000." -> "$15,000" + "."
    # Our paths keep the sentence period inside the money word (PARITY row P3).
    check("eval: $15,000. splits its sentence period",
          eval_units("Total $15,000. Next"), ["Total", "$15,000", ".", "Next"],
          xfail="P3")
    check("v9:   $15,000. splits its sentence period",
          v9_units("Total $15,000. Next"), ["Total", "$15,000", ".", "Next"],
          xfail="P3")

    # PR #714 forward-binding currency. This is the bug fixed in v9_rl_prep this
    # pass: "2026 $ 9,353" used to segment as "2026 $" + "9,353".
    check("v9:   bare $ binds forward to its amount",
          v9_units("Contributions 2026 $ 9,353 Expected"),
          ["Contributions", "2026", "$ 9,353", "Expected"])
    check("v9:   two amounts in a column do not weld together",
          v9_units("Issuance $\n\n119\n\n$\n\n439"),
          ["Issuance", "$\n\n119", "$\n\n439"])
    check("v9:   percent still binds backward", v9_units("margin 12.5 % flat"),
          ["margin", "12.5 %", "flat"])
    check("v9:   bps still binds backward", v9_units("spread 25 bps wider"),
          ["spread", "25 bps", "wider"])
    check("v9:   whitespace-split thousands separator merges",
          v9_units("Cost 1, 904, 220 total"), ["Cost", "1, 904, 220", "total"])

    # The eval path has no cross-whitespace merge at all: a column-aligned
    # "$   3,698" is two independent keep/drop decisions, so the $ can be
    # dropped off its own amount (PARITY row P2).
    check("eval: bare $ binds forward to its amount",
          eval_units("expense $ 3,698 total"), ["expense", "$ 3,698", "total"],
          xfail="P2")
    check("eval: whitespace-split thousands separator merges",
          eval_units("Cost 1, 904 total"), ["Cost", "1, 904", "total"],
          xfail="P2")


# --------------------------------------------------------------------------- #
# 2. cross-implementation agreement on identical text
# --------------------------------------------------------------------------- #
def test_cross_implementation(tok):
    section("eval grouping vs v9 grouping on identical text")

    for label, text in [("prose", PROSE), ("table", TABLE), ("seam", SEAM)]:
        ev = [surface(w, tok) for _, w in eval_words_for_text(text, tok)]
        ev_nonempty = [s for s in ev if s]
        v9 = [text[a:b] for a, b in segment_words(text)]
        # newlines are word-internal in v9 and get stripped from the eval surface;
        # compare on whitespace-squeezed forms so only the SEGMENTATION differs.
        v9_sq = [re.sub(r"\s+", " ", s).strip() for s in v9]
        ev_sq = [re.sub(r"\s+", " ", s).strip() for s in ev_nonempty]
        # prose and running text agree; only column-aligned money diverges (P2).
        xf = "P2" if label == "table" else None
        check(f"{label}: identical word lists", ev_sq, v9_sq, xfail=xf)

    # Whitespace-only "words". Column padding tokenizes to a run of bare '▁'
    # pieces, each of which opens a word in the eval grouping and then renders
    # as nothing -- but still consumes a slot in the budget variants (row P4).
    ev = [surface(w, tok) for _, w in eval_words_for_text(TABLE, tok)]
    check("eval: no empty words in a column-aligned table",
          sum(1 for s in ev if s == ""), 0, xfail="P4")
    check("v9:   no empty words in a column-aligned table",
          sum(1 for a, b in segment_words(TABLE) if not TABLE[a:b].strip()), 0)

    # Newline pollution of the pool. The eval path tokenizes each line with its
    # trailing "\n" and the newline token opens no word, so it lands in the last
    # word of the line -- under MIN pooling that caps the row's final number
    # (row P5). The training path maps newline tokens to word_id -1.
    polluted = [surface(w, tok) for _, w in eval_words_for_text(TABLE, tok)
                if any("\n" in t for t in w) and surface(w, tok)]
    check("eval: no content word absorbs a newline token", polluted, [], xfail="P5")

    _, _, word_id, enc = v9_words_for_text(TABLE, tok)
    nl_tokens = [i for i, t in enumerate(tok.convert_ids_to_tokens(enc["input_ids"]))
                 if "\n" in t]
    check("v9:   newline tokens are non-content (word_id -1)",
          sorted({word_id[i] for i in nl_tokens}) or [-1], [-1])


# --------------------------------------------------------------------------- #
# 3. chunking and seams
# --------------------------------------------------------------------------- #
def test_seams(tok):
    section("chunking / seam handling")

    words = [w for _, w in eval_words_for_text(TABLE + PROSE + SEAM, tok)]
    chunks = eval_chunks(words, budget=40)
    check("eval: more than one chunk in the fixture", len(chunks) > 1, True)

    # A word must live in exactly one chunk. Prod's chunk_text_exact can cut
    # between two subwords of the same word (row P8); ours cannot, and that is
    # the invariant worth locking down on our side.
    owners = {}
    for ci, cmap in enumerate(chunks):
        for wi in cmap:
            owners.setdefault(wi, set()).add(ci)
    check("eval: no word is split across chunks",
          [wi for wi, cs in owners.items() if len(cs) > 1], [])
    check("eval: every word lands in a chunk", len(owners), len(words))

    # render_mask restores the seam: dropping a word must not fuse its
    # neighbours, and newlines survive a dropped word (prod's
    # reconstruct_seam_chunk_preserves_leading_space, in word space).
    w, nl, _, _ = v9_words_for_text(SEAM, tok)
    keep = [x != "The" for x in w]
    out = render_mask(w, nl, keep)
    check("v9:   dropping a word does not fuse its neighbours",
          "amendmentwas" in out or "courts.The" in out, False)
    check("v9:   newline survives a dropped word", out.count("\n"), 1)

    keep_all = [True] * len(w)
    check("v9:   all-keep round-trips the line count",
          render_mask(w, nl, keep_all).count("\n"), 1)

    # Same three-row table, same keep mask, through both renderers.
    mini = "Total revenue 3,605\nCost of revenue 1,755\nNet income 900"
    lines = mini.split("\n")
    eval_lines = []
    for li, line in enumerate(lines):
        payload = line + ("\n" if li < len(lines) - 1 else "")
        ids = tok(payload, add_special_tokens=False)["input_ids"]
        eval_lines.append(group_words_eval(tok.convert_ids_to_tokens(ids)))

    mw, mnl, _, _ = v9_words_for_text(mini, tok)
    line_of, cur = [], 0
    for nl in mnl:
        line_of.append(cur)
        cur += nl

    # Mask A: keep only the row LABEL of rows 0 and 2, drop row 1 entirely. No
    # line-final word survives, so nothing muddies the line-slot question.
    keep_a_eval = [w[:1] if li != 1 else [] for li, w in enumerate(eval_lines)]
    keep_a_v9 = [ln != 1 and mw[i] in ("Total", "Net") for i, ln in enumerate(line_of)]
    ra, rv = eval_render(keep_a_eval, tok), render_mask(mw, mnl, keep_a_v9)
    # P6: the eval render appends only non-empty lines, so a row that keeps
    # nothing loses its line slot and the rows below it slide up.
    check("eval: a dropped row keeps its line slot", len(ra.split("\n")), 3,
          xfail="P6")
    check("v9:   a dropped row keeps its line slot", len(rv.split("\n")), 3)

    # Mask B: keep everything. Line geometry must round-trip exactly.
    rb = eval_render(eval_lines, tok)
    rv_b = render_mask(mw, mnl, [True] * len(mw))
    # P7: convert_tokens_to_string re-emits the first token's '▁' as a space and
    # the eval render never strips it.
    check("eval: rendered lines have no leading space",
          [l for l in rb.split("\n") if l.startswith(" ")], [], xfail="P7")
    check("v9:   rendered lines have no leading space",
          [l for l in rv_b.split("\n") if l.startswith(" ")], [])
    # P8: the line's trailing "\n" token lives inside that line's last word, so a
    # surviving last word emits a newline AND the "\n".join adds another one.
    check("eval: all-keep round-trips the table exactly", rb, mini, xfail="P8")
    check("v9:   all-keep round-trips the table exactly", rv_b, mini)


# --------------------------------------------------------------------------- #
# 4. unicode
# --------------------------------------------------------------------------- #
def test_unicode(tok):
    section("unicode")

    ev = [surface(w, tok) for _, w in eval_words_for_text(UNICODE, tok)]
    check("eval: byte-fallback placeholder never leaks",
          [s for s in ev if "<0x" in s], [])
    check("eval: OOV char decodes to its character", "伾" in "".join(ev), True)

    v9 = [UNICODE[a:b] for a, b in segment_words(UNICODE)]
    check("v9:   OOV char survives segmentation", "伾" in v9, True)

    # prod splits unspaced scripts per token (postprocess.rs
    # merge_gives_per_token_words_for_unspaced_scripts: 9 words). Neither python
    # path has that rule, so a CJK sentence is a single keep/drop unit (row P9).
    cjk = "该航空公司的机队迅速扩大了。"
    cjk_ev = [surface(w, tok) for _, w in eval_words_for_text(cjk, tok) if surface(w, tok)]
    check("eval: CJK sentence is more than one unit", len(cjk_ev) > 1, True, xfail="P9")
    check("v9:   CJK sentence is more than one unit",
          len(segment_words(cjk)) > 1, True, xfail="P9")


# --------------------------------------------------------------------------- #
# 5. no dangling subwords in the training windows
# --------------------------------------------------------------------------- #
def test_no_dangling(tok):
    section("window edges (v11 snap_to_words)")
    try:
        from v11_pretokenize import prepare_source, slice_window, snap_to_words
    except Exception as exc:                                    # v11 may be mid-write
        print(f"  skip  v11_pretokenize unavailable ({type(exc).__name__})")
        return

    src = prepare_source({"chunk_text": TABLE + PROSE}, tok)
    check("v11:  source prepared", src is not None, True)
    if src is None:
        return

    bad = []
    for s in range(0, max(1, src["n_tok"] - 20), 3):
        a, b = snap_to_words(src, s, s + 20)
        if b <= a:
            continue
        win = slice_window(src, a, b)
        if win is None:
            continue
        first, last = src["first_tok"], src["last_tok"]
        w0, w1 = win["word_start"], win["word_end"]
        if first[w0] < a or last[w1] >= b:
            bad.append((s, w0, w1))
    check("v11:  no window edge lands inside a word", bad, [])


# --------------------------------------------------------------------------- #
# 6. the mirrors above still match the real files
# --------------------------------------------------------------------------- #
def guard_mirrors_match_source():
    section("mirror drift guard")
    for name in EVAL_SOURCES:
        p = HERE / name
        if not p.exists():
            print(f"  skip  {name} absent")
            continue
        src = p.read_text()
        check(f"{name}: word-boundary rule unchanged",
              EVAL_BOUNDARY_SRC in src, True)
    # Not an endorsement of min-pooling -- this pins the CURRENT state so the
    # migration in PARITY.md P1 is a deliberate edit, not a drift.
    p = HERE / "v8_eval_precompress.py"
    if p.exists():
        check("v8_eval_precompress.py: still MIN-pools subwords (P1 unmigrated)",
              EVAL_POOL_SRC in p.read_text(), True)


# --------------------------------------------------------------------------- #
# 7. eval-v2 (EVAL_V2=1) closes P4/P5/P6/P7/P8
#
# These import the real implementation out of v8_eval_precompress, so this is
# the shipped code path, not a mirror of it.
# --------------------------------------------------------------------------- #
def eval_v2_words_for_text(text, tok):
    """The v2 word list for a document: same per-line tokenization as v1, with
    the non-content pending buffer carried across line boundaries."""
    from v8_eval_precompress import group_document_v2
    lines = text.split("\n")
    per_line = []
    for li, line in enumerate(lines):
        payload = line + ("\n" if li < len(lines) - 1 else "")
        ids = tok(payload, add_special_tokens=False)["input_ids"]
        per_line.append((li, tok.convert_ids_to_tokens(ids) if ids else []))
    words, line_of_word = [], []
    for li, ws in group_document_v2(per_line):
        words.extend(ws)
        line_of_word.extend([li] * len(ws))
    nl_after = [(line_of_word[i + 1] - line_of_word[i]) if i + 1 < len(line_of_word) else 0
                for i in range(len(line_of_word))]
    return words, nl_after


def test_eval_v2(tok):
    section("eval-v2 (EVAL_V2=1)")
    try:
        from v8_eval_precompress import group_words_v2, is_content_tok, render_v2
    except Exception as exc:
        print(f"  skip  v8_eval_precompress unimportable ({type(exc).__name__}: {exc})")
        return

    check("v2:   '\\n' is not content", is_content_tok("\n"), False)
    check("v2:   bare '▁' is not content", is_content_tok("▁"), False)
    check("v2:   '▁Total' is content", is_content_tok("▁Total"), True)

    # P4 / P5 on the real column-aligned fixture.
    words, nl_after = eval_v2_words_for_text(TABLE, tok)
    strs = [surface(w_content_v2(w), tok) for w in words]
    check("v2:   no whitespace-only words (P4)", sum(1 for s in strs if s == ""), 0)
    polluted = [s for w, s in zip(words, strs)
                if s and any("\n" in t for t, f in zip(*w) if f)]
    check("v2:   no content token is a newline (P5)", polluted, [])

    # The encoder input is unchanged -- that is what keeps the A/B honest.
    v1_words = [w for _, w in eval_words_for_text(TABLE, tok)]
    v1_stream = [t for w in v1_words for t in w]
    v2_stream = [t for w in words for t in w[0]]
    check("v2:   encoder token stream identical to v1", v2_stream, v1_stream)

    # ...but the CHUNKING is not. infer_probs packs whole words to a token
    # budget, and a v2 word carries the whitespace run that preceded it, so the
    # greedy cuts land in different places. Measured on a real 59k-token filing:
    # 30 chunks either way, 12 of 30 starts identical, the rest off by 1-3
    # tokens. Documented in PARITY.md section 6 -- probs are near-identical, not
    # bit-identical, and any small v1/v2 delta has this underneath it.
    def pack(sizes, budget):
        starts, at, cur = [0], 0, 0
        for n in sizes:
            if cur + n > budget and cur:
                at += cur
                starts.append(at)
                cur = 0
            cur += n
        return starts

    b1 = pack([len(w) for w in v1_words], 24)
    b2 = pack([len(w[0]) for w in words], 24)
    check("v2:   chunk starts identical to v1", b2, b1, xfail="section 6")
    check("v2:   chunk count unchanged", len(b2), len(b1))

    # P6 / P7 / P8: all-keep round-trips, and a dropped row keeps its slot.
    mini = "Total revenue 3,605\nCost of revenue 1,755\nNet income 900"
    mw, mnl = eval_v2_words_for_text(mini, tok)
    mstrs = [surface(w_content_v2(w), tok) for w in mw]
    check("v2:   all-keep round-trips the table exactly (P8)",
          render_v2(mstrs, mnl, [True] * len(mw)), mini)
    check("v2:   no leading space on any line (P7)",
          [l for l in render_v2(mstrs, mnl, [True] * len(mw)).split("\n")
           if l.startswith(" ")], [])
    drop_mid = [s not in ("Cost", "of", "1,755") and not (s == "revenue" and i == 5)
                for i, s in enumerate(mstrs)]
    check("v2:   a dropped row keeps its line slot (P6)",
          len(render_v2(mstrs, mnl, drop_mid).split("\n")), 3)

    # Blank source lines survive the pending carry rather than vanishing.
    blank = "Header\n\nBody text"
    bw, bnl = eval_v2_words_for_text(blank, tok)
    bstrs = [surface(w_content_v2(w), tok) for w in bw]
    check("v2:   blank source line round-trips",
          render_v2(bstrs, bnl, [True] * len(bw)), blank)
    check("v2:   blank line's newlines stay in the encoder input",
          sum(t.count("\n") for w in bw for t in w[0]), 2)

    # Budget accounting: content tokens only.
    check("v2:   word token cost excludes padding",
          sum(sum(w[1]) for w in words) < sum(len(w[0]) for w in words), True)


def w_content_v2(w):
    return [t for t, f in zip(*w) if f]


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    guard_mirrors_match_source()
    test_atomicity(tok)
    test_cross_implementation(tok)
    test_seams(tok)
    test_unicode(tok)
    test_no_dangling(tok)
    test_eval_v2(tok)

    print(f"\n{len(_fails)} failed, {len(_xfails)} known divergences (PARITY.md), "
          f"{len(_xpasses)} stale xfail")
    if _xfails:
        rows = sorted({r for _, r in _xfails})
        print(f"  documented rows: {', '.join(rows)}")
    if _xpasses:
        print("\nA documented divergence now PASSES. Update PARITY.md and drop the "
              "xfail:\n  " + "\n  ".join(f"{r}: {n}" for n, r in _xpasses))
    if _fails or _xpasses:
        print("\nPARITY FAIL")
        return 1
    print("\nPARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
