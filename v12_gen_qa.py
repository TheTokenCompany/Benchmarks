#!/usr/bin/env python3
"""v12 synthetic QA: 12k-20k validated question/answer pairs over the v12 corpus.

Emits the v8 QA schema EXACTLY -- qa_id, file, question, gold_answer,
evidence_abs_lines, qtype -- because v10_build_targets/v11_pretokenize consume
that shape and the whole point of v12 is a bigger corpus, not a new format.

WHY EVIDENCE LINES ARE THE HARD PART
------------------------------------
The label policy (v10 build_labels) scores lines by how well they match the gold
answer's rare facts. If evidence_abs_lines is off by even a few lines, the
"evidence" the model is trained to keep is the wrong row of the wrong table, and
every downstream keep/drop label inherits that error. Two defenses:

  1. The model never guesses line numbers. It is shown a NUMBERED excerpt --
     "L4821| Net revenues $ 12,345 $ 10,987" -- and cites the numbers it sees.
     Absolute indices into the stored .txt are used throughout, so a citation is
     directly checkable against the file on disk.
  2. Every QA is validated before it is written: the gold answer's key figures
     must actually appear on the cited lines. A numeric answer needs at least one
     of its numbers on a cited line (normalized: $, commas, %, and accounting
     parens stripped); a qualitative answer needs two content words. Failures are
     dropped, not repaired -- a repaired citation is a guess wearing a hat.

WHICH WINDOWS GET ASKED ABOUT
-----------------------------
Questions are only as useful as the lines they point at, so window selection is
driven by the wave-1 blind-spot diagnostic (evrank.json / blindspots.py: 228 gold
evidence numbers the champion ranks below half the filing) rather than by raw
density. Three findings from it are encoded here:

  * 29% of blind spots are table rows with TWO numeric cells -- "Total liabilities
    11,755.0 11,490.3". v8's is_table_line needs three, so a density-only selector
    scores those rows zero and walks away from the lines the compressor most
    reliably drops. line_score credits two-cell rows equally.
  * They cluster in the middle of the document (48% at 0.25-0.50, 29% at
    0.50-0.75), not at the front where the cover tables are.
  * 90.4% are PRIMARY financial-statement rows, not notes. Windows are therefore
    weighted TOWARD primary statements (2x) with notes/schedules a secondary
    weight (1x) -- the target is >=50% of numeric questions on primary-statement
    rows and ~25% on notes, so the model learns to keep those rows when they are
    relevant rather than never being asked about them.

Measured effect over 120 real 10-K/10-Qs: two-cell rows rise from 9.4% to 19.2%
of selected lines, blind-spot-section lines from 10.1% to 16.4%, and the mean
window position moves from 0.59 to 0.50.

The QA mix is steered the same way -- see MULTISTEP_FOCUS for why multistep is
oversampled and why single-line "multistep" items are dropped rather than kept.

BATCHING AND SPEND
------------------
Generation runs on the Message Batches API (50% off) with claude-sonnet-5,
thinking disabled (the task is extraction, not reasoning, and thinking would
double the bill), and a json_schema output format so parsing cannot drift.

The budget cap is enforced BEFORE submission, not after: batches are async, so
by the time actual usage is known the money is spent. Requests are therefore
priced up front from a token estimate at the conservative non-introductory rate,
and submission stops when the projection crosses the cap. Actual usage is
reconciled from each batch's results and reported alongside the estimate.

Resumable: state.json records every submitted batch id and every harvested one.
Rerunning polls outstanding batches and picks up where it stopped -- a killed
process never loses a batch that was already paid for.

Run:
    .venv/bin/python v12_gen_qa.py --selftest
    .venv/bin/python v12_gen_qa.py --corpus $SCRATCH/v12corpus-smoke --max-docs 20 \\
        --windows-per-doc 2 --budget 2
    .venv/bin/python v12_gen_qa.py --corpus $SCRATCH/v12corpus   # full run
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SCRATCH = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-"
           "b409-cb03154f2355-financebench-check/3224cbfc-fe21-4c45-9714-a9878f8f7d10/"
           "scratchpad")
DEFAULT_CORPUS = f"{SCRATCH}/v12corpus"

MODEL = "claude-sonnet-5"
# Batches are 50% off. These are the NON-introductory sonnet-5 rates halved; the
# introductory rate is lower, so pricing against these can only overshoot the
# estimate, never the cap.
IN_PER_TOK = 3.00 / 2 / 1e6
OUT_PER_TOK = 15.00 / 2 / 1e6
MAX_OUT_TOKENS = 2000

QTYPES = ["metrics-extraction", "multistep-numerical", "domain-qualitative"]

SECRET_ID = "otso-personal-anthropic-api-key"
AWS_PROFILE, AWS_REGION = "default", "us-west-2"

NUM_WORD = re.compile(r"\d")
# numbers as they appear in filings: $1,234.5  (17,746)  12.3%  1,234
FIGURE = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\s?%?\)?")
STOP = set("""the a an and or of for to in on at by with from as is are was were be been
this that these those its it their there which what when how much many total company
during period year quarter three six nine twelve months ended reported approximately
about over under than then also into per share value net non due basic diluted""".split())

SYSTEM = """You write training questions over SEC filings for a document-compression \
model. You are given a NUMBERED excerpt of one filing. Every line is prefixed with its \
absolute line number in the source file, like:

L4821| Net revenues $ 12,345 $ 10,987

Write questions whose answers are fully supported by lines IN THIS EXCERPT, and cite \
the exact absolute line numbers that carry the answer.

Question types (produce a mix; skip a type if the excerpt does not support it):
  metrics-extraction   one reported figure read off one line. The answer is that
                       figure, with its unit/scale as the filing states it.
  multistep-numerical  a figure computed from numbers on 2-4 DIFFERENT lines (a
                       sum, a difference, a ratio, a margin, a subtotal checked
                       against its components). Cite EVERY line whose number
                       enters the computation, and state the result. A statement
                       row usually carries both periods side by side, so a
                       year-over-year change read off ONE row is NOT multistep --
                       that is metrics-extraction. Multistep means the inputs
                       live on separate lines.
  domain-qualitative   a substantive question about what the filing says -- a
                       driver of results, a risk, an accounting policy, a segment
                       narrative. The answer is one or two sentences.

Rules that decide whether the item is kept:
  * evidence_abs_lines must be the L-numbers shown in the excerpt, nothing else.
  * The literal figures in gold_answer must appear on the lines you cite. If you
    compute a value, cite the input lines AND put the inputs in the answer, e.g.
    "$2,468 million (1,234 + 1,234)".
  * The question must name the company, statement, or period specifically enough
    to be answerable against the whole filing -- not just this excerpt. Never
    write "according to the excerpt" or reference line numbers in the question.
  * Skip cover pages, signature blocks, exhibit indexes, and boilerplate legends.
    If the excerpt is all boilerplate, return an empty list.

Return 2-4 items."""

# Appended to the user turn on a share of windows. Wave-1 reader failures cluster
# on COMPONENT JUXTAPOSITION -- the answer needs two numbers that live on different
# lines, and a compressor that keeps one of them scores as if it kept neither. A
# corpus that is one-third multistep under-trains exactly that. metrics-extraction
# is the cheapest type for the model to emit, so left alone the mix drifts toward
# it; this directive pushes the other way on the windows that can support it.
MULTISTEP_FOCUS = """
PRIORITY FOR THIS EXCERPT: at least half of your items must be
multistep-numerical, and each of those MUST cite two or more different L-numbers
whose values both enter the computation (a subtotal against its components, one
statement line against another, a segment against the consolidated total, a
balance against the same balance in another section). Show the inputs in the
answer. Combining two figures printed on the SAME row -- the two period columns
of one line item -- does not qualify; that is metrics-extraction. Only fall back
to other types if this excerpt genuinely has no two lines worth relating."""

# Measured on the densest v12 windows (16 sync calls, cross-line rule enforced):
# without the directive 29% of validated items come back multistep, with it 66%.
# The directive share needed for a target mix is derived from these rather than
# guessed, because asking for "45% multistep" and applying the directive to 45%
# of windows would land at ~46-54% -- the directive is far stronger than 1:1.
MS_RATE_BASE, MS_RATE_FOCUS = 0.29, 0.66


def focus_fraction(target):
    """Share of windows that should carry MULTISTEP_FOCUS to reach `target` mix."""
    return max(0.0, min(1.0, (target - MS_RATE_BASE) / (MS_RATE_FOCUS - MS_RATE_BASE)))


SCHEMA = {
    "type": "object",
    "properties": {
        "qa": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "gold_answer": {"type": "string"},
                    "evidence_abs_lines": {"type": "array",
                                           "items": {"type": "integer"}},
                    "qtype": {"type": "string", "enum": QTYPES},
                },
                "required": ["question", "gold_answer", "evidence_abs_lines", "qtype"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# window selection
# --------------------------------------------------------------------------- #
CID_BAD = re.compile(r"[^a-zA-Z0-9_-]")


def custom_id_for(file, lo):
    """Batch custom_ids are constrained to ^[a-zA-Z0-9_-]{1,64}$, so the doc
    filename cannot be used raw (it has a dot, and could have anything else).
    This is the single source of truth for the id, used when building requests,
    when reloading already-done windows, and when keying the result index."""
    return f"{CID_BAD.sub('-', file.rsplit('.', 1)[0])}-w{lo}"[-64:]


# Where the questions should land, from the wave-1 blind-spot artifact
# (evrank.json -> blindspots.py, 228 under-ranked evidence numbers): 90.4% of
# blind-spot lines are PRIMARY financial-statement table rows, not notes. The
# corpus therefore has to make the model keep primary-statement rows when they are
# relevant, rather than steering questions away from them -- so primary statements
# are weighted ABOVE notes here, not below.
#
# Within the primary statements the balance sheet (37 hits) and cash flow (18)
# dominate the income statement (9), which the two-cell fix in line_score already
# corrects for: income statements carry three year-columns and always scored as
# tables, balance sheets carry two date-columns and never did.
PRIMARY_SECTION = re.compile(
    r"statements? of operations|statements? of income|comprehensive income|"
    r"balance sheets?|financial position|cash flows?|stockholders.? equity|"
    r"net revenue|net sales|total revenue|cost of (revenue|sales|goods|products)|"
    r"gross profit|operating (income|loss|expenses)|research and development|"
    r"selling, general|net (income|loss|earnings)|earnings per share|"
    r"total (assets|liabilit|equity|current)|accounts (receivable|payable)|"
    r"inventor|accrued|deferred revenue|cash and cash equivalents|"
    r"income taxes payable|non-?current|depreciation|amortization|"
    r"(operating|investing|financing) activities|by segment|reportable segment",
    re.I)
NOTE_SECTION = re.compile(
    r"^note \d|notes to|schedule of|consist of the following|summari[sz]es|"
    r"fair value|maturit|goodwill|intangible|operating lease|"
    r"commitments and contingencies|stock-based compensation", re.I)


def line_score(line):
    """How much a line is worth asking a question about.

    Two-numeric-cell rows score as highly as three-cell ones, and that is the
    whole point rather than a detail. The v8 is_table_line rule needs >=3 numeric
    words, so "Total liabilities 11,755.0 11,490.3" -- a two-column balance-sheet
    row -- scores zero under it. Those rows are 29% of the measured blind spots
    and include the single worst-ranked evidence number in the wave-1 diagnostic,
    so a selector that cannot see them steers windows away from exactly the
    lines the compressor already fails on.

    Long prose lines still carry the qualitative questions; short labels, page
    furniture and cover-page text stay at zero."""
    w = line.split()
    if not w:
        return 0
    nnum = sum(1 for x in w if NUM_WORD.search(x) or x in ("$", "%"))
    if nnum >= 3:
        return 3
    if nnum == 2 and len(w) >= 3:
        return 3
    if len(w) >= 25:
        return 2
    return 0


def pick_windows(lines, n_windows, win_lines):
    """-> list of (start, end) absolute line ranges, non-overlapping, best first.

    A window is measured in NON-BLANK lines (blank lines are dropped from the
    excerpt anyway) so every window carries a comparable amount of real text
    regardless of how airy that part of the document is."""
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if len(idx) < 20:
        return []
    scores = [line_score(lines[i]) for i in idx]
    # prefix sums over the non-blank sequence -> score of any window in O(1)
    pre = [0]
    for s in scores:
        pre.append(pre[-1] + s)

    # Blind-spot weighting, from the wave-1 diagnostic: 77% of under-ranked
    # evidence sits between 25% and 75% of the way through the filing, and the
    # sections it sits in are balance sheet / cash flow / notes rather than the
    # income statement. Pure density peaks at the front, where the income
    # statement and the cover-page tables are -- so density alone walks straight
    # past the region the compressor actually fails on.
    # Primary-statement rows are worth double a note/schedule row: the target mix
    # is >=50% of numeric questions on primary statements and ~25% on notes.
    sect = [2 if PRIMARY_SECTION.search(lines[i])
            else (1 if NOTE_SECTION.search(lines[i]) else 0) for i in idx]
    presect = [0]
    for s in sect:
        presect.append(presect[-1] + s)

    cands = []
    step = max(1, win_lines // 3)
    for a in range(0, max(1, len(idx) - win_lines // 2), step):
        b = min(len(idx), a + win_lines)
        if b - a < win_lines // 2:
            break
        base = pre[b] - pre[a]
        mid = (a + b) / 2 / max(1, len(idx))
        pos_w = 1.25 if 0.25 <= mid < 0.75 else 1.0
        sect_w = 1.0 + 0.5 * min(1.0, (presect[b] - presect[a]) / max(1, (b - a) * 0.15))
        cands.append((base * pos_w * sect_w, a, b))
    cands.sort(key=lambda c: -c[0])

    out, taken = [], []
    for _, a, b in cands:
        if any(a < tb and ta < b for ta, tb in taken):
            continue
        taken.append((a, b))
        out.append((idx[a], idx[b - 1]))
        if len(out) >= n_windows:
            break
    return out


def render_excerpt(lines, lo, hi):
    """Numbered excerpt of lines[lo..hi]. Blank lines are omitted -- they cost
    tokens and carry nothing -- but the numbers stay absolute, so a citation
    still indexes straight into the stored .txt."""
    return "\n".join(f"L{i}| {lines[i]}"
                     for i in range(lo, hi + 1) if lines[i].strip())


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def norm_figure(s):
    """'$ (1,234.50)%' -> '1234.5'. Trailing zeros in the decimal are dropped so
    '1,234.50' from an answer matches '1,234.5' on the line."""
    # order matters: currency/percent/space must go before the accounting parens
    # are stripped, or "$ (1,234.50)%" keeps its brackets and never matches.
    for ch in "$,% \t":
        s = s.replace(ch, "")
    s = s.strip("()").lstrip("-")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def figures(text):
    out = set()
    for m in FIGURE.findall(text):
        n = norm_figure(m)
        # single digits match everything; they prove nothing about the citation
        if len(n) >= 2 and any(c.isdigit() for c in n):
            out.add(n)
    return out


def content_words(text):
    return {w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in STOP}


def validate(qa, lines, lo, hi):
    """-> (ok, reason). The one check that matters: is the answer actually ON the
    lines the item points at?"""
    ev = qa.get("evidence_abs_lines") or []
    if not isinstance(ev, list) or not ev:
        return False, "no_evidence"
    ev = [e for e in dict.fromkeys(ev) if isinstance(e, int)]
    if not ev:
        return False, "no_evidence"
    if len(ev) > 8:
        return False, "too_many_evidence"
    if any(not (lo <= e <= hi) for e in ev):
        return False, "evidence_out_of_window"
    if any(not lines[e].strip() for e in ev):
        return False, "evidence_on_blank_line"

    q, a = qa.get("question", ""), qa.get("gold_answer", "")
    if len(q) < 25 or len(a) < 2:
        return False, "too_short"
    if re.search(r"\b(excerpt|line \d|L\d{2,})\b", q, re.I):
        return False, "question_references_excerpt"
    if qa.get("qtype") not in QTYPES:
        return False, "bad_qtype"
    # A multistep item citing one line is a same-row period comparison mislabelled
    # as multistep. Those are the easy shape; the training signal we are short of
    # is component juxtaposition ACROSS lines, so single-line multistep is dropped
    # rather than relabelled -- relabelling would quietly inflate the easy class.
    if qa["qtype"] == "multistep-numerical" and len(ev) < 2:
        return False, "multistep_single_line"

    cited = " ".join(lines[e] for e in ev)
    want = figures(a)
    if want:
        have = figures(cited)
        if not (want & have):
            return False, "no_figure_on_cited_lines"
    else:
        if len(content_words(a) & content_words(cited)) < 2:
            return False, "no_content_overlap"
    qa["evidence_abs_lines"] = sorted(ev)
    return True, "ok"


# --------------------------------------------------------------------------- #
# spend
# --------------------------------------------------------------------------- #
# Measured against messages.count_tokens on real v12 excerpts: 2.5-3.6 chars per
# token depending on how table-dense the window is (digit-heavy rows tokenize far
# worse than prose). 2.4 is below the densest case observed, so the projection
# always runs high -- which is the only safe direction when the cap has to be
# enforced before an async batch is submitted.
CHARS_PER_TOKEN = 2.4


def est_input_tokens(system, user):
    return int((len(system) + len(user)) / CHARS_PER_TOKEN) + 32


def cost(n_in, n_out):
    return n_in * IN_PER_TOK + n_out * OUT_PER_TOK


def get_api_key():
    if os.environ.get("V12_USE_ENV_KEY") and os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    out = subprocess.run(
        ["aws", "secretsmanager", "get-secret-value", "--secret-id", SECRET_ID,
         "--profile", AWS_PROFILE, "--region", AWS_REGION,
         "--query", "SecretString", "--output", "text"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


# --------------------------------------------------------------------------- #
# batch driving
# --------------------------------------------------------------------------- #
def table_density(lines, lo, hi):
    """Share of non-blank lines in a window that read as table rows. Multistep
    questions need numbers on separate lines to relate, so this is what decides
    which windows get the multistep directive."""
    body = [l for l in lines[lo:hi + 1] if l.strip()]
    if not body:
        return 0.0
    return sum(1 for l in body if line_score(l) == 3) / len(body)


def is_shell(row):
    """Blank-check shells file constantly and their statements are near-empty
    templates. They stay in the corpus as document diversity but are capped to one
    QA window, so ~5% of documents cannot become ~5% of the questions."""
    return "Blank Check" in (row.get("sic") or "")


def build_requests(corpus, max_docs, windows_per_doc, win_lines, seed, done_ids,
                   multistep_target=0.45, covered_files=frozenset()):
    """-> (requests, index). index maps custom_id -> {file, lo, hi} so results
    can be validated against the same window the model saw.

    Two sampling policies are applied here rather than in the prompt:
      * shell filers get at most one window (see is_shell);
      * the most table-dense windows get MULTISTEP_FOCUS (share derived from
        `multistep_target` via focus_fraction), since
        a window with no numbers to relate cannot produce a good multistep item
        no matter what the prompt asks for.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    man = [json.loads(l) for l in (corpus / "manifest.jsonl").read_text().splitlines()
           if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(man)
    if max_docs:
        man = man[:max_docs]

    # pass 1: enumerate candidate windows and score them for table density
    cands = []
    for row in man:
        path = corpus / "docs" / row["file"]
        if not path.exists():
            continue
        lines = path.read_text().split("\n")
        n_win = 1 if is_shell(row) else windows_per_doc
        for lo, hi in pick_windows(lines, n_win, win_lines):
            cid = custom_id_for(row["file"], lo)
            if cid in done_ids:
                continue
            excerpt = render_excerpt(lines, lo, hi)
            if len(excerpt) < 800:
                continue
            cands.append({"row": row, "lo": lo, "hi": hi, "cid": cid,
                          "excerpt": excerpt,
                          "density": table_density(lines, lo, hi)})

    # Docs with no QA yet come first. The budget cap trims this list from the end,
    # so ordering decides what survives: corpus-wide coverage beats a second helping
    # for documents an earlier round already covered.
    cands.sort(key=lambda c: (c["row"]["file"] in covered_files, -c["density"]))

    # pass 2: the densest `multistep_frac` of windows carry the multistep directive
    n_focus = int(round(len(cands) * focus_fraction(multistep_target)))
    focus = {c["cid"] for c in
             sorted(cands, key=lambda c: -c["density"])[:n_focus]}

    reqs, index = [], {}
    for c in cands:
        row, cid = c["row"], c["cid"]
        user = (f"Filing: {row['name']} ({row['ticker']}) {row['form']}, "
                f"period {row['period']}.\n\nNUMBERED EXCERPT:\n{c['excerpt']}")
        if cid in focus:
            user += "\n" + MULTISTEP_FOCUS
        reqs.append(Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_OUT_TOKENS,
                system=SYSTEM,
                thinking={"type": "disabled"},
                output_config={"format": {"type": "json_schema",
                                          "schema": SCHEMA}},
                messages=[{"role": "user", "content": user}],
            )))
        index[cid] = {"file": row["file"], "lo": c["lo"], "hi": c["hi"],
                      "ticker": row["ticker"], "cik": row["cik"],
                      "shell": is_shell(row), "multistep_focus": cid in focus,
                      "est_in": est_input_tokens(SYSTEM, user)}
    return reqs, index


def harvest(client, batch_id, index, corpus, out_f, stats):
    """Stream one finished batch's results into qa.jsonl. -> (actual_in, actual_out)."""
    a_in = a_out = 0
    cache = {}
    for res in client.messages.batches.results(batch_id):
        cid = res.custom_id
        meta = index.get(cid)
        if res.result.type != "succeeded":
            stats[f"batch_{res.result.type}"] += 1
            continue
        msg = res.result.message
        a_in += msg.usage.input_tokens + (msg.usage.cache_creation_input_tokens or 0)
        a_out += msg.usage.output_tokens
        if meta is None:
            stats["unknown_custom_id"] += 1
            continue
        if msg.stop_reason == "refusal":
            stats["refusal"] += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            stats["unparseable"] += 1
            continue

        path = corpus / "docs" / meta["file"]
        if meta["file"] not in cache:
            cache[meta["file"]] = path.read_text().split("\n")
        lines = cache[meta["file"]]

        for k, qa in enumerate(payload.get("qa", [])):
            stats["generated"] += 1
            ok, why = validate(qa, lines, meta["lo"], meta["hi"])
            stats[f"drop_{why}" if not ok else "validated"] += 1
            if not ok:
                continue
            out_f.write(json.dumps({
                "qa_id": f"{meta['file'][:-4]}_w{meta['lo']}_q{k:02d}",
                "file": meta["file"],
                "question": qa["question"],
                "gold_answer": qa["gold_answer"],
                "evidence_abs_lines": qa["evidence_abs_lines"],
                "qtype": qa["qtype"],
                "ticker": meta["ticker"], "cik": meta["cik"],
                "window": [meta["lo"], meta["hi"]], "validated": True,
            }) + "\n")
    out_f.flush()
    return a_in, a_out


def poll(client, batch_id, quiet=False):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            return b
        if not quiet:
            c = b.request_counts
            print(f"      {batch_id}: {b.processing_status} "
                  f"(done {c.succeeded}, err {c.errored}, left {c.processing})",
                  flush=True)
        time.sleep(30)


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest():
    lines = (["ACME CORP", "", "FORM 10-Q", ""]
             + ["boilerplate legend line"] * 14
             + ["", "CONSOLIDATED STATEMENTS OF OPERATIONS", "",
                "September 2025 September 2024", "",
                "Net revenues $ 12,345 $ 10,987",
                "Cost of products sold $ 5,000 $ 4,500",
                "Operating earnings $ 2,100 $ 1,900",
                "Net (loss) income $ (17,746) $ 2,655", ""]
             + ["Management believes results reflect pricing actions and volume "
                "growth across the segments during the reported period, offset by "
                "unfavourable currency effects and higher input costs." ] * 3)

    assert line_score("Net revenues $ 12,345 $ 10,987") == 3
    assert line_score("FORM 10-Q") == 0
    # the wave-1 blind-spot shape: a two-column balance-sheet row. v8's
    # is_table_line scores this 0; the selector must not.
    assert line_score("Total liabilities 11,755.0  11,490.3") == 3
    assert line_score("Other non-current liabilities 223.2 241.0") == 3
    assert line_score("2024 2023") == 0          # bare period header, not a row
    assert PRIMARY_SECTION.search("Total liabilities 11,755.0 11,490.3")
    assert PRIMARY_SECTION.search("Net revenues $ 12,345 $ 10,987")
    assert PRIMARY_SECTION.search("CONSOLIDATED STATEMENTS OF CASH FLOWS")
    assert NOTE_SECTION.search("NOTE 7 - Fair value measurements")
    assert not PRIMARY_SECTION.search("The registrant is a large accelerated filer")
    wins = pick_windows(lines, 1, 14)
    assert wins, "no window picked"
    lo, hi = wins[0]
    ex = render_excerpt(lines, lo, hi)
    assert "L" in ex and "| " in ex and "\n\n" not in ex, "blank lines leaked"
    rev = [l for l in ex.split("\n") if "Net revenues" in l]
    assert len(rev) == 1 and rev[0].startswith(f"L{lines.index('Net revenues $ 12,345 $ 10,987')}|"), rev

    assert norm_figure("$ (1,234.50)%") == "1234.5"
    assert figures("$2,468 million (1,234 + 1,234)") == {"2468", "1234"}

    nrev = lines.index("Net revenues $ 12,345 $ 10,987")
    npro = lines.index([l for l in lines if l.startswith("Management")][0])
    good = {"question": "What were ACME's net revenues for the period ended "
                        "September 2025 per the statements of operations?",
            "gold_answer": "$12,345", "evidence_abs_lines": [nrev],
            "qtype": "metrics-extraction"}
    assert validate(dict(good), lines, 0, len(lines) - 1) == (True, "ok")

    for bad, why in [
        ({**good, "gold_answer": "$99,999"}, "no_figure_on_cited_lines"),
        ({**good, "evidence_abs_lines": [nrev + 1]}, "no_figure_on_cited_lines"),
        ({**good, "evidence_abs_lines": []}, "no_evidence"),
        ({**good, "evidence_abs_lines": [1]}, "evidence_on_blank_line"),
        ({**good, "evidence_abs_lines": [nrev], "question": "What is on line 5 of "
          "the excerpt for this filing period?"}, "question_references_excerpt"),
        ({**good, "qtype": "nonsense"}, "bad_qtype"),
    ]:
        ok, got = validate(dict(bad), lines, 0, len(lines) - 1)
        assert not ok and got == why, (why, got)

    single = {**good, "qtype": "multistep-numerical"}
    assert validate(dict(single), lines, 0, len(lines) - 1) == (
        False, "multistep_single_line")
    pair = {**good, "qtype": "multistep-numerical",
            "gold_answer": "$17,345 (12,345 + 5,000)",
            "evidence_abs_lines": [nrev, nrev + 1]}
    assert validate(dict(pair), lines, 0, len(lines) - 1) == (True, "ok")

    qual = {"question": "According to management, what drove the change in operating "
                        "results during the reported period?",
            "gold_answer": "Pricing actions and volume growth across the segments, "
                           "offset by unfavourable currency effects.",
            "evidence_abs_lines": [npro], "qtype": "domain-qualitative"}
    assert validate(dict(qual), lines, 0, len(lines) - 1) == (True, "ok")
    qual_bad = {**qual, "gold_answer": "A litigation settlement with the regulator."}
    assert validate(dict(qual_bad), lines, 0, len(lines) - 1)[1] == "no_content_overlap"

    dense = ["Net revenues $ 12,345 $ 10,987"] * 8 + ["short"] * 2
    prose = ["a plain sentence with no numbers at all in it here"] * 10
    assert table_density(dense, 0, 9) == 0.8, table_density(dense, 0, 9)
    assert table_density(prose, 0, 9) == 0.0
    assert table_density([""], 0, 0) == 0.0
    assert is_shell({"sic": "Blank Checks"}) and not is_shell({"sic": "Pharmaceutical"})
    assert not is_shell({})

    assert abs(focus_fraction(0.45) - 0.432) < 0.01, focus_fraction(0.45)
    assert focus_fraction(0.20) == 0.0 and focus_fraction(0.99) == 1.0

    c = cost(4500, 600)
    assert 0.010 < c < 0.013, c
    print("selftest OK: window scoring, numbered excerpt, figure normalization, "
          f"7 validator rejections, multistep cross-line rule, shell/density gates, "
          f"cost model (${c:.4f}/req)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=None, help="default <corpus>/qa.jsonl")
    ap.add_argument("--budget", type=float, default=150.0, help="USD hard cap")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--windows-per-doc", type=int, default=3)
    ap.add_argument("--win-lines", type=int, default=110,
                    help="non-blank lines per excerpt. Filing table rows are long, "
                         "so 110 lines is already ~8k input tokens on a 10-K")
    ap.add_argument("--multistep-target", type=float, default=0.45,
                    help="target multistep-numerical share of the QA mix; the "
                         "directive share is derived from measured response rates")
    ap.add_argument("--batch-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1212)
    ap.add_argument("--submit-only", action="store_true",
                    help="submit batches and exit without polling. Lets a second "
                         "round queue while the first is still processing; a "
                         "single later --harvest-only run collects them all, so "
                         "two concurrent processes never race on state.json")
    ap.add_argument("--harvest-only", action="store_true",
                    help="submit nothing; poll and harvest every outstanding batch")
    ap.add_argument("--dry-run", action="store_true",
                    help="build requests, print the cost projection, submit nothing")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    corpus = Path(args.corpus)
    out_path = Path(args.out) if args.out else corpus / "qa.jsonl"
    state_path = corpus / "qa_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "batches": {}, "spend_actual": 0.0, "spend_est": 0.0}

    done_ids, covered_files = set(), set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_ids.add(custom_id_for(r["file"], r["window"][0]))
                covered_files.add(r["file"])
    # a submitted-but-unharvested batch must not have its windows rebuilt
    for b in state["batches"].values():
        if b.get("status") != "harvested":
            done_ids.update(b.get("custom_ids", []))
            covered_files.update(b.get("files", []))

    print(f"[1/4] building requests from {corpus}")
    reqs, index = build_requests(corpus, args.max_docs, args.windows_per_doc,
                                 args.win_lines, args.seed, done_ids,
                                 args.multistep_target, covered_files)
    # A resumed run rebuilds NO requests for windows it already submitted (they
    # are in done_ids), so the fresh index has no entry for them -- and harvest
    # keys on the index. Without persisting it, resuming after a crash would poll
    # a batch that was already paid for and then discard every result as an
    # unknown custom_id. The index is therefore merged across runs on disk.
    index_path = corpus / "qa_index.json"
    if index_path.exists():
        index = {**json.loads(index_path.read_text()), **index}
    index_path.write_text(json.dumps(index))
    est_in = sum(index[r["custom_id"]]["est_in"] for r in reqs)
    est_out = len(reqs) * 850          # measured: ~790 output tokens per window
    proj = cost(est_in, est_out)
    n_focus = sum(1 for r in reqs if index[r["custom_id"]]["multistep_focus"])
    n_shell = sum(1 for r in reqs if index[r["custom_id"]]["shell"])
    print(f"      {len(reqs)} windows over "
          f"{len({index[r['custom_id']]['file'] for r in reqs})} docs "
          f"({n_focus} multistep-focused, {n_shell} from shell filers)")
    print(f"      projected: {est_in/1e6:.2f}M in + {est_out/1e6:.2f}M out "
          f"= ${proj:.2f} (cap ${args.budget:.2f}, already spent "
          f"${state['spend_actual']:.2f})")

    # trim to the cap BEFORE submitting: batches are async, so an over-budget
    # submission cannot be recalled.
    remaining = args.budget - state["spend_actual"]
    if proj > remaining:
        keep, acc = 0, 0.0
        for r in reqs:
            c = cost(index[r["custom_id"]]["est_in"], 850)
            if acc + c > remaining:
                break
            acc += c
            keep += 1
        print(f"      ! cap-limited: submitting {keep}/{len(reqs)} windows "
              f"(${acc:.2f} of ${remaining:.2f} remaining)")
        reqs = reqs[:keep]
    if not reqs and not any(b.get("status") != "harvested"
                            for b in state["batches"].values()):
        print("nothing to do")
        return
    if args.dry_run:
        print("dry-run: nothing submitted")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key())

    if args.harvest_only:
        reqs = []
    print(f"[2/4] submitting {len(reqs)} requests in batches of {args.batch_size}")
    for i in range(0, len(reqs), args.batch_size):
        chunk = reqs[i:i + args.batch_size]
        b = client.messages.batches.create(requests=chunk)
        state["batches"][b.id] = {
            "status": "submitted", "n": len(chunk),
            "custom_ids": [r["custom_id"] for r in chunk],
            "files": sorted({index[r["custom_id"]]["file"] for r in chunk}),
            "est": cost(sum(index[r["custom_id"]]["est_in"] for r in chunk),
                        len(chunk) * 850)}
        state["spend_est"] += state["batches"][b.id]["est"]
        state_path.write_text(json.dumps(state))
        print(f"      {b.id}  {len(chunk)} reqs  est ${state['batches'][b.id]['est']:.2f}")

    if args.submit_only:
        print(f"[3/4] --submit-only: {len(state['batches'])} batch(es) on record; "
              f"rerun with --harvest-only to collect")
        return
    print("[3/4] polling + harvesting")
    stats = Counter()
    out_f = out_path.open("a")
    try:
        for bid, meta in list(state["batches"].items()):
            if meta.get("status") == "harvested":
                continue
            poll(client, bid)
            a_in, a_out = harvest(client, bid, index, corpus, out_f, stats)
            spent = cost(a_in, a_out)
            meta.update(status="harvested", actual_in=a_in, actual_out=a_out,
                        actual=spent)
            state["spend_actual"] += spent
            state_path.write_text(json.dumps(state))
            print(f"      {bid} harvested: {a_in/1e3:.0f}k in {a_out/1e3:.0f}k out "
                  f"= ${spent:.2f} (running ${state['spend_actual']:.2f})")
    finally:
        out_f.close()
        state_path.write_text(json.dumps(state))

    n_qa = sum(1 for l in out_path.read_text().splitlines() if l.strip())
    by_t = Counter(json.loads(l)["qtype"] for l in out_path.read_text().splitlines()
                   if l.strip())
    print(f"[4/4] {n_qa} validated QA in {out_path}")
    ms = by_t.get("multistep-numerical", 0) / max(1, n_qa)
    print(f"      qtypes: {dict(by_t)}  (multistep {ms:.1%}, target "
          f"{args.multistep_target:.0%})")
    print(f"      this run: generated={stats['generated']} "
          f"validated={stats['validated']} "
          f"yield={stats['validated']/max(1,stats['generated']):.1%}")
    for k, v in sorted(stats.items()):
        if k.startswith("drop_") or k in ("refusal", "unparseable"):
            print(f"        {k}: {v}")
    print(f"      spend: ${state['spend_actual']:.2f} of ${args.budget:.2f}")


if __name__ == "__main__":
    main()
