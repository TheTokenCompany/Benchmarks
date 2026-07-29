#!/usr/bin/env python3
"""v12 corpus fetch: 1,500-2,500 clean EDGAR filings, zero FinanceBench overlap.

The v8/v11 corpus (300 filings) was built by an agent whose scratchpad was wiped;
only pretokenized tensors survived. v12 rebuilds it 5-10x larger AND from scratch,
with the exclusion policy enforced in code rather than by hand.

CLEAN DATA POLICY (the whole point of this file)
------------------------------------------------
FinanceBench is the held-out eval. Any of its 33 companies appearing in the
training corpus makes every FinanceBench number a lie. Exclusion is by CIK, and
CIKs are resolved from a hand-written ticker map rather than name matching --
"JOHNSON" in a doc_name could be Johnson & Johnson or Johnson Controls, and a
fuzzy match that guesses wrong either leaks a company in or drops a clean one.
Name-substring matching runs as a SECOND net on top of the CIK check, so a
subsidiary or renamed filer that shares a name still gets dropped.

The exclusion set is written to exclusions.json next to the corpus, and
v12_verify.py re-asserts it against the finished manifest.

WHAT GETS FETCHED
-----------------
Four form families, because the compressor serves all of them and a 10-K-only
corpus teaches only one document shape:
    10-K / 10-K405 / 10-KSB   annual, long, statement-heavy
    10-Q                       quarterly, shorter, same table grammar
    8-K                        event-driven, mostly prose, often tiny
    EX-99 (from 8-K)           earnings press releases: the densest tables in EDGAR

Each filer's documents are sampled ACROSS YEARS rather than newest-first (see
pick_filings): the obvious "take the latest N" produced a corpus that was 98%
2025, since every active filer's most recent 10-Q is from the current year.

Blank-check shells (SPACs) are KEPT -- they are legitimate document diversity --
but v12_gen_qa caps them to one QA window each, so ~5% of documents cannot become
~5% of the questions off the back of near-empty template statements.

Filers are drawn from the SEC's own ticker->exchange file (~10k listed filers),
shuffled with a fixed seed, and walked until the per-form quotas fill. That is a
deliberate approximation of "S&P 400/600 + random filers": the SEC publishes no
index membership, and the mid/small-cap tail is what the random draw is mostly
made of anyway once the ~30 FinanceBench mega-caps are removed. Mega-cap bias is
further reduced by capping documents per filer (--per-filer), so no single large
company can dominate.

TEXT EXTRACTION
---------------
EDGAR HTML is inline-XBRL: every reported number is wrapped in <ix:nonFraction>
with a hidden <ix:header> block carrying the whole taxonomy. Naive text
extraction yields either the taxonomy dump or numbers glued to their labels.
Here: <ix:header> is deleted, every other ix:* tag is unwrapped in place (the
number stays, the markup goes), then tables are rendered ROW PER LINE with cells
space-joined and paragraphs are rendered ONE PER LINE -- the line grammar
v8_build_masks/v10_build_targets already parse (is_table_line = >=3 numeric
words on a line).

OUTPUT
------
$OUT/docs/<cik>_<form>_<period>[_<n>].txt   plain text, one doc per file
$OUT/manifest.jsonl                          one row per doc (append-only)
$OUT/exclusions.json                         the enforced FinanceBench blocklist

Resumable: manifest is read on startup, already-fetched accessions are skipped,
and every new doc is flushed to disk + manifest immediately. Ctrl-C is safe.

Run:
    .venv/bin/python v12_fetch_corpus.py --selftest
    .venv/bin/python v12_fetch_corpus.py --target 20 --out $SCRATCH/v12corpus-smoke
    .venv/bin/python v12_fetch_corpus.py                    # full 2000-doc build
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SCRATCH = ("/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-"
           "b409-cb03154f2355-financebench-check/3224cbfc-fe21-4c45-9714-a9878f8f7d10/"
           "scratchpad")
DEFAULT_OUT = f"{SCRATCH}/v12corpus"

# EDGAR etiquette: declared UA with a real contact, <=10 req/s. We run at 8.
CONTACT = "otso.veistera@gmail.com"
USER_AGENT = f"TTC-compression-research {CONTACT}"
MAX_RPS = 8.0

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_DIR = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}"

FORM_FAMILIES = {
    "10-K": ("10-K", "10-K405", "10-KSB", "10-K/A"),
    "10-Q": ("10-Q", "10-Q/A"),
    "8-K": ("8-K", "8-K/A"),
}
DATE_LO, DATE_HI = "2019-01-01", "2025-12-31"

# FinanceBench company prefix -> ticker. Hand-written on purpose: the doc_name
# prefixes are ambiguous ("JOHNSON", "BLOCK", "AES") and a fuzzy resolver that
# guesses wrong silently poisons the corpus. BLOCK renamed SQ -> XYZ in 2025 and
# ACTIVISIONBLIZZARD was acquired by Microsoft in 2023, so both tickers are
# checked and a miss on those two alone is tolerated (see resolve_exclusions).
FB_TICKERS = {
    "3M": ["MMM"], "ACTIVISIONBLIZZARD": ["ATVI"], "ADOBE": ["ADBE"], "AES": ["AES"],
    "AMAZON": ["AMZN"], "AMCOR": ["AMCR"], "AMD": ["AMD"], "AMERICANEXPRESS": ["AXP"],
    "AMERICANWATERWORKS": ["AWK"], "BESTBUY": ["BBY"], "BLOCK": ["XYZ", "SQ"],
    "BOEING": ["BA"], "COCACOLA": ["KO"], "CORNING": ["GLW"], "COSTCO": ["COST"],
    "CVSHEALTH": ["CVS"], "FOOTLOCKER": ["FL"], "GENERALMILLS": ["GIS"],
    "JOHNSON": ["JNJ"], "JPMORGAN": ["JPM"], "KRAFTHEINZ": ["KHC"],
    "LOCKHEEDMARTIN": ["LMT"], "MGMRESORTS": ["MGM"], "MICROSOFT": ["MSFT"],
    "NETFLIX": ["NFLX"], "NIKE": ["NKE"], "PAYPAL": ["PYPL"], "PEPSICO": ["PEP"],
    "PFIZER": ["PFE"], "ULTABEAUTY": ["ULTA"], "VERIZON": ["VZ"], "WALMART": ["WMT"],
}
# Second net: any filer whose SEC name contains one of these (normalized, no
# spaces/punct) is dropped regardless of CIK. Catches subsidiaries, renamed
# entities, and the acquired-company case.
FB_NAME_FRAGMENTS = [
    "3M", "ACTIVISIONBLIZZARD", "ADOBE", "AESCORP", "AMAZONCOM", "AMCOR",
    "ADVANCEDMICRODEVICES", "AMERICANEXPRESS", "AMERICANWATERWORKS", "BESTBUY",
    "BLOCKINC", "SQUAREINC", "BOEING", "COCACOLA", "CORNINGINC", "COSTCOWHOLESALE",
    "CVSHEALTH", "FOOTLOCKER", "GENERALMILLS", "JOHNSONJOHNSON", "JPMORGAN",
    "KRAFTHEINZ", "LOCKHEEDMARTIN", "MGMRESORTS", "MICROSOFT", "NETFLIX", "NIKEINC",
    "PAYPAL", "PEPSICO", "PFIZER", "ULTABEAUTY", "VERIZON", "WALMART",
]

WS = re.compile(r"[ \t\xa0  ]+")
BLANKRUN = re.compile(r"\n{3,}")
NONALNUM = re.compile(r"[^A-Z0-9]")
PAREN_OPEN = re.compile(r"\(\s+")
PAREN_CLOSE = re.compile(r"\s+\)")


# --------------------------------------------------------------------------- #
# rate-limited HTTP
# --------------------------------------------------------------------------- #
class Throttle:
    """Global token bucket. Shared by every worker thread so the process-wide
    rate stays under the SEC limit no matter how many threads run."""

    def __init__(self, rps):
        self.min_gap = 1.0 / rps
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.min_gap
        gap = at - time.monotonic()
        if gap > 0:
            time.sleep(gap)


class Fetcher:
    def __init__(self, throttle, retries=4):
        self.throttle = throttle
        self.retries = retries
        self.local = threading.local()
        self.n_req = 0
        self.n_bytes = 0
        self.counter_lock = threading.Lock()

    @property
    def session(self):
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT,
                              "Accept-Encoding": "gzip, deflate"})
            self.local.s = s
        return s

    def get(self, url, as_json=False):
        last = None
        for attempt in range(self.retries):
            self.throttle.wait()
            try:
                r = self.session.get(url, timeout=45)
            except requests.RequestException as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
                continue
            with self.counter_lock:
                self.n_req += 1
                self.n_bytes += len(r.content)
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429, 500, 502, 503, 504):
                # 403 from EDGAR means "you are going too fast", not "forbidden"
                time.sleep(2.0 * (attempt + 1))
                last = RuntimeError(f"HTTP {r.status_code}")
                continue
            r.raise_for_status()
            return r.json() if as_json else r
        raise RuntimeError(f"giving up on {url}: {last}")


# --------------------------------------------------------------------------- #
# exclusions
# --------------------------------------------------------------------------- #
def norm_name(s):
    return NONALNUM.sub("", (s or "").upper())


def resolve_exclusions(filers, strict=True):
    """-> (set of excluded CIKs, report dict). Hard-fails if too many misses.

    filers: list of {cik, name, ticker, exchange}. A FinanceBench ticker that
    resolves to no CIK is only tolerated for companies known to have left the
    ticker file (delisted / acquired / renamed) -- those are still caught by the
    name-fragment net, which is why the tolerance exists at all.
    """
    by_ticker = {}
    for f in filers:
        by_ticker.setdefault((f["ticker"] or "").upper(), f)

    cik_excl, resolved, missing = set(), {}, []
    for company, tickers in FB_TICKERS.items():
        hit = next((by_ticker[t] for t in tickers if t in by_ticker), None)
        if hit is None:
            missing.append(company)
            continue
        cik_excl.add(int(hit["cik"]))
        resolved[company] = {"cik": int(hit["cik"]), "ticker": hit["ticker"],
                             "sec_name": hit["name"]}

    # name net, applied to the whole filer universe
    frag_excl = set()
    for f in filers:
        n = norm_name(f["name"])
        if any(frag in n for frag in FB_NAME_FRAGMENTS):
            frag_excl.add(int(f["cik"]))

    if strict and len(missing) > 3:
        raise SystemExit(f"exclusion resolution failed: {len(missing)} FinanceBench "
                         f"companies unresolved ({missing}). Refusing to fetch -- a "
                         f"silent miss here contaminates the eval.")
    report = {"resolved": resolved, "unresolved_tickers": missing,
              "n_cik_by_ticker": len(cik_excl), "n_cik_by_name": len(frag_excl),
              "n_cik_total": len(cik_excl | frag_excl)}
    return cik_excl | frag_excl, report


# --------------------------------------------------------------------------- #
# HTML / iXBRL -> line-oriented plain text
# --------------------------------------------------------------------------- #
BLOCK_TAGS = ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "blockquote", "hr", "ul", "ol", "dl", "dt", "dd")
IX_JUNK = ("header", "hidden", "references", "resources", "relationship")


def is_ixbrl_junk(tag):
    n = (tag.name or "").lower()
    return (n.startswith("ix:") or n.startswith("xbrl") or n.startswith("link:")
            or n.startswith("xbrli:")) and any(j in n for j in IX_JUNK)


def is_ixbrl_wrapper(tag):
    n = (tag.name or "").lower()
    return n.startswith("ix:") or n.startswith("xbrli:")


def html_to_text(html):
    """EDGAR HTML -> text with the line grammar the label policy expects:
    one line per table row (cells space-joined), one line per paragraph."""
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    # EDGAR serves plenty of XHTML; the HTML parser handles it fine and the
    # warning would fire once per document.
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")

    for t in soup.find_all(["script", "style"]):
        t.decompose()
    # inline XBRL: the header is a taxonomy dump (delete), every other ix:* tag
    # is a wrapper around a real reported number (unwrap, keep the number).
    for t in soup.find_all(is_ixbrl_junk):
        t.decompose()
    for t in soup.find_all(attrs={"style": re.compile(r"display\s*:\s*none", re.I)}):
        t.decompose()
    for t in soup.find_all(is_ixbrl_wrapper):
        t.unwrap()

    out = []

    def cell_text(el):
        return WS.sub(" ", el.get_text(" ", strip=True)).strip()

    for tbl in soup.find_all("table"):
        rows = []
        for tr in tbl.find_all("tr"):
            cells = [cell_text(td) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                # "$" stays its own word (v8's is_numeric_word counts it, and the
                # v8 corpus rows look like "Net revenues $ 12,345 $ 10,987"), but
                # a bracket split across cells is reattached so a negative reads
                # as one word "(1,234)" instead of three.
                rows.append(PAREN_OPEN.sub("(", PAREN_CLOSE.sub(
                    ")", " ".join(cells))))
        tbl.replace_with("\n\n" + "\n".join(rows) + "\n\n" if rows else "\n")

    # A joined-on-newline get_text() would split every paragraph that contains a
    # <span> or <b> -- and EDGAR wraps dates, amounts, and emphasis in spans
    # constantly, so "For the fiscal year ended December 31, 2024" arrives as
    # three lines. Break only on BLOCK elements; inline tags contribute no
    # separator, which is what keeps a paragraph on one line.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in list(soup.find_all(BLOCK_TAGS)):
        blk.insert_before("\n")
        blk.insert_after("\n")

    text = soup.get_text("")
    lines = []
    for raw in text.split("\n"):
        ln = WS.sub(" ", raw).strip()
        lines.append(ln)
    text = "\n".join(lines)
    text = BLANKRUN.sub("\n\n", text)
    return text.strip()


def pdf_to_text(blob):
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(blob))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def doc_to_text(resp):
    ct = (resp.headers.get("Content-Type") or "").lower()
    url = resp.url.lower()
    if "pdf" in ct or url.endswith(".pdf"):
        return pdf_to_text(resp.content)
    if url.endswith(".txt") and "<" not in resp.text[:2000]:
        return WS.sub(" ", resp.text).strip()
    return html_to_text(resp.text)


def approx_tokens(text):
    """Word count is the unit the mask/label policy operates on; a real
    tokenizer count is a v12_pretokenize concern, not a fetch concern."""
    return len(text.split())


# --------------------------------------------------------------------------- #
# EDGAR walking
# --------------------------------------------------------------------------- #
def iter_filings(sub):
    """Yield (form, filing_date, period, accession, primary_doc) from a
    submissions JSON `recent` block. Older shards are ignored: they are pre-2019
    for almost every filer, and 2019-2025 is the window."""
    rec = sub.get("filings", {}).get("recent", {})
    n = len(rec.get("accessionNumber", []))
    for i in range(n):
        yield (rec["form"][i], rec["filingDate"][i], rec.get("reportDate", [""] * n)[i],
               rec["accessionNumber"][i], rec.get("primaryDocument", [""] * n)[i])


def pick_filings(sub, quotas, rng):
    """-> list of (family, form, date, period, accn, primary), SPREAD ACROSS YEARS.

    Taking the newest N filings per filer looks reasonable and is not: every
    established filer's newest 10-Q is from the current year, so a whole corpus
    built that way lands ~98% in one year (measured: 1309/1335 in 2025 on the
    first attempt). The submissions `recent` block reaches back to 2014-2021 for
    most filers, so the history is right there -- it just has to be sampled
    rather than truncated. Filings are bucketed by year and the quota is filled
    round-robin over a shuffled year order, so a filer contributing 2 documents
    contributes them from 2 different years wherever it can."""
    by_fam = defaultdict(lambda: defaultdict(list))
    for form, fdate, period, accn, primary in iter_filings(sub):
        if not (DATE_LO <= fdate <= DATE_HI):
            continue
        fam = next((f for f, forms in FORM_FAMILIES.items() if form in forms), None)
        if fam is None:
            continue
        by_fam[fam][fdate[:4]].append((fam, form, fdate, period or fdate, accn, primary))

    picked = []
    for fam, quota in quotas.items():
        buckets = by_fam.get(fam)
        if not buckets:
            continue
        years = sorted(buckets)
        rng.shuffle(years)
        n_avail = sum(len(v) for v in buckets.values())
        i = 0
        while quota > 0 and n_avail > 0:
            y = years[i % len(years)]
            b = buckets[y]
            if b:
                picked.append(b.pop(rng.randrange(len(b))))
                quota -= 1
                n_avail -= 1
            i += 1
    return picked


def ex99_docs(fetcher, cik, accn_nodash):
    """EX-99 exhibits on an 8-K -- the earnings press releases. Returns
    [(filename, description)]. The filing index is one extra request per 8-K,
    which is why 8-K quotas stay small."""
    idx = fetcher.get(f"{ARCHIVE_DIR.format(cik=cik, accn=accn_nodash)}/index.json",
                      as_json=True)
    if not idx:
        return []
    out = []
    for item in idx.get("directory", {}).get("item", []):
        name = item.get("name", "")
        if not name.lower().endswith((".htm", ".html", ".txt")):
            continue
        if re.match(r"^ex-?99", name, re.I) or "ex99" in name.lower():
            out.append(name)
    return out[:2]


def safe_stem(cik, form, period, seq):
    form_s = re.sub(r"[^A-Z0-9]", "", form.upper())
    per = (period or "").replace("-", "")[:8] or "NA"
    return f"{cik}_{form_s}_{per}" + (f"_{seq}" if seq else "")


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def fetch_filer(fetcher, filer, quotas, seen_accn, docs_dir, min_words, want_ex99,
                seed=0):
    """-> list of manifest rows for one filer. Never raises: a filer that 404s,
    times out, or serves unparseable HTML is skipped, not fatal."""
    cik = int(filer["cik"])
    rows = []
    try:
        sub = fetcher.get(SUBMISSIONS_URL.format(cik=cik), as_json=True)
    except Exception:
        return rows
    if not sub:
        return rows
    name = sub.get("name") or filer["name"]
    sic = sub.get("sicDescription", "")
    rng = random.Random(f"{cik}:{seed}")
    for fam, form, fdate, period, accn, primary in pick_filings(sub, quotas, rng):
        accn_nodash = accn.replace("-", "")
        targets = []
        if primary:
            targets.append((primary, fam, form))
        if fam == "8-K" and want_ex99:
            for ex in ex99_docs(fetcher, cik, accn_nodash):
                targets.append((ex, "EX-99", "EX-99"))

        for seq, (fname, out_fam, out_form) in enumerate(targets):
            key = f"{accn}:{fname}"
            if key in seen_accn:
                continue
            url = f"{ARCHIVE_DIR.format(cik=cik, accn=accn_nodash)}/{fname}"
            try:
                resp = fetcher.get(url)
                if resp is None:
                    continue
                text = doc_to_text(resp)
            except Exception:
                continue
            nw = approx_tokens(text)
            if nw < min_words:
                continue
            stem = safe_stem(cik, out_form, period, seq)
            path = docs_dir / f"{stem}.txt"
            n = 1
            while path.exists():
                path = docs_dir / f"{stem}-{n}.txt"
                n += 1
            path.write_text(text)
            rows.append({
                "file": path.name, "cik": cik, "name": name, "ticker": filer["ticker"],
                "exchange": filer.get("exchange", ""), "sic": sic,
                "family": out_fam, "form": out_form, "filing_date": fdate,
                "period": period, "accession": accn, "source_url": url,
                "words": nw, "lines": text.count("\n") + 1,
            })
            seen_accn.add(key)
    return rows


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
SAMPLE_IXBRL = """
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><style>p{color:red}</style>
<ix:header><ix:hidden><ix:nonNumeric name="dei:X">JUNK TAXONOMY</ix:nonNumeric>
</ix:hidden></ix:header></head><body>
<p>CONSOLIDATED STATEMENTS OF OPERATIONS</p>
<table><tr><th></th><th>2025</th><th>2024</th></tr>
<tr><td>Net revenues</td><td>$</td><td><ix:nonFraction>12,345</ix:nonFraction></td>
<td>$</td><td><ix:nonFraction>10,987</ix:nonFraction></td></tr>
<tr><td>Operating income</td><td></td><td><ix:nonFraction>2,100</ix:nonFraction></td>
<td></td><td><ix:nonFraction>1,900</ix:nonFraction></td></tr></table>
<div style="display:none">HIDDEN JUNK</div>
<p>Revenue grew on   pricing   and volume during the period.</p>
<div>For the fiscal year ended <span>December 31</span><span>, 2024</span></div>
</body></html>
"""


def selftest():
    txt = html_to_text(SAMPLE_IXBRL)
    lines = [l for l in txt.split("\n") if l.strip()]
    assert "JUNK TAXONOMY" not in txt, "ix:header survived"
    assert "HIDDEN JUNK" not in txt, "display:none survived"
    assert "color:red" not in txt, "style survived"
    rev = [l for l in lines if l.startswith("Net revenues")]
    assert len(rev) == 1, f"table row not one line: {rev}"
    assert "12,345" in rev[0] and "10,987" in rev[0], f"ix numbers lost: {rev}"
    assert any(l == "CONSOLIDATED STATEMENTS OF OPERATIONS" for l in lines), lines
    prose = [l for l in lines if l.startswith("Revenue grew")]
    assert prose == ["Revenue grew on pricing and volume during the period."], prose
    # inline <span>s must not split a sentence across lines
    fy = [l for l in lines if l.startswith("For the fiscal")]
    assert fy == ["For the fiscal year ended December 31, 2024"], fy

    sys.path.insert(0, str(Path(__file__).parent))
    from v8_build_masks import is_table_line, is_title_line
    assert is_table_line(rev[0].split()), "row not seen as a table line by v8 policy"
    assert is_title_line("CONSOLIDATED STATEMENTS OF OPERATIONS".split())

    fake = [{"cik": i, "name": n, "ticker": t, "exchange": "NYSE"}
            for i, (n, t) in enumerate([("MICROSOFT CORP", "MSFT"),
                                        ("Pfizer Inc.", "PFE"),
                                        ("CLEAN WIDGETS INC", "CWI")], start=1)]
    excl, rep = resolve_exclusions(fake + [{"cik": 99, "name": "3M CO", "ticker": "MMM",
                                            "exchange": "NYSE"}], strict=False)
    assert 1 in excl and 2 in excl and 99 in excl and 3 not in excl, (excl, rep)
    # CWI resolves through neither net; MSFT/PFE by ticker, 3M by both
    assert rep["resolved"]["MICROSOFT"]["cik"] == 1, rep

    sub = {"filings": {"recent": {
        "form": ["10-Q"] * 12, "accessionNumber": [f"a{i}" for i in range(12)],
        "primaryDocument": ["d.htm"] * 12,
        "reportDate": [""] * 12,
        "filingDate": [f"{y}-0{m}-01" for y in (2021, 2022, 2023, 2024)
                       for m in (2, 5, 8)]}}}
    got = pick_filings(sub, {"10-Q": 3}, random.Random(0))
    yrs = {d[:4] for _, _, d, _, _, _ in got}
    assert len(got) == 3 and len(yrs) == 3, (got, yrs)
    got_all = pick_filings(sub, {"10-Q": 99}, random.Random(0))
    assert len(got_all) == 12, len(got_all)   # quota above supply takes everything

    t = Throttle(50.0)
    t0 = time.monotonic()
    for _ in range(10):
        t.wait()
    assert time.monotonic() - t0 >= 0.16, "throttle not throttling"
    print("selftest OK: ixbrl strip, table/prose lines, v8 line grammar, "
          "exclusions, year-stratified filing pick, throttle")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target", type=int, default=2000, help="documents to collect")
    ap.add_argument("--per-filer", type=int, default=4,
                    help="max primary docs per filer (spreads the corpus)")
    ap.add_argument("--min-words", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1212)
    ap.add_argument("--no-ex99", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    out = Path(args.out)
    docs = out / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.jsonl"

    throttle = Throttle(MAX_RPS)
    fetcher = Fetcher(throttle)

    print(f"[1/4] filer universe from {TICKERS_URL}")
    raw = fetcher.get(TICKERS_URL, as_json=True)
    fields = raw["fields"]
    filers = [dict(zip(fields, row)) for row in raw["data"]]
    print(f"      {len(filers)} listed filers")

    excl_ciks, excl_report = resolve_exclusions(filers)
    (out / "exclusions.json").write_text(json.dumps(
        {"cik": sorted(excl_ciks), "report": excl_report,
         "name_fragments": FB_NAME_FRAGMENTS}, indent=1))
    print(f"[2/4] FinanceBench exclusions: {len(excl_ciks)} CIKs "
          f"({excl_report['n_cik_by_ticker']} by ticker, "
          f"{excl_report['n_cik_by_name']} by name); "
          f"unresolved={excl_report['unresolved_tickers']}")

    pool = [f for f in filers if int(f["cik"]) not in excl_ciks
            and f.get("exchange") in ("NYSE", "Nasdaq", "NYSE American", "CBOE")]
    random.Random(args.seed).shuffle(pool)
    print(f"      {len(pool)} candidate filers after exclusion")

    rows, seen_accn, seen_cik = [], set(), Counter()
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                seen_accn.add(f"{r['accession']}:{r['source_url'].rsplit('/', 1)[-1]}")
                seen_cik[r["cik"]] += 1
        print(f"      resuming: {len(rows)} docs already in manifest")

    # per-filer quotas: 10-Q is the most abundant form, 8-K the cheapest signal
    q = max(1, args.per_filer // 4)
    quotas = {"10-K": q, "10-Q": max(1, q * 2), "8-K": q}

    todo = [f for f in pool if seen_cik[int(f["cik"])] == 0]
    print(f"[3/4] fetching to {args.target} docs "
          f"(quotas {quotas}, {args.workers} workers, <={MAX_RPS} req/s)")

    man = man_path.open("a")
    lock = threading.Lock()
    t0 = time.monotonic()
    stop = threading.Event()

    def work(filer):
        if stop.is_set():
            return []
        return fetch_filer(fetcher, filer, quotas, seen_accn, docs,
                           args.min_words, not args.no_ex99, args.seed)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            it = iter(todo)
            futs = {}
            for f in it:
                futs[ex.submit(work, f)] = f
                if len(futs) >= args.workers * 4:
                    break
            while futs:
                done = next(as_completed(futs))
                futs.pop(done)
                try:
                    got = done.result()
                except Exception as e:
                    got = []
                    print(f"  ! worker error: {type(e).__name__}: {e}")
                with lock:
                    for r in got:
                        rows.append(r)
                        man.write(json.dumps(r) + "\n")
                    man.flush()
                    n = len(rows)
                if n >= args.target:
                    stop.set()
                    break
                if n and n % 50 < len(got):
                    el = time.monotonic() - t0
                    print(f"      {n}/{args.target} docs  {fetcher.n_req} req  "
                          f"{fetcher.n_bytes/1e6:.0f} MB  {el:.0f}s  "
                          f"({n/max(el,1)*60:.0f} docs/min)")
                nxt = next(it, None)
                if nxt is not None and not stop.is_set():
                    futs[ex.submit(work, nxt)] = nxt
    except KeyboardInterrupt:
        print("\ninterrupted -- manifest is consistent, rerun to resume")
    finally:
        man.close()

    forms = Counter(r["family"] for r in rows)
    print(f"[4/4] {len(rows)} docs from {len({r['cik'] for r in rows})} filers")
    print(f"      forms: {dict(forms)}")
    print(f"      words: total={sum(r['words'] for r in rows):,} "
          f"median={sorted(r['words'] for r in rows)[len(rows)//2]:,}" if rows else "")
    print(f"      -> {man_path}")


if __name__ == "__main__":
    main()
