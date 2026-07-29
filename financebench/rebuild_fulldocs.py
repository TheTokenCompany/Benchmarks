#!/usr/bin/env python3
"""Rebuild fulldoc_contexts.json (qid -> full filing text) for the fd- fleets.

Downloads each FinanceBench item's source PDF (doc_link) and extracts text with
pypdf (AES-encrypted docs are opened with the empty owner password). Output goes
to $SCRATCH/fulldoc_contexts.json. Resumable; PDFs cached in $SCRATCH/fb_pdfs.

The scratchpad is tmp-cleaned between sessions — this script has now been
needed three times; it lives in the repo for that reason.

Usage: SCRATCH=<dir> .venv/bin/python financebench/rebuild_fulldocs.py
"""

import json
import os

import requests
from datasets import load_dataset
from pypdf import PdfReader

SCRATCH = os.environ["SCRATCH"]
PDF_DIR = os.path.join(SCRATCH, "fb_pdfs")
OUT = os.path.join(SCRATCH, "fulldoc_contexts.json")
os.makedirs(PDF_DIR, exist_ok=True)

ds = load_dataset("PatronusAI/financebench", split="train")
items = list(ds)[:150]

out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for i, item in enumerate(items):
    qid = str(item.get("question_id", i))
    if qid in out and len(out[qid]) > 1000:
        continue
    pdf_path = os.path.join(PDF_DIR, item["doc_name"] + ".pdf")
    if not os.path.exists(pdf_path):
        # GitHub mirror first (issuer IR sites rate-limit and time out), doc_link fallback
        urls = [
            f"https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{item['doc_name']}.pdf",
            item["doc_link"],
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=60, headers={"User-Agent": "financebench-eval"})
                if r.ok and r.content[:4] == b"%PDF":
                    open(pdf_path, "wb").write(r.content)
                    break
            except requests.RequestException:
                continue
        else:
            print(f"  SKIP {qid} ({item['doc_name']}): all sources failed")
            continue
    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        reader.decrypt("")
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    out[qid] = text
    if len(out) % 10 == 0:
        json.dump(out, open(OUT, "w"))
        print(f"{len(out)}/150")

json.dump(out, open(OUT, "w"))
print(f"done: {len(out)} contexts -> {OUT}")
