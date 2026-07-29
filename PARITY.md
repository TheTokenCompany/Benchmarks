# Train/serve parity: bear compression stack

Four times now, a training run has been paid for before anyone noticed that
training-time and serving-time semantics disagreed:

1. evidence-centered training windows against sliding-window serving (an RL run
   that read +15 on val and -14 on the benchmark),
2. subword→word pooling — our eval MIN-pools, prod `merge_words` MAX-pools,
3. number atomicity — prod merges digit-runs atomically since PR #714, the
   a3-lineage training data split them at digit separators,
4. a rendering action-space swap at inference (line-atomic render of
   word-trained scores), worth 3-5pp.

This file is the audit: what each implementation actually does, where they
differ, and which behaviour is canonical. `parity_test.py` asserts every row
that is mechanically checkable; rows still open are `xfail`ed there against the
`P<n>` ids used here, so the suite is green today and turns red the moment a
*new* divergence appears — or the moment a documented one is fixed and this
file goes stale.

Implementations audited:

| tag | file | role |
| --- | --- | --- |
| **prod** | `packages/otsofier/otsofier-rs/crates/otsofier-core/src/{postprocess,tokenizer,segment,merge}.rs` | what customers get (read-only reference) |
| **eval** | `v8_eval_precompress.py`, `precompress_v72_focus.py`, `precompress_v71.py` | every benchmark number we quote |
| **train** | `v9_rl_prep.py` (segmentation + render), `v10_build_targets.py`, `v11_pretokenize.py`, `v9_rwr_train.py::pool_words` | what the model is fitted to |

`v10` and `v11` import `segment_words` / `newlines_after` /
`map_tokens_to_words` / `render_mask` straight from `v9_rl_prep`, so the whole
training lineage shares one semantics. The three eval scripts share a second
one (`group_words` is copy-pasted between them, byte-identical). Prod is a
third. **No two of the three agree on how to cut a document into keep/drop
units.**

---

## 1. Semantics matrix

| stage | prod (otsofier-rs) | eval (v8/v71/v72) | train (v9/v10/v11) |
| --- | --- | --- | --- |
| tokenizer | mmBERT `tokenizer.json`, `[CLS] chunk [SEP]` | same vocab, `[CLS] q [SEP] chunk [SEP]` | same, `[CLS] q [SEP] chunk [SEP]` |
| unit of tokenization | whole document | **one line at a time**, trailing `\n` included in the payload | whole chunk, offsets kept |
| word starts on | `▁` marker, ASCII punctuation, force-token, unspaced-script char | `▁` or `Ġ` marker **only** | whitespace run boundary (on raw text) |
| apostrophe `'` | glued when letter-`'`-letter and next has no `▁` | glued (punctuation never opens a word) | glued (no whitespace around it) |
| digit separators `,` `.` | glued when digit-`,`-digit and next has no `▁` | glued | glued |
| trailing sentence `.` | **splits** into its own word | glued into the preceding word | glued into the preceding word |
| `%`, `bps`, `$` as separate tokens | `%` splits (ASCII punct) | glued | merged into the number (`12.5 %`, `25 bps`, `$ 3,698`) |
| number split across whitespace | not merged (never sees the whitespace) | **not merged** | merged (`1, 904`, `$ 1,904`) |
| unspaced scripts (CJK/Thai/…) | one word per token | whole passage = one word | whole passage = one word |
| byte-fallback `<0xNN>` | collapsed to real chars, **max** over the run, before merging | decoded at render time by `convert_tokens_to_string`, min-pooled as bytes | non-issue (words are raw text slices) |
| specials in the word stream | filtered before the lookahead | never present | `word_id = -1` |
| whitespace tokens | folded into the preceding word | **each bare `▁` opens its own empty word** | `word_id = -1` |
| newline tokens | folded into the preceding word | **folded into the line's last word** | `word_id = -1` |
| subword → word pooling | **max** | **min** | **min** (`cfg.pool`, `max` available) |
| keep rule | `prob > threshold` (strict) | `prob >= aggr` | top-scoring words to a **token** budget |
| ratio mode | `r`-th percentile of **word** probs | fixed `aggr`, or a token budget in the `budget*` variants | token budget (`target_keep`) |
| force keeps | `force_tokens` → 1.0, `force_reserve_digit` → any digit word 1.0 | `safenum` / `safetab` variants approximate it | none |
| protected spans | `<ttc_safe>`, URLs, optional JSON never compressed | none | none |
| render | char-level, `▁`→space, punctuation gets no leading space, `.`/`,`+digit suppresses space | detokenize each line's kept tokens, `"\n".join` non-empty lines | space-join kept words, re-emit `\n` for dropped words too |
| chunk seam | `preserve_leading_space` on chunks after the first | words never cross a chunk | words never cross a window (`snap_to_words`) |
| chunking | sequential, cut at the last `.`/`\n` token before the cap | pack whole words to a token budget | evidence-centered (v9) / position-sampled (v11) |

### Measured on 120 real FinanceBench chunks (`val_meta.jsonl`)

- eval produces **9.2% more words** than train on identical text (148,683 vs 136,131)
- **only 14 / 120 documents** get the same word list from both
- **5.7%** of eval words are whitespace-only phantoms
- **6.2%** of eval words are content words polluted by a newline token

---

## 2. Divergences

Impact is marked **measured** (observed on real text or in a past incident) or
**potential** (reasoned from the code, not yet quantified).

### P1 — subword→word pooling: min vs max
`merge_words` (postprocess.rs:817) folds subword probs with `f32::max`; the
comment is explicit that one informative subword should carry the word.
`v8_eval_precompress.py:132` uses `min(...)`, and `v9_rwr_train.py:148` defaults
`pool="min"` *specifically to match the eval*. So the model is trained on, and
scored on, a statistic prod never computes.

Under min, one weak subword vetoes the word. mmBERT emits digits one per token,
so a 7-token amount like `3,605,357` needs **all seven** subwords above
threshold to survive our eval, but only **one** to survive prod. Our numbers are
therefore pessimistic against prod at the same aggressiveness, and — worse — the
policy is being optimised to raise its *weakest* subword rather than its
strongest.
**Impact: measured (incident 2). This is the headline row.**

### P2 — word segmentation: three different definitions
Prod cuts on `▁` + punctuation with intra-word glue. Eval cuts on `▁` only.
Train cuts on whitespace runs plus the PR #714 number merges.

The consequential case is a column-aligned filing table. `"expense  $  3,698"`:

| | units |
| --- | --- |
| prod | `expense` · `$` · `3,698` |
| eval | `expense` · `$` · `3,698` |
| train | `expense` · `$ 3,698` |

and `"Cost 1, 904 total"` (a thousands separator split across column padding):

| | units |
| --- | --- |
| eval | `Cost` · `1,` · `904` · `total` |
| train | `Cost` · `1, 904` · `total` |

The eval and prod paths can drop `$` off its own amount, or keep `1,` and drop
`904`, producing a corrupted number in the output. This is exactly the failure
PR #714 fixed *inside* a token run — it is still open *across* whitespace, in
both prod and eval. The training path is the only one that gets it right.
**Impact: measured — 106/120 documents segment differently between eval and train.**

### P3 — trailing sentence punctuation
Prod's `merge_glues_real_per_digit_tokenization` test pins `"$15,000."` →
`▁$15,000` + `.`; the period is sentence punctuation and gets its own decision.
Both python paths glue it in, so `$15,000.` lives or dies as one unit.
Low harm (the period rides along with its number) but it shifts every word
count and therefore every retention denominator.
**Impact: potential.**

### P4 — phantom whitespace words (eval only)
Column padding tokenizes to a run of bare `▁` pieces. Each one starts a new word
in `group_words`, renders as the empty string, and still occupies a slot in the
`budget22` / `budget33` variants, where `wlen[(ref, wi)] = len(w) = 1` token.
So a fraction of every token budget is spent keeping nothing.
**Impact: measured — 5.7% of eval words (8,508 / 148,683).**

### P5 — newline tokens contaminate the eval pool
The eval path deliberately tokenizes `line + "\n"` so the model sees line
structure — but `"\n"` carries no `▁`, so `group_words` folds it into the line's
**last** word. Combined with P1's min-pool, the newline's keep-prob caps the
row's final number, which in a filing table is the value the question is about.
The training path maps newline tokens to `word_id = -1` and excludes them.
**Impact: measured — 6.2% of eval words are content words carrying a newline;
in a table that is disproportionately the numeric column.**

### P6 — eval render deletes empty lines
`"\n".join(kept_line_texts)` only appends non-empty lines, so a row that keeps
nothing loses its line slot and everything below slides up. `render_mask`
re-emits `"\n" * nl_after` for dropped words too, so the table keeps its shape.
Under compression, most rows drop something — the reader LLM sees a table whose
row alignment is a function of the keep mask.
**Impact: potential; related to incident 4 (rendering action space).**

### P7 — eval render emits a leading space per line
`convert_tokens_to_string` re-emits the first token's `▁` as a space and the
eval render never strips it. Prod's `reconstruct` strips it on the first chunk
and preserves it only at a seam (`preserve_leading_space`).
**Impact: measured, cosmetic — every rendered line starts with a space.**

### P8 — eval render doubles newlines between surviving rows
The line's trailing `"\n"` token lives inside that line's last word (P5). If
that word survives, the rendered line already ends in `"\n"` and then
`"\n".join` adds another. All-keep does not round-trip:

```
in:  "Total revenue 3,605\nCost of revenue 1,755"
out: " Total revenue 3,605\n\n Cost of revenue 1,755"
```

Rows whose last word was dropped get a single newline. So blank lines appear
and disappear according to the keep mask, and every compressed context we have
ever benchmarked carries this.
**Impact: measured. Cheapest real fix in the file (see migration step 2).**

### P9 — unspaced scripts
Prod splits CJK/Thai/Lao/Khmer/… per token (`is_unspaced_script`,
`starts_unspaced_word`), because `▁` never appears mid-run there. Neither python
path has any such rule: `该航空公司的机队迅速扩大了。` is **one** keep/drop word
for us and **nine** for prod. Irrelevant to FinanceBench, blocking for any CJK
customer, and it means our eval cannot measure prod's behaviour on that traffic
at all.
**Impact: potential, out-of-scope for the current campaign.**

### P10 — thresholding and the action space
Three different keep rules: prod `prob > t` strict, eval `prob >= aggr`, train
"top words until the token budget is spent". The strict-vs-inclusive difference
is an edge case; the budget-vs-cutoff difference is not.

Prod's ratio mode drops the `r`-th **word** percentile. Our `budget*` variants
and the RL policy spend a **token** budget. On a table where a number is 4-8
subwords and its label is 1-2, these select materially different sets at the
same nominal retention — and `budget_mask_from` is explicit that token weighting
was chosen on purpose, because a word-count reward teaches the policy to keep
cheap words and drop expensive numbers. That reasoning is right; it just is not
what prod does.
**Impact: measured (incident 4, 3-5pp).**

### P11 — chunking and seams
Prod chunks sequentially, cutting at the last `.` or `\n` token before
`max_seq_len - 2`, and passes `preserve_leading_space = chunk_idx > 0` so the
seam space survives the empty-string join. Two consequences:

- The cut is at a *token*, not a word. `chunk_end_tokens = [".", "\n"]` and `.`
  is the same vocab piece that appears inside `3.14`, so prod can split a number
  across chunks; `glues_intraword` only looks within a chunk, so the halves get
  independent keep/drop. Rare (needs a decimal point at the chunk cap) but it is
  the P2 failure mode again.
- If no boundary token exists in a whole window, `ed` stays at `st + max_len`
  and the chunk is `max_len + 1` tokens; `chunk_and_tokenize` then clamps to
  `max_seq_len - 2` and the overflow token is silently dropped.

Our eval chunker packs whole words only, and v11's `snap_to_words` shrinks a
window until both edges fall on whole words. Both are strictly better than prod
here. The real seam risk is upstream: v9 windows are evidence-**centered** and
v11 windows are position-**sampled**, while prod's are sequential and
position-agnostic. v11's position buckets exist to fix exactly this.
**Impact: measured (incident 1).**

### P12 — force keeps
Prod supports `force_tokens` (prob → 1.0) and `force_reserve_digit` (any word
containing a digit → 1.0). Our `safenum` / `safetab` variants approximate
`force_reserve_digit` at eval time. Training has no force mechanism at all —
v10 up-weights numeric words in the loss but never pins them. If prod ships with
`force_reserve_digit = true`, the model is being scored under a rule it was
never trained against, and the aggressiveness sweep means something different
on each side.
**Impact: potential.**

### P13 — protected segments
`segment_protected_batch` keeps `<ttc_safe>` regions, URLs, and (optionally)
JSON out of the classifier entirely, and `merge_segments_batch` re-inserts a
space at segment boundaries. Neither eval nor training models this, so no
benchmark we run exercises it.
**Impact: potential.**

### P14 — byte-fallback
Prod collapses `<0xNN>` runs to real characters **before** merging, one word per
decoded character, `max` over each character's bytes. Our python paths leave the
placeholders in the token stream (they glue onto the preceding word since they
carry no `▁`) and rely on `convert_tokens_to_string` to decode at render time.
Verified no placeholder leaks into our output, but an OOV character is pooled
into its neighbour rather than getting its own decision, and it is min-pooled.
**Impact: potential, low.**

---

## 3. Canonical semantics

**Prod is the reference wherever the difference is arbitrary.** Prod is what the
customer receives; a research number that does not predict prod behaviour is not
worth the GPU hours. Three deliberate exceptions, in the order they should be
resolved:

1. **Word segmentation: the training path wins, and prod should adopt it.**
   `segment_words`' cross-whitespace merges (`1, 904`, `$ 3,698`, `12.5 %`) are
   the correct reading of PR #714's intent — a number is one keep/drop decision
   — applied to the column-aligned tables that prod's token-local rule cannot
   see. This is the one place research should push a change *into* prod rather
   than conform to it.

2. **Token-weighted budgets stay.** `budget_mask_from`'s reasoning holds: a
   word-count objective teaches the policy to keep cheap words and drop
   expensive numbers. Prod should grow a token-budget mode rather than research
   adopting prod's word percentile. Until then, keep the absolute-cutoff
   variants in every sweep so there is a comparable number on both sides.

3. **Windowing legitimately differs, and must stay measured.** Training needs
   the evidence inside the window or there is nothing to learn; serving cannot
   know where the evidence is. v11's position buckets are the bridge. What must
   never happen again is a training distribution that puts evidence in one place
   every time (incident 1).

Everything else conforms to prod: **max-pool**, punctuation opens a word,
trailing sentence punctuation splits, whitespace and newline tokens are
non-content, unspaced scripts split per token, renders round-trip under
all-keep, strict `>` thresholding.

---

## 4. Migration order

Ordered by (value) / (risk × cost). Steps 1-3 do not touch a model.

1. **Do not migrate anything mid-campaign.** `v8_eval_precompress.py` is under
   active edit and every committed `financebench/results_*` number came off the
   current semantics. Land these behind a flag, re-run one config as an A/B,
   and only then flip the default.

2. **P8, then P5, then P4 — eval render and pooling hygiene (no retraining).**
   All three are the same root cause: non-content tokens are inside content
   words. Strip whitespace-only and newline-only tokens out of `group_words`'
   output words (keeping them in the model input, which is the part that was
   deliberate), and the doubled newlines, the min-pool contamination, and the
   phantom budget slots go together. Changes retention denominators, so re-run
   the sweep — cheap, no GPU training.
   **DONE — this is eval-v2 (`EVAL_V2=1`), see section 6.** Still opt-in; the
   remaining work is the A/B that decides whether it becomes the default.

3. **P2 — give the eval path the training path's segmentation.** Replace the
   per-line `group_words` with `segment_words` + `map_tokens_to_words`, which
   already exist and are already tested. This makes eval and training measure
   the same units for the first time, and it is a prerequisite for any honest
   read of a v9/v10/v11 checkpoint. Expect the retention curve to move; that
   movement is the P2 bug being removed, not a regression.

4. **P1 — min → max pool, as a deliberate A/B.** Only after 2 and 3, because
   min-pool's damage is concentrated in exactly the words P4/P5 corrupt, and
   flipping first would attribute their fix to the pool change. Run
   `pool=max` against `pool=min` on a fixed checkpoint before retraining
   anything. **A checkpoint headed for prod must be trained under max.**

5. **P10 / P12 — add prod's absolute-cutoff and `force_reserve_digit` modes to
   the eval harness** so the sweep produces a number that predicts prod
   directly, instead of one that has to be mentally translated.

6. **P3, P7, P6 — cosmetic render conformance.** Do these with 3, since they
   touch the same function.

7. **P9, P13, P14 — prod-only features (unspaced scripts, protected segments,
   byte-fallback pooling).** No FinanceBench impact. Needed before any CJK or
   API-doc customer, and they need their own benchmark first.

Not scheduled: **P11**'s prod-side gaps (mid-number chunk splits, the
`max_len + 1` overflow drop). They are prod bugs, not parity gaps — file them
against otsofier-rs rather than working around them here.

---

## 5. Prod action items (for taaha — PR #714 follow-ups)

Two of these are correctness bugs in `otsofier-rs` that our research stack cannot
work around, and one is a capability gap. They are listed here with repro lines
so they can go straight onto a ticket. **None of them should be patched from
this repo.**

### PA-1 — a currency symbol can be dropped off its own amount (severity: high)

PR #714 made a number atomic *within a token run*: `glues_intraword` joins
digit-`,`-digit and letter-`'`-letter when the right neighbour carries no `▁`.
It cannot see across whitespace, and filing tables are column-aligned, so the
symbol and its amount are separate keep/drop words:

```
input:  "Income tax expense    $   3,698"
tokens: ▁Income ▁tax ▁expense ▁ ▁ ▁ ▁$ ▁ ▁ ▁ 3 , 6 9 8
prod merge_words -> ["▁Income", "▁tax", "▁expense", "▁$", "▁3,698"]
                                                     ^^^^  ^^^^^^^
                                    two independent keep/drop decisions
```

Drop the `▁$` word and the output says `3,698` where the filing said `$3,698`.
The same shape breaks a thousands separator that column padding split:
`"Cost of revenue   1, 904, 220"` becomes `1,` · `904,` · `220`, and keeping a
subset produces a number that never existed.

Our training path already fixes this — `v9_rl_prep.segment_words` merges across
whitespace (`$ 1,904`, `1, 904`, `12.5 %`) — and we think that rule is the
correct reading of #714's intent, so the suggested fix is to port it rather than
invent a new one. Rust-side it is a lookahead in `merge_words`: when the current
word is exactly a currency symbol and the next word begins with a digit, do not
close the word.

Repro (python, our side, showing the intended output):
```
.venv/bin/python -c "from v9_rl_prep import segment_words as s; \
  t='Income tax expense    \$   3,698'; print([t[a:b] for a,b in s(t)])"
# ['Income', 'tax', 'expense', '$   3,698']
```

### PA-2 — the chunker can split a number across a chunk seam (severity: medium, rare)

`pipeline.rs:359` passes `chunk_end_tokens = [".", "\n"]`, and
`chunk_text_exact` / `chunk_and_tokenize` cut immediately *after* the last
matching token id at or before the cap. The `.` piece it matches is the same
vocab piece that appears inside `3.14` and inside `$1,234.56`. When a decimal
point happens to be the last boundary token before `max_seq_len - 2`, the chunk
ends between `3` `.` and `14`; `glues_intraword` only ever looks within one
chunk, so the halves become independent keep/drop words in two different
`reconstruct` calls. This is PA-1's failure mode again, arriving by a different
route.

Fix direction: after choosing `ed`, walk back to a token that is not a
digit-adjacent separator, or require the boundary token to be followed by a `▁`
piece (which is what makes a `.` sentence punctuation rather than a decimal
point) — the same test `glues_intraword` already applies.

### PA-3 — a chunk with no boundary token silently loses a token (severity: low)

In the same loop, if no `chunk_end_ids` match anywhere in the window, `ed` keeps
its initial value `st + max_len` and the range pushed is `(st, ed + 1)` —
`max_len + 1` tokens. `chunk_and_tokenize` then clamps with
`content_len = chunk_ids.len().min(self.max_seq_len - 2)` and the overflow token
is dropped from the output with no warning. Needs a whole 8190-token window with
no `.` and no `\n` (minified JSON, a base64 blob, a long CJK passage), so it is
rare, but the failure is silent text loss rather than an error.

Fix direction: clamp the range when it is built, or emit the remainder as its
own chunk.

### PA-4 — no token-budget compression mode (capability gap, not a bug)

Prod offers an absolute probability cutoff and a **word**-percentile ratio mode
(`postprocess.rs:422`). Our budget variants and the RL policy spend a **token**
budget, for the reason spelled out in `budget_mask_from`: a word-count objective
teaches the policy to keep cheap one-token words and drop the four-subword
numbers that carry the answer. On a filing table the two select materially
different sets at the same nominal retention, which is why a research number
does not translate directly. Adding a token-budget mode to prod would let the
two sides quote the same number. See P10.

---

## 6. eval-v2: the opt-in fix for P4/P5/P6/P7/P8

`EVAL_V2=1` on `v8_eval_precompress.py`. Default OFF — every committed
`financebench/results_*` number came off v1, so v2 caches take a `-v2` suffix
(`fd-v9kl03-budget22-v2--0.5`) and the two sit side by side.

**What it changes.** One idea: v1 folds non-content tokens into content words.
v2 keeps every token in the *encoder input* but marks whitespace/newline pieces
non-content, which takes them out of the min-pool, out of the budget, and out of
the render. Rendering then uses the canonical rule from section 3 (the same one
as `v9_rl_prep.render_mask`), which is what closes P6/P7/P8.
`parity_test.py` asserts the encoder-stream equality explicitly.

**How close the A/B really is.** The concatenated token stream is byte-identical
(verified on a 59,353-token filing). Keep-probs are *not* bit-identical, though:
`infer_probs` packs whole words to a token budget, and v2 words absorb their
preceding whitespace as a prefix instead of leaving it as a standalone word, so
the greedy packer lands its cuts in slightly different places — on that filing,
30 chunks either way, but only 12 of 30 chunk starts identical, the rest shifted
by 1-3 tokens. Every token still appears in some chunk with essentially the same
neighbours; a handful near each seam change chunk membership. So the honest
framing is *same model, same tokens, windowing perturbed by a few positions* —
close enough that the A/B measures the accounting and the render, not a
different model, but not the exact-equality claim it would be nice to make.

Two bugs surfaced while building it, both caught by the golden suite rather than
by a benchmark:

- the trailing newline of the last line never reached the encoder (the pending
  buffer was not flushed when the final line tokenized to nothing);
- **a bare `▁` is simultaneously whitespace and the marker that opens the next
  word.** Demoting it to non-content without also treating it as a boundary
  glued `3,605` onto the preceding row label — a *worse* atomicity bug than the
  one being fixed. `group_words_v2` opens a word on a pending non-content run
  for exactly this reason.

**Measured, 150 full filings, `fd-v9kl03`, gemini-3.5-flash-lite corpus:**

| | budget22 v1 → v2 | budget33 v1 → v2 |
| --- | --- | --- |
| denominator (`original_tokens`) | 18,626,567 → 15,847,519 (−14.9%) | same |
| bear tokens kept | 4,097,763 → 3,486,381 (−14.9%) | 6,146,681 → 5,229,624 (−14.9%) |
| reader-visible chars | 17,082,668 → 15,819,478 (−7.4%) | 25,737,398 → 24,175,705 (−6.1%) |
| output lines | 305,978 → 216,495 | 496,445 → 326,631 |
| lines starting with a space | 205,489 → **0** | 311,957 → **0** |
| blank lines | 117,628 → 31,303 | 212,607 → 44,716 |
| gold-number survival | 0.532 → 0.548 (**+0.017**) | 0.568 → 0.565 (**−0.003**) |

So **~15% of what the v1 accounting called a token was whitespace**, and the
render was emitting 205k lines that begin with a space plus 86k spurious blank
lines. That is the "slightly lossy lens" quantified.

**Consequence for how the campaign's numbers are worded.** Every fixed-budget
cache spends part of its budget on whitespace phantoms, so the *absolute*
retention labels are optimistic about how much real content the budget buys:
"74.0% at 22% kept" is 22% of tokens-including-padding, which is roughly 19% of
content. The *rankings* are unaffected — every config shared the bug, and
adaptive-dose verified a regenerated budget22 is byte-identical to the existing
`fd-v9kl03-budget22` cache across all 150 questions — so no comparison in the
campaign is invalidated. Only the absolute "% kept" wording should be read as
"% of the padded token count".

### Fleet result: the bug tax was cost, not accuracy

Three arms, full 150, gemini-3.5-flash-lite reader, same gpt-5-mini judge, all
five result sets complete (150 rows, 0 unjudged, 0 empty answers, 0 duplicate
qids). Both arms of each pair answer the *same* questions, so the read is the
paired McNemar exact test on the discordant pairs, not the marginal gap.

| arm | v1 | v2 | delta | fixed / broke | McNemar p | reader chars |
| --- | --- | --- | --- | --- | --- | --- |
| A. nominal 22% | 74.0% | 74.0% | +0.0pp | 5 / 5 | 1.000 | **−7.4%** |
| B. matched cost (v2 @ 24%) | 74.0% | 73.3% | −0.7pp | 5 / 6 | 1.000 | +1.5% |
| C. nominal 33% | 75.3% | 77.3% | +2.0pp | 9 / 6 | 0.607 | **−6.1%** |

Pooled across the two same-nominal arms: 14 fixed, 11 broke, p=0.690.

**Nothing here is significant, including arm C.** +2.0pp is the largest number
in the table and the most likely to be misread: it rests on 15 discordant pairs
split 9/6, which is what chance produces at this sample size. With 150 binary
items the standard error at p≈0.75 is 3.5pp, and E42 measures 6-11% run-to-run
verdict flips, so nothing below roughly 5pp is readable from single runs. Arm C
is not evidence that v2 helps at 33%; it is consistent with the same null as A
and B.

**Arm B is the informative one.** It is cost-matched to within 1.5% of v1's
reader characters, and it came back −0.7pp. So the 7.4% of context the fix
reclaims buys nothing when it is spent — the accuracy/retention curve is already
flat at this operating point *for the content this model ranks 22nd-24th
percentile*. That is a narrower claim than "more content does not help": what
gets bought at the margin is low-ranked filler, not the kind of adjacent column
header the AMCOR case showed mattering.

**Conclusion for the migration.** v2 is accuracy-neutral and 6-7% cheaper at
equal nominal budget. Adopt it for new work on the cost argument and on
correctness (all-keep now round-trips, which it did not before); do not restate
any campaign number as an accuracy improvement.

**Gold-number survival predicted this correctly, and it was free.** It is the
fraction of the gold answer's numeric tokens still present in the compressed
context, over the 119 items whose answer contains a number — an upper bound on
accuracy computable with no API calls. It read +0.017 at budget22 and −0.003 at
budget33 *before* the fleet ran, i.e. flat, and flat is what 450 reader calls
and 450 judge calls then confirmed. Worth reaching for first the next time a
change looks like it should move accuracy: it beat two rounds of intuition here.

Its blind spot is real but did not bite: survival asks whether the number is
*present*, not whether the reader can *use* it, so it scores the AMCOR case
(v1 dropping "million" off the column header above a surviving `2,018`) as a
hit. That was the mechanism by which the layout cleanup might have moved
accuracy independently of survival. Arms A-C say it did not, at least not
measurably at 150 items.

**Why `budget24` exists.** v2 at nominal 22% spends less than v1 at nominal 22%,
because the denominator no longer counts padding, so comparing those two arms
alone confounds "the fix helped" with "we spent less". `budget24` matches v1's
reader characters to within 1.5% (`budget26` matches bear tokens instead;
`budget24` is the fair one, since the reader pays per character). It is built and
fleeted; `budget26` is built but was not needed.

---

## 7. Fixed in this pass

`v9_rl_prep.segment_words` — a bare currency symbol was matching the
backward-binding `trailing_unit` rule (`$` is in both `CURRENCY` and `UNITS`),
so it welded to the *preceding* number and then blocked the forward
`bare_currency` merge:

```
"Contributions 2026 $ 9,353"  ->  ["Contributions", "2026 $", "9,353"]   (was)
                              ->  ["Contributions", "2026", "$ 9,353"]   (now)
"Issuance $\n\n119\n\n$\n\n439" -> ["Issuance", "$\n\n119\n\n$", "439"]  (was)
                                -> ["Issuance", "$\n\n119", "$\n\n439"]  (now)
```

Currency binds forward, units (`%`, `bps`) bind backward. This contradicted the
function's own docstring and defeated the PR #714 atomicity it exists to
provide. `v10_build_targets.selftest()` and `v11_pretokenize.selftest()` both
pass with the fix; **any dataset built from `v9_rl_prep` before this change has
mis-segmented currency columns and should be rebuilt.**

`v8_eval_precompress.py` — the budget variant read its fraction as
`0.33 if variant.startswith("budget33") else 0.22`, so every budget other than
22 or 33 silently ran at 22%. Now `int(re.search(r"budget(\d+)", variant))/100`,
which is what `linebudget` already did. `budget22` and `budget33` are unchanged
bit-for-bit, so no measured number moves; `budget24` / `budget26` now mean what
they say, which is what the cost-matched v2 arm needs.
