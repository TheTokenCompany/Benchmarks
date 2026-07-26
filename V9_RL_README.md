# v9 RL: reward-weighted regression for the finance compressor

RWR fine-tuning of a wave-2 SFT bear checkpoint against an **end-to-end** reward: a
candidate keep-mask is rendered back into text, a reader model (gemini-3.1-flash-lite)
is asked the question against that text, and the reward is whether the gold answer
survived. Not token recall against a mask — answerability, which is what FinanceBench
scores.

Forked from `v8_train.py` (Modal scaffolding, volumes, checkpoint layout) and taaha's
`training/bear_rl/modal_train_bear_rl_rwr.py` (RWR mechanics). His plain REINFORCE
collapsed; RWR did not, so the anti-collapse parts are kept: candidates sampled from
the current policy, rewards softmaxed into *weights* (never a return gradient), KL to a
frozen reference as the trust region, and a keep floor that kills degenerate candidates.

## Files

| File | Runs | What |
|---|---|---|
| `v9_rl_prep.py` | locally | synthetic QA -> evidence-centered 2048-token chunks + word geometry + gold facts; uploads to `otso-v8-data:/v9-rl` |
| `v9_rwr_train.py` | Modal `otso-v9-rl` | the RWR trainer (B200, H100 fallback) + a CPU `reader_probe` |
| `v9_eval_export.py` | Modal `otso-v9-eval-export` | verifies a checkpoint loads the way `v8_eval_precompress.py` loads it |

## Launch

**Data is already built and uploaded** (2000 train / 200 val, stratified 667/667/666 by
qtype, every chunk contains its evidence). Rebuild only if the QA corpus changes:

```bash
.venv/bin/python v9_rl_prep.py --upload                       # -> otso-v8-data:/v9-rl
```

### The ettin variant (`v9-rl-ettin`)

`jhu-clsp/ettin-encoder-400m` uses a different tokenizer (vocab 50368, GPT-2-style BPE,
CLS/SEP/PAD 50281/50282/50283) from mmBERT's 256k multilingual vocab, so it needs its own
build. `--reuse-from` rebuilds **the same items** rather than re-running selection:
`chunk_text`, `words`, `nl_after` and `gold_fact_words` are char-level facts about the
filing and carry over untouched; only the question ids, chunk ids and token→word map are
recomputed. Re-running the full selection under a different tokenizer would re-window
every filing and pick a different 2000 items, making the two RL runs incomparable.

```bash
.venv/bin/python v9_rl_prep.py \
    --reuse-from <scratch>/v8data/v9-rl \
    --tokenizer <ettin-ckpt-dir-or-jhu-clsp/ettin-encoder-400m> \
    --out-dir <scratch>/v8data/v9-rl-ettin --upload
```

Already built and uploaded to `otso-v8-data:/v9-rl-ettin`. Verified against the mmBERT
build: identical `qa_id` order and `source_id`, identical qtype split, every item's words
a prefix of the mmBERT item's, max token id 50283 (inside ettin's 50368 vocab), zero
out-of-range or non-contiguous `word_id` runs, zero unreachable gold facts.

Two differences worth knowing. Ettin averages 1.535 tokens/word against mmBERT's ~1.62,
but it is *less* uniform, so 422/2000 train and 38/200 val items overflow the 2048 budget
and lose tail words (mean 9.8 of ~1123, max 205); exactly one item lost gold facts (3 of
them). And only 78.6% of numeric words are multi-subword under ettin versus 96% under
mmBERT — atomic-number merging still matters, just slightly less.

**1. Reader probe first (CPU, ~48 calls, no GPU).** Confirms the secret attaches and
the reward has dynamic range before spending GPU hours:

```bash
.venv/bin/modal run v9_rwr_train.py::reader_probe --n 12
```

Already run — the numbers to compare against: `keep=1.0 -> 0.833`, `keep=0.33 -> 0.083`,
`keep=0.22 -> 0.167`, `keep=0.12 -> 0.000`, zero errors.

**2. GPU smoke (stub reader, 8 items, 1 epoch, no reader calls).** The GPU path has
**not** been executed yet — run this before the real launch:

```bash
.venv/bin/modal run v9_rwr_train.py --init-from exp-20260725-235952-w2-140-s2/best --smoke 1
```

**3. Real run.** `--init-from` is resolved against `/output` (`otso-v8-training`) then
`/models` (`compression-models`, read-only); absolute paths are honoured. Nothing is
hardcoded — pick the checkpoint at launch:

```bash
.venv/bin/modal run --detach v9_rwr_train.py \
    --init-from exp-20260725-235952-w2-140-s2 \
    --tag s0 --lam 1.0 --kl-coef 0.05 --k 6 --epochs 4 \
    --max-llm-calls 150000 --max-hours 6.0
```

Note the path has **no `/best` suffix**: v8 writes its best checkpoint at the *run root*
(`config.json` + `model.safetensors` + tokenizer + `best_metrics.json` directly under
`exp-*`), and `exp-20260725-235952-w2-140-s2/best` does not exist on the volume. The
epoch snapshots are `epoch_1 … epoch_12`.

**Baselines to beat.** Both SFT inits read on the full 200 val items with real reader
calls and min-pooling (1438 calls; per-item results in `anchor_read_200.json`, verdicts
cached in `anchor_read_cache.jsonl`). Both val sets hold the same 200 `qa_id`s in the same
order, so this is a true per-item pairing:

Cells are `answer_survival / fact_survival`:

| keep | w2-140-s2 **epoch_4** (record lineage) | w2-140-s2 root = epoch_9 (AUPRC-best) | w2-ettin (AUPRC-best) |
|---|---|---|---|
| 1.00 | 0.915 / 1.000 | 0.915 / 1.000 | 0.905 / 1.000 |
| 0.33 | **0.740** / 0.969 | 0.715 / 0.923 | 0.705 / 0.922 |
| 0.22 | **0.665** / 0.940 | 0.635 / 0.873 | 0.635 / 0.867 |
| 0.12 | **0.610** / 0.872 | 0.420 / 0.739 | 0.510 / 0.771 |

Paired deltas against epoch_4 (McNemar on discordant items):

| keep | ettin − ep4 | disc | p | ep9 − ep4 | disc | p |
|---|---|---|---|---|---|---|
| 1.00 | −0.010 | 3/1 | 0.625 | +0.000 | 0/0 | 1.000 |
| 0.33 | −0.035 | 24/17 | 0.349 | −0.025 | 17/12 | 0.458 |
| 0.22 | −0.030 | 28/22 | 0.480 | −0.030 | 23/17 | 0.430 |
| 0.12 | **−0.100** | 45/25 | **0.022** | **−0.190** | 42/4 | **0.00002** |

**The ceiling is 0.915**, so RL has ~25 points of headroom at keep 0.22 from epoch_4. An
earlier 12-item probe put the ceiling at 0.833; that was small-sample error. `val_ceiling`
in the logs reads ~0.91 — read the val curve against that, not against 1.0.

**val AUPRC is selecting the wrong checkpoint.** epoch_9 won the SFT run on val AUPRC, yet
epoch_4 answers better at *every* retention, by +19.0pp at keep 0.12 (42 items only
epoch_4 answered against 4 only epoch_9, p=0.00002) and +3.0pp at 0.22. `fact_survival`
moves the same way (0.940 vs 0.873 at 0.22), so this is not a reader artifact. This is a
quantitative replication of the early-epoch-beats-full inversion the wave-2 fleet already
saw three times — and it says the v8 selection metric actively costs answerability. Future
SFT waves should select on something answerability-based, not AUPRC.

**Against the right anchor, ettin has no band where it wins.** It is behind epoch_4 at
every keep, significantly so at 0.12 (−10.0pp, p=0.022). An earlier read had ettin *ahead*
by 9pp at 0.12 — that was measured against the AUPRC-selected epoch_9, and the apparent
ettin advantage was really epoch_9's weakness. Combined with 140M vs 400M (~3x cheaper and
faster to serve), **w2-140-s2/epoch_4 is the init to prefer at every retention tested.**

Use `--detach` or the run dies with your laptop.

**3b. The ettin run.** 400M backbone, so micro-batch 8 as the SFT used:

```bash
.venv/bin/modal run --detach v9_rwr_train.py \
    --init-from exp-20260726-000202-w2-ettin \
    --data-subdir v9-rl-ettin --batch 8 \
    --tag ettin-s0 --lam 1.0 --kl-coef 0.3 --k 6 --epochs 4 \
    --max-llm-calls 150000 --max-hours 6.0
```

Verified on CPU with the real 400M checkpoint at batch 8: forward runs, min-pooling gives
word probs mean 0.395 / std 0.430 (not collapsed), the deterministic anchor lands at
0.219-0.220 against the 0.22 target, candidates sit at 0.12/0.22/0.33, KL to an identical
ref is ~0, and the gradient reaches 174 tensors (28 layers vs mmBERT's 22).

Its exploration headroom is *better* than mmBERT's — at `--gumbel-temp 0.3`, exploration
beats the deterministic anchor on **100%** of items (best R 1.874 vs the anchor's 1.188,
anchor softmax weight 0.081). One caution: on the same 10 val items with real reader
calls, the ettin SFT anchor scores answer_survival **0.500** at keep 0.22 against
w2-140-s2's 0.600. That is a one-item difference at n=10 so it is not conclusive, but the
SFT fleet ranked ettin *above* mmBERT on val AUPRC (0.8274 vs 0.8247) while the
end-to-end reward ranks it below — the usual reminder that AUPRC and answerability are
not the same ordering. Worth a 200-item `reader_probe`-style read before betting the night
on it.

**4. Verify + eval.** The checkpoint layout is v8's exactly, so the existing eval needs
no changes:

```bash
.venv/bin/modal run v9_eval_export.py --exp-path exp-<ts>-v9-rwr-s0
MODEL_VOL=otso-v8-training MODEL_VOL_PATH=exp-<ts>-v9-rwr-s0 MODEL_ALIAS=v9rwr \
    .venv/bin/modal run precompress_v72_focus.py
```

## Resume

Re-run the **same command** (same `--tag`). `RLConfig.__post_init__` reuses the existing
`exp-*-<run_name>` directory, the policy reloads from `latest/`, and `state.json`
restores the epoch, step and reader-call count. The reader verdict cache lives on the
data volume, so a resume re-scores nothing it has already scored.

The KL anchor is re-loaded from the **original** `init_from`, which `state.json` records
— never from `latest`. Re-anchoring to the current policy on each resume would turn the
trust region into a no-op and let the run drift arbitrarily far from the SFT model.

Not restored: optimizer state (AdamW moments for 140M params are ~1.1 GB, too heavy to
write every half-epoch). A resumed run restarts the optimizer.

## Reward

```
R = 2.0*answer_survival + 0.5*fact_survival - lam*keep_frac - 2.0*[keep_frac < 0.04]
```

- **answer_survival** — the reader answers the question from the rendered candidate;
  scored 1/0 by *deterministic* comparison to the gold answer. No LLM judge: numeric
  golds are compared numerically with 2% relative tolerance after stripping `$ % ,`,
  and across scale swaps (`$1,904 million` vs a reply of `1.904 billion` is a hit).
  Non-numeric golds fall back to content-word overlap at 60%. A rubric or a judge would
  make the reward drift between epochs, and RWR cannot optimize a moving target.
- **fact_survival** — fraction of `gold_fact_words` (numbers from the evidence lines,
  their adjacent label words, and the answer's own content words) present in the
  rendered text. Pure string check, free, computed on every candidate.
- **keep_frac** is **token**-weighted, not word-weighted: retention is billed in
  tokens, and a policy rewarded on a word count learns to keep cheap one-token words
  and drop the 4-subword numbers that carry the answer.

## Knobs

| Flag | Default | Notes |
|---|---|---|
| `--init-from` | *(required)* | SFT checkpoint; policy **and** KL anchor |
| `--k` | 6 | candidates per item; a 7th deterministic anchor is always added |
| `--sampler` | `gumbel` | `gumbel` \| `bernoulli` — see below |
| `--gumbel-temp` | 1.0 | exploration noise on the keep-logit |
| `--target-keep` | 0.22 | the deterministic anchor's token budget |
| `--lam` | 1.0 | keep_frac penalty |
| `--kl-coef` | 0.05 | trust region |
| `--reward-temp` | 0.10 | reward -> weight softmax |
| `--lr` | 5e-6 | below v8's 2e-5: RWR moves a converged policy |
| `--pool` | `max` | subword -> word pooling |
| `--epochs` / `--batch` | 4 / 4 | ~500 steps/epoch, ~28 reader calls/step |
| `--max-llm-calls` | 150000 | clean stop (checkpoint written), not a timeout |
| `--max-hours` | 6.0 | ditto |
| `--stub-reader 1` | off | reward from gold-answer containment, no API calls |

### Why `gumbel` is the default, against the original Bernoulli spec

The Gumbel sampler is taaha's fixed-budget mechanic: perturb the word keep-logit with
Gumbel noise, take top-k to a **token budget** drawn from `cand_keeps`
(0.12/0.22/0.33 — the val keeps). Candidates then differ in *which* words, at the
retentions the eval actually measures. `--sampler bernoulli` restores the original spec.

Measured on the **real SFT init with real reader calls** (10 val items), comparing each
sampler's best exploration candidate against the deterministic anchor:

| sampler | beats anchor | best exploration R | winner's keep_frac | anchor's softmax weight |
|---|---|---|---|---|
| `bernoulli` | 40% | 1.276 | 0.411 | 0.524 |
| `gumbel --gumbel-temp 0.3` | **70%** | **1.681** | 0.191 | **0.194** |
| `gumbel --gumbel-temp 1.0` | 50% | 1.441 | 0.180 | 0.309 |

(anchor reward 1.439 at keep 0.22.) Bernoulli's problem is not diversity but *where* the
diversity lands: summing ~1100 independent coin flips concentrates the keep rate, so its
winning candidates sit at roughly twice the 0.22 target, and imitating them drags
retention up and away from the point the eval scores.

`gumbel_temp` is a real knob now — exploration beats the anchor on 70% of items at 0.3,
50% at 1.0, 40% at 2.0 and only 10% at 4.0, where the noise swamps the policy's ranking.
Default 0.3.

### The Gumbel NaN trap (fixed here, still live in taaha's trainer)

The Gumbel draw was forked verbatim from `modal_train_bear_rl_rwr.py` and is silently
broken there:

```python
g = -torch.log(-torch.log(u).clamp(min=1e-9))     # WRONG: every draw is NaN
```

`-torch.log(u).clamp(min=1e-9)` parses as `-(torch.log(u).clamp(min=1e-9))`. `log(u)` is
negative, `clamp(min=1e-9)` raises it to `+1e-9`, the unary minus makes it `-1e-9`, and
the outer `log` of a negative number is NaN. **Every** Gumbel draw comes out NaN, so
`argsort` orders by nothing and the sampler returns arbitrary masks that do not respond
to `gumbel_temp` at all — measured top-22% Jaccard of 0.005 against the anchor at every
temperature from 0.01 to 10, and identical rewards across all of them. The fix clamps
`u` on both sides and drops the inner clamp:

```python
u = torch.rand_like(wlogit).clamp(1e-9, 1 - 1e-9)
g = -torch.log(-torch.log(u))                     # mean 0.553, std 1.265
```

against Gumbel's true 0.577 / 1.283. **Worth telling taaha** — his RWR exploration has
been noise-free (i.e. absent) this whole time.

## Selection and logging

Validation runs every half epoch: deterministic masks at keeps {0.12, 0.22, 0.33} over
the 200 val items -> `fact_survival` + `answer_survival`. Appended to
`<save_path>/metrics.jsonl` and logged to W&B. Best checkpoint = highest
`answer_survival@0.22`, written to the run root (v8 layout); `latest/` is written every
validation regardless.

Watch, in order: `val/answer_survival@0.22` (the selection metric) → `train/keep_frac`
(sliding to the floor with flat answer_survival means `lam` is too high, or the reader is
down and scoring everything 0) → `train/reward_spread` → `train/kl` (should stay small
and bounded) → `reader/errors` and `reader/cache_hit_rate`.

## Assumptions and things to know

- **The Gemini key is a Modal secret.** `gemini-api-secret` (taaha's) supplies
  `GEMINI_API_KEY`. It is **not** in AWS Secrets Manager and not in the worktree `.env`,
  so nothing reader-related can be smoke-tested locally without one — that is what
  `--stub-reader 1` is for. Reached via the OpenAI-compat endpoint
  `https://generativelanguage.googleapis.com/v1beta/openai/` (the `openai` package),
  matching the rest of the harness.
- **Pooling is MIN**, matching `v8_eval_precompress.py` / `precompress_v72_focus.py`,
  the path tonight's benchmark numbers come from. Measured on the real SFT init (12 val
  items, real reader calls) the choice is close to free: answer_survival is identical at
  keep 0.22 (0.667 both), max is one item better at 0.12 and 0.33, min is marginally
  better on fact_survival — all within noise at n=12. **Before any prod ship:**
  otsofier's `merge_words` (PR #714) MAX-pools glued words, so an RL checkpoint headed
  for prod has to revisit this. `--pool max` remains available for the A/B.
- **The reward ceiling is 0.833, not 1.0.** One val item in six is unanswerable even
  from the *uncompressed* chunk. Those items can only ever teach "drop more" (the answer
  term is pinned at 0, leaving `-lam*keep_frac`), so `prefilter_full` (default on) reads
  every train chunk uncompressed once and drops the unanswerable ones — ~1.3% of the
  call budget, cached, free on resume. Val is deliberately **not** filtered so it stays
  comparable to the eval harness; the ceiling is logged as `val_ceiling` instead.
- **Numbers are atomic.** Word segmentation is whitespace spans plus merges for split
  thousands separators (`1, 904`), bare currency symbols (`$ 1,904`) and trailing units
  (`12.5 %`), per otsofier PR #714. Verified on the built dataset: 40,605 of 42,239
  numeric words span more than one subword (96%), every token run is contiguous, and
  all subwords of a word share one keep decision. At token level, 96% of numbers could
  be half-kept.
- **Frozen-ref log-probs are precomputed once** (~16 MB fp16 on CPU) instead of a ref
  forward per step. Exact — the ref is frozen and the dataset fixed — and it removes a
  third of the step cost.
- Reader errors score 0.0 rather than crashing: an overnight run must survive a
  transient upstream failure, and a zero on one candidate only removes it from that
  item's reward softmax. Failed calls are **not** cached. Watch `reader/errors` — a
  silent reader outage looks exactly like a policy that stopped preserving answers.
- The budget guard is checked at the top of each step, so a validation pass can overshoot
  it slightly. The final validation is always allowed to run.
- `word_id` (`-1` on question/specials/padding) doubles as v8's `loss_mask`: content
  tokens are exactly `word_id >= 0`.

## What has and has not been tested

Verified locally (no GPU): the full prep build (2000/200, all evidence contained, zero
unplaceable), 35 units on answer matching / rendering / fact survival / pooling / budget
masks / reward shape, number atomicity on the real dataset, and the whole RWR step
pipeline (forward -> pool -> sample -> render -> reward -> weighted imitation + KL ->
backward) on the real mmBERT-small model with real data: finite loss, gradient reaching
138 tensors, KL-to-identical-ref ~1e-6, deterministic mask hitting 0.217-0.219 against a
0.22 target. The reader probe ran for real on Modal CPU.

**Not executed:** the trainer on a GPU. Step 2 above is the first thing to run.
