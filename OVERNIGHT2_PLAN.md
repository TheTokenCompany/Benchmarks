# OVERNIGHT-2 (26.07.2026) — RL campaign, target ≥90%, +$10k approved

User mandate (verbatim intent): "REALLY REALLY GOOD PARETO, RLd to the max, at least
90% quality. spend up to 10k more." User asleep; full autonomy; keep iterating.
ALWAYS update the 🐻 artifact (claude.ai/code/artifact/21572692-f0b5-4ed8-b358-dc937f240f15)
with every result batch. ALWAYS run experiments on full 150 filings (fd- configs).

## Honest ceilings (established 25.07)
- Single-run ceiling with Gemini-3.5-FL answerer + gpt-5-mini judge: ~88.7% (union of 60+ configs).
- E29: Fable-5 extractive oracle = 0/17 flips (selection exhausted).
- E30: Fable-5 abstractive = 3/17 flips (representation lever real, needs CORRECT labels).
- E31: heuristic banner labels = NEGATIVE (-2 / -5.3 pp). Correctness threshold is real.
- 90% single-run requires: reliability on the noisy ~10 marginal Qs + representation wins.

## Current champions (full 150, Gemini): dscale-s1-safetab@0.3 84.0% @71%;
v7.1-safetab@0.5 82.7% @57%. gpt-5-mini reader (E28): 86.0% vs own control 85.3%.

## Workstreams
1. LEDGAR MCQ regen (Alan's pipeline, sandbox/finance_domain_research/ledgar_bench @
   b7a8f1e3, worktree at $SCRATCH/research-finance). Task b0w1kryxj running
   (quicktest→full, $40 guard). Then SECQUE (~$60 Opus). Alan's produced data is
   gitignored+unmirrored → regen is the only path. His sweep: r≥0.3 saturated →
   reward must run at 5-25% retention band.
2. Wave-2 SFT evals: w2-140 (AUPRC .806 ep8), w2-140-s2 (.825 ep9), w2-ettin (.827 ep8),
   12-epoch runs, one Modal app still active. On finish: v8_eval_precompress budget22/33
   ±safetab + early-epoch w2-140-s2 → Gemini fleet full-150 → pick RL init.
   Wakeup scheduled. REMEMBER wave-1 transfer inversion: eval early epochs too.
3. v9 RL build: agent "v9-rl-builder" building v9_rl_prep.py + v9_rwr_train.py
   (RWR/GRPO-style: K=6 sampled word-masks, softmax-weighted imitation, KL-to-ref,
   reward = 2·answer_survival(gemini-3.1-flash-lite, deterministic numeric match, NO judge)
   + 0.5·fact_survival − λ·keep_frac − degen penalty; λ→10-25% keep band; LLM cache;
   6h/150k-call guards). Init from best wave-2 checkpoint. Modal app otso-v9-rl, B200.
4. Contamination: diff SECQUE 45 CIK×year filings vs v8 corpus before using as reward.
5. Periodic: eval RL checkpoints on FinanceBench full-150 (v8_eval_precompress →
   fleet), artifact update each batch.

## Keys / env
- ANTHROPIC: SM otso-personal-anthropic-api-key (default profile us-west-2).
- GEMINI: SM otsofy-llm-api-keys-usw2 → google_ai_api_key.
- OPENAI (judge): worktree .env. Prod bear API: SM ttc-prod-api-key-otso.
- SCRATCH=/private/tmp/claude-501/-Users-otsov--superset-worktrees-8144834a-c76f-41f6-b409-cb03154f2355-financebench-check/838fe3de-3a8f-4846-ba75-9e663316210c/scratchpad
- Fleet pattern: cd financebench && FULLDOC_JSON=$SCRATCH/fulldoc_contexts.json
  RESULTS_SUBDIR=results_gemini35fl_fulldoc OPENAI_MODEL=gemini-3.5-flash-lite
  ANSWER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  ANSWER_API_KEY=$GEMINI_API_KEY BEAR_MODELS=<name> AGGR_LEVELS=<levels>
  QUESTION_CONCURRENCY=6 ../.venv/bin/python run_benchmark.py --config <name>--<lvl>
  (429 poisoning → purge ERROR records, retry at concurrency 2-4).

## Morning deliverable
Artifact: updated master charts + RL section; summary message: best Pareto table,
what RL bought, spend estimate, honest gap-to-90 statement.

## Mid-night state (26.07 ~03:30)
- RL RECORDS: v9-kl03 stable ckpt (exp-20260726-023910-v9-rwr-s2ep4-kl03) = 74.0% @ 22%
  (4.7pp from control), 75.3% @ 33%, 80.0% @ 55% safetab. Unstable peak s2ep4 = 73.3/76.0.
  KL 0.3 = stability fix (collapse at kl 0.05 after ep1; NaN-Gumbel bug in taaha's RWR
  found+fixed, memory ttc_taaha_rwr_gumbel_nan_bug).
- Builder's 200-item paired anchor read (anchor_read_200.json in SCRATCH): inits TIED at
  keep 0.22 (0.635 both, p=1.0); ettin +9.0pp at keep 0.12 (p=.015 raw, ~.06 Bonferroni);
  TRUE reward ceiling 0.915 (not 0.833). Serving decision rule: mmBERT-140M for 22% band
  (3x cheaper), ettin-400m only for a 12%-band product.
- Running: kl015 (mmBERT, chasing >0.705 stably), iter2-from-peak (re-anchored RWR),
  ettin-s0 (4ep, kl 0.3). Each: on finish → cache (v8_eval_precompress, MODEL_VOL_PATH=
  exp dir, ALIAS_PREFIX=fd-) → fleet budget22/33 ±safetab → artifact.
- Wave-2 SFT verdict (E32): ettin best SFT (78.7 @ 51) but no wave-2 beat wave-1 deep;
  inversion (early epoch > full) replicated 3x.
- Artifact current through kl03 records (label kl03-stable-74-at-22).

## v11 label-policy record (28.07, final)
Seam-window survivor rule: v11_pretokenize.py IMPLEMENTS the qtype-conditional
tightening (metrics/multistep labels-only seam windows -> relabelled negative;
check_window raises on violations). On the v11 corpus this is INERT — zero
relabels; all 62 labels-only seam windows are domain-qualitative, which the rule
keeps by design. Decision (option a): keep the tightening live as a guard. It
costs nothing here and catches the bad case if a future corpus (v12) produces
numeric-qtype labels-only seams. Recorded so no future rebuild "discovers" the
branch. Tensors sha256-verified identical across final uploads; word-boundary
golden fixtures run at the start of every build.

## E43 record: adaptive dosing KILLED (28.07, ~$5)
Every confidence-dose policy loses to fixed budget at matched mean retention
(best: -2.0pp at tightest band; pure tau -4.7). Mechanism: accuracy-retention is
steeply concave -> Jensen penalty on retention variance; starving (-16.7pp)
costs 3x feeding (+5.7); router signal Spearman 0.014 = uninformative.
Oracle "90.7%" debunked: above reader ceiling; best-of-3-identical-runs =
85-87% from 6-11% verdict flips alone. RULE: no per-question oracles over
config banks. Budget22 failure split: 49% evidence PRESENT but reader failed
(unreachable), 51% evidence lost (recall -> v12 RL target). safetab-on-kept
+4.6pp at same retention: WHAT you keep dominates HOW MUCH.

## E44 record: evidence-rank segmentation (28.07, ~$1) — V12 IS DATA-FIRST
r_star = retention at which each of 3876 gold-evidence numbers first survives.
Failures are enriched 3.3x in BLIND SPOTS (r*>50%) vs only 1.7x in near misses
(22-33%); near misses are largely benign (reader answers anyway). Median dropped
number on failed questions needs ~43% retention. => RL sharpening buys the least
distinguishing bucket; v12 = training-data coverage of blind-spot line types
first, RL second. Blind-spot (question, number, r_star) list in scratchpad
evrank.json. ALSO: 8.1% of gold evidence numbers absent from extracted text
entirely — PDF extraction loss, hard recall ceiling; try pdfplumber/marker vs
pypdf before v12 evals.
