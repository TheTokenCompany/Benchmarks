# Overnight campaign 25.07.2026: query-adaptive finance compressor

## Target
Best Pareto on real FinanceBench full-doc (150 filings, Gemini 3.5 FL answerer, gpt-5-mini judge).
North star: 90% @ 22% retention (= measured oracle-router ceiling 90.7% @ 22%).
User-approved: budget $10k; CLEAN data (no benchmark companies/filings/questions in training);
2048-ctx OK; any model size may win; keep iterating past morning until 90/22 or diminishing returns.

## Why this is possible (evidence from 24-25.07 study, see 🐻 artifact)
- Oracle router over existing configs: 90.7% @ 22% mean retention → sufficient small keep-sets EXIST per question.
- Minimal sufficient retention distribution: 30 q <5%, 33 q 5-15%, 54 q 15-30%, 19 q >30% → model must emit
  per-question adaptive mass (query-conditioned, calibrated). Fixed-threshold models cannot.
- Key content findings: number+row-label+period units are load-bearing; statement titles are labels (don't drop);
  boilerplate over-survives; welding/windows useless; line-level structure matters.
- dscale-financial-0.5M (finance-domain, tiny) leads full-doc board with rules: 84.0% @ 71% → domain data works.

## Phases
P0 research-taaha agent: extract taaha's training loop/GRPO/datagen specifics. [running]
P1 edgar-corpus agent: 200-300 clean 10-K/10-Q texts, NO FinanceBench companies. [running]
P1b synthetic QA: generate FinanceBench-style questions w/ KNOWN evidence spans (generation gives ground truth);
    verify answerability (answer from evidence page + judge). ~2-5k questions.
P1c masks: evidence lines full + row labels + statement titles + periods + numbers in evidence region + graded halo;
    per-question retention target varies (needle ~2-10%, qualitative 20-40%). Validate sufficiency on a subsample
    via compress→answer→judge. Held-out synthetic val split for checkpoint selection (never select on real FB).
P2 ~10 training runs, Modal B200, volumes prefixed otso-v8 (never write taaha's volumes):
    R1 SFT 2048 warm-start v7.1 | R2 SFT 2048 from mmBERT-small | R3 SFT warm-start bear-4.0-focus-1-ctx2048
    R4 SFT 512 (serving-compat) | R5 focal/imbalance variant | R6 retention-conditioning token variant
    R7 hard-negative windows | R8 curriculum general→finance | R9 ettin-400m capacity probe | R10 best-recipe seed 2
    Conventions: content-loss masking (CLS/SEP/PAD excluded), [CLS] q [SEP] chunk [SEP], macro-F1/AUPRC selection.
P3 auto-eval: each checkpoint → precompress real-FB fd (threshold sweep) → Gemini fleet (QUESTION_CONCURRENCY=4,
    purge-retry pattern) → Pareto; also +safetab overlay. Compare vs champions:
    82.7@57 (safetab), 84.0@71 (dscale+safetab), 77.3@46 (e16b), 66@30 (linemax), 37.3@12 (ws72@0.7).
P4 artifact update (🐻 https://claude.ai/code/artifact/21572692-f0b5-4ed8-b358-dc937f240f15) + morning report.

## Infra notes
- Harness: financebench/run_benchmark.py (concurrent, ANSWER_* env, FULLDOC_JSON, RESULTS_SUBDIR, BEAR_MODELS/AGGR_LEVELS env).
- fulldoc_contexts.json in scratchpad (rebuildable via rebuild_fulldocs.py).
- Gemini key: otsofy-llm-api-keys-usw2 (google_ai_api_key). Full-doc fleets throttle >~40 concurrent: use conc 4 + purge-retry.
- Precompress scripts (all support FULLDOC_JSON/ALIAS_PREFIX): v71, v40_focus, v40_fix, fuse, e16_20, v72_focus, v72_rlfocus.
- Modal: tokenco workspace, .venv/bin/modal. B200 via app functions; caches → financebench/compression_cache/.
