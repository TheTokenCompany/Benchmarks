# Latency Benchmark

Measures time-to-first-token, time-to-first-byte, total latency, and streaming
throughput for chat completions across providers, models, and input sizes.

## Matrix

**Providers:**
- `native` — OpenAI API (for `gpt-*` models) or Anthropic Messages API (for `claude-*` models)
- `openrouter` — OpenRouter
- `ttc_none` — TTC Gateway, no compression (`aggressiveness=0`)
- `ttc_01` — TTC Gateway, `aggressiveness=0.1`
- `ttc_05` — TTC Gateway, `aggressiveness=0.5`

**Models:**
- `gpt-5.4`, `gpt-5-mini` (OpenAI)
- `claude-sonnet-4.6`, `claude-opus-4.6` (Anthropic)

**Input sizes:**
- `short` (~200 tokens)
- `medium` (~2000 tokens)
- `long` (~10000 tokens)

Inputs are generated automatically the first time the benchmark runs (via the
TTC gateway using `gpt-5-mini`) and cached in `inputs.json`.

Total: 4 models × 5 providers × 3 sizes = **60 combinations**, each run 3× by default = 180 streamed requests.

## Setup

From the benchmarks root (`benchmarks/`):

```bash
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, TTC_API_KEY
```

Install deps:

```bash
pip install -r requirements.txt
# or with uv: uv pip install -r requirements.txt
```

## Run

```bash
cd latency
python benchmark.py
```

Options:

```bash
python benchmark.py --runs 5              # 5 runs per combo instead of 3
python benchmark.py --provider ttc_01     # only TTC with aggressiveness=0.1
python benchmark.py --model gpt-5-mini    # only one model
python benchmark.py --size short          # only short inputs
# combine filters
python benchmark.py --model gpt-5-mini --size short --runs 1
```

## Metrics

Per run:
- `time_to_first_byte_ms` — first HTTP response byte after request send
- `time_to_first_token_ms` — first non-empty content token in the stream
- `total_time_ms` — from request send to stream close
- `chunks_received` — SSE chunks with content
- `content_chars` — total characters in streamed response
- `chars_per_sec` — sustained streaming throughput (excluding initial latency)

Results are saved to `results.json` with both per-run data and averaged
summaries, plus a formatted summary table printed at the end.

## Model IDs

If the benchmark fails on a specific provider with "model not found", update
the corresponding ID in the `MODELS` list at the top of `benchmark.py`. The
defaults assume:

- Anthropic native API: `claude-sonnet-4-6`, `claude-opus-4-6`
- OpenRouter: `anthropic/claude-sonnet-4.6`, `openai/gpt-5.4`, etc.
- TTC Gateway: same format as OpenRouter

## Notes

- Tests run sequentially to avoid network contention distorting measurements.
- Max output is capped at 200 tokens so response-side latency doesn't dominate.
- The same input text is used across all providers for a given size so
  compression wins are real (TTC may be faster on long inputs even though it
  adds compression overhead, because the LLM sees fewer tokens).
