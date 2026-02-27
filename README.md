# Bear Compression Benchmarks

Measures how [Bear](https://thetokencompany.com) input token compression affects gpt-5-mini accuracy across multiple datasets.

## Benchmarks

| Benchmark | Dataset | Questions | Answer Format | Evaluation |
|-----------|---------|-----------|---------------|------------|
| [`financebench/`](financebench/) | [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) | 150 | Free-text financial | LLM-as-judge |
| [`longbench_v2/`](longbench_v2/) | [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) | 503 | Multiple-choice (A/B/C/D) | Regex extraction |
| [`squad_v2/`](squad_v2/) | [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | 11,900 | Extractive QA + unanswerable | LLM-as-judge |
| [`coqa/`](coqa/) | [stanfordnlp/coqa](https://huggingface.co/datasets/stanfordnlp/coqa) | ~7,500 | Conversational free-text QA | LLM-as-judge |

Each benchmark compares a control baseline (raw context) against multiple Bear compression aggressiveness levels to measure the accuracy/compression tradeoff.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Bear API key](https://thetokencompany.com)

## Setup

API keys are stored in a single `.env` file at the repo root (shared by all benchmarks):

```bash
cp .env.example .env
# Edit .env with your API keys
```

Or just run any benchmark — it will prompt for keys on first run.

## Running a Benchmark

```bash
cd financebench   # or longbench_v2
./run.sh
```

The script handles virtual environment creation, dependency installation, and execution.

### Options

```bash
./run.sh --limit 5              # Test with 5 questions
./run.sh --config control       # Run a single config
./run.sh --config bear_0.1      # Run a single config
./run.sh --config bear_0.3 --limit 10  # Combine
```

### Configs

Each benchmark runs these configurations by default:

| Config | Context | Bear Aggressiveness |
|--------|---------|---------------------|
| `control` | Raw | — |
| `bear_0.05` | Compressed | 0.05 |
| `bear_0.1` | Compressed | 0.1 |
| `bear_0.3` | Compressed | 0.3 |

Additional levels (`bear_0.4`, `bear_0.5`, `bear_0.7`) are available via `--config`.

### Resume Support

If a run is interrupted, re-run the same command. Completed questions are skipped automatically.

## Results

Each benchmark writes results to `<benchmark>/results/<config>.json` and prints:
- Accuracy or score per config
- Compression ratio and token savings (for Bear configs)
- Breakdowns by dataset-specific categories
- Comparative summary across all configs

See each benchmark's README for dataset-specific details and evaluation methodology.
