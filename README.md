# Bear Compression Benchmarks

Measures how [Bear](https://thetokencompany.com) input token compression affects gpt-5-mini accuracy across multiple datasets.

## Benchmarks

| Benchmark | Dataset | Questions | Answer Format | Evaluation |
|-----------|---------|-----------|---------------|------------|
| [`financebench/`](financebench/) | [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) | 150 | Free-text financial | LLM-as-judge |
| [`longbench_v2/`](longbench_v2/) | [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) | 503 | Multiple-choice (A/B/C/D) | Regex extraction |
| [`squad_v2/`](squad_v2/) | [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | 11,900 | Extractive QA + unanswerable | LLM-as-judge |
| [`coqa/`](coqa/) | [stanfordnlp/coqa](https://huggingface.co/datasets/stanfordnlp/coqa) | ~7,500 | Conversational free-text QA | LLM-as-judge |

Each benchmark compares a control baseline (raw context) against multiple Bear model × aggressiveness combinations to measure the accuracy/compression tradeoff.

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

## Configuration

All settings live in [`config.yaml`](config.yaml) at the repo root:

- **Shared settings**: LLM model, Bear API URL, retry policy, token limits
- **Bear models**: list of Bear model versions to evaluate (e.g. `bear-1.2`)
- **Aggressiveness levels**: list of compression levels (e.g. `[0.1, 0.3, 0.4, 0.5, 0.7]`)
- **Per-benchmark**: dataset name, system prompt

Configs are **auto-generated** from `bear_models × aggressiveness_levels`. For example, with one model and five levels, the generated configs are: `control`, `bear-1.2--0.1`, `bear-1.2--0.3`, `bear-1.2--0.4`, `bear-1.2--0.5`, `bear-1.2--0.7`.

To add a new Bear model, just add it to the `bear_models` list in `config.yaml`.

## Running a Benchmark

```bash
cd financebench   # or longbench_v2, squad_v2, coqa
./run.sh
```

The script handles virtual environment creation, dependency installation, and execution.

### Options

```bash
./run.sh --limit 5                          # Test with 5 questions
./run.sh --config control                   # Run a single config
./run.sh --config bear-1.2--0.1             # Specific model + aggressiveness
./run.sh --config bear-1.2--0.3 --limit 10  # Combine
```

### Resume Support

If a run is interrupted, re-run the same command. Completed questions are skipped automatically.

## Results

Each benchmark writes results to `<benchmark>/results/<config>.json` (e.g. `results/bear-1.2--0.3.json`) and prints:
- Accuracy or score per config
- Compression ratio and token savings (for Bear configs)
- Breakdowns by dataset-specific categories
- Comparative summary across all configs

See each benchmark's README for dataset-specific details and evaluation methodology.
