# FinanceBench × Bear Compression Benchmark

Measures how [Bear](https://thetokencompany.com) input token compression affects GPT-5.2 accuracy on 150 financial Q&A tasks from [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench).

## Quick Start

```bash
# Clone and run — the script handles everything
./run.sh
```

That's it. On first run it will:
1. Ask for your **OpenAI** and **Bear** API keys
2. Create a uv virtual environment and install dependencies
3. Run all 4 benchmark configs (control + 3 compression levels)

## What You Need

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Bear API key](https://thetokencompany.com)

## What It Runs

| Config | Context | Bear Aggressiveness |
|--------|---------|---------------------|
| `control` | Raw evidence pages | — |
| `bear_0.05` | Compressed | 0.05 |
| `bear_0.1` | Compressed | 0.1 |
| `bear_0.3` | Compressed | 0.3 |

Each config sends all 150 FinanceBench questions to GPT-5.2 with oracle context (the gold evidence pages). A GPT-5.2 judge then scores each answer against the verified gold answer.

## Options

```bash
# Run everything (all 4 configs, all 150 questions)
./run.sh

# Test with 5 questions first
./run.sh --limit 5

# Run a single config
./run.sh --config control
./run.sh --config bear_0.1

# Combine
./run.sh --config bear_0.3 --limit 10
```

## Output

Results are saved to `results/<config>.json`. Each run also prints:
- Accuracy per config
- Compression ratio (for Bear configs)
- Breakdown by question type and reasoning category
- Side-by-side comparison across all configs

## Resume Support

If a run is interrupted, just re-run the same command. It picks up where it left off — completed questions are skipped automatically.

## Manual Setup (if you prefer)

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
.venv/bin/python run_benchmark.py
```
