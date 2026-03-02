#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

# 1. Check for .env at repo root
if [ ! -f "$ROOT/.env" ]; then
    echo ""
    echo "No .env file found. Let's set up your API keys."
    echo ""
    cp "$ROOT/.env.example" "$ROOT/.env"
    read -p "Enter your OPENAI_API_KEY: " openai_key
    read -p "Enter your BEAR_API_KEY: " bear_key
    sed -i.bak "s|your-openai-api-key-here|$openai_key|" "$ROOT/.env"
    sed -i.bak "s|your-bear-api-key-here|$bear_key|" "$ROOT/.env"
    rm -f "$ROOT/.env.bak"
    echo "Keys saved to $ROOT/.env"
    echo ""
fi

# 2. Shared venv at repo root
if [ ! -d "$ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    uv venv "$ROOT/.venv"
fi

echo "Installing dependencies..."
uv pip install -q -p "$ROOT/.venv" -r "$ROOT/requirements.txt"
echo ""

# 3. Run benchmark
echo "Starting benchmark..."
echo ""
"$ROOT/.venv/bin/python" run_benchmark.py "$@"
