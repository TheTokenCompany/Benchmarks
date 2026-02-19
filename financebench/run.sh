#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# 1. Check for .env
if [ ! -f .env ]; then
    echo ""
    echo "No .env file found. Let's set up your API keys."
    echo ""
    cp .env.example .env
    read -p "Enter your OPENAI_API_KEY: " openai_key
    read -p "Enter your BEAR_API_KEY: " bear_key
    sed -i.bak "s|your-openai-api-key-here|$openai_key|" .env
    sed -i.bak "s|your-bear-api-key-here|$bear_key|" .env
    rm -f .env.bak
    echo "Keys saved to .env"
    echo ""
fi

# 2. Set up uv venv and install dependencies
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    uv venv
fi

echo "Installing dependencies..."
uv pip install -q -r requirements.txt
echo ""

# 3. Run benchmark
echo "Starting benchmark..."
echo ""
.venv/bin/python run_benchmark.py "$@"
