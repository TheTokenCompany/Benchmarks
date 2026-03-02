"""Shared configuration loader for all benchmarks.

Loads config.yaml from the repo root, merges shared + benchmark-specific
sections, and auto-generates CONFIGS from bear_models × aggressiveness_levels.

Usage in any benchmark's config.py:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from shared_config import load_config
    globals().update(load_config("financebench"))
"""

import os

import yaml
from dotenv import load_dotenv

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(benchmark_name: str) -> dict:
    """Load and merge shared + benchmark-specific config from config.yaml.

    Auto-generates CONFIGS dict from bear_models × aggressiveness_levels:
        "control"        -> {compressed: False, aggressiveness: None, bear_model: None}
        "bear-1.2--0.1"  -> {compressed: True,  aggressiveness: 0.1, bear_model: "bear-1.2"}
        ...

    Returns flat dict of all configuration constants, ready for
    globals().update() in each benchmark's config.py.
    """
    # Load .env from repo root
    load_dotenv(os.path.join(_ROOT_DIR, ".env"))

    # Load YAML
    yaml_path = os.path.join(_ROOT_DIR, "config.yaml")
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    shared = raw["shared"]
    bench = raw["benchmarks"][benchmark_name]

    cfg = {}

    # API keys (always from env)
    cfg["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    cfg["BEAR_API_KEY"] = os.getenv("BEAR_API_KEY")

    # Shared values (env vars can override)
    cfg["OPENAI_MODEL"] = os.getenv("OPENAI_MODEL", shared["openai_model"])
    cfg["JUDGE_MODEL"] = os.getenv("JUDGE_MODEL", shared["judge_model"])
    cfg["BEAR_API_URL"] = shared["bear_api_url"]
    cfg["MAX_INPUT_TOKENS"] = shared["max_input_tokens"]
    cfg["MAX_RETRIES"] = shared["max_retries"]
    cfg["RETRY_BASE_DELAY"] = shared["retry_base_delay"]
    cfg["BEAR_MODELS"] = shared["bear_models"]
    cfg["AGGRESSIVENESS_LEVELS"] = shared["aggressiveness_levels"]

    # Results dir: <root>/<benchmark_name>/results
    cfg["RESULTS_DIR"] = os.path.join(_ROOT_DIR, benchmark_name, "results")

    # Benchmark-specific values
    cfg["DATASET_NAME"] = bench["dataset_name"]
    cfg["SYSTEM_PROMPT"] = bench["system_prompt"]
    cfg["NUM_QUESTIONS_LIMIT"] = bench.get("num_questions_limit")  # None means all

    # Auto-generate CONFIGS from bear_models × aggressiveness_levels
    configs = {"control": {"compressed": False, "aggressiveness": None, "bear_model": None}}
    for model in shared["bear_models"]:
        for agg in shared["aggressiveness_levels"]:
            name = f"{model}--{agg}"
            configs[name] = {
                "compressed": True,
                "aggressiveness": agg,
                "bear_model": model,
            }
    cfg["CONFIGS"] = configs

    return cfg
