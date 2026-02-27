#!/usr/bin/env python3
"""MRCR v2 (8-needle) benchmark runner with Bear compression."""

import argparse
import json
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

import config
from compress import compress_text
from evaluate import evaluate_answer


def load_mrcr():
    """Load the MRCR dataset, filtered to 8-needle samples."""
    ds = load_dataset(config.DATASET_NAME, split="train")
    ds = ds.filter(lambda x: x["n_needles"] == config.N_NEEDLES)
    return ds


def extract_context_and_question(item) -> tuple[str, str]:
    """Parse the prompt JSON and separate context from the final question.

    The prompt is a JSON string containing a list of OpenAI-format messages.
    The last user message is the retrieval question; everything before it
    is the conversation context that may be compressed.
    """
    messages = json.loads(item["prompt"])

    # Last message is the retrieval question
    question = messages[-1]["content"]

    # Serialize prior messages as text
    context_parts = []
    for msg in messages[:-1]:
        role = msg["role"].capitalize()
        context_parts.append(f"{role}: {msg['content']}")
    context = "\n\n".join(context_parts)

    return context, question


def build_prompt(context: str, question: str) -> list[dict]:
    """Build the chat messages for the LLM."""
    user_content = f"{context}\n\n{question}"
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def query_llm(messages: list[dict]) -> str:
    """Send messages to OpenAI and return the response text."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=0,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  LLM API error (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"LLM API failed after {config.MAX_RETRIES} retries: {e}")


def load_existing_results(results_path: str) -> list[dict]:
    """Load existing results for resume support."""
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return []


def save_results(results_path: str, results: list[dict]):
    """Save results to JSON."""
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def get_completed_ids(results: list[dict]) -> set:
    """Get set of question IDs already completed."""
    return {r["question_id"] for r in results if "question_id" in r}


def get_context_bin(n_chars: int) -> str:
    """Map character count to a context-length bin label."""
    # Approximate token bins (chars / ~4 ≈ tokens) matching MRCR's 8 bins
    bins = [
        (0, 32768, "4k-8k tokens"),
        (32768, 65536, "8k-16k tokens"),
        (65536, 131072, "16k-32k tokens"),
        (131072, 262144, "32k-64k tokens"),
        (262144, 524288, "64k-128k tokens"),
        (524288, 1048576, "128k-256k tokens"),
        (1048576, 2097152, "256k-512k tokens"),
        (2097152, float("inf"), "512k-1M tokens"),
    ]
    for low, high, label in bins:
        if low <= n_chars < high:
            return label
    return "unknown"


def run_single_config(config_name: str, dataset, limit: int | None = None):
    """Run benchmark for a single configuration."""
    cfg = config.CONFIGS[config_name]
    is_compressed = cfg["compressed"]
    aggressiveness = cfg["aggressiveness"]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.RESULTS_DIR, f"{config_name}.json")

    # Resume support
    results = load_existing_results(results_path)
    completed_ids = get_completed_ids(results)

    items = list(dataset)
    if limit is not None:
        items = items[:limit]

    remaining = []
    for i, item in enumerate(items):
        qid = str(i)
        if qid not in completed_ids:
            remaining.append((i, item))

    if not remaining:
        print(f"  [{config_name}] All {len(items)} questions already completed. Skipping.")
        return results

    print(f"  [{config_name}] {len(results)} done, {len(remaining)} remaining")

    for i, item in tqdm(remaining, desc=config_name, unit="q"):
        qid = str(i)
        gold_answer = item["answer"]
        random_string = item["random_string_to_prepend"]
        desired_msg_index = item.get("desired_msg_index", 0)
        total_messages = item.get("total_messages", 0)
        n_chars = item.get("n_chars", 0)
        context_bin = get_context_bin(n_chars)

        # Extract context and question from the prompt JSON
        raw_context, question = extract_context_and_question(item)

        # Compress if needed
        compression_info = {}
        if is_compressed:
            try:
                comp_result = compress_text(raw_context, aggressiveness)
                context_for_llm = comp_result["compressed_text"]
                compression_info = {
                    "original_tokens": comp_result["original_tokens"],
                    "compressed_tokens": comp_result["compressed_tokens"],
                    "compression_ratio": (
                        comp_result["compressed_tokens"] / comp_result["original_tokens"]
                        if comp_result["original_tokens"] > 0
                        else 1.0
                    ),
                }
            except RuntimeError as e:
                print(f"  Compression failed for question {qid}: {e}")
                context_for_llm = raw_context
                compression_info = {"error": str(e)}
        else:
            context_for_llm = raw_context

        # Query LLM
        messages = build_prompt(context_for_llm, question)
        try:
            model_answer = query_llm(messages)
        except RuntimeError as e:
            print(f"  LLM failed for question {qid}: {e}")
            model_answer = f"ERROR: {e}"

        # Evaluate
        eval_result = evaluate_answer(gold_answer, model_answer, random_string)

        result = {
            "question_id": qid,
            "n_needles": config.N_NEEDLES,
            "desired_msg_index": desired_msg_index,
            "total_messages": total_messages,
            "n_chars": n_chars,
            "context_bin": context_bin,
            "gold_answer": gold_answer[:200] + "..." if len(gold_answer) > 200 else gold_answer,
            "model_answer": model_answer[:200] + "..." if len(model_answer) > 200 else model_answer,
            "score": eval_result["score"],
            "hash_present": eval_result["hash_present"],
            "evaluation_explanation": eval_result["explanation"],
            "config": config_name,
            "compressed": is_compressed,
            "aggressiveness": aggressiveness,
            **compression_info,
        }

        results.append(result)
        # Incremental save
        save_results(results_path, results)

    return results


def print_summary(config_name: str, results: list[dict]):
    """Print summary statistics for a config run."""
    if not results:
        print(f"\n--- {config_name}: No results ---")
        return

    total = len(results)
    scores = [r["score"] for r in results if r.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0
    hash_missing = sum(1 for r in results if not r.get("hash_present", True))

    print(f"\n{'=' * 60}")
    print(f"  {config_name} — Summary")
    print(f"{'=' * 60}")
    print(f"  Total questions:       {total}")
    print(f"  Average score:         {avg_score:.3f}")
    print(f"  Hash missing:          {hash_missing}")

    # Compression stats
    compressed_results = [r for r in results if "original_tokens" in r]
    if compressed_results:
        total_original = sum(r["original_tokens"] for r in compressed_results)
        total_compressed = sum(r["compressed_tokens"] for r in compressed_results)
        tokens_saved = total_original - total_compressed
        print(f"  Original tokens:       {total_original:,}")
        print(f"  Compressed tokens:     {total_compressed:,}")
        if tokens_saved > 0:
            savings_pct = tokens_saved / total_original * 100
            print(f"  Tokens saved:          {tokens_saved:,} ({savings_pct:.1f}% reduction)")
        else:
            print(f"  Tokens saved:          0 (no effective compression at this aggressiveness)")

    # Breakdown by context_bin
    bins = set(r.get("context_bin") for r in results)
    bins.discard("")
    bins.discard(None)
    if bins:
        print(f"\n  By context length:")
        for b in sorted(bins):
            subset = [r for r in results if r.get("context_bin") == b]
            bin_scores = [r["score"] for r in subset if r.get("score") is not None]
            bin_avg = sum(bin_scores) / len(bin_scores) if bin_scores else 0
            print(f"    {b}: avg={bin_avg:.3f} (n={len(subset)})")

    # Breakdown by desired_msg_index (which needle was requested)
    indices = set(r.get("desired_msg_index") for r in results)
    indices.discard(0)
    indices.discard(None)
    if indices:
        print(f"\n  By desired_msg_index:")
        for idx in sorted(indices):
            subset = [r for r in results if r.get("desired_msg_index") == idx]
            idx_scores = [r["score"] for r in subset if r.get("score") is not None]
            idx_avg = sum(idx_scores) / len(idx_scores) if idx_scores else 0
            print(f"    index {idx}: avg={idx_avg:.3f} (n={len(subset)})")

    print()


def main():
    parser = argparse.ArgumentParser(description="MRCR v2 (8-needle) benchmark with Bear compression")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(config.CONFIGS.keys()),
        help="Run a single config (default: run all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions per config (for testing)",
    )
    args = parser.parse_args()

    # Validate API keys
    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key.")
        return

    configs_to_run = [args.config] if args.config else list(config.CONFIGS.keys())

    # Check Bear API key for compressed configs
    compressed_configs = [c for c in configs_to_run if config.CONFIGS[c]["compressed"]]
    if compressed_configs and not config.BEAR_API_KEY:
        print("ERROR: BEAR_API_KEY not set but compressed configs requested:", compressed_configs)
        print("Copy .env.example to .env and fill in your Bear API key.")
        return

    print("Loading MRCR dataset (8-needle)...")
    dataset = load_mrcr()
    print(f"Loaded {len(dataset)} questions\n")

    all_results = {}
    for cfg_name in configs_to_run:
        print(f"Running config: {cfg_name}")
        results = run_single_config(cfg_name, dataset, limit=args.limit)
        all_results[cfg_name] = results
        print_summary(cfg_name, results)

    # Print comparative summary if multiple configs
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  Comparative Summary")
        print(f"{'=' * 60}")
        for cfg_name, results in all_results.items():
            scores = [r["score"] for r in results if r.get("score") is not None]
            avg_score = sum(scores) / len(scores) if scores else 0
            compressed_results = [r for r in results if "original_tokens" in r]
            if compressed_results:
                total_original = sum(r["original_tokens"] for r in compressed_results)
                total_compressed = sum(r["compressed_tokens"] for r in compressed_results)
                tokens_saved = total_original - total_compressed
                if tokens_saved > 0:
                    savings_pct = tokens_saved / total_original * 100
                    print(f"  {cfg_name:15s}  avg_score={avg_score:.3f}  tokens_saved={tokens_saved:,} ({savings_pct:.1f}% reduction)")
                else:
                    print(f"  {cfg_name:15s}  avg_score={avg_score:.3f}  (no effective compression)")
            else:
                print(f"  {cfg_name:15s}  avg_score={avg_score:.3f}  (no compression)")
        print()


if __name__ == "__main__":
    main()
