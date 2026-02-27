#!/usr/bin/env python3
"""LongBench V2 benchmark runner with Bear compression."""

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


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return len(text) // 4


def load_longbench_v2():
    """Load the LongBench V2 dataset."""
    ds = load_dataset(config.DATASET_NAME, split="train")
    return ds


def extract_context(item) -> str:
    """Extract context directly from the context field."""
    return item["context"].strip()


def build_prompt(context: str, question: str, choice_a: str, choice_b: str,
                 choice_c: str, choice_d: str) -> list[dict]:
    """Build the chat messages for the LLM."""
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"A) {choice_a}\n"
        f"B) {choice_b}\n"
        f"C) {choice_c}\n"
        f"D) {choice_d}"
    )
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

                max_completion_tokens=1024,
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


def run_single_config(config_name: str, items: list, limit: int | None = None):
    """Run benchmark for a single configuration."""
    cfg = config.CONFIGS[config_name]
    is_compressed = cfg["compressed"]
    aggressiveness = cfg["aggressiveness"]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.RESULTS_DIR, f"{config_name}.json")

    # Resume support
    results = load_existing_results(results_path)
    completed_ids = get_completed_ids(results)

    if limit is not None:
        items = items[:limit]

    remaining = []
    for i, item in enumerate(items):
        qid = item.get("_id", str(i))
        if qid not in completed_ids:
            remaining.append((i, item))

    if not remaining:
        print(f"  [{config_name}] All {len(items)} questions already completed. Skipping.")
        return results

    print(f"  [{config_name}] {len(results)} done, {len(remaining)} remaining")

    for i, item in tqdm(remaining, desc=config_name, unit="q"):
        qid = item.get("_id", str(i))
        question = item["question"]
        gold_answer = item["answer"]
        domain = item.get("domain", "")
        sub_domain = item.get("sub_domain", "")
        difficulty = item.get("difficulty", "")
        length_category = item.get("length", "")
        choice_a = item.get("choice_A", "")
        choice_b = item.get("choice_B", "")
        choice_c = item.get("choice_C", "")
        choice_d = item.get("choice_D", "")

        # Extract context
        raw_context = extract_context(item)

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
        messages = build_prompt(context_for_llm, question,
                                choice_a, choice_b, choice_c, choice_d)
        try:
            model_answer = query_llm(messages)
        except RuntimeError as e:
            print(f"  LLM failed for question {qid}: {e}")
            model_answer = f"ERROR: {e}"

        # Evaluate
        eval_result = evaluate_answer(gold_answer, model_answer)

        result = {
            "question_id": qid,
            "question": question,
            "domain": domain,
            "sub_domain": sub_domain,
            "difficulty": difficulty,
            "length_category": length_category,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "extracted_answer": eval_result["extracted_answer"],
            "correct": eval_result["correct"],
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
    evaluated = [r for r in results if r.get("correct") is not None]
    correct = sum(1 for r in evaluated if r["correct"])
    accuracy = correct / len(evaluated) if evaluated else 0
    failed_extractions = sum(1 for r in results if r.get("correct") is None)

    print(f"\n{'=' * 60}")
    print(f"  {config_name} — Summary")
    print(f"{'=' * 60}")
    print(f"  Total questions:       {total}")
    print(f"  Evaluated:             {len(evaluated)}")
    print(f"  Extraction failures:   {failed_extractions}")
    print(f"  Correct:               {correct}")
    print(f"  Accuracy:              {accuracy:.1%}")

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

    # Breakdown by domain
    domains = set(r.get("domain") for r in evaluated)
    domains.discard("")
    domains.discard(None)
    if domains:
        print(f"\n  By domain:")
        for d in sorted(domains):
            subset = [r for r in evaluated if r.get("domain") == d]
            d_correct = sum(1 for r in subset if r["correct"])
            d_acc = d_correct / len(subset) if subset else 0
            print(f"    {d}: {d_correct}/{len(subset)} ({d_acc:.1%})")

    # Breakdown by difficulty
    difficulties = set(r.get("difficulty") for r in evaluated)
    difficulties.discard("")
    difficulties.discard(None)
    if difficulties:
        print(f"\n  By difficulty:")
        for diff in sorted(difficulties):
            subset = [r for r in evaluated if r.get("difficulty") == diff]
            diff_correct = sum(1 for r in subset if r["correct"])
            diff_acc = diff_correct / len(subset) if subset else 0
            print(f"    {diff}: {diff_correct}/{len(subset)} ({diff_acc:.1%})")

    # Breakdown by length category
    lengths = set(r.get("length_category") for r in evaluated)
    lengths.discard("")
    lengths.discard(None)
    if lengths:
        print(f"\n  By length:")
        for ln in sorted(lengths):
            subset = [r for r in evaluated if r.get("length_category") == ln]
            ln_correct = sum(1 for r in subset if r["correct"])
            ln_acc = ln_correct / len(subset) if subset else 0
            print(f"    {ln}: {ln_correct}/{len(subset)} ({ln_acc:.1%})")

    print()


def main():
    parser = argparse.ArgumentParser(description="LongBench V2 benchmark with Bear compression")
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

    print("Loading LongBench V2 dataset...")
    dataset = load_longbench_v2()
    total_loaded = len(dataset)
    print(f"Loaded {total_loaded} questions")

    # Filter out questions whose uncompressed prompt exceeds the model's input token limit
    items = list(dataset)
    if args.limit is not None:
        items = items[:args.limit]
    filtered_items = []
    for item in items:
        context = extract_context(item)
        prompt_text = (
            config.SYSTEM_PROMPT + "\n"
            + f"Context:\n{context}\n\n"
            + f"Question: {item['question']}\n\n"
            + f"A) {item.get('choice_A', '')}\n"
            + f"B) {item.get('choice_B', '')}\n"
            + f"C) {item.get('choice_C', '')}\n"
            + f"D) {item.get('choice_D', '')}"
        )
        if estimate_tokens(prompt_text) <= config.MAX_INPUT_TOKENS:
            filtered_items.append(item)
    discarded = len(items) - len(filtered_items)
    print(f"Filtered: {len(filtered_items)} questions within {config.MAX_INPUT_TOKENS:,}-token limit, "
          f"{discarded} discarded ({discarded}/{len(items)})\n")

    all_results = {}
    for cfg_name in configs_to_run:
        print(f"Running config: {cfg_name}")
        results = run_single_config(cfg_name, filtered_items, limit=None)
        all_results[cfg_name] = results
        print_summary(cfg_name, results)

    # Print comparative summary if multiple configs
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  Comparative Summary")
        print(f"{'=' * 60}")
        for cfg_name, results in all_results.items():
            evaluated = [r for r in results if r.get("correct") is not None]
            correct = sum(1 for r in evaluated if r["correct"])
            accuracy = correct / len(evaluated) if evaluated else 0
            compressed_results = [r for r in results if "original_tokens" in r]
            if compressed_results:
                total_original = sum(r["original_tokens"] for r in compressed_results)
                total_compressed = sum(r["compressed_tokens"] for r in compressed_results)
                tokens_saved = total_original - total_compressed
                if tokens_saved > 0:
                    savings_pct = tokens_saved / total_original * 100
                    print(f"  {cfg_name:15s}  accuracy={accuracy:.1%}  tokens_saved={tokens_saved:,} ({savings_pct:.1f}% reduction)")
                else:
                    print(f"  {cfg_name:15s}  accuracy={accuracy:.1%}  (no effective compression)")
            else:
                print(f"  {cfg_name:15s}  accuracy={accuracy:.1%}  (no compression)")
        print()


if __name__ == "__main__":
    main()
