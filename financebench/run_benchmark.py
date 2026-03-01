#!/usr/bin/env python3
"""FinanceBench benchmark runner with Bear compression."""

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
from evaluate import judge_answer


def load_financebench():
    """Load the FinanceBench dataset."""
    ds = load_dataset(config.DATASET_NAME, split="train")
    return ds


def extract_context(item) -> str:
    """Extract oracle context from evidence pages."""
    evidence_list = item["evidence"]
    pages = []
    for ev in evidence_list:
        text = ev.get("evidence_text_full_page", "")
        if text:
            pages.append(text.strip())
    return "\n\n---\n\n".join(pages)


def build_prompt(context: str, question: str) -> list[dict]:
    """Build the chat messages for the LLM."""
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
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


def run_single_config(config_name: str, dataset, limit: int | None = None):
    """Run benchmark for a single configuration."""
    cfg = config.CONFIGS[config_name]
    is_compressed = cfg["compressed"]
    aggressiveness = cfg["aggressiveness"]
    bear_model = cfg["bear_model"]

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
        qid = item.get("question_id", str(i))
        if qid not in completed_ids:
            remaining.append((i, item))

    if not remaining:
        print(f"  [{config_name}] All {len(items)} questions already completed. Skipping.")
        return results

    print(f"  [{config_name}] {len(results)} done, {len(remaining)} remaining")

    for i, item in tqdm(remaining, desc=config_name, unit="q"):
        qid = item.get("question_id", str(i))
        question = item["question"]
        gold_answer = item["answer"]
        question_type = item.get("question_type", "")
        question_reasoning = item.get("question_reasoning", "")

        # Extract context
        raw_context = extract_context(item)

        # Compress if needed
        compression_info = {}
        if is_compressed:
            try:
                comp_result = compress_text(raw_context, aggressiveness, bear_model)
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
        try:
            eval_result = judge_answer(question, gold_answer, model_answer)
        except RuntimeError as e:
            print(f"  Judge failed for question {qid}: {e}")
            eval_result = {"correct": None, "explanation": f"ERROR: {e}"}

        result = {
            "question_id": qid,
            "question": question,
            "question_type": question_type,
            "question_reasoning": question_reasoning,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "correct": eval_result["correct"],
            "judge_explanation": eval_result["explanation"],
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

    print(f"\n{'=' * 60}")
    print(f"  {config_name} — Summary")
    print(f"{'=' * 60}")
    print(f"  Total questions:  {total}")
    print(f"  Evaluated:        {len(evaluated)}")
    print(f"  Correct:          {correct}")
    print(f"  Accuracy:         {accuracy:.1%}")

    # Compression stats
    compressed_results = [r for r in results if "original_tokens" in r]
    if compressed_results:
        total_original = sum(r["original_tokens"] for r in compressed_results)
        total_compressed = sum(r["compressed_tokens"] for r in compressed_results)
        tokens_saved = total_original - total_compressed
        print(f"  Original tokens:  {total_original:,}")
        print(f"  Compressed tokens:{total_compressed:,}")
        if tokens_saved > 0:
            savings_pct = tokens_saved / total_original * 100
            print(f"  Tokens saved:     {tokens_saved:,} ({savings_pct:.1f}% reduction)")
        else:
            print(f"  Tokens saved:     0 (no effective compression at this aggressiveness)")

    # Breakdown by question_type
    types = set(r.get("question_type") for r in evaluated)
    types.discard("")
    types.discard(None)
    if types:
        print(f"\n  By question_type:")
        for qt in sorted(types):
            subset = [r for r in evaluated if r.get("question_type") == qt]
            qt_correct = sum(1 for r in subset if r["correct"])
            qt_acc = qt_correct / len(subset) if subset else 0
            print(f"    {qt}: {qt_correct}/{len(subset)} ({qt_acc:.1%})")

    # Breakdown by question_reasoning
    reasonings = set(r.get("question_reasoning") for r in evaluated)
    reasonings.discard("")
    reasonings.discard(None)
    if reasonings:
        print(f"\n  By question_reasoning:")
        for qr in sorted(reasonings):
            subset = [r for r in evaluated if r.get("question_reasoning") == qr]
            qr_correct = sum(1 for r in subset if r["correct"])
            qr_acc = qr_correct / len(subset) if subset else 0
            print(f"    {qr}: {qr_correct}/{len(subset)} ({qr_acc:.1%})")

    print()


def main():
    parser = argparse.ArgumentParser(description="FinanceBench benchmark with Bear compression")
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

    # CLI --limit overrides config.yaml limit
    limit = args.limit if args.limit is not None else config.NUM_QUESTIONS_LIMIT

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

    print("Loading FinanceBench dataset...")
    dataset = load_financebench()
    print(f"Loaded {len(dataset)} questions\n")

    all_results = {}
    for cfg_name in configs_to_run:
        print(f"Running config: {cfg_name}")
        results = run_single_config(cfg_name, dataset, limit=limit)
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
