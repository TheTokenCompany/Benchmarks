#!/usr/bin/env python3
"""CoQA benchmark runner with Bear compression."""

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


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return len(text) // 4


def load_coqa():
    """Load the CoQA validation set."""
    ds = load_dataset(config.DATASET_NAME, split="validation")
    return ds


def flatten_conversations(dataset) -> list[dict]:
    """Flatten conversations into individual question items.

    Each conversation has a story and multiple Q&A turns. We expand each turn
    into a separate item that includes the full prior conversation history.

    Returns list of dicts with keys:
        story_id, source, story, question, gold_answer, turn_number,
        prior_turns (list of (question, answer) tuples)
    """
    items = []
    conversations = list(dataset)

    for conv_idx, conv in enumerate(conversations):
        story = conv["story"].strip()
        source = conv.get("source", "unknown")
        questions = conv["questions"]
        answers = conv["answers"]["input_text"]
        num_turns = len(questions)

        for turn_idx in range(num_turns):
            prior_turns = [
                (questions[t], answers[t]) for t in range(turn_idx)
            ]
            items.append({
                "story_id": str(conv_idx),
                "question_id": f"{conv_idx}_{turn_idx}",
                "source": source,
                "story": story,
                "question": questions[turn_idx],
                "gold_answer": answers[turn_idx],
                "turn_number": turn_idx + 1,
                "prior_turns": prior_turns,
            })

    return items


def build_prompt(story: str, prior_turns: list[tuple], question: str) -> list[dict]:
    """Build the chat messages for the LLM.

    Includes the story and any prior conversation turns for context.
    """
    parts = [f"Story:\n{story}"]

    if prior_turns:
        history = []
        for i, (q, a) in enumerate(prior_turns, 1):
            history.append(f"Q{i}: {q}")
            history.append(f"A{i}: {a}")
        parts.append("Conversation so far:\n" + "\n".join(history))

    parts.append(f"Current question: {question}")
    user_content = "\n\n".join(parts)

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


def run_single_config(config_name: str, items: list[dict]):
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

    remaining = [item for item in items if item["question_id"] not in completed_ids]

    if not remaining:
        print(f"  [{config_name}] All {len(items)} questions already completed. Skipping.")
        return results

    print(f"  [{config_name}] {len(results)} done, {len(remaining)} remaining")

    # Cache compressed stories to avoid re-compressing the same story for each turn
    story_cache = {}

    for item in tqdm(remaining, desc=config_name, unit="q"):
        story = item["story"]

        # Compress story if needed (cached per story_id)
        compression_info = {}
        if is_compressed:
            story_id = item["story_id"]
            if story_id in story_cache:
                story_for_llm = story_cache[story_id]["compressed_text"]
                compression_info = {
                    "original_tokens": story_cache[story_id]["original_tokens"],
                    "compressed_tokens": story_cache[story_id]["compressed_tokens"],
                    "compression_ratio": story_cache[story_id]["compression_ratio"],
                }
            else:
                try:
                    comp_result = compress_text(story, aggressiveness, bear_model)
                    story_for_llm = comp_result["compressed_text"]
                    ratio = (
                        comp_result["compressed_tokens"] / comp_result["original_tokens"]
                        if comp_result["original_tokens"] > 0
                        else 1.0
                    )
                    compression_info = {
                        "original_tokens": comp_result["original_tokens"],
                        "compressed_tokens": comp_result["compressed_tokens"],
                        "compression_ratio": ratio,
                    }
                    story_cache[story_id] = {
                        "compressed_text": story_for_llm,
                        "original_tokens": comp_result["original_tokens"],
                        "compressed_tokens": comp_result["compressed_tokens"],
                        "compression_ratio": ratio,
                    }
                except RuntimeError as e:
                    print(f"  Compression failed for {item['question_id']}: {e}")
                    story_for_llm = story
                    compression_info = {"error": str(e)}
        else:
            story_for_llm = story

        # Query LLM
        messages = build_prompt(story_for_llm, item["prior_turns"], item["question"])
        try:
            model_answer = query_llm(messages)
        except RuntimeError as e:
            print(f"  LLM failed for {item['question_id']}: {e}")
            model_answer = f"ERROR: {e}"

        # Evaluate
        try:
            eval_result = judge_answer(item["question"], item["gold_answer"], model_answer)
        except RuntimeError as e:
            print(f"  Judge failed for {item['question_id']}: {e}")
            eval_result = {"correct": None, "explanation": f"ERROR: {e}"}

        result = {
            "question_id": item["question_id"],
            "story_id": item["story_id"],
            "source": item["source"],
            "turn_number": item["turn_number"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
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

    # Breakdown by source domain
    sources = sorted(set(r.get("source", "unknown") for r in evaluated))
    if len(sources) > 1:
        print(f"\n  By source domain:")
        for src in sources:
            subset = [r for r in evaluated if r.get("source") == src]
            s_correct = sum(1 for r in subset if r["correct"])
            s_acc = s_correct / len(subset) if subset else 0
            print(f"    {src}: {s_correct}/{len(subset)} ({s_acc:.1%})")

    # Breakdown by turn number
    turn_numbers = sorted(set(r.get("turn_number", 0) for r in evaluated))
    if len(turn_numbers) > 1:
        print(f"\n  By turn number:")
        for tn in turn_numbers:
            subset = [r for r in evaluated if r.get("turn_number") == tn]
            t_correct = sum(1 for r in subset if r["correct"])
            t_acc = t_correct / len(subset) if subset else 0
            print(f"    turn {tn}: {t_correct}/{len(subset)} ({t_acc:.1%})")

    print()


def main():
    parser = argparse.ArgumentParser(description="CoQA benchmark with Bear compression")
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
        help="Limit number of questions (for testing)",
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

    print("Loading CoQA dataset (validation split)...")
    dataset = load_coqa()
    print(f"Loaded {len(dataset)} conversations")

    print("Flattening conversations into individual questions...")
    items = flatten_conversations(dataset)
    total_questions = len(items)
    if args.limit is not None:
        items = items[:args.limit]
    print(f"Expanded to {total_questions} questions total, using {len(items)}")

    # Filter out questions whose uncompressed prompt exceeds the model's input token limit
    filtered_items = []
    for item in items:
        messages = build_prompt(item["story"], item["prior_turns"], item["question"])
        prompt_text = messages[0]["content"] + "\n" + messages[1]["content"]
        if estimate_tokens(prompt_text) <= config.MAX_INPUT_TOKENS:
            filtered_items.append(item)
    discarded = total_questions - len(filtered_items)
    print(f"Filtered: {len(filtered_items)} questions within {config.MAX_INPUT_TOKENS:,}-token limit, "
          f"{discarded} discarded ({discarded}/{total_questions})\n")

    all_results = {}
    for cfg_name in configs_to_run:
        print(f"Running config: {cfg_name}")
        results = run_single_config(cfg_name, filtered_items)
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
