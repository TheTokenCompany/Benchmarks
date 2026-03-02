# LongBench V2

503 long-context multiple-choice tasks from [LongBench V2](https://huggingface.co/datasets/zai-org/LongBench-v2).

## Dataset

Each question includes a long context (8k to 2M words) and 4 answer choices (A/B/C/D). Covers 6 domains: single-doc QA, multi-doc QA, long in-context learning, long dialogue, code repository, and structured data understanding.

## Evaluation

Uses **regex-based letter extraction** to parse the model's answer choice (A/B/C/D) and compare against the gold answer. No LLM judge required.

## Summary Breakdowns

- By `domain` (6 task categories)
- By `difficulty` (easy / hard)
- By `length` (short / medium / long)
