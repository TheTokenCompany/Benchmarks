# FinanceBench

150 financial Q&A tasks from [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench).

## Dataset

Each question is paired with oracle context (gold evidence pages from SEC filings). The model answers free-form financial questions based on this context.

## Evaluation

Uses **LLM-as-judge** (gpt-5-mini) to compare the model's answer against the verified gold answer. The judge handles equivalent number formats ("3.5 billion" == "$3.5B"), percentage notation, and minor rounding differences (within 2%).

## Summary Breakdowns

- By `question_type` (e.g. numerical extraction, comparison)
- By `question_reasoning` (e.g. single-step, multi-step)
