# SQuAD 2.0

11,900 reading comprehension questions (validation split) from [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2).

## Dataset

Each question is paired with a Wikipedia paragraph as context. SQuAD 2.0 combines ~100k answerable questions (extractive — the answer is a span from the context) with ~50k unanswerable questions written adversarially to look plausible. The model must either extract the correct answer or identify that the question cannot be answered from the context.

## Evaluation

Uses **LLM-as-judge** (gpt-5-mini) to compare the model's answer against the gold answer. The judge handles synonym equivalency, phrasing differences, and extra detail. For unanswerable questions, the model must indicate the question cannot be answered (e.g. "unanswerable", "cannot be determined").

## Summary Breakdowns

- By `answerability` (answerable vs. unanswerable)
- By `title` (Wikipedia article — shown when 50 or fewer distinct articles)
