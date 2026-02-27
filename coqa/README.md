# CoQA

~7,500 conversational reading comprehension questions (validation split, 500 conversations) from [CoQA](https://huggingface.co/datasets/stanfordnlp/coqa).

## Dataset

Each example is a **conversation** — a story (passage) paired with ~15 sequential Q&A turns. Questions are conversational and context-dependent: later questions reference prior answers (e.g., Q3: "for what subjects?" follows Q2: "what is the library for?"). Stories span five domains: Wikipedia, CNN, Gutenberg, RACE, and MCTest.

## Compression Strategy

Bear compression is applied only to the **story** (source passage). The conversation history (prior Q&A turns) is always included uncompressed, since it represents the model's own prior answers rather than input context. This isolates the effect of compressing the source document on conversational comprehension.

## Evaluation

Uses **LLM-as-judge** (gpt-5-mini) to compare the model's answer against the gold answer. The judge handles synonym equivalency, phrasing differences, yes/no matching, and short extractive answers.

## Summary Breakdowns

- By `source` domain (wikipedia, cnn, gutenberg, race, mctest)
- By `turn_number` (early vs. late turns in the conversation)
