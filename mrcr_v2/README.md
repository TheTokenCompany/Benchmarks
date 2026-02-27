# MRCR v2 (8-needle)

~800 multi-round coreference resolution tasks from the 8-needle variant of [MRCR](https://huggingface.co/datasets/openai/mrcr).

## Dataset

A long multi-turn conversation contains 8 identical creative writing requests (e.g. "write a poem about tapirs"). The model must return the Kth specific instance, prefixed with a required 10-character hash string. Context lengths range from ~4k to ~1M tokens across 8 bins.

## Evaluation

Uses **`difflib.SequenceMatcher`** similarity ratio (0.0–1.0), following the official MRCR scoring protocol. Score is automatically 0.0 if the required hash prefix is missing from the response.

## Summary Breakdowns

- By context-length bin (8 bins from 4k to 1M tokens)
- By `desired_msg_index` (which of the 8 identical instances was requested)
