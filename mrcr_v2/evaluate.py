from difflib import SequenceMatcher


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Score a response against the gold answer using SequenceMatcher.

    Follows the official MRCR evaluation protocol:
    1. If the response doesn't start with the required hash, score is 0.
    2. Otherwise, strip the hash from both and compute similarity ratio.

    Returns a float between 0.0 and 1.0.
    """
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def evaluate_answer(gold_answer: str, model_response: str, random_string: str) -> dict:
    """Evaluate an MRCR response.

    Args:
        gold_answer: The gold response (with hash prefix).
        model_response: The full model response text.
        random_string: The 10-char hash that must prefix the response.

    Returns:
        dict with keys:
            score (float): Similarity score 0.0–1.0.
            hash_present (bool): Whether the response started with the required hash.
            explanation (str): Human-readable explanation.
    """
    hash_present = model_response.startswith(random_string)
    score = grade(model_response, gold_answer, random_string)

    if not hash_present:
        explanation = f"Hash '{random_string}' missing from response start -> score 0.0"
    else:
        explanation = f"Hash present. Similarity: {score:.3f}"

    return {
        "score": score,
        "hash_present": hash_present,
        "explanation": explanation,
    }
