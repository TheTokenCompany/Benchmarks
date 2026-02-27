import re


# Ordered from most specific to least specific
_PATTERNS = [
    re.compile(r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*:?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"[Aa]nswer\s*:\s*\(?([A-D])\)?"),
    re.compile(r"\b([A-D])\)\s+is\s+(?:the\s+)?correct", re.IGNORECASE),
    re.compile(r"^\(?([A-D])\)?$", re.MULTILINE),
    re.compile(r"^\(?([A-D])\)?\s*[\.:]", re.MULTILINE),
    re.compile(r"\b([A-D])\b"),
]


def extract_answer(model_response: str) -> str | None:
    """Extract a single answer letter (A/B/C/D) from a model response.

    Tries patterns from most specific to least specific.
    Returns the letter (uppercase) or None if no answer found.
    """
    text = model_response.strip()
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def evaluate_answer(gold_answer: str, model_response: str) -> dict:
    """Evaluate a multiple-choice answer by extracting the letter and comparing.

    Args:
        gold_answer: The correct letter (A, B, C, or D).
        model_response: The full model response text.

    Returns:
        dict with keys:
            correct (bool | None): True if correct, False if wrong, None if extraction failed.
            extracted_answer (str | None): The extracted letter, or None.
            explanation (str): Human-readable explanation of the evaluation.
    """
    extracted = extract_answer(model_response)

    if extracted is None:
        return {
            "correct": None,
            "extracted_answer": None,
            "explanation": f"Could not extract answer letter from response: {model_response[:200]}",
        }

    gold = gold_answer.strip().upper()
    is_correct = extracted == gold

    return {
        "correct": is_correct,
        "extracted_answer": extracted,
        "explanation": f"Extracted '{extracted}', gold is '{gold}' -> {'CORRECT' if is_correct else 'INCORRECT'}",
    }
