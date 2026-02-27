import time
from openai import OpenAI
import config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge. Compare a model's answer to the gold (correct) answer for a financial question.

Rules:
- Numbers that are equivalent should be treated as correct (e.g., "3.5 billion" == "$3,500,000,000" == "$3.5B").
- Percentages: "15%" == "15 percent" == "0.15" (when context makes the meaning clear).
- Minor rounding differences are acceptable (e.g., "$3.47B" vs "$3.5B" is acceptable if within 2%).
- The model answer must convey the same key facts as the gold answer.
- Extra detail in the model answer is fine as long as the core answer is correct.
- If the gold answer is a specific number/fact, the model answer must include that number/fact.

Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief explanation."""


def judge_answer(question: str, gold_answer: str, model_answer: str) -> dict:
    """Use LLM-as-judge to evaluate if model_answer matches gold_answer.

    Returns dict with keys: correct (bool), explanation (str)
    """
    client = _get_client()
    user_msg = (
        f"Question: {question}\n\n"
        f"Gold Answer: {gold_answer}\n\n"
        f"Model Answer: {model_answer}"
    )

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],

                max_completion_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            correct = text.upper().startswith("CORRECT")
            return {"correct": correct, "explanation": text}
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Judge API error (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Judge API failed after {config.MAX_RETRIES} retries: {e}")
