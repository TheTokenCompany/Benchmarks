import time
import requests
import config


def compress_text(text: str, aggressiveness: float) -> dict:
    """Compress text using the Bear API.

    Returns dict with keys:
        compressed_text, original_tokens, compressed_tokens
    """
    payload = {
        "model": config.BEAR_MODEL,
        "input": text,
        "compression_settings": {"aggressiveness": aggressiveness},
    }
    headers = {
        "Authorization": f"Bearer {config.BEAR_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(config.MAX_RETRIES):
        try:
            resp = requests.post(
                config.BEAR_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            original = data["original_input_tokens"]
            compressed = data["output_tokens"]
            # API bug: output_tokens sometimes exceeds input; cap to original
            if compressed > original:
                compressed = original
            return {
                "compressed_text": data["output"],
                "original_tokens": original,
                "compressed_tokens": compressed,
            }
        except (requests.RequestException, KeyError) as e:
            if attempt < config.MAX_RETRIES - 1:
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Bear API error (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Bear API failed after {config.MAX_RETRIES} retries: {e}")
