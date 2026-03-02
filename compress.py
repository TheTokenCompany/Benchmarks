"""Bear compression API wrapper (shared across all benchmarks)."""

import os
import time

import requests
import yaml
from dotenv import load_dotenv

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

# Load retry settings and API URL from config.yaml
with open(os.path.join(_ROOT_DIR, "config.yaml"), "r") as _f:
    _cfg = yaml.safe_load(_f)["shared"]

_BEAR_API_URL = _cfg["bear_api_url"]
_MAX_RETRIES = _cfg["max_retries"]
_RETRY_BASE_DELAY = _cfg["retry_base_delay"]
_BEAR_API_KEY = os.getenv("BEAR_API_KEY")


def compress_text(text: str, aggressiveness: float, model: str) -> dict:
    """Compress text using the Bear API.

    Args:
        text: The text to compress.
        aggressiveness: Compression aggressiveness (0.0–1.0).
        model: Bear model name (e.g. "bear-1.2").

    Returns dict with keys:
        compressed_text, original_tokens, compressed_tokens
    """
    payload = {
        "model": model,
        "input": text,
        "compression_settings": {"aggressiveness": aggressiveness},
    }
    headers = {
        "Authorization": f"Bearer {_BEAR_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(
                _BEAR_API_URL,
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
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Bear API error (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Bear API failed after {_MAX_RETRIES} retries: {e}")
