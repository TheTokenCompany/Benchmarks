"""
Long-running latency benchmark for TTC Gateway.

Runs every 15 minutes via cron, testing gpt-5-mini and claude-sonnet-4.6
through the TTC gateway with compression=none. Appends results to a
JSONL log file for later analysis.

Best practices applied:
  - Deterministic short prompt with fixed expected output length
  - Streaming to detect connection/buffering issues
  - Fresh connection per request (no pooling) to measure real cold latency
  - Tracks TTFB, TTFT, total latency, chars/s, error type
  - Logs timestamp + all metrics as one JSON line per request
  - Rotates through input sizes to reduce cache effects

Usage (called by cron every 15 min):
  python3.11 longrun.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# Load .env from home dir (on EC2)
from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")

TTC_API_KEY = os.getenv("TTC_API_KEY", "")
TTC_BASE_URL = os.getenv("TTC_BASE_URL", "")

LOG_FILE = Path.home() / "longrun-results.jsonl"
STATE_FILE = Path.home() / "longrun-state.json"

# Models to test each cycle
MODELS = [
    {"label": "gpt-5-mini",       "ttc_id": "openai/gpt-5-mini",          "vendor": "openai"},
    {"label": "claude-sonnet-4.6", "ttc_id": "anthropic/claude-sonnet-4.6", "vendor": "anthropic"},
    {"label": "gemini-2.5-flash",  "ttc_id": "google/gemini-2.5-flash",    "vendor": "google"},
]

# Compression levels to cycle through
COMPRESSIONS = ["none", "low", "high"]

# Short deterministic prompt — asks for a fixed-format response
PROMPT = (
    "Respond with exactly this format and nothing else:\n"
    "STATUS: OK\n"
    "DATE: today's date in YYYY-MM-DD format\n"
    "WORD: a single random English word\n"
)

REQUEST_TIMEOUT = 120.0
MAX_OUTPUT_TOKENS = 2000


@dataclass
class RunResult:
    timestamp: str
    model: str
    compression: str
    success: bool
    error: Optional[str] = None
    status_code: Optional[int] = None
    time_to_first_byte_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    chunks_received: int = 0
    content_chars: int = 0
    chars_per_sec: Optional[float] = None
    response_preview: Optional[str] = None



def run_one(model: dict, compression: str) -> RunResult:
    """Synchronous streaming request to TTC gateway."""
    ts = datetime.now(timezone.utc).isoformat()
    result = RunResult(timestamp=ts, model=model["label"], compression=compression, success=False)

    if not TTC_API_KEY:
        result.error = "TTC_API_KEY not set"
        return result

    model_field = f"{model['ttc_id']}?compression={compression}"

    tok_key = "max_completion_tokens" if model["vendor"] == "openai" else "max_tokens"
    body = {
        "model": model_field,
        "messages": [{"role": "user", "content": PROMPT + f" [nonce:{uuid.uuid4().hex[:8]}]"}],
        tok_key: MAX_OUTPUT_TOKENS,
        "stream": True,
    }
    if model["vendor"] == "openai":
        body["reasoning_effort"] = "minimal"

    headers = {
        "Authorization": f"Bearer {TTC_API_KEY}",
        "Content-Type": "application/json",
    }

    t_start = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            with client.stream("POST", f"{TTC_BASE_URL}/chat/completions",
                               json=body, headers=headers) as response:
                result.status_code = response.status_code
                if response.status_code != 200:
                    body_bytes = response.read()
                    result.error = f"HTTP {response.status_code}: {body_bytes.decode(errors='replace')[:300]}"
                    result.total_time_ms = (time.perf_counter() - t_start) * 1000
                    return result

                content_parts = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    now = time.perf_counter()
                    if result.time_to_first_byte_ms is None:
                        result.time_to_first_byte_ms = (now - t_start) * 1000

                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        if result.time_to_first_token_ms is None:
                            result.time_to_first_token_ms = (now - t_start) * 1000
                        result.chunks_received += 1
                        result.content_chars += len(content)
                        content_parts.append(content)

        result.total_time_ms = (time.perf_counter() - t_start) * 1000
        if result.content_chars > 0 and result.time_to_first_token_ms is not None:
            stream_secs = (result.total_time_ms - result.time_to_first_token_ms) / 1000
            if stream_secs > 0:
                result.chars_per_sec = round(result.content_chars / stream_secs, 1)
        result.success = result.time_to_first_token_ms is not None
        result.response_preview = "".join(content_parts)[:100]
        if not result.success and result.error is None:
            result.error = "no content in stream"
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.total_time_ms = (time.perf_counter() - t_start) * 1000

    return result


def get_cycle_index() -> int:
    """Get and increment the cycle counter to rotate compression levels."""
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    idx = state.get("cycle", 0)
    state["cycle"] = idx + 1
    STATE_FILE.write_text(json.dumps(state))
    return idx


def main():
    cycle = get_cycle_index()
    compression = COMPRESSIONS[cycle % len(COMPRESSIONS)]

    print(f"[{datetime.now(timezone.utc).isoformat()}] Cycle {cycle}, compression={compression}")

    for model in MODELS:
        result = run_one(model, compression)

        # Append to JSONL log
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        if result.success:
            print(f"  {model['label']}: ttft={result.time_to_first_token_ms:.0f}ms "
                  f"total={result.total_time_ms:.0f}ms chars={result.content_chars}")
        else:
            print(f"  {model['label']}: FAIL — {result.error}")

    print()


if __name__ == "__main__":
    main()
