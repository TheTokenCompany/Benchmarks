"""
Latency benchmark for chat completions across providers.

Measures time-to-first-token, time-to-first-byte, total latency, and streaming
throughput for:

  - OpenAI native API (gpt-5.4, gpt-5-mini)
  - Anthropic native API (claude-sonnet-4.6, claude-opus-4.6)
  - OpenRouter (all 4 models)
  - TTC Gateway, no compression (all 4 models)
  - TTC Gateway, aggressiveness=0.1 (all 4 models)
  - TTC Gateway, aggressiveness=0.5 (all 4 models)

Each combo is run with 3 input sizes (short ~200 tokens, medium ~2000 tokens,
long ~10000 tokens). Inputs are generated once and cached in inputs.json.

Run: uv run python benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from dotenv import load_dotenv

# Load .env from benchmarks root (one level up)
HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")

# ---------------------------- Config ----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TTC_API_KEY = os.getenv("TTC_API_KEY", "")
TTC_BASE_URL = os.getenv("TTC_BASE_URL", "https://api.thetokencompany.com:8443/v1")

INPUTS_FILE = HERE / "inputs.json"
RESULTS_FILE = HERE / "results.json"

# Models to test. Each entry maps a label to provider-specific model IDs.
MODELS: list[dict] = [
    {
        "label": "gpt-5.4",
        "vendor": "openai",
        "openai_id": "gpt-5.4",
        "openrouter_id": "openai/gpt-5.4",
        "ttc_id": "openai/gpt-5.4",
    },
    {
        "label": "gpt-5-mini",
        "vendor": "openai",
        "openai_id": "gpt-5-mini",
        "openrouter_id": "openai/gpt-5-mini",
        "ttc_id": "openai/gpt-5-mini",
    },
    {
        "label": "claude-sonnet-4.6",
        "vendor": "anthropic",
        "anthropic_id": "claude-sonnet-4-6",
        "openrouter_id": "anthropic/claude-sonnet-4.6",
        "ttc_id": "anthropic/claude-sonnet-4.6",
    },
    {
        "label": "claude-opus-4.6",
        "vendor": "anthropic",
        "anthropic_id": "claude-opus-4-6",
        "openrouter_id": "anthropic/claude-opus-4.6",
        "ttc_id": "anthropic/claude-opus-4.6",
    },
]

# Provider variants to test.
# Each has a "kind" (which http client it uses) and extra config.
PROVIDERS: list[dict] = [
    {"key": "native",     "kind": "native",     "label": "Native API"},
    {"key": "openrouter", "kind": "openrouter", "label": "OpenRouter"},
    {"key": "ttc_none",   "kind": "ttc",        "label": "TTC (no compression)",   "aggressiveness": 0.0},
    {"key": "ttc_01",     "kind": "ttc",        "label": "TTC (aggressiveness=0.1)", "aggressiveness": 0.1},
    {"key": "ttc_05",     "kind": "ttc",        "label": "TTC (aggressiveness=0.5)", "aggressiveness": 0.5},
]

# Input size targets (approximate input token counts).
INPUT_SIZES = {
    "short":  200,
    "medium": 2000,
    "long":   10000,
}

# How many times to run each combo (results are averaged).
DEFAULT_RUNS = 3

# Max response tokens. Needs to be large enough for OpenAI reasoning models
# (gpt-5*, o*) which use tokens for internal thinking before emitting content.
# The actual task asks for one sentence so non-reasoning models won't use this
# full budget.
MAX_OUTPUT_TOKENS = 2000

# Timeout per request.
REQUEST_TIMEOUT = 180.0


# ---------------------------- Metrics ----------------------------

@dataclass
class RunMetrics:
    success: bool
    error: Optional[str] = None
    status_code: Optional[int] = None
    time_to_first_byte_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    chunks_received: int = 0
    content_chars: int = 0
    chars_per_sec: Optional[float] = None


@dataclass
class ComboResult:
    model: str
    provider: str
    input_size: str
    input_tokens: int
    runs: list[RunMetrics] = field(default_factory=list)

    def summary(self) -> dict:
        successful = [r for r in self.runs if r.success]
        n = len(successful)
        def avg(field_name):
            vals = [getattr(r, field_name) for r in successful if getattr(r, field_name) is not None]
            return round(statistics.mean(vals), 1) if vals else None

        def median(field_name):
            vals = [getattr(r, field_name) for r in successful if getattr(r, field_name) is not None]
            return round(statistics.median(vals), 1) if vals else None

        return {
            "model": self.model,
            "provider": self.provider,
            "input_size": self.input_size,
            "input_tokens": self.input_tokens,
            "runs_ok": n,
            "runs_failed": len(self.runs) - n,
            "ttfb_ms_avg":  avg("time_to_first_byte_ms"),
            "ttft_ms_avg":  avg("time_to_first_token_ms"),
            "ttft_ms_med":  median("time_to_first_token_ms"),
            "total_ms_avg": avg("total_time_ms"),
            "chars_per_sec_avg": avg("chars_per_sec"),
            "errors": list({r.error for r in self.runs if r.error}),
        }


# ---------------------------- Providers ----------------------------

async def stream_openai_format(
    url: str,
    headers: dict,
    body: dict,
) -> RunMetrics:
    """Stream from an OpenAI-compatible /chat/completions endpoint (OpenAI,
    OpenRouter, TTC Gateway)."""
    metrics = RunMetrics(success=False)
    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                metrics.status_code = response.status_code
                if response.status_code != 200:
                    body_bytes = await response.aread()
                    metrics.error = f"HTTP {response.status_code}: {body_bytes.decode(errors='replace')[:300]}"
                    return metrics

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    now = time.perf_counter()
                    if metrics.time_to_first_byte_ms is None:
                        metrics.time_to_first_byte_ms = (now - t_start) * 1000

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
                        if metrics.time_to_first_token_ms is None:
                            metrics.time_to_first_token_ms = (now - t_start) * 1000
                        metrics.chunks_received += 1
                        metrics.content_chars += len(content)

        metrics.total_time_ms = (time.perf_counter() - t_start) * 1000
        if metrics.content_chars > 0 and metrics.time_to_first_token_ms is not None:
            stream_secs = (metrics.total_time_ms - metrics.time_to_first_token_ms) / 1000
            if stream_secs > 0:
                metrics.chars_per_sec = metrics.content_chars / stream_secs
        metrics.success = metrics.time_to_first_token_ms is not None
        if not metrics.success and metrics.error is None:
            metrics.error = "no content in stream"
    except Exception as e:
        metrics.error = f"{type(e).__name__}: {e}"
        metrics.total_time_ms = (time.perf_counter() - t_start) * 1000
    return metrics


async def stream_anthropic_native(model: str, messages: list[dict]) -> RunMetrics:
    """Stream from Anthropic's native /v1/messages endpoint (different format)."""
    metrics = RunMetrics(success=False)
    if not ANTHROPIC_API_KEY:
        metrics.error = "ANTHROPIC_API_KEY not set"
        return metrics

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }

    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                metrics.status_code = response.status_code
                if response.status_code != 200:
                    body_bytes = await response.aread()
                    metrics.error = f"HTTP {response.status_code}: {body_bytes.decode(errors='replace')[:300]}"
                    return metrics

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    now = time.perf_counter()
                    if metrics.time_to_first_byte_ms is None:
                        metrics.time_to_first_byte_ms = (now - t_start) * 1000

                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type")
                    if event_type == "content_block_delta":
                        delta = data.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text = delta.get("text") or ""
                            if text:
                                if metrics.time_to_first_token_ms is None:
                                    metrics.time_to_first_token_ms = (now - t_start) * 1000
                                metrics.chunks_received += 1
                                metrics.content_chars += len(text)
                    elif event_type == "message_stop":
                        break

        metrics.total_time_ms = (time.perf_counter() - t_start) * 1000
        if metrics.content_chars > 0 and metrics.time_to_first_token_ms is not None:
            stream_secs = (metrics.total_time_ms - metrics.time_to_first_token_ms) / 1000
            if stream_secs > 0:
                metrics.chars_per_sec = metrics.content_chars / stream_secs
        metrics.success = metrics.time_to_first_token_ms is not None
        if not metrics.success and metrics.error is None:
            metrics.error = "no content in stream"
    except Exception as e:
        metrics.error = f"{type(e).__name__}: {e}"
        metrics.total_time_ms = (time.perf_counter() - t_start) * 1000
    return metrics


def _tokens_param(vendor: str) -> str:
    """Newer OpenAI models (gpt-5*, o1*, o3*) require max_completion_tokens."""
    return "max_completion_tokens" if vendor == "openai" else "max_tokens"


def _openai_extra(model_id: str) -> dict:
    """Extra params for OpenAI reasoning models to minimize thinking time,
    so the benchmark measures LLM latency not reasoning latency.

    gpt-5 / gpt-5-mini / gpt-5-nano: "minimal" (fastest)
    gpt-5.1+ / o*: "none" (fastest, "minimal" not supported)
    """
    if model_id.startswith(("gpt-5-mini", "gpt-5-nano")) or model_id == "gpt-5":
        return {"reasoning_effort": "minimal"}
    if model_id.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"reasoning_effort": "none"}
    return {}


async def run_one(provider: dict, model: dict, messages: list[dict]) -> RunMetrics:
    """Run a single test against one provider+model combo."""
    kind = provider["kind"]
    vendor = model["vendor"]
    tok_key = _tokens_param(vendor)

    if kind == "native":
        if vendor == "openai":
            if not OPENAI_API_KEY:
                return RunMetrics(success=False, error="OPENAI_API_KEY not set")
            body = {
                "model": model["openai_id"],
                "messages": messages,
                tok_key: MAX_OUTPUT_TOKENS,
                "stream": True,
                **_openai_extra(model["openai_id"]),
            }
            return await stream_openai_format(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                body=body,
            )
        elif vendor == "anthropic":
            return await stream_anthropic_native(model["anthropic_id"], messages)

    elif kind == "openrouter":
        if not OPENROUTER_API_KEY:
            return RunMetrics(success=False, error="OPENROUTER_API_KEY not set")
        body = {
            "model": model["openrouter_id"],
            "messages": messages,
            tok_key: MAX_OUTPUT_TOKENS,
            "stream": True,
        }
        if vendor == "openai":
            body.update(_openai_extra(model["openai_id"]))
        return await stream_openai_format(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://thetokencompany.com",
                "X-Title": "TTC latency benchmark",
            },
            body=body,
        )

    elif kind == "ttc":
        if not TTC_API_KEY:
            return RunMetrics(success=False, error="TTC_API_KEY not set")
        agg = provider.get("aggressiveness", 0.0)
        model_field = model["ttc_id"]
        if agg > 0.0:
            model_field = f"{model_field}?aggressiveness={agg}"
        body = {
            "model": model_field,
            "messages": messages,
            tok_key: MAX_OUTPUT_TOKENS,
            "stream": True,
        }
        if vendor == "openai":
            body.update(_openai_extra(model["openai_id"]))
        return await stream_openai_format(
            url=f"{TTC_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {TTC_API_KEY}",
                "Content-Type": "application/json",
            },
            body=body,
        )

    return RunMetrics(success=False, error=f"unknown provider kind: {kind}")


# ---------------------------- Inputs ----------------------------

INPUT_PROMPTS = {
    "short":  "Write a very short factual paragraph (around 100 words) about the history of llamas. Plain prose, no formatting.",
    "medium": "Write a detailed article of around 1500 words explaining the history and design of LZ77, LZ78, Huffman coding, and modern data compression. Plain prose, no headings, no bullet points, no formatting.",
    "long":   "Write a comprehensive essay of around 7500 words about the evolution of large language models from the transformer architecture in 2017 through to the current state of the art. Cover the key architectural innovations, training techniques, scaling laws, alignment methods, and deployment considerations. Plain prose, no headings, no bullet points, no formatting.",
}


async def generate_inputs() -> dict[str, str]:
    """Generate short/medium/long input texts using native OpenAI + gpt-5-mini
    (streamed, to avoid timeouts on long generations). Caches results in inputs.json."""
    if INPUTS_FILE.exists():
        with INPUTS_FILE.open() as f:
            cached = json.load(f)
        if set(cached.keys()) == set(INPUT_PROMPTS.keys()):
            print(f"[inputs] Loaded from cache: {INPUTS_FILE}")
            for k, v in cached.items():
                print(f"  {k}: {len(v)} chars")
            return cached

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required to generate inputs")

    print("[inputs] Generating via OpenAI native (gpt-5-mini, streamed)...")
    inputs: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        for size, prompt in INPUT_PROMPTS.items():
            print(f"  Generating {size}...", flush=True)
            body = {
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 16000,
                "stream": True,
            }
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            content_parts: list[str] = []
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json=body,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    body_bytes = await response.aread()
                    raise RuntimeError(
                        f"Input generation failed ({size}): HTTP {response.status_code}: "
                        f"{body_bytes.decode(errors='replace')[:500]}"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
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
                    text = delta.get("content") or ""
                    if text:
                        content_parts.append(text)

            content = "".join(content_parts)
            if not content:
                raise RuntimeError(f"Input generation for {size} returned empty content")
            inputs[size] = content
            print(f"    → {len(content)} chars", flush=True)

    INPUTS_FILE.write_text(json.dumps(inputs, indent=2))
    print(f"[inputs] Cached to {INPUTS_FILE}")
    return inputs


def build_messages(input_text: str) -> list[dict]:
    """Build a standard chat messages list with the given input content."""
    return [
        {
            "role": "user",
            "content": (
                "Summarize the following text in one short sentence (max 20 words). "
                "Reply with just the sentence, nothing else.\n\n"
                f"{input_text}"
            ),
        }
    ]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


# ---------------------------- Runner ----------------------------

async def main(runs: int, only_provider: Optional[str], only_model: Optional[str], only_size: Optional[str]):
    inputs = await generate_inputs()

    # Build the test matrix
    combos: list[ComboResult] = []
    for model in MODELS:
        if only_model and model["label"] != only_model:
            continue
        for provider in PROVIDERS:
            if only_provider and provider["key"] != only_provider:
                continue
            for size, target_tokens in INPUT_SIZES.items():
                if only_size and size != only_size:
                    continue
                combos.append(ComboResult(
                    model=model["label"],
                    provider=provider["key"],
                    input_size=size,
                    input_tokens=estimate_tokens(inputs[size]),
                ))

    print()
    print("=" * 80)
    print(f"Running {len(combos)} combos × {runs} runs = {len(combos) * runs} requests")
    print("=" * 80)

    # Run each combo sequentially (to avoid network contention affecting timing)
    model_by_label = {m["label"]: m for m in MODELS}
    provider_by_key = {p["key"]: p for p in PROVIDERS}

    import asyncio as _asyncio

    for i, combo in enumerate(combos, 1):
        model = model_by_label[combo.model]
        provider = provider_by_key[combo.provider]
        messages = build_messages(inputs[combo.input_size])
        print(f"\n[{i}/{len(combos)}] {combo.model} | {provider['label']} | {combo.input_size} ({combo.input_tokens} tokens)", flush=True)

        # Run the N runs in parallel (they're independent) to speed up the benchmark
        results = await _asyncio.gather(
            *[run_one(provider, model, messages) for _ in range(runs)],
            return_exceptions=False,
        )
        for r, m in enumerate(results):
            combo.runs.append(m)
            if m.success:
                print(f"  run {r+1}: ttft={m.time_to_first_token_ms:.0f}ms total={m.total_time_ms:.0f}ms chars={m.content_chars}", flush=True)
            else:
                print(f"  run {r+1}: FAIL — {m.error}", flush=True)

    # Save full results + summary
    raw = [
        {
            "model": c.model,
            "provider": c.provider,
            "input_size": c.input_size,
            "input_tokens": c.input_tokens,
            "runs": [asdict(r) for r in c.runs],
        }
        for c in combos
    ]
    summaries = [c.summary() for c in combos]
    RESULTS_FILE.write_text(json.dumps({
        "summaries": summaries,
        "raw": raw,
    }, indent=2))

    # Print pretty summary table
    print()
    print("=" * 120)
    print("SUMMARY (averaged over runs)")
    print("=" * 120)
    header = f"{'model':<20} {'provider':<14} {'size':<7} {'in_tok':>7} {'ttft_ms':>10} {'ttft_med':>10} {'total_ms':>10} {'chars/s':>10} {'ok':>4}"
    print(header)
    print("-" * 120)
    for s in summaries:
        def fmt(val):
            return f"{val:.0f}" if isinstance(val, (int, float)) else "-"
        print(
            f"{s['model']:<20} {s['provider']:<14} {s['input_size']:<7} {s['input_tokens']:>7} "
            f"{fmt(s['ttft_ms_avg']):>10} {fmt(s['ttft_ms_med']):>10} {fmt(s['total_ms_avg']):>10} "
            f"{fmt(s['chars_per_sec_avg']):>10} {s['runs_ok']:>2}/{s['runs_ok']+s['runs_failed']:<2}"
        )
    print()
    print(f"Full results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Runs per combo (default {DEFAULT_RUNS})")
    parser.add_argument("--provider", type=str, default=None, help="Run only this provider key (e.g. ttc_01)")
    parser.add_argument("--model", type=str, default=None, help="Run only this model label (e.g. gpt-5-mini)")
    parser.add_argument("--size", type=str, default=None, choices=list(INPUT_SIZES.keys()), help="Run only this input size")
    args = parser.parse_args()

    asyncio.run(main(args.runs, args.provider, args.model, args.size))
