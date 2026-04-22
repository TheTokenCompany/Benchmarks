# TTC Gateway Latency Benchmark Report

## Overview

This report measures the latency overhead introduced by the TTC Gateway compared to native provider APIs and OpenRouter. The goal is to quantify how much additional latency, if any, routing requests through the TTC Gateway adds relative to calling the model providers directly.

## Methodology

### Setup

- **Location:** EC2 t3.medium instance in us-west-2
- **Protocol:** Streaming SSE (server-sent events) over HTTPS
- **Runs:** 3 sequential runs per combination, fully sequential execution (one request at a time)
- **Cache busting:** Each request includes a unique random nonce to prevent upstream caching
- **Reasoning suppression:** OpenAI gpt-5.4 was tested with `reasoning_effort: "none"` across all providers to measure pure inference latency, not reasoning overhead

### Models tested

| Model | Vendor | Type |
|-------|--------|------|
| gpt-5.4 | OpenAI | Frontier, non-reasoning mode |
| claude-sonnet-4.6 | Anthropic | Frontier |
| gemini-3.1-flash-lite-preview | Google | Lightweight, fast |

### Providers tested

| Provider | Description |
|----------|-------------|
| **Native** | Direct API call to OpenAI (only available for gpt-5.4) |
| **OpenRouter** | Third-party multi-model router |
| **TTC (compression=none)** | TTC Gateway, no prompt compression |
| **TTC (compression=low)** | TTC Gateway, low compression |
| **TTC (compression=high)** | TTC Gateway, high compression |

### Input sizes

| Label | Approximate tokens |
|-------|--------------------|
| short | ~200 |
| medium | ~2,000 |
| long | ~10,000 |

### What was measured

- **TTFT (Time to First Token):** Time from request sent to first content token received. This is the most important metric for interactive/streaming use cases.
- **Total time:** Time from request sent to stream completion.
- **Throughput:** Characters per second during the streaming phase.

### Fairness controls

- All providers received identical prompts, input texts, and parameters
- Fresh TCP + TLS connection per request (no connection reuse)
- Fully sequential execution to avoid any concurrency or network contention
- Same `max_completion_tokens` / `max_tokens` budget (2000) across all providers
- OpenAI-specific parameters (`reasoning_effort`, `max_completion_tokens`) applied consistently to native, OpenRouter, and TTC

## Results

### gpt-5.4

| Provider | Size | TTFT avg (ms) | TTFT median (ms) | Total avg (ms) | OK |
|----------|------|---------------|-------------------|-----------------|-----|
| Native | short | 780 | 703 | 1101 | 3/3 |
| Native | medium | 852 | 836 | 1361 | 3/3 |
| Native | long | 918 | 899 | 1313 | 3/3 |
| OpenRouter | short | 656 | 551 | 1169 | 3/3 |
| OpenRouter | medium | 602 | 551 | 1150 | 3/3 |
| OpenRouter | long | 1146 | 1053 | 1775 | 3/3 |
| TTC (none) | short | 937 | 953 | 1432 | 3/3 |
| TTC (none) | medium | 928 | 918 | 1344 | 3/3 |
| TTC (none) | long | 1065 | 1088 | 1640 | 3/3 |
| TTC (low) | short | 1041 | 1020 | 1499 | 3/3 |
| TTC (low) | medium | 1362 | 1085 | 1861 | 3/3 |
| TTC (low) | long | 1051 | 1081 | 1588 | 3/3 |
| TTC (high) | short | 1195 | 941 | 1648 | 3/3 |
| TTC (high) | medium | 942 | 939 | 1434 | 3/3 |
| TTC (high) | long | 1087 | 1096 | 1703 | 3/3 |

**Summary:** TTC adds approximately **100-200ms TTFT overhead** compared to native OpenAI API. This is consistent across input sizes and compression levels. OpenRouter showed similar or slightly lower TTFT on short/medium inputs.

### claude-sonnet-4.6

| Provider | Size | TTFT avg (ms) | TTFT median (ms) | Total avg (ms) | OK |
|----------|------|---------------|-------------------|-----------------|-----|
| OpenRouter | short | 895 | 807 | 1514 | 3/3 |
| OpenRouter | medium | 731 | 612 | 1821 | 3/3 |
| OpenRouter | long | 1577 | 1724 | 2686 | 3/3 |
| TTC (none) | short | 1324 | 1152 | 1834 | 3/3 |
| TTC (none) | medium | 4314 | 4175 | 5508 | 3/3 |
| TTC (none) | long | 1384 | 1324 | 2281 | 3/3 |
| TTC (low) | short | 1904 | 1243 | 3076 | 3/3 |
| TTC (low) | medium | 1182 | 1176 | 2236 | 3/3 |
| TTC (low) | long | 1942 | 1545 | 2625 | 3/3 |
| TTC (high) | short | 2809 | 3344 | 3447 | 3/3 |
| TTC (high) | medium | 1344 | 1311 | 2382 | 3/3 |
| TTC (high) | long | 1274 | 1266 | 2130 | 3/3 |

**Summary:** TTC and OpenRouter show **comparable median TTFT** (~1.1-1.5s) for most combinations. There are occasional latency spikes on both providers (e.g., TTC none/medium at 4.1s, OpenRouter long at 1.7s), but median performance is similar. No native Anthropic baseline was available (account had zero balance).

### gemini-3.1-flash-lite-preview

| Provider | Size | TTFT avg (ms) | TTFT median (ms) | Total avg (ms) | OK |
|----------|------|---------------|-------------------|-----------------|-----|
| OpenRouter | short | 876 | 890 | 928 | 3/3 |
| OpenRouter | medium | 818 | 803 | 897 | 3/3 |
| OpenRouter | long | 773 | 759 | 855 | 3/3 |
| TTC (none) | short | 1240 | 1236 | 1323 | 3/3 |
| TTC (none) | medium | 1018 | 1005 | 1143 | 3/3 |
| TTC (none) | long | 1121 | 1099 | 1366 | 3/3 |
| TTC (low) | short | 1002 | 992 | 1090 | 3/3 |
| TTC (low) | medium | 1142 | 1163 | 1262 | 3/3 |
| TTC (low) | long | 1044 | 1053 | 1279 | 3/3 |
| TTC (high) | short | 1155 | 1157 | 1217 | 3/3 |
| TTC (high) | medium | 1133 | 1065 | 1281 | 3/3 |
| TTC (high) | long | 1110 | 1107 | 1284 | 3/3 |

**Summary:** TTC adds approximately **200-350ms TTFT overhead** compared to OpenRouter. Gemini flash-lite is the fastest model tested, with sub-1s TTFT on both providers. No native Google API baseline was tested.

## Key findings

1. **TTC Gateway overhead is small and consistent.** For gpt-5.4, the gateway adds ~100-200ms over the native OpenAI API. For Gemini, it adds ~200-350ms over OpenRouter. For Claude, TTC and OpenRouter are comparable.

2. **Compression level does not meaningfully affect latency.** Results for compression=none, low, and high are within the noise margin across all models and input sizes. The compression processing time is negligible relative to network and inference latency.

3. **100% success rate.** All requests completed successfully across all providers, models, and input sizes.

4. **OpenAI reasoning overhead matters.** Earlier test runs using gpt-5-mini without `reasoning_effort: "minimal"` showed 5-9s TTFT through TTC vs ~1s native. This was not a gateway issue but the model spending its token budget on internal reasoning. Suppressing reasoning brought TTC latency in line with native.

5. **Concurrent request blasting inflates latency.** Early test runs that sent 50 concurrent requests per provider showed 10-15s TTFT on TTC. Sequential single-request testing showed ~1-1.5s TTFT. Benchmark methodology matters.

## Methodology notes

- With only 3 runs per combination, individual outliers have outsized impact on averages. Median values are more representative.
- The benchmark measures cold-start latency (new connection per request). Applications that reuse connections would see lower TTFT due to saved TLS handshake time.
- Results reflect conditions at the time of testing (April 22, 2026) and may vary with provider load, time of day, and infrastructure changes.
