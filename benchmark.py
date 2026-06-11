#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible chat-completions server.

The script intentionally uses only the Python standard library so it can run
inside minimal training/serving environments. It fires concurrent
``/v1/chat/completions`` requests, reports latency percentiles, and estimates
output-token throughput from the response usage object when available.

Example:

    python benchmark.py --base-url http://127.0.0.1:8000 --requests 256 --concurrency 64 --long-tail
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RequestResult:
    """Per-request measurement emitted by a worker thread."""

    ok: bool
    latency_s: float
    output_tokens: int
    bucket: str
    requested_tokens: int
    error: str | None = None


def main() -> None:
    """Parse flags, run the benchmark, and print aggregate metrics."""

    args = parse_args()
    cases = build_cases(args)
    payloads = [make_payload(args, case) for case in cases]

    start = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(post_chat_completion, args.base_url, payload, case["bucket"], int(case["max_tokens"]), args.timeout_s)
            for payload, case in zip(payloads, cases, strict=True)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed_s = time.perf_counter() - start

    print_summary(results, elapsed_s)


def parse_args() -> argparse.Namespace:
    """Return command-line options for one benchmark run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL, without /v1 suffix.")
    parser.add_argument("--model", default="benchmark", help="Model name sent in the request payload.")
    parser.add_argument("--requests", type=int, default=256, help="Total number of requests to send.")
    parser.add_argument("--concurrency", type=int, default=64, help="Maximum concurrent client requests.")
    parser.add_argument("--max-tokens", type=int, default=128, help="max_tokens for each completion.")
    parser.add_argument("--short-max-tokens", type=int, default=64, help="Short-tail max_tokens when --long-tail is enabled.")
    parser.add_argument("--medium-max-tokens", type=int, default=512, help="Medium-tail max_tokens when --long-tail is enabled.")
    parser.add_argument("--long-max-tokens", type=int, default=4096, help="Long-tail max_tokens when --long-tail is enabled.")
    parser.add_argument("--long-tail", action="store_true", help="Mix short/medium/long requests to stress tail handling.")
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Approximate prompt length in whitespace tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top_p.")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Per-request HTTP timeout.")
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if min(args.short_max_tokens, args.medium_max_tokens, args.long_max_tokens) <= 0:
        parser.error("--short/medium/long max token values must be positive")
    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    return args


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build deterministic benchmark cases, optionally with long-tail lengths."""

    cases = []
    for idx in range(args.requests):
        bucket, max_tokens = request_shape(args, idx)
        cases.append(
            {
                "bucket": bucket,
                "max_tokens": max_tokens,
                "prompt": build_prompt(idx, args.prompt_tokens, bucket, max_tokens),
            }
        )
    return cases


def request_shape(args: argparse.Namespace, idx: int) -> tuple[str, int]:
    """Return the latency bucket and max token budget for one request."""

    if not args.long_tail:
        return "uniform", args.max_tokens
    # 75/20/5 split. The sparse long bucket is designed to expose tail effects.
    slot = idx % 20
    if slot == 0:
        return "long", args.long_max_tokens
    if slot in {1, 2, 3, 4}:
        return "medium", args.medium_max_tokens
    return "short", args.short_max_tokens


def build_prompt(idx: int, prompt_tokens: int, bucket: str, max_tokens: int) -> str:
    """Build a more complex prompt that makes long-tail decoding visible."""

    facts = " ".join(f"constraint_{i}=value_{(idx + i) % 17}" for i in range(prompt_tokens))
    return f"""You are evaluating a local LLM serving engine under concurrent load.

Request id: {idx}
Target bucket: {bucket}
Token budget: {max_tokens}

Task:
Given the synthetic project brief and constraints below, produce a structured
analysis with these sections:
1. assumptions
2. dependency graph
3. risk table
4. execution plan
5. final recommendation

Rules:
- Be concrete and internally consistent.
- Use the provided constraint names.
- For long requests, expand the dependency graph and risk table.
- Do not mention that this is a benchmark.

Synthetic constraints:
{facts}
"""


def make_payload(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    """Build one OpenAI-compatible chat-completions JSON payload."""

    return {
        "model": args.model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": int(case["max_tokens"]),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }


def post_chat_completion(base_url: str, payload: dict[str, Any], bucket: str, requested_tokens: int, timeout_s: float) -> RequestResult:
    """POST one completion request and return latency plus token counts."""

    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer benchmark"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        latency_s = time.perf_counter() - start
        return RequestResult(ok=True, latency_s=latency_s, output_tokens=completion_tokens(data), bucket=bucket, requested_tokens=requested_tokens)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        latency_s = time.perf_counter() - start
        return RequestResult(ok=False, latency_s=latency_s, output_tokens=0, bucket=bucket, requested_tokens=requested_tokens, error=str(exc))


def completion_tokens(data: dict[str, Any]) -> int:
    """Best-effort completion-token count from OpenAI-compatible responses."""

    usage = data.get("usage") or {}
    if isinstance(usage.get("completion_tokens"), int):
        return int(usage["completion_tokens"])
    choices = data.get("choices") or []
    if not choices:
        return 0
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return len(str(content).split())


def print_summary(results: list[RequestResult], elapsed_s: float) -> None:
    """Print compact benchmark metrics."""

    ok = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    latencies = sorted(result.latency_s for result in ok)
    output_tokens = sum(result.output_tokens for result in ok)
    total = len(results)

    print(f"requests_total={total}")
    print(f"requests_ok={len(ok)}")
    print(f"requests_failed={len(failed)}")
    print(f"elapsed_s={elapsed_s:.3f}")
    print(f"request_rps={len(ok) / elapsed_s if elapsed_s > 0 else 0.0:.3f}")
    print(f"output_tokens={output_tokens}")
    print(f"output_tokens_per_s={output_tokens / elapsed_s if elapsed_s > 0 else 0.0:.3f}")
    if latencies:
        print(f"latency_mean_s={statistics.fmean(latencies):.3f}")
        print(f"latency_p50_s={percentile(latencies, 0.50):.3f}")
        print(f"latency_p90_s={percentile(latencies, 0.90):.3f}")
        print(f"latency_p99_s={percentile(latencies, 0.99):.3f}")
        print(f"latency_max_s={latencies[-1]:.3f}")
    print_bucket_summaries(ok)
    if failed:
        print("first_errors:")
        for result in failed[:5]:
            print(f"- {result.error}")


def print_bucket_summaries(results: list[RequestResult]) -> None:
    """Print latency/token metrics by request length bucket."""

    buckets = sorted({result.bucket for result in results})
    for bucket in buckets:
        rows = [result for result in results if result.bucket == bucket]
        latencies = sorted(result.latency_s for result in rows)
        tokens = sum(result.output_tokens for result in rows)
        requested = sorted({result.requested_tokens for result in rows})
        print(
            f"bucket={bucket} count={len(rows)} requested_tokens={requested} "
            f"output_tokens={tokens} latency_p50_s={percentile(latencies, 0.50):.3f} "
            f"latency_p90_s={percentile(latencies, 0.90):.3f} latency_max_s={latencies[-1]:.3f}"
        )


def percentile(values: list[float], q: float) -> float:
    """Return nearest-rank percentile for already sorted values."""

    if not values:
        return 0.0
    idx = min(max(int(round(q * (len(values) - 1))), 0), len(values) - 1)
    return values[idx]


if __name__ == "__main__":
    main()
