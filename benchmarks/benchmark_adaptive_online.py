# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online serving benchmark for adaptive draft model switching.

Starts a vLLM OpenAI-compatible API server for each variant (base, int8, fp8,
adaptive) and drives it with Poisson-distributed concurrent requests sampled
from ShareGPT. Measures TTFT, ITL, end-to-end latency, and output throughput
to compare how each draft-model configuration performs under realistic traffic.

Unlike the wave benchmark (which runs requests in discrete synchronized batches
with ignore_eos=True), this script:
  - Uses real EOS tokens - requests stop when the model finishes
  - Uses Poisson inter-arrival times - batch sizes vary organically
  - Uses vLLM's online scheduler - scheduling decisions are realistic
  - Measures latency (TTFT, ITL) in addition to throughput

Usage:
    python benchmarks/benchmark_adaptive_online.py \\
        --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
        --request-rate 4 \\
        --duration 180
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np


@dataclass
class RequestResult:
    prompt_len: int
    output_tokens: int
    ttft: float           # seconds to first token
    itl: list[float]      # per-token inter-token latencies (seconds), excluding first
    e2el: float           # end-to-end latency (seconds)
    success: bool
    error: str = ""


@dataclass
class OnlineMetrics:
    completed: int
    failed: int
    output_throughput: float    # tok/s over measurement window
    request_goodput: float      # successful req/s over measurement window
    ttft_p50_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p99_ms: float
    e2el_p50_ms: float
    e2el_p99_ms: float


def load_sharegpt(
    dataset_path: str,
    num_samples: int,
    max_model_len: int,
    tokenizer,
    seed: int,
) -> list[tuple[str, int, int]]:
    """Return (prompt_text, prompt_len, max_output_len) tuples from ShareGPT."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    random.seed(seed)
    random.shuffle(data)
    results: list[tuple[str, int, int]] = []
    for entry in data:
        if len(results) >= num_samples:
            break
        prompt_text = entry["conversations"][0]["value"]
        completion_text = entry["conversations"][1]["value"]
        prompt_ids = tokenizer(prompt_text).input_ids
        completion_ids = tokenizer(completion_text).input_ids
        if len(prompt_ids) < 4 or len(completion_ids) < 4:
            continue
        if len(prompt_ids) + len(completion_ids) > max_model_len:
            continue
        results.append((prompt_text, len(prompt_ids), len(completion_ids)))
    return results


def make_spec_config(
    variant: str,
    draft_model_base: str,
    draft_model_fp8: str,
    draft_model_int8: str,
    num_spec_tokens: int,
    threshold: int,
    low_threshold: int,
    ema_alpha: float,
) -> dict | None:
    """Build the --speculative-config dict for a given variant, or None for base."""
    if variant == "base":
        return None
    draft = {
        "draft_base": draft_model_base,
        "int8": draft_model_int8,
        "fp8": draft_model_fp8,
        "adaptive": draft_model_fp8,
    }[variant]
    cfg: dict = {
        "method": "draft_model",
        "model": draft,
        "num_speculative_tokens": num_spec_tokens,
    }
    if variant == "adaptive":
        cfg["alt_model"] = draft_model_int8
        cfg["adaptive_threshold"] = threshold
        cfg["adaptive_low_threshold"] = low_threshold
        cfg["adaptive_ema_alpha"] = ema_alpha
    return cfg


def build_serve_cmd(
    target_model: str,
    max_num_seqs: int,
    max_model_len: int,
    port: int,
    spec_config: dict | None,
) -> list[str]:
    """Build the vllm serve subprocess command."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", target_model,
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_model_len),
        "--port", str(port),
        "--no-enable-log-requests",
    ]
    if spec_config:
        cmd += ["--speculative-config", json.dumps(spec_config)]
    return cmd


def _clear_compile_cache() -> None:
    import shutil
    cache_dir = Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def start_server(cmd: list[str], log_path: str) -> subprocess.Popen:
    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for subprocess lifetime
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )


def kill_server(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        # Always follow up with SIGKILL to ensure GPU memory is released.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)
    except (ProcessLookupError, PermissionError):
        pass


async def wait_for_server(
    base_url: str,
    proc: subprocess.Popen,
    log_path: str,
    timeout: int = 600,
) -> bool:
    """Poll /health until 200, process death, or timeout.

    Prints progress dots and the log file path so the user can tail it.
    """
    print(f"    (server log: {log_path})")
    deadline = time.monotonic() + timeout
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"\n  ERROR: server process exited (code {proc.returncode}). "
                      f"Check log: {log_path}")
                return False
            try:
                async with session.get(
                    f"{base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        print()
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(10)
    print()
    return False


_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


async def send_one_request(
    session: aiohttp.ClientSession,
    completions_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send one streaming completions request; return timing metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    output_tokens = 0
    ttft = 0.0
    itl: list[float] = []
    first_token = False
    success = False
    error = ""

    st = time.perf_counter()
    most_recent = st
    buf = ""

    try:
        async with session.post(
            completions_url, json=payload, timeout=_REQUEST_TIMEOUT
        ) as response:
            if response.status != 200:
                body = await response.text()
                return RequestResult(
                    prompt_len=0, output_tokens=0, ttft=0, itl=[], e2el=0,
                    success=False, error=f"HTTP {response.status}: {body[:200]}",
                )

            async for chunk_bytes in response.content.iter_any():
                buf += chunk_bytes.decode("utf-8", errors="replace")
                # Process complete SSE messages (separated by blank lines).
                while "\n\n" in buf:
                    message, buf = buf.split("\n\n", 1)
                    for line in message.splitlines():
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if data.get("choices"):
                            ts = time.perf_counter()
                            if not first_token:
                                first_token = True
                                ttft = ts - st
                            else:
                                itl.append(ts - most_recent)
                            most_recent = ts

                        if usage := data.get("usage"):
                            output_tokens = usage.get("completion_tokens", 0)

            success = first_token

    except Exception as exc:
        error = str(exc)

    e2el = time.perf_counter() - st
    return RequestResult(
        prompt_len=0,
        output_tokens=output_tokens,
        ttft=ttft,
        itl=itl,
        e2el=e2el,
        success=success,
        error=error,
    )


async def run_load(
    base_url: str,
    model: str,
    prompts: list[tuple[str, int, int]],
    request_rate: float,
    duration: float,
    warmup_sec: float,
) -> list[RequestResult]:
    """Drive the server with Poisson arrivals; discard warmup results."""
    completions_url = f"{base_url}/v1/completions"
    results: list[RequestResult] = []
    lock = asyncio.Lock()

    async def worker(prompt_text: str, max_tokens: int, is_warmup: bool) -> None:
        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(connector=connector) as session:
            r = await send_one_request(
                session, completions_url, model, prompt_text, max_tokens
            )
        if not is_warmup:
            async with lock:
                results.append(r)

    tasks: list[asyncio.Task] = []
    loop_start = asyncio.get_event_loop().time()
    idx = 0

    while True:
        now = asyncio.get_event_loop().time()
        elapsed = now - loop_start
        if elapsed >= warmup_sec + duration:
            break

        is_warmup = elapsed < warmup_sec
        prompt_text, _, max_tokens = prompts[idx % len(prompts)]
        idx += 1

        task = asyncio.create_task(worker(prompt_text, max_tokens, is_warmup))
        tasks.append(task)

        if request_rate > 0 and not (request_rate == float("inf")):
            interval = random.expovariate(request_rate)
            await asyncio.sleep(interval)
        else:
            await asyncio.sleep(0)

    await asyncio.gather(*tasks, return_exceptions=True)
    return results


def compute_metrics(results: list[RequestResult], duration: float) -> OnlineMetrics:
    """Compute aggregate statistics from a list of completed request results."""
    successful = [r for r in results if r.success]
    failed = len(results) - len(successful)

    if not successful:
        return OnlineMetrics(
            completed=0, failed=failed,
            output_throughput=0.0, request_goodput=0.0,
            ttft_p50_ms=0.0, ttft_p99_ms=0.0,
            itl_p50_ms=0.0, itl_p99_ms=0.0,
            e2el_p50_ms=0.0, e2el_p99_ms=0.0,
        )

    total_output = sum(r.output_tokens for r in successful)
    ttfts_ms = np.array([r.ttft * 1000.0 for r in successful])
    itls_ms = np.array([v * 1000.0 for r in successful for v in r.itl])
    e2els_ms = np.array([r.e2el * 1000.0 for r in successful])

    def pct(arr: "np.ndarray", p: float) -> float:
        return float(np.percentile(arr, p)) if len(arr) > 0 else 0.0

    return OnlineMetrics(
        completed=len(successful),
        failed=failed,
        output_throughput=total_output / duration if duration > 0 else 0.0,
        request_goodput=len(successful) / duration if duration > 0 else 0.0,
        ttft_p50_ms=pct(ttfts_ms, 50),
        ttft_p99_ms=pct(ttfts_ms, 99),
        itl_p50_ms=pct(itls_ms, 50),
        itl_p99_ms=pct(itls_ms, 99),
        e2el_p50_ms=pct(e2els_ms, 50),
        e2el_p99_ms=pct(e2els_ms, 99),
    )


def format_results_table(
    all_metrics: dict[str, OnlineMetrics],
    variant_labels: list[str],
) -> str:
    from tabulate import tabulate

    headers = [
        "Variant", "Completed", "Failed",
        "TTFT p50 (ms)", "TTFT p99 (ms)",
        "ITL p50 (ms)", "ITL p99 (ms)",
        "E2EL p50 (ms)", "E2EL p99 (ms)",
        "Out tok/s", "Goodput (req/s)",
    ]
    rows = []
    for lbl in variant_labels:
        m = all_metrics.get(lbl)
        if m is None:
            rows.append([lbl] + ["N/A"] * (len(headers) - 1))
            continue
        rows.append([
            lbl,
            m.completed,
            m.failed,
            f"{m.ttft_p50_ms:.1f}",
            f"{m.ttft_p99_ms:.1f}",
            f"{m.itl_p50_ms:.2f}",
            f"{m.itl_p99_ms:.2f}",
            f"{m.e2el_p50_ms:.0f}",
            f"{m.e2el_p99_ms:.0f}",
            f"{m.output_throughput:.1f}",
            f"{m.request_goodput:.2f}",
        ])
    return tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)


def save_results(
    output_path: str,
    config: dict,
    all_metrics: dict[str, OnlineMetrics],
    variant_labels: list[str],
) -> None:
    data = {
        "config": config,
        "results": {
            lbl: {
                "completed": all_metrics[lbl].completed,
                "failed": all_metrics[lbl].failed,
                "output_throughput": all_metrics[lbl].output_throughput,
                "request_goodput": all_metrics[lbl].request_goodput,
                "ttft_p50_ms": all_metrics[lbl].ttft_p50_ms,
                "ttft_p99_ms": all_metrics[lbl].ttft_p99_ms,
                "itl_p50_ms": all_metrics[lbl].itl_p50_ms,
                "itl_p99_ms": all_metrics[lbl].itl_p99_ms,
                "e2el_p50_ms": all_metrics[lbl].e2el_p50_ms,
                "e2el_p99_ms": all_metrics[lbl].e2el_p99_ms,
            }
            for lbl in variant_labels
            if lbl in all_metrics
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online serving benchmark for adaptive draft model switching."
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model-base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-model-fp8", default="Qwen/Qwen3-1.7B-FP8")
    parser.add_argument("--draft-model-int8", default="Qwen/Qwen3-1.7B-GPTQ-Int8")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--num-prompts", type=int, default=500,
        help="ShareGPT samples to pre-load (requests cycle through them)",
    )
    parser.add_argument(
        "--request-rate", type=float, default=4.0,
        help="Target request arrival rate (req/s, Poisson distributed)",
    )
    parser.add_argument(
        "--duration", type=float, default=120.0,
        help="Measurement window per variant (seconds, excluding warmup)",
    )
    parser.add_argument(
        "--warmup", type=float, default=30.0,
        help="Warmup period per variant (seconds, excluded from metrics)",
    )
    parser.add_argument("--num-spec-tokens", type=int, default=5)
    parser.add_argument(
        "--threshold", type=int, default=16,
        help="EMA batch-size threshold for switching TO fp8 (high threshold)",
    )
    parser.add_argument(
        "--low-threshold", type=int, default=8,
        help="EMA batch-size threshold for switching BACK to int8 (low threshold)",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-num-seqs", type=int, default=128,
        help="Maximum concurrent sequences passed to vllm serve",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="results/adaptive_online_results.json",
    )
    parser.add_argument(
        "--variants", nargs="+",
        default=["base", "draft_base", "int8", "fp8", "adaptive"],
        choices=["base", "draft_base", "int8", "fp8", "adaptive"],
        help="Which variants to benchmark",
    )
    return parser.parse_args(argv)


def _unique_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(p.with_stem(f"{p.stem}_{stamp}"))


def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {args.target_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    print(f"Loading up to {args.num_prompts} ShareGPT prompts ...")
    prompts = load_sharegpt(
        dataset_path=args.dataset,
        num_samples=args.num_prompts,
        max_model_len=args.max_model_len,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    if not prompts:
        raise ValueError("No valid prompts loaded from dataset.")
    print(f"  Loaded {len(prompts)} prompts")

    base_url = f"http://localhost:{args.port}"
    variant_labels: list[str] = args.variants
    all_metrics: dict[str, OnlineMetrics] = {}

    for variant in variant_labels:
        print(f"\n{'=' * 60}")
        print(f"Variant: {variant}")
        print(f"{'=' * 60}")

        _clear_compile_cache()

        spec_config = make_spec_config(
            variant=variant,
            draft_model_base=args.draft_model_base,
            draft_model_fp8=args.draft_model_fp8,
            draft_model_int8=args.draft_model_int8,
            num_spec_tokens=args.num_spec_tokens,
            threshold=args.threshold,
            low_threshold=args.low_threshold,
            ema_alpha=args.ema_alpha,
        )

        cmd = build_serve_cmd(
            target_model=args.target_model,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            port=args.port,
            spec_config=spec_config,
        )
        print(f"  cmd: {' '.join(cmd[:5])} ...")

        log_path = f"/tmp/vllm_serve_{variant}.log"
        proc = start_server(cmd, log_path)
        try:
            print("  Waiting for server to be ready (up to 600s) ...")
            ready = asyncio.run(wait_for_server(base_url, proc, log_path, timeout=600))
            if not ready:
                print(f"  ERROR: server did not become ready; skipping {variant}")
                continue

            print(
                f"  Server ready. "
                f"warmup={args.warmup}s | duration={args.duration}s | "
                f"rate={args.request_rate} req/s"
            )
            results = asyncio.run(run_load(
                base_url=base_url,
                model=args.target_model,
                prompts=prompts,
                request_rate=args.request_rate,
                duration=args.duration,
                warmup_sec=args.warmup,
            ))

            metrics = compute_metrics(results, args.duration)
            all_metrics[variant] = metrics
            print(
                f"  completed={metrics.completed}  failed={metrics.failed}\n"
                f"  TTFT p50/p99: {metrics.ttft_p50_ms:.1f}/{metrics.ttft_p99_ms:.1f} ms\n"
                f"  ITL  p50/p99: {metrics.itl_p50_ms:.2f}/{metrics.itl_p99_ms:.2f} ms\n"
                f"  E2EL p50/p99: {metrics.e2el_p50_ms:.0f}/{metrics.e2el_p99_ms:.0f} ms\n"
                f"  Output throughput: {metrics.output_throughput:.1f} tok/s"
            )

        finally:
            print("  Shutting down server ...")
            kill_server(proc)
            time.sleep(20)

    if not all_metrics:
        print("No variants completed successfully.")
        return

    print("\n" + "=" * 60)
    print("Online serving results")
    print("=" * 60)
    print(format_results_table(all_metrics, [v for v in variant_labels if v in all_metrics]))

    config = {
        "target_model": args.target_model,
        "draft_model_base": args.draft_model_base,
        "draft_model_fp8": args.draft_model_fp8,
        "draft_model_int8": args.draft_model_int8,
        "request_rate": args.request_rate,
        "duration": args.duration,
        "warmup": args.warmup,
        "num_prompts": args.num_prompts,
        "num_spec_tokens": args.num_spec_tokens,
        "threshold": args.threshold,
        "low_threshold": args.low_threshold,
        "ema_alpha": args.ema_alpha,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "variants": variant_labels,
    }
    output_path = _unique_path(args.output)
    save_results(output_path, config, all_metrics, variant_labels)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
