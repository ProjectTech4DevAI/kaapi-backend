"""Load-test Vertex STT/TTS: concurrent requests within a time window.

Fires `--concurrency` worker threads for `--duration` seconds, each looping
execute() as fast as it can. Reports throughput (requests/min), latency
percentiles, and when the first 429 lands — i.e. how many concurrent
requests Vertex sustains before rate-limiting. In `mixed` mode each request
randomly picks STT or TTS.

provider.execute() is blocking (requests-based), so threads — not asyncio —
give real parallelism for I/O-bound HTTP calls.

Usage:
    uv run python -m scripts.bench_vertex_rate_limit mixed -c 20 -d 60
    uv run python -m scripts.bench_vertex_rate_limit tts -c 50 -d 60 --out load.csv
"""

import argparse
import csv
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.core.audio_utils import AudioRef, pcm_to_wav
from app.models.llm import NativeCompletionConfig, QueryParams
from app.models.llm.constants import CompletionType
from app.services.llm.providers.google_ai import GoogleVertexAIProvider

# Representative of a RAG answer that gets synthesised to speech.
TTS_TEXT = (
    "Based on the retrieved documents, crop rotation improves soil health by "
    "alternating nutrient demands across seasons, which reduces pest buildup "
    "and preserves nitrogen. Farmers in the region typically pair legumes with "
    "cereals over a three-year cycle to maximise yield while limiting "
    "fertiliser use. Let me know if you would like the specifics for your soil "
    "type."
)
# STT input: 0.1s of silence (24kHz, 16-bit PCM) — smallest valid audio.
STT_AUDIO = AudioRef(bytes_=pcm_to_wav(b"\x00\x00" * 2400), mime_type="audio/wav")

RATE_LIMIT_MARKER = "code: 429"
CSV_FIELDS = ["ts", "worker", "kind", "status", "latency_s", "detail"]


def build_call(kind: str) -> tuple[NativeCompletionConfig, QueryParams, Any]:
    if kind == "tts":
        return (
            NativeCompletionConfig(
                provider="google-native", params={}, type=CompletionType.TTS
            ),
            QueryParams(input=TTS_TEXT),
            TTS_TEXT,
        )
    return (
        NativeCompletionConfig(
            provider="google-native", params={}, type=CompletionType.STT
        ),
        QueryParams(input="transcribe"),
        STT_AUDIO,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tts", "stt", "mixed"])
    ap.add_argument(
        "-c", "--concurrency", type=int, default=10, help="parallel workers"
    )
    ap.add_argument(
        "-d", "--duration", type=float, default=60.0, help="load window in seconds"
    )
    ap.add_argument(
        "--out", default="vertex_load.csv", help="per-request CSV output path"
    )
    args = ap.parse_args()

    # credentials={} -> falls back to platform settings (GCP_VERTEX_API_KEY, etc.)
    provider = GoogleVertexAIProvider(GoogleVertexAIProvider.create_client({}))

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    first_429: dict[str, float] = {}  # {"t": seconds-into-run}
    start = time.monotonic()
    deadline = start + args.duration

    def worker(wid: int) -> None:
        while time.monotonic() < deadline:
            kind = random.choice(("tts", "stt")) if args.mode == "mixed" else args.mode
            config, query, resolved_input = build_call(kind)

            t0 = time.monotonic()
            _, error = provider.execute(config, query, resolved_input)
            dt = time.monotonic() - t0

            rate_limited = bool(error) and RATE_LIMIT_MARKER in error
            status = "rate_limited" if rate_limited else ("error" if error else "ok")
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "worker": wid,
                "kind": kind,
                "status": status,
                "latency_s": round(dt, 3),
                "detail": (error or "")[:300],
            }
            with lock:
                results.append(row)
                if rate_limited and "t" not in first_429:
                    first_429["t"] = t0 - start

    print(
        f"[load] mode={args.mode} concurrency={args.concurrency} duration={args.duration}s -> {args.out}"
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for f in [pool.submit(worker, w) for w in range(args.concurrency)]:
            f.result()
    elapsed = time.monotonic() - start

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    ok = by_status.get("ok", 0)
    latencies = sorted(r["latency_s"] for r in results)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(p / 100 * len(latencies)))
        return latencies[idx]

    print(f"\n[load] === summary ({elapsed:.1f}s, {args.concurrency} workers) ===")
    print(f"  total requests : {total}")
    print(f"  by status      : {by_status}")
    print(
        f"  throughput     : {total / elapsed * 60:.0f} req/min ({ok / elapsed * 60:.0f} ok/min)"
    )
    if latencies:
        print(
            f"  latency p50/p95/p99 : {pct(50):.2f}s / {pct(95):.2f}s / {pct(99):.2f}s "
            f"(mean {statistics.mean(latencies):.2f}s)"
        )
    if "t" in first_429:
        print(
            f"  first 429      : {first_429['t']:.1f}s into run "
            f"(after ~{ok} successful requests)"
        )
    else:
        print("  first 429      : none — not rate-limited at this concurrency")


if __name__ == "__main__":
    main()
