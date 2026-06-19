#!/usr/bin/env python
"""
Staging load simulator. Hits the staging API via HTTP with X-API-KEY.
Local-only — env vars (STAGING_API_URL, STAGING_API_KEY) kept on disk, not pushed.

Usage:
    export STAGING_API_URL=https://staging-api.example/api/v1
    export STAGING_API_KEY=...
    python scripts/load_sim.py --load llm:1.0 --duration 300
    python scripts/load_sim.py --load llm:1.0,stt:0.3,tts:0.3,sts:0.2 --duration 600

Buckets: llm, stt, tts, sts (speech-to-speech via /llm/chain).
"""
import argparse
import logging

# import os
import random
import threading
import time
import uuid
from dataclasses import dataclass

import httpx

logger = logging.getLogger("load_sim")

CONFIG_ID = ""
KB_ID = ""
CALLBACK_URL = ""


def _body_llm(run_id: str) -> dict:
    return {
        "query": {
            "input": {
                "type": "text",
                "content": {
                    "format": "text",
                    "value": "What does AMAN Foundation do in disaster resilience?",
                },
            }
        },
        "config": {"id": CONFIG_ID, "version": 1},
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {"test_id": f"sim-{run_id}-llm", "user": "load_sim"},
    }


def _body_stt(run_id: str) -> dict:
    return {
        "query": {
            "input": {
                "type": "audio",
                "content": {"format": "base64", "value": AUDIO_B64},
            }
        },
        "config": {
            "blob": {
                "completion": {
                    "provider": "google",
                    "type": "stt",
                    "params": {"model": "gemini-2.5-pro"},
                }
            }
        },
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {"test_id": f"sim-{run_id}-stt", "user": "load_sim"},
    }


def _body_tts(run_id: str) -> dict:
    return {
        "query": {
            "input": {
                "type": "text",
                "content": {
                    "format": "text",
                    "value": "Hello. The earth is almost round. Nobody should die of hunger.",
                },
            }
        },
        "config": {
            "blob": {
                "completion": {
                    "provider": "google",
                    "type": "tts",
                    "params": {
                        "model": "gemini-2.5-pro-preview-tts",
                        "response_format": "ogg",
                    },
                }
            }
        },
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {"test_id": f"sim-{run_id}-tts", "user": "load_sim"},
    }


def _body_sts(run_id: str) -> dict:
    return {
        "query": {
            "type": "audio",
            "content": {
                "format": "base64",
                "value": AUDIO_B64,
                "mime_type": "audio/ogg",
            },
        },
        "knowledge_base_ids": [KB_ID],
        "callback_url": CALLBACK_URL,
        "request_metadata": {"test": f"sim-{run_id}-sts", "user": "load_sim"},
    }


# bucket -> (endpoint path, body builder)
BUCKETS = {
    "llm": ("/llm/call", _body_llm),
    "stt": ("/llm/call", _body_stt),
    "tts": ("/llm/call", _body_tts),
    "sts": ("/llm/chain/sts", _body_sts),
}


@dataclass
class LoadSpec:
    bucket: str
    rate: float


def parse_load(spec: str) -> list[LoadSpec]:
    out: list[LoadSpec] = []
    for entry in spec.split(","):
        b, r = entry.strip().split(":")
        if b not in BUCKETS:
            raise ValueError(f"unknown bucket {b!r}. options: {sorted(BUCKETS)}")
        out.append(LoadSpec(bucket=b, rate=float(r)))
    return out


def next_sleep(rate: float, mode: str, jitter_pct: float) -> float:
    base = 1.0 / rate
    if mode == "none":
        return base
    if mode == "uniform":
        return base * random.uniform(1 - jitter_pct, 1 + jitter_pct)
    if mode == "poisson":
        return random.expovariate(rate)
    raise ValueError(mode)


def enqueue_one(client: httpx.Client, load: LoadSpec, run_id: str) -> None:
    path, build = BUCKETS[load.bucket]
    try:
        t0 = time.monotonic()
        r = client.post(path, json=build(run_id))
        dt = (time.monotonic() - t0) * 1000
        logger.info(
            f"[enqueue] run_id={run_id} bucket={load.bucket} status={r.status_code} latency_ms={dt:.0f}"
        )
        if r.status_code >= 400:
            logger.error(f"[enqueue] body={r.text[:300]}")
    except Exception as e:
        logger.error(f"[enqueue] bucket={load.bucket} failed: {e}")


def run_bucket(
    client: httpx.Client,
    load: LoadSpec,
    stop: threading.Event,
    run_id: str,
    jitter: str,
    jitter_pct: float,
) -> None:
    while not stop.is_set():
        enqueue_one(client, load, run_id)
        remaining = next_sleep(load.rate, jitter, jitter_pct)
        while remaining > 0 and not stop.is_set():
            chunk = min(remaining, 0.5)
            time.sleep(chunk)
            remaining -= chunk


def _selftest() -> None:
    assert [l.bucket for l in parse_load("llm:1.0,stt:0.5")] == ["llm", "stt"]
    try:
        parse_load("nope:1")
        raise AssertionError
    except ValueError:
        pass
    for m in ("none", "uniform", "poisson"):
        assert 0 < next_sleep(2.0, m, 0.3) < 100
    for path, build in BUCKETS.values():
        body = build("test")
        assert isinstance(body, dict) and path.startswith("/")
    print("selftest ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--load", help="comma list bucket:rate, e.g. llm:1.0,stt:0.3,sts:0.2"
    )
    p.add_argument("--duration", type=int, help="seconds")
    p.add_argument(
        "--jitter", choices=["poisson", "uniform", "none"], default="poisson"
    )
    p.add_argument("--jitter-pct", type=float, default=0.3)
    p.add_argument("--run-id", default=uuid.uuid4().hex[:8])
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.selftest:
        _selftest()
        return
    if not args.load or not args.duration:
        p.error("--load and --duration required")

    base_url = ""
    api_key = ""
    loads = parse_load(args.load)
    logger.info(
        f"[main] run_id={args.run_id} base_url={base_url} duration={args.duration}s loads={loads}"
    )

    stop = threading.Event()
    with httpx.Client(
        base_url=base_url, headers={"X-API-KEY": api_key}, timeout=args.timeout
    ) as client:
        threads = [
            threading.Thread(
                target=run_bucket,
                args=(client, l, stop, args.run_id, args.jitter, args.jitter_pct),
                daemon=True,
            )
            for l in loads
        ]
        for t in threads:
            t.start()
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            logger.info("[main] interrupted")
        stop.set()
        for t in threads:
            t.join(timeout=2)

    logger.info(f"[main] done. run_id={args.run_id}")


# ponytail: opus sample inline. One short sample reused for stt/sts. Swap for
# longer or varied clips if you need realistic duration spread per request.
AUDIO_B64 = "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"


if __name__ == "__main__":
    main()
