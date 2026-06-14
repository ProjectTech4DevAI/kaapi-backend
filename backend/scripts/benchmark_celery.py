#!/usr/bin/env python3
"""
Benchmark script for the /llm/call endpoint (send-side only).

What this script captures:
  - Per-request: send time, kaapi ack status, ack latency, ack-side error class.
  - Per-window: requests dispatched + ack-class histogram.

What it deliberately does NOT capture:
  - Vertex outcome (success / 429 / INTERNAL / etc). Kaapi returns 200 on the
    initial /llm/call as soon as the job is enqueued; the real provider result
    arrives asynchronously in the webhook callback. To classify those, export
    webhook.site requests as CSV and run scripts/parse_webhook_export.py
    against it, then join on benchmark_request_id.

Config is read from TEST_JSON.md (scripts/TEST_JSON.md relative to the repo
root, or override via env var TEST_JSON_PATH).

Usage (from repo root):
    python backend/scripts/benchmark_celery.py
"""

import csv
import os
import random
import re
import threading
import time
import uuid
from datetime import datetime, timezone

import requests

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = ""
# BASE_URL="http://localhost:8000"
API_KEY = ""
CALLBACK_URL = "https://play.svix.com/in/e_aF8nyc5E9KknpA9KSFo7FOEmX34/"

# TTS text (the Odia sample from TEST_JSON.md)
TTS_INPUT_TEXT = "ରୋଜ ସଞ୍ଜବେଳୁଁ ପାଖରେ ବସାଇ ରାତି ଦୁଇ ଘଡ଼ି ଯାଏ ପଢ଼ାନ୍ତି"

TOTAL_DURATION_SECS = 5 * 60  # 1 minute
REQUESTS_PER_WINDOW = 50
WINDOW_SECS = 60
BASE_INTERVAL_SECS = WINDOW_SECS / REQUESTS_PER_WINDOW
JITTER_SECS = 0.12

ACK_CLASSES = (
    "ack_ok",
    "ack_4xx",
    "ack_5xx",
    "ack_timeout",
    "ack_network",
    "ack_unknown",
)

CSV_FIELDS = [
    "request_id",
    "request_type",
    "seq",
    "window_num",
    "sent_at",
    "http_status",
    "ack_class",
    "ack_latency_ms",
    "error",
]

MINUTE_CSV_FIELDS = ["minute", "requests_sent", "cumulative_sent"] + [
    f"cls_{c}" for c in ACK_CLASSES
]


def _classify_ack(status: int | None, exc: Exception | None) -> str:
    """Classify the kaapi enqueue response only. Vertex outcomes go elsewhere."""
    if exc is not None:
        name = type(exc).__name__.lower()
        if "timeout" in name:
            return "ack_timeout"
        return "ack_network"
    if status is None:
        return "ack_unknown"
    if 200 <= status < 300:
        return "ack_ok"
    if 400 <= status < 500:
        return "ack_4xx"
    if 500 <= status < 600:
        return "ack_5xx"
    return "ack_unknown"


RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Shared state ───────────────────────────────────────────────────────────────
_lock = threading.Lock()
_completed: list[dict] = []  # one row per finished request
_status_counts: dict[str, int] = {}  # cumulative ack_class -> count
_window_status_counts: dict[str, int] = {}  # current window ack_class -> count


# ── Payload builders ───────────────────────────────────────────────────────────
def _build_metadata(
    *,
    request_id: str,
    request_type: str,
    seq: int,
    window_num: int,
    sent_at_iso: str,
    sent_ts: float,
    payload_chars: int | None = None,
    payload_bytes: int | None = None,
) -> dict:
    """Everything needed to reconstruct latency from webhook.site alone."""
    return {
        "benchmark_request_id": request_id,
        "test_id": f"vertex-{request_type}-benchmark",
        "user": "Prajna",
        "request_type": request_type,
        "seq": seq,
        "window_num": window_num,
        "sent_at": sent_at_iso,
        "sent_ts_ms": int(sent_ts * 1000),
        "payload_chars": payload_chars,
        "payload_bytes": payload_bytes,
        "provider": "google",
        "run_id": RUN_ID,
    }


def _build_tts_payload(
    request_id: str, seq: int, window_num: int, sent_at_iso: str, sent_ts: float
) -> dict:
    return {
        "query": {"input": TTS_INPUT_TEXT},
        "config": {
            "blob": {
                "completion": {
                    "provider": "google",
                    "type": "tts",
                    "params": {"model": "gemini-2.5-pro-tts"},
                }
            }
        },
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": _build_metadata(
            request_id=request_id,
            request_type="tts",
            seq=seq,
            window_num=window_num,
            sent_at_iso=sent_at_iso,
            sent_ts=sent_ts,
            payload_chars=len(TTS_INPUT_TEXT),
        ),
    }


def _build_stt_payload(
    request_id: str,
    audio_b64: str,
    seq: int,
    window_num: int,
    sent_at_iso: str,
    sent_ts: float,
    audio_bytes: int,
) -> dict:
    return {
        "query": {
            "input": {
                "type": "audio",
                "content": {
                    "format": "base64",
                    "value": audio_b64,
                },
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
        "request_metadata": _build_metadata(
            request_id=request_id,
            request_type="stt",
            seq=seq,
            window_num=window_num,
            sent_at_iso=sent_at_iso,
            sent_ts=sent_ts,
            payload_bytes=audio_bytes,
        ),
    }


# ── Request sender (runs in its own thread) ────────────────────────────────────
def _send_request(
    request_type: str,
    audio_b64: str,
    seq: int,
    window_num: int,
    audio_bytes: int,
) -> None:
    request_id = str(uuid.uuid4())
    sent_ts = time.time()
    sent_at_iso = datetime.fromtimestamp(sent_ts, tz=timezone.utc).isoformat()

    payload = (
        _build_tts_payload(request_id, seq, window_num, sent_at_iso, sent_ts)
        if request_type == "tts"
        else _build_stt_payload(
            request_id, audio_b64, seq, window_num, sent_at_iso, sent_ts, audio_bytes
        )
    )

    http_status = None
    ack_latency_ms = None
    error = None
    exc_obj: Exception | None = None

    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/llm/call",
            json=payload,
            headers={"X-API-KEY": API_KEY, "Content-Type": "application/json"},
            timeout=30,
        )
        ack_latency_ms = round((time.time() - sent_ts) * 1000, 2)
        http_status = resp.status_code
        if not resp.ok:
            error = resp.text[:300]
    except Exception as exc:
        ack_latency_ms = round((time.time() - sent_ts) * 1000, 2)
        error = str(exc)
        exc_obj = exc

    ack_class = _classify_ack(http_status, exc_obj)

    print(
        f"[SENT]     {sent_at_iso}  type={request_type}  id={request_id}"
        f"  seq={seq}  status={http_status}  class={ack_class}  ack={ack_latency_ms} ms"
        + (f"  error={error}" if error else "")
    )

    with _lock:
        _status_counts[ack_class] = _status_counts.get(ack_class, 0) + 1
        _window_status_counts[ack_class] = _window_status_counts.get(ack_class, 0) + 1
        _completed.append(
            {
                "request_id": request_id,
                "request_type": request_type,
                "seq": seq,
                "window_num": window_num,
                "sent_at": sent_at_iso,
                "http_status": http_status,
                "ack_class": ack_class,
                "ack_latency_ms": ack_latency_ms,
                "error": error,
            }
        )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_stt_audio(test_json_path: str) -> str:
    print(f"Loading STT audio blob from {test_json_path} ...")
    with open(test_json_path, "r", errors="replace") as fh:
        for line in fh:
            m = re.search(r'"value"\s*:\s*"([A-Za-z0-9+/=]{200,})"', line)
            if m:
                blob = m.group(1)
                print(f"Loaded audio blob ({len(blob):,} chars)")
                return blob
    raise RuntimeError("Could not find base64 audio blob in TEST_JSON.md")


def _flush_minute_csv(results_file: str) -> None:
    minute_file = results_file.replace(".csv", "_per_minute.csv")
    with _lock:
        rows = list(_minute_stats)
    with open(minute_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MINUTE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-minute stats → {minute_file}")


def _flush_csv(results_file: str) -> None:
    with _lock:
        rows = list(_completed)
    with open(results_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# Per-window send stats (declared here so _flush_minute_csv can see it; written
# by main()).
_minute_stats: list[dict] = []


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_json_path = os.environ.get(
        "TEST_JSON_PATH",
        os.path.join(repo_root, "scripts", "TEST_JSON.md"),
    )

    audio_b64 = _load_stt_audio(test_json_path)
    audio_bytes = len(audio_b64) * 3 // 4  # rough decoded byte size

    results_file = os.path.join(
        repo_root,
        f"benchmark_llm_{RUN_ID}.csv",
    )

    print(f"\nResults → {results_file}")
    print(
        f"Config  → {REQUESTS_PER_WINDOW} req/window, "
        f"{BASE_INTERVAL_SECS}s interval ±{JITTER_SECS}s jitter, "
        f"{TOTAL_DURATION_SECS // 60} min total\n"
    )
    print(
        "Note: Vertex error classification is intentionally omitted here — "
        "kaapi returns 200 on enqueue regardless of Vertex outcome. Export "
        "webhook.site CSV and run scripts/parse_webhook_export.py to bucket "
        "real provider outcomes.\n"
    )

    benchmark_start = time.time()
    total_sent = 0
    window_num = 0
    seq_counter = 0

    try:
        while time.time() - benchmark_start < TOTAL_DURATION_SECS:
            window_start = time.time()
            window_num += 1
            sent_this_window = 0

            for i in range(REQUESTS_PER_WINDOW):
                if time.time() - benchmark_start >= TOTAL_DURATION_SECS:
                    break

                seq_counter += 1
                req_type = random.choice(["tts", "stt"])
                threading.Thread(
                    target=_send_request,
                    args=(req_type, audio_b64, seq_counter, window_num, audio_bytes),
                    daemon=True,
                ).start()
                total_sent += 1
                sent_this_window += 1

                # Sleep between requests (skip after the last one in the window)
                if i < REQUESTS_PER_WINDOW - 1:
                    jitter = random.uniform(-JITTER_SECS, JITTER_SECS)
                    sleep_secs = max(1.0, BASE_INTERVAL_SECS + jitter)
                    time.sleep(sleep_secs)

            # Wait out the remainder of the 60-second window
            elapsed = time.time() - window_start
            remaining = WINDOW_SECS - elapsed
            time_left_overall = TOTAL_DURATION_SECS - (time.time() - benchmark_start)
            if remaining > 0 and time_left_overall > 0:
                time.sleep(min(remaining, time_left_overall))

            with _lock:
                window_hist = dict(_window_status_counts)
                row = {
                    "minute": window_num,
                    "requests_sent": sent_this_window,
                    "cumulative_sent": total_sent,
                }
                for cls in ACK_CLASSES:
                    row[f"cls_{cls}"] = window_hist.get(cls, 0)
                _minute_stats.append(row)
                _window_status_counts.clear()
            print(
                f"[MINUTE {window_num:2d}] sent={sent_this_window}"
                f"  cumulative={total_sent}  ack_status={window_hist}"
            )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — flushing results.")

    # Brief grace period for in-flight ack threads to land before we flush.
    print("\n[INFO] Waiting 5s for in-flight ack threads to land...")
    time.sleep(5)

    _flush_csv(results_file)
    _flush_minute_csv(results_file)

    with _lock:
        n_total = len(_completed)
        cumulative = dict(_status_counts)

    print(f"\n{'─' * 60}")
    print("Benchmark complete")
    print(f"  Requests sent       : {total_sent}")
    print(f"  Ack rows captured   : {n_total}")
    print(f"  Cumulative ack hist : {cumulative}")
    print(f"  Results saved to    : {results_file}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
