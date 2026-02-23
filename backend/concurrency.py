"""
Multimodal Load Testing Framework for LLM Backend (ECS 2vCPU/4GB)

Supports three test modes:
1. TEXT: Standard text-based LLM calls (existing)
2. STT: Speech-to-Text - takes local audio file path
3. TTS: Text-to-Speech - reads from CSV 'question' column

Usage:
    # Text mode
    python concurrency.py --mode text --csv requests_data.csv --workers 10 --duration 600

    # STT mode (audio file path required)
    python concurrency.py --mode stt --audio-file audio.wav --workers 5 --duration 300

    # TTS mode (reads from CSV 'question' column)
    python concurrency.py --mode tts --csv requests_data.csv --workers 8 --duration 600

    # MIXED mode (realistic production workload)
    python concurrency.py --mode mixed --csv requests_data.csv --audio-file audio.wav \
        --workers 15 --duration 600 --rate 60 \
        --text-ratio 0.7 --tts-ratio 0.2 --stt-ratio 0.1 \
        --pattern bursty

Metrics captured to identify ECS resource saturation:
- latency_ms: end-to-end request time
- ttfb_ms: time to first byte (if streaming)
- queue_wait_ms: time between task submission and start (Celery backpressure)
- worker_memory_mb: process memory at request time
- cpu_percent: CPU usage at request time
- real_time_factor: latency_ms / (audio_duration_s * 1000) [STT only]
- concurrency_level: active workers at request time

Memory pressure threshold: 3276 MB (80% of 4096 MB ECS limit)
"""

import argparse
import csv
import json
import os
import random
import time
import psutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock
from dataclasses import dataclass, asdict
import wave
import contextlib

# API Configuration
API_URL = "https://api-staging.kaapi.ai/api/v1/llm/call"
API_KEY = "ApiKeyi8"
CALLBACK_URL = "httpv"

# ECS Resource Limits
ECS_MEMORY_MB = 4096
ECS_MEMORY_PRESSURE_THRESHOLD = int(ECS_MEMORY_MB * 0.8)  # 3276 MB
ECS_CPU_COUNT = 2

# Global metrics lock
metrics_lock = Lock()
active_workers = 0


@dataclass
class RequestMetrics:
    """Metrics captured for each request"""

    request_id: int
    test_type: str  # 'text', 'stt', 'tts'
    timestamp: int
    config_id: str

    # Timing metrics
    latency_ms: float
    ttfb_ms: Optional[float] = None
    queue_wait_ms: Optional[float] = None

    # Resource metrics
    worker_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    concurrency_level: int = 0

    # Audio-specific metrics
    audio_duration_s: Optional[float] = None
    audio_file_size_kb: Optional[float] = None
    input_text_length: Optional[int] = None
    real_time_factor: Optional[float] = None

    # Response metrics
    success: bool = False
    status_code: int = 0
    error_type: Optional[str] = None
    response_size_bytes: Optional[int] = None

    # Flags
    memory_pressure: bool = False


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Get duration of audio file in seconds"""
    try:
        with contextlib.closing(wave.open(audio_path, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Warning: Could not get audio duration for {audio_path}: {e}")
        return None


def load_csv_data(csv_file_path: str, max_requests: int) -> List[Dict]:
    """Load data from CSV file containing config_id and question columns"""
    data = []
    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({"config_id": row["config_id"], "question": row["question"]})

    # Adjust size to exactly max_requests
    if len(data) < max_requests:
        missing = max_requests - len(data)
        extra_rows = random.choices(data, k=missing)
        data.extend(extra_rows)
    elif len(data) > max_requests:
        data = random.sample(data, max_requests)

    random.shuffle(data)
    return data


def create_text_payload(config_id: str, question: str, request_id: int) -> Dict:
    """Create payload for text-based LLM call"""
    return {
        "query": {"input": question},
        "config": {"id": config_id, "version": 1},
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {
            "timestamp": int(time.time()),
            "request_id": request_id,
            "test_type": "text",
        },
    }


def create_stt_payload(config_id: str, audio_path: str, request_id: int) -> Dict:
    """Create payload for STT (Speech-to-Text) call"""
    return {
        "query": {"input": audio_path},  # Local file path
        "config": {"id": config_id, "version": 1},
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {
            "timestamp": int(time.time()),
            "request_id": request_id,
            "test_type": "stt",
            "audio_path": audio_path,
        },
    }


def create_tts_payload(config_id: str, text: str, request_id: int) -> Dict:
    """Create payload for TTS (Text-to-Speech) call"""
    return {
        "query": {"input": text},
        "config": {"id": config_id, "version": 1},
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {
            "timestamp": int(time.time()),
            "request_id": request_id,
            "test_type": "tts",
            "input_text_length": len(text),
        },
    }


def assign_modality_mix(
    num_requests: int,
    text_ratio: float = 0.7,
    tts_ratio: float = 0.2,
    stt_ratio: float = 0.1,
) -> List[str]:
    """
    Assign modality types to requests based on specified ratios.
    Returns a randomized list of modality assignments.

    Args:
        num_requests: Total number of requests
        text_ratio: Proportion of text requests (default: 70%)
        tts_ratio: Proportion of TTS requests (default: 20%)
        stt_ratio: Proportion of STT requests (default: 10%)
    """
    # Validate ratios sum to 1.0
    total_ratio = text_ratio + tts_ratio + stt_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Calculate counts
    text_count = int(num_requests * text_ratio)
    tts_count = int(num_requests * tts_ratio)
    stt_count = num_requests - text_count - tts_count  # Remainder goes to STT

    # Create assignments
    assignments = ["text"] * text_count + ["tts"] * tts_count + ["stt"] * stt_count

    # Shuffle to randomize order
    random.shuffle(assignments)

    return assignments


def distribute_requests_bursty(
    total_requests: int, duration_seconds: int
) -> List[float]:
    """
    Create a bursty traffic pattern with random spikes and quiet periods.
    Simulates realistic production load with variable intensity.
    """
    schedule = []

    # Divide duration into 30-second windows
    window_size = 30
    num_windows = int(duration_seconds / window_size)

    requests_per_window = total_requests // num_windows

    for window_idx in range(num_windows):
        window_start = window_idx * window_size
        window_end = window_start + window_size

        # Randomly decide if this window is a burst (30% chance)
        is_burst = random.random() < 0.3

        if is_burst:
            # Burst: 2-3x normal rate, concentrated in first half of window
            burst_multiplier = random.uniform(2.0, 3.0)
            burst_requests = int(requests_per_window * burst_multiplier)
            burst_duration = window_size / 2

            for _ in range(burst_requests):
                if len(schedule) >= total_requests:
                    break
                schedule.append(window_start + random.uniform(0, burst_duration))
        else:
            # Normal: evenly distributed
            for _ in range(requests_per_window):
                if len(schedule) >= total_requests:
                    break
                schedule.append(random.uniform(window_start, window_end))

    # Fill remaining requests if any
    while len(schedule) < total_requests:
        schedule.append(random.uniform(0, duration_seconds))

    schedule.sort()
    return schedule[:total_requests]


def distribute_requests_rampup(
    total_requests: int, duration_seconds: int
) -> List[float]:
    """
    Create a gradual ramp-up pattern starting slow and increasing to target rate.
    Useful for testing how system handles increasing load.
    """
    schedule = []

    # Ramp up over first 25% of duration, then sustain
    rampup_duration = duration_seconds * 0.25
    sustain_duration = duration_seconds * 0.75

    # 20% of requests during ramp-up, 80% during sustain
    rampup_requests = int(total_requests * 0.2)
    sustain_requests = total_requests - rampup_requests

    # Ramp-up phase: quadratic distribution (slow start, faster end)
    for i in range(rampup_requests):
        progress = (i / rampup_requests) ** 2  # Quadratic curve
        schedule.append(progress * rampup_duration)

    # Sustain phase: uniform distribution
    for _ in range(sustain_requests):
        schedule.append(rampup_duration + random.uniform(0, sustain_duration))

    schedule.sort()
    return schedule


def distribute_requests_spike(
    total_requests: int, duration_seconds: int
) -> List[float]:
    """
    Create a spike pattern: normal load with periodic sharp spikes.
    Simulates events like batch job triggers or user activity surges.
    """
    schedule = []

    # Place 3-5 spikes throughout the test
    num_spikes = random.randint(3, 5)
    spike_requests = int(total_requests * 0.4)  # 40% of requests in spikes
    normal_requests = total_requests - spike_requests

    # Generate spike times
    spike_times = sorted(
        random.sample(range(30, duration_seconds - 30, 30), num_spikes)
    )
    requests_per_spike = spike_requests // num_spikes

    # Add spike requests (concentrated in 5-second windows)
    for spike_time in spike_times:
        for _ in range(requests_per_spike):
            schedule.append(spike_time + random.uniform(0, 5))

    # Add normal requests (distributed throughout)
    for _ in range(normal_requests):
        schedule.append(random.uniform(0, duration_seconds))

    schedule.sort()
    return schedule[:total_requests]


def distribute_requests_uniform(
    total_requests: int, minutes: int, per_minute: int
) -> List[float]:
    """
    Create a uniform random schedule of when to send each request.
    Returns a sorted list of timestamps (in seconds from start) for each request.
    """
    schedule = []

    for minute in range(minutes):
        # Generate random times within this minute for the requests
        minute_start = minute * 60
        minute_end = minute_start + 60

        # Generate random timestamps within this minute
        for _ in range(per_minute):
            random_time = random.uniform(minute_start, minute_end)
            schedule.append(random_time)

    # Sort the schedule so requests are sent in order
    schedule.sort()
    return schedule


def send_request(
    payload: Dict,
    request_id: int,
    test_type: str,
    audio_path: Optional[str] = None,
    input_text: Optional[str] = None,
) -> RequestMetrics:
    """Send a single API request and capture detailed metrics"""
    global active_workers

    # Start timing and resource monitoring
    start_time = time.time()
    queue_submit_time = time.time()

    # Capture resource usage at request start
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    worker_memory_mb = memory_info.rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=0.1)

    # Track active workers
    with metrics_lock:
        active_workers += 1
        current_concurrency = active_workers

    # Audio-specific metrics
    audio_duration_s = None
    audio_file_size_kb = None
    if test_type == "stt" and audio_path:
        audio_duration_s = get_audio_duration(audio_path)
        if os.path.exists(audio_path):
            audio_file_size_kb = os.path.getsize(audio_path) / 1024

    input_text_length = None
    if test_type == "tts" and input_text:
        input_text_length = len(input_text)

    # Prepare headers
    config_id = payload["config"]["id"]
    if config_id == "0b558ecb-1cd1-4eb7-9714-a3036e0da908":
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": "ApiKey Special",
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-KEY": API_KEY,
        }

    provider_timestamp = payload["request_metadata"]["timestamp"]
    queue_wait_ms = (time.time() - queue_submit_time) * 1000

    print(
        f"[{test_type.upper()}] Sending request {request_id} | "
        f"Memory: {worker_memory_mb:.1f}MB | CPU: {cpu_percent:.1f}% | "
        f"Workers: {current_concurrency}"
    )

    # Initialize metrics
    metrics = RequestMetrics(
        request_id=request_id,
        test_type=test_type,
        timestamp=provider_timestamp,
        config_id=config_id,
        latency_ms=0,
        queue_wait_ms=queue_wait_ms,
        worker_memory_mb=worker_memory_mb,
        cpu_percent=cpu_percent,
        concurrency_level=current_concurrency,
        audio_duration_s=audio_duration_s,
        audio_file_size_kb=audio_file_size_kb,
        input_text_length=input_text_length,
        memory_pressure=(worker_memory_mb > ECS_MEMORY_PRESSURE_THRESHOLD),
    )

    try:
        # Send request with streaming to capture TTFB
        response = requests.post(
            API_URL, json=payload, headers=headers, timeout=120, stream=True
        )

        # Capture TTFB (time to first byte)
        ttfb_time = time.time()
        metrics.ttfb_ms = (ttfb_time - start_time) * 1000

        # Read full response
        response_content = response.content
        end_time = time.time()

        # Calculate latency
        metrics.latency_ms = (end_time - start_time) * 1000
        metrics.status_code = response.status_code
        metrics.success = response.status_code == 200
        metrics.response_size_bytes = len(response_content)

        # Calculate real-time factor for STT
        if test_type == "stt" and audio_duration_s and audio_duration_s > 0:
            metrics.real_time_factor = metrics.latency_ms / (audio_duration_s * 1000)

        if not metrics.success:
            metrics.error_type = f"HTTP_{response.status_code}"

    except requests.exceptions.Timeout:
        end_time = time.time()
        metrics.latency_ms = (end_time - start_time) * 1000
        metrics.status_code = 0
        metrics.success = False
        metrics.error_type = "TIMEOUT"
        print(f"Request {request_id} timed out after {metrics.latency_ms:.0f}ms")

    except requests.exceptions.ConnectionError as e:
        end_time = time.time()
        metrics.latency_ms = (end_time - start_time) * 1000
        metrics.status_code = 0
        metrics.success = False
        metrics.error_type = "CONNECTION_ERROR"
        print(f"Request {request_id} connection error: {str(e)[:100]}")

    except Exception as e:
        end_time = time.time()
        metrics.latency_ms = (end_time - start_time) * 1000
        metrics.status_code = 0
        metrics.success = False
        metrics.error_type = f"EXCEPTION_{type(e).__name__}"
        print(f"Request {request_id} failed: {str(e)[:100]}")

    finally:
        # Decrement active workers
        with metrics_lock:
            active_workers -= 1

    return metrics


def worker_task(
    task_data: Dict, request_id: int, test_type: str, audio_path: Optional[str] = None
) -> RequestMetrics:
    """Worker function executed by thread pool"""
    if test_type == "text":
        payload = create_text_payload(
            config_id=task_data["config_id"],
            question=task_data["question"],
            request_id=request_id,
        )
        return send_request(payload, request_id, test_type)

    elif test_type == "stt":
        if not audio_path:
            raise ValueError("STT mode requires audio_path")
        payload = create_stt_payload(
            config_id=task_data.get("config_id", "default-stt-config"),
            audio_path=audio_path,
            request_id=request_id,
        )
        return send_request(payload, request_id, test_type, audio_path=audio_path)

    elif test_type == "tts":
        text = task_data["question"]
        payload = create_tts_payload(
            config_id=task_data["config_id"], text=text, request_id=request_id
        )
        return send_request(payload, request_id, test_type, input_text=text)

    else:
        raise ValueError(f"Unknown test type: {test_type}")


def run_load_test(
    test_type: str,
    max_workers: int,
    duration_seconds: int,
    csv_file: Optional[str] = None,
    audio_file: Optional[str] = None,
    requests_per_minute: int = 50,
    traffic_pattern: str = "uniform",
    text_ratio: float = 0.7,
    tts_ratio: float = 0.2,
    stt_ratio: float = 0.1,
) -> List[RequestMetrics]:
    """
    Run concurrent load test with specified parameters

    Args:
        test_type: 'text', 'stt', 'tts', or 'mixed'
        max_workers: Number of concurrent workers
        duration_seconds: Test duration in seconds
        csv_file: Path to CSV file (for text/tts/mixed modes)
        audio_file: Path to audio file (for stt/mixed modes)
        requests_per_minute: Target request rate
        traffic_pattern: 'uniform', 'bursty', 'rampup', or 'spike'
        text_ratio: Proportion of text requests in mixed mode (default: 0.7)
        tts_ratio: Proportion of TTS requests in mixed mode (default: 0.2)
        stt_ratio: Proportion of STT requests in mixed mode (default: 0.1)
    """
    print("=" * 80)
    print(f"MULTIMODAL LOAD TEST - {test_type.upper()} MODE")
    print("=" * 80)
    print(f"ECS Configuration: {ECS_CPU_COUNT} vCPU, {ECS_MEMORY_MB} MB RAM")
    print(f"Memory Pressure Threshold: {ECS_MEMORY_PRESSURE_THRESHOLD} MB")
    print(f"Workers: {max_workers}")
    print(f"Duration: {duration_seconds}s ({duration_seconds/60:.1f} minutes)")
    print(f"Target Rate: {requests_per_minute} req/min")
    print(f"Traffic Pattern: {traffic_pattern}")
    if test_type == "mixed":
        print(
            f"Modality Mix: TEXT={text_ratio*100:.0f}% | TTS={tts_ratio*100:.0f}% | STT={stt_ratio*100:.0f}%"
        )
    print("-" * 80)

    # Validate inputs
    if test_type in ["text", "tts", "mixed"] and not csv_file:
        raise ValueError(f"{test_type.upper()} mode requires --csv-file")

    if test_type in ["stt", "mixed"] and not audio_file:
        raise ValueError(f"{test_type.upper()} mode requires --audio-file")

    if test_type in ["stt", "mixed"] and audio_file and not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Calculate total requests
    total_minutes = duration_seconds / 60
    total_requests = int(requests_per_minute * total_minutes)

    # Load data
    if test_type in ["text", "tts", "mixed"]:
        if not csv_file:
            raise ValueError(f"{test_type.upper()} mode requires csv_file")
        print(f"Loading data from {csv_file}...")
        data = load_csv_data(csv_file, total_requests)
        print(f"Loaded {len(data)} records from CSV")
    else:  # stt
        # For STT, create dummy data entries (we'll use the same audio file)
        data = [
            {"config_id": "default-stt-config", "question": ""}
            for _ in range(total_requests)
        ]
        print(f"Prepared {len(data)} STT requests for {audio_file}")
        if audio_file:
            duration = get_audio_duration(audio_file)
            if duration:
                print(f"Audio duration: {duration:.2f}s")

    # Assign modality mix for mixed mode
    modality_assignments = None
    if test_type == "mixed":
        print(f"\nAssigning modality mix...")
        modality_assignments = assign_modality_mix(
            total_requests,
            text_ratio=text_ratio,
            tts_ratio=tts_ratio,
            stt_ratio=stt_ratio,
        )
        text_count = modality_assignments.count("text")
        tts_count = modality_assignments.count("tts")
        stt_count = modality_assignments.count("stt")
        print(f"Assigned: {text_count} text, {tts_count} TTS, {stt_count} STT requests")

    # Create schedule based on traffic pattern
    print(
        f"\nCreating {traffic_pattern} traffic schedule for {total_requests} requests..."
    )
    if traffic_pattern == "uniform":
        schedule = distribute_requests_uniform(
            total_requests, int(total_minutes), requests_per_minute
        )
    elif traffic_pattern == "bursty":
        schedule = distribute_requests_bursty(total_requests, duration_seconds)
    elif traffic_pattern == "rampup":
        schedule = distribute_requests_rampup(total_requests, duration_seconds)
    elif traffic_pattern == "spike":
        schedule = distribute_requests_spike(total_requests, duration_seconds)
    else:
        raise ValueError(f"Unknown traffic pattern: {traffic_pattern}")

    # Execute load test
    results: List[RequestMetrics] = []
    start_time = time.time()

    print(f"\nStarting load test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i, (task_data, scheduled_time) in enumerate(zip(data, schedule), start=1):
            # Wait until scheduled time
            current_elapsed = time.time() - start_time
            wait_time = scheduled_time - current_elapsed

            if wait_time > 0:
                time.sleep(wait_time)

            # Determine modality for this request
            current_test_type = test_type
            if test_type == "mixed" and modality_assignments:
                current_test_type = modality_assignments[i - 1]

            # Submit task
            future = executor.submit(
                worker_task,
                task_data=task_data,
                request_id=i,
                test_type=current_test_type,
                audio_path=audio_file if current_test_type == "stt" else None,
            )
            futures[future] = i

        # Collect results as they complete
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                metrics = future.result()
                results.append(metrics)

                # Print progress every 50 requests
                if len(results) % 50 == 0:
                    elapsed_minutes = (time.time() - start_time) / 60
                    success_count = sum(1 for r in results if r.success)
                    memory_pressure_count = sum(1 for r in results if r.memory_pressure)
                    avg_latency = sum(r.latency_ms for r in results[-50:]) / 50

                    print(
                        f"Progress: {len(results)}/{total_requests} | "
                        f"Elapsed: {elapsed_minutes:.1f}min | "
                        f"Success: {success_count} | "
                        f"Avg Latency: {avg_latency:.0f}ms | "
                        f"Memory Pressure: {memory_pressure_count}"
                    )
            except Exception as e:
                print(f"Error processing request {request_id}: {e}")

    end_time = time.time()
    total_duration = end_time - start_time

    # Print summary
    print("-" * 80)
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total requests: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.success)}")
    print(f"Failed: {sum(1 for r in results if not r.success)}")
    print(f"Memory pressure events: {sum(1 for r in results if r.memory_pressure)}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration / 60:.2f} minutes)")
    print(f"Actual rate: {len(results) / (total_duration / 60):.2f} requests/minute")

    if results:
        latencies = [r.latency_ms for r in results if r.success]
        if latencies:
            print(f"\nLatency Stats:")
            print(f"  Min: {min(latencies):.0f}ms")
            print(f"  Max: {max(latencies):.0f}ms")
            print(f"  Mean: {sum(latencies)/len(latencies):.0f}ms")
            print(f"  Median: {sorted(latencies)[len(latencies)//2]:.0f}ms")

        if test_type == "stt":
            rtfs = [r.real_time_factor for r in results if r.real_time_factor]
            if rtfs:
                print(f"\nReal-Time Factor (STT):")
                print(f"  Min: {min(rtfs):.2f}x")
                print(f"  Max: {max(rtfs):.2f}x")
                print(f"  Mean: {sum(rtfs)/len(rtfs):.2f}x")

    print("=" * 80)

    return results


def save_results(results: List[RequestMetrics], output_file: str):
    """Save detailed metrics to JSON file"""
    output_data = {
        "summary": {
            "total_requests": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "memory_pressure_events": sum(1 for r in results if r.memory_pressure),
            "test_type": results[0].test_type if results else None,
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed metrics saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Load Testing Framework for LLM Backend"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["text", "stt", "tts", "mixed"],
        help="Test mode: text, stt (speech-to-text), tts (text-to-speech), or mixed (realistic production workload)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Test duration in seconds (default: 600s = 10min)",
    )
    parser.add_argument(
        "--rate", type=int, default=50, help="Target requests per minute (default: 50)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with config_id and question columns (required for text/tts/mixed modes)",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file (required for stt/mixed modes)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="load_test_results.json",
        help="Output JSON file for detailed metrics (default: load_test_results.json)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="uniform",
        choices=["uniform", "bursty", "rampup", "spike"],
        help="Traffic pattern: uniform (steady), bursty (random spikes), rampup (gradual increase), spike (periodic surges)",
    )
    parser.add_argument(
        "--text-ratio",
        type=float,
        default=0.7,
        help="Proportion of text requests in mixed mode (default: 0.7 = 70%%)",
    )
    parser.add_argument(
        "--tts-ratio",
        type=float,
        default=0.2,
        help="Proportion of TTS requests in mixed mode (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--stt-ratio",
        type=float,
        default=0.1,
        help="Proportion of STT requests in mixed mode (default: 0.1 = 10%%)",
    )

    args = parser.parse_args()

    try:
        results = run_load_test(
            test_type=args.mode,
            max_workers=args.workers,
            duration_seconds=args.duration,
            csv_file=args.csv,
            audio_file=args.audio_file,
            requests_per_minute=args.rate,
            traffic_pattern=args.pattern,
            text_ratio=args.text_ratio,
            tts_ratio=args.tts_ratio,
            stt_ratio=args.stt_ratio,
        )

        save_results(results, args.output)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
