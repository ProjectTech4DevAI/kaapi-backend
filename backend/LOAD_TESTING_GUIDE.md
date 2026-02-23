# Multimodal Load Testing Guide

## Overview

This load testing framework supports stress testing for three modalities:
- **TEXT**: Standard text-based LLM calls
- **STT**: Speech-to-Text (audio file → text)
- **TTS**: Text-to-Speech (text → audio)

The framework captures detailed metrics to identify ECS resource saturation on a **2 vCPU / 4GB** shared task with Celery + FastAPI services.

## Architecture

```
┌─────────────────┐         ┌──────────────┐         ┌─────────────┐
│  concurrency.py │────────>│ ECS Backend  │────────>│ callback.py │
│  (Load Test)    │  HTTP   │ 2vCPU/4GB   │ Webhook │  (Receiver) │
└─────────────────┘         └──────────────┘         └─────────────┘
         │                                                    │
         │ Generates                                          │ Generates
         v                                                    v
 load_test_results.json                          callback_responses.json
         │                                                    │
         └────────────────────┬───────────────────────────────┘
                              v
                        analyze.py
                              │
                              v
                    Analysis Report (text/json/csv)
```

## Quick Start

### 1. Set Up Callback Server

Start the webhook receiver to capture server responses:

```bash
# Terminal 1: Start callback server
python callback.py

# Server runs on http://localhost:8001
# Logs callbacks to callback_responses.json
```

### 2. Run Load Tests

#### Text Mode (Existing)

```bash
python concurrency.py \
  --mode text \
  --csv requests_data.csv \
  --workers 10 \
  --duration 600 \
  --rate 50 \
  --output text_load_test.json
```

#### STT Mode (Speech-to-Text)

```bash
python concurrency.py \
  --mode stt \
  --audio-file /path/to/audio.wav \
  --workers 5 \
  --duration 300 \
  --rate 30 \
  --output stt_load_test.json
```

**Requirements**:
- Audio file must be accessible on local filesystem
- Supported formats: WAV, MP3, etc. (depends on provider)
- Same audio file is used for all STT requests

#### TTS Mode (Text-to-Speech)

```bash
python concurrency.py \
  --mode tts \
  --csv requests_data.csv \
  --workers 8 \
  --duration 600 \
  --rate 40 \
  --output tts_load_test.json
```

**Requirements**:
- CSV must have `question` column (text to synthesize)
- CSV must have `config_id` column (TTS config identifier)

### 3. Analyze Results

```bash
# Text output to console
python analyze.py load_test_results.json

# Save report to file
python analyze.py load_test_results.json --output report.txt

# JSON output for programmatic analysis
python analyze.py load_test_results.json --format json --output report.json

# Export detailed metrics to CSV
python analyze.py load_test_results.json --format csv --output metrics.csv
```

## Metrics Captured

### Request-Level Metrics

| Metric | Description | Unit | Use Case |
|--------|-------------|------|----------|
| `latency_ms` | End-to-end request time | ms | Identify slow requests |
| `ttfb_ms` | Time to first byte | ms | Detect network/routing issues |
| `queue_wait_ms` | Time in queue before processing | ms | Celery backpressure indicator |
| `worker_memory_mb` | Worker process memory | MB | Memory saturation detection |
| `cpu_percent` | CPU usage at request time | % | CPU saturation detection |
| `concurrency_level` | Active workers at request time | count | Concurrency analysis |
| `success` | Request succeeded | boolean | Success rate calculation |
| `error_type` | Error classification | string | Failure mode analysis |

### Audio-Specific Metrics

#### STT (Speech-to-Text)
| Metric | Description | Formula |
|--------|-------------|---------|
| `audio_duration_s` | Input audio duration | Extracted from WAV |
| `audio_file_size_kb` | Input file size | File size / 1024 |
| `real_time_factor` | Processing speed vs real-time | `latency_ms / (audio_duration_s * 1000)` |

**Real-Time Factor Interpretation**:
- `< 1.0x`: Faster than real-time (ideal)
- `1.0x - 2.0x`: Acceptable performance
- `> 2.0x`: Slow processing (potential issue)

#### TTS (Text-to-Speech)
| Metric | Description |
|--------|-------------|
| `input_text_length` | Character count of input text |
| `response_size_bytes` | Size of generated audio |

### Resource Saturation Indicators

#### Memory Pressure
- **Threshold**: 3276 MB (80% of 4096 MB ECS limit)
- **Flag**: `memory_pressure = true` when `worker_memory_mb > 3276`
- **Critical**: OOM kill occurs at ~4096 MB (hard limit)

#### CPU Saturation
- **Warning**: CPU > 80% sustained
- **Critical**: CPU maxed at 200% (2 vCPU)

#### Celery Queue Backpressure
- **Warning**: `queue_wait_ms > 1000` (>1s wait)
- **Indicator**: Workers can't keep up with request rate

## Analysis Report Sections

### 1. Test Summary
- Total requests, success rate, memory pressure events

### 2. Latency Analysis (by test type)
- Min, max, mean, median, p90, p95, p99
- TTFB statistics (if available)

### 3. Real-Time Factor (STT only)
- RTF distribution (p50, p90, p95, p99)
- Count of requests < 1x (faster than real-time)
- Count of requests > 2x (slow processing)

### 4. Resource Saturation Analysis
- **Memory**: Usage stats, pressure events, warnings
- **CPU**: Usage stats, high CPU events
- **Warnings**: Auto-generated alerts for saturation

### 5. Queue Wait Time
- Distribution of queue wait times
- High wait event detection (>1s)
- Celery backpressure indicators

### 6. Concurrency Levels
- Worker concurrency distribution during test

### 7. Error Analysis
- Total failures, failure rate
- Error type breakdown (TIMEOUT, HTTP_XXX, etc.)

## Sample Workflows

### Finding Memory Exhaustion Point

```bash
# Run increasing worker counts until memory saturates
for workers in 5 10 15 20 25; do
  echo "Testing with $workers workers..."
  python concurrency.py \
    --mode text \
    --csv data.csv \
    --workers $workers \
    --duration 300 \
    --output "results_workers_${workers}.json"

  # Analyze
  python analyze.py "results_workers_${workers}.json" --output "report_workers_${workers}.txt"
done

# Compare memory_p99 across reports to find saturation point
```

### Comparing STT Performance Across Audio Lengths

```bash
# Test with different audio durations
python concurrency.py --mode stt --audio-file short_5s.wav --workers 10 --output stt_short.json
python concurrency.py --mode stt --audio-file medium_30s.wav --workers 10 --output stt_medium.json
python concurrency.py --mode stt --audio-file long_120s.wav --workers 10 --output stt_long.json

# Compare real_time_factor across files
python analyze.py stt_short.json | grep "RTF"
python analyze.py stt_medium.json | grep "RTF"
python analyze.py stt_long.json | grep "RTF"
```

### Identifying Optimal Request Rate

```bash
# Test different rates to find maximum sustainable throughput
for rate in 30 40 50 60 70; do
  python concurrency.py \
    --mode tts \
    --csv data.csv \
    --workers 10 \
    --rate $rate \
    --duration 600 \
    --output "results_rate_${rate}.json"

  python analyze.py "results_rate_${rate}.json"
done

# Look for:
# - Increasing failure_rate
# - Increasing queue_wait_ms
# - Memory pressure events
# These indicate you've exceeded capacity
```

## CSV File Format

### For TEXT and TTS modes

```csv
config_id,question
0b558ecb-1cd1-4eb7-9714-a3036e0da908,"What is the capital of France?"
a1b2c3d4-5e6f-7890-abcd-ef1234567890,"Explain quantum computing in simple terms."
```

**Required columns**:
- `config_id`: LLM configuration identifier
- `question`: Input text (prompt for TEXT, text to synthesize for TTS)

## Callback Server Schema

The callback server (`callback.py`) receives responses with this structure:

```json
{
  "timestamp": 1704067200,
  "test_type": "stt",
  "request_id": 42,
  "headers": {...},
  "payload": {
    "metadata": {
      "request_id": 42,
      "test_type": "stt",
      "timestamp": 1704067200
    },
    "data": {
      "response": {...},
      "usage": {...}
    }
  }
}
```

**Audio handling**: For TTS responses, base64 audio data is replaced with `<base64_audio_data:N_bytes>` to keep log files manageable. The `_audio_size_bytes` field preserves size information.

## Interpreting Results

### Healthy System
- Success rate > 99%
- Memory pressure events: 0%
- Queue wait p99 < 500ms
- CPU p99 < 150% (75% per core)
- Real-time factor (STT) p90 < 1.5x

### Memory Saturation
- Memory pressure events > 10%
- Max memory approaching 4096 MB
- OOM kills in logs
- **Action**: Reduce workers or increase ECS memory

### CPU Saturation
- CPU p99 > 180% (90% per core)
- High queue wait times
- Increasing latency under load
- **Action**: Reduce workers or increase ECS CPU

### Celery Queue Backpressure
- Queue wait p99 > 1000ms
- High queue wait rate
- **Action**: Check RabbitMQ, increase Celery workers, or reduce request rate

### Network/API Issues
- High TTFB relative to latency
- Many TIMEOUT errors
- **Action**: Check network, API timeouts, or upstream service health

## Tips

1. **Start conservative**: Begin with low worker counts (5-10) and gradually increase
2. **Monitor ECS metrics**: Use AWS CloudWatch to correlate with test results
3. **Run during off-hours**: Avoid impacting production traffic
4. **Test one modality at a time**: Isolate STT/TTS/text to understand each
5. **Capture baseline**: Run a low-load baseline test before stress testing
6. **Watch RabbitMQ**: Monitor queue depth and consumer count on EC2
7. **Use realistic data**: Audio files and text should match production patterns

## Troubleshooting

### "Request timed out after Xms"
- Increase `timeout` in `send_request()` function (default: 120s)
- Check if backend is responding

### "Audio file not found"
- Ensure audio file path is absolute or relative to script location
- Verify file exists: `ls -la /path/to/audio.wav`

### "CSV file not found"
- Check CSV path is correct
- Verify CSV has `config_id` and `question` columns

### High memory usage in test script itself
- Reduce worker count
- Process is tracking active requests - this is expected

### Callback server not receiving responses
- Verify `CALLBACK_URL` in `concurrency.py` is correct
- Check firewall/network allows backend to reach callback server
- Ensure callback server is running (`python callback.py`)

## Integration with CI/CD

```bash
#!/bin/bash
# Example: Automated load test in CI

# Start callback server in background
python callback.py &
CALLBACK_PID=$!

# Run load test
python concurrency.py \
  --mode text \
  --csv test_data.csv \
  --workers 10 \
  --duration 300 \
  --output results.json

# Analyze
python analyze.py results.json --format json --output report.json

# Parse report for failures
FAILURE_RATE=$(cat report.json | jq '.summary.failed / .summary.total_requests * 100')

if (( $(echo "$FAILURE_RATE > 5" | bc -l) )); then
  echo "FAIL: Failure rate ${FAILURE_RATE}% exceeds 5% threshold"
  kill $CALLBACK_PID
  exit 1
fi

# Cleanup
kill $CALLBACK_PID
echo "PASS: Load test successful"
```

## Advanced Configuration

### Modify API endpoint

Edit `concurrency.py`:
```python
API_URL = "https://your-api-endpoint.com/api/v1/llm/call"
API_KEY = "your-api-key"
CALLBACK_URL = "http://your-callback-server:8001"
```

### Adjust resource thresholds

Edit constants in `concurrency.py` and `analyze.py`:
```python
ECS_MEMORY_MB = 8192  # If you have 8GB ECS task
ECS_MEMORY_PRESSURE_THRESHOLD = int(ECS_MEMORY_MB * 0.8)
ECS_CPU_COUNT = 4  # If you have 4 vCPU
```

### Change request distribution

Modify `distribute_requests_randomly()` in `concurrency.py` for custom scheduling patterns (e.g., spike testing, gradual ramp-up).

## Files Reference

| File | Purpose |
|------|---------|
| `concurrency.py` | Load test execution (generates metrics) |
| `callback.py` | Webhook server (receives server responses) |
| `analyze.py` | Metrics analysis and reporting |
| `load_test_results.json` | Client-side metrics output |
| `callback_responses.json` | Server-side response log |
| `requests_data.csv` | Input data (TEXT/TTS modes) |

## Support

For issues or questions:
1. Check CloudWatch logs for backend errors
2. Review RabbitMQ management UI for queue issues
3. Verify ECS task metrics in AWS Console
4. Check Celery worker logs

---

**Last Updated**: 2026-02-23
