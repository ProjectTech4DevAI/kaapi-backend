# Mixed Mode & Traffic Patterns Guide

## Overview

The load testing framework now supports **realistic production workloads** with:

1. **Mixed Mode**: Combine text, STT, and TTS requests in a single test
2. **Traffic Patterns**: Simulate realistic load patterns (bursty, rampup, spike)
3. **Configurable Ratios**: Control the proportion of each modality

This addresses the reality that **production traffic is never homogeneous** - users generate a mix of text queries, voice transcriptions, and speech synthesis requests with variable intensity.

---

## Quick Start: Mixed Mode

### Basic Mixed Mode Test

```bash
python concurrency.py \
  --mode mixed \
  --csv requests_data.csv \
  --audio-file audio.wav \
  --workers 15 \
  --duration 600 \
  --rate 60
```

**Default mix**: 70% text, 20% TTS, 10% STT

### Custom Ratios

```bash
python concurrency.py \
  --mode mixed \
  --csv requests_data.csv \
  --audio-file audio.wav \
  --workers 15 \
  --duration 600 \
  --rate 60 \
  --text-ratio 0.5 \
  --tts-ratio 0.3 \
  --stt-ratio 0.2
```

**Custom mix**: 50% text, 30% TTS, 20% STT

### With Traffic Pattern

```bash
python concurrency.py \
  --mode mixed \
  --csv requests_data.csv \
  --audio-file audio.wav \
  --workers 15 \
  --duration 600 \
  --rate 60 \
  --pattern bursty \
  --text-ratio 0.7 \
  --tts-ratio 0.2 \
  --stt-ratio 0.1
```

---

## Traffic Patterns

### 1. Uniform (Default)

**Behavior**: Evenly distributed requests over time
**Use case**: Baseline testing, steady-state performance

```bash
--pattern uniform
```

**Characteristics**:
- Requests spread evenly throughout test duration
- Consistent load on system
- Good for establishing baseline metrics

**Example**: 60 req/min = 1 request every second, uniformly distributed

---

### 2. Bursty

**Behavior**: Random spikes and quiet periods
**Use case**: Simulating real production traffic

```bash
--pattern bursty
```

**Characteristics**:
- Divides time into 30-second windows
- 30% of windows are "burst" windows
- Burst windows: 2-3x normal rate, concentrated in first 15 seconds
- Normal windows: evenly distributed
- **Most realistic for production**

**Example**:
- Normal window: 30 requests spread over 30s
- Burst window: 60-90 requests in first 15s, quiet for remaining 15s

---

### 3. Ramp-Up

**Behavior**: Gradual increase from low to target rate
**Use case**: Testing how system handles increasing load

```bash
--pattern rampup
```

**Characteristics**:
- First 25% of duration: slow ramp-up (20% of total requests)
- Remaining 75% of duration: sustained target rate (80% of total requests)
- Quadratic curve for ramp (starts very slow, accelerates)

**Example**: 10-minute test
- Minutes 0-2.5: Slowly increasing from ~5 req/min to target
- Minutes 2.5-10: Sustained at target 60 req/min

---

### 4. Spike

**Behavior**: Normal load with periodic sharp spikes
**Use case**: Simulating batch jobs, scheduled tasks, user surges

```bash
--pattern spike
```

**Characteristics**:
- 3-5 random spike events during test
- 40% of requests concentrated in 5-second spike windows
- 60% of requests distributed throughout test
- Spikes placed at 30-second boundaries (not at edges)

**Example**: 10-minute test
- 4 spikes at random times (e.g., 1:30, 3:00, 5:30, 8:00)
- Each spike: 100 requests in 5 seconds = 1200 req/min burst
- Background: steady 36 req/min between spikes

---

## Mixed Mode Requirements

### Required Files

1. **CSV file** (for text and TTS requests):
   ```csv
   config_id,question
   config-123,What is the weather?
   config-456,Tell me a joke
   ```

2. **Audio file** (for STT requests):
   - Any audio format supported by your provider (WAV, MP3, etc.)
   - Same audio file used for all STT requests
   - File must be accessible on local filesystem

### How It Works

1. **Request Generation**:
   - Total requests calculated: `requests_per_minute * duration_minutes`
   - Modality assigned to each request based on ratios
   - Assignments shuffled randomly

2. **Scheduling**:
   - Traffic pattern determines when each request is sent
   - Independent of modality assignment

3. **Execution**:
   - Requests sent according to schedule
   - Each request uses its assigned modality
   - Text/TTS: reads from CSV, uses `question` column
   - STT: uses provided audio file

---

## Example Use Cases

### Use Case 1: Production Simulation

**Scenario**: Your production traffic is 60% text queries, 25% TTS, 15% STT with bursty patterns

```bash
python concurrency.py \
  --mode mixed \
  --csv prod_questions.csv \
  --audio-file sample_voice.wav \
  --workers 20 \
  --duration 1800 \
  --rate 80 \
  --pattern bursty \
  --text-ratio 0.6 \
  --tts-ratio 0.25 \
  --stt-ratio 0.15 \
  --output prod_simulation.json
```

**Result**: 30-minute test with 2,400 requests (1,440 text, 600 TTS, 360 STT) with realistic burst patterns

---

### Use Case 2: Morning Rush Simulation

**Scenario**: Gradual traffic increase during morning hours

```bash
python concurrency.py \
  --mode mixed \
  --csv requests.csv \
  --audio-file audio.wav \
  --workers 25 \
  --duration 3600 \
  --rate 100 \
  --pattern rampup \
  --text-ratio 0.7 \
  --tts-ratio 0.2 \
  --stt-ratio 0.1 \
  --output morning_rush.json
```

**Result**: 1-hour test ramping from low traffic to 100 req/min

---

### Use Case 3: Batch Job Impact

**Scenario**: Testing impact of periodic batch job triggers

```bash
python concurrency.py \
  --mode mixed \
  --csv requests.csv \
  --audio-file audio.wav \
  --workers 30 \
  --duration 1200 \
  --rate 60 \
  --pattern spike \
  --text-ratio 0.8 \
  --tts-ratio 0.15 \
  --stt-ratio 0.05 \
  --output batch_impact.json
```

**Result**: 20-minute test with 3-5 sudden spikes simulating batch jobs

---

## Comparing Traffic Patterns

Run the same test with different patterns to understand system behavior:

```bash
# Baseline: uniform
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 15 --duration 600 --rate 60 --pattern uniform \
  --output baseline_uniform.json

# Realistic: bursty
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 15 --duration 600 --rate 60 --pattern bursty \
  --output realistic_bursty.json

# Stress: spike
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 15 --duration 600 --rate 60 --pattern spike \
  --output stress_spike.json

# Compare
python analyze.py baseline_uniform.json > report_uniform.txt
python analyze.py realistic_bursty.json > report_bursty.txt
python analyze.py stress_spike.json > report_spike.txt
```

**Look for**:
- Higher failure rates in spike pattern
- Higher p99 latency in bursty pattern
- Memory pressure events during spikes
- Queue wait times during bursts

---

## Finding Resource Saturation with Mixed Mode

### Step 1: Establish Baseline (Uniform)

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 10 --duration 600 --rate 50 --pattern uniform \
  --output baseline.json
```

### Step 2: Test Realistic Load (Bursty)

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 15 --duration 600 --rate 60 --pattern bursty \
  --output realistic.json
```

### Step 3: Push to Limits (Spike)

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 25 --duration 600 --rate 80 --pattern spike \
  --output stress.json
```

### Step 4: Analyze Results

```bash
python analyze.py stress.json
```

**Saturation indicators**:
- Memory pressure > 10% of requests
- CPU p99 > 180% (90% per core)
- Queue wait p99 > 1000ms
- Failure rate > 5%

---

## Tips for Mixed Mode Testing

### 1. Start Conservative

```bash
# Begin with low rate
--workers 10 --rate 40 --duration 300
```

Gradually increase workers and rate until saturation.

### 2. Match Production Ratios

Analyze your production metrics:
```sql
SELECT
  modality,
  COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM requests
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY modality;
```

Use actual ratios:
```bash
--text-ratio 0.65 --tts-ratio 0.22 --stt-ratio 0.13
```

### 3. Use Bursty Pattern by Default

Production traffic is never uniform:
```bash
--pattern bursty  # Most realistic
```

### 4. Test Multiple Patterns

Different patterns stress different system components:
- **Uniform**: Tests sustained throughput
- **Bursty**: Tests burst handling and queue management
- **Rampup**: Tests scaling behavior
- **Spike**: Tests absolute peak capacity

### 5. Monitor ECS Metrics

Correlate test results with AWS CloudWatch:
- ECS Task CPU Utilization
- ECS Task Memory Utilization
- RabbitMQ Queue Depth
- Celery Worker Count

---

## Analysis Tips for Mixed Mode

### Modality Breakdown

Mixed mode results include `test_type` field for each request:

```bash
python analyze.py mixed_results.json --format csv --output metrics.csv
```

Then analyze in Python:
```python
import pandas as pd

df = pd.read_csv('metrics.csv')

# Latency by modality
print(df.groupby('test_type')['latency_ms'].describe())

# Success rate by modality
print(df.groupby('test_type')['success'].mean())

# Memory pressure by modality
print(df.groupby('test_type')['memory_pressure'].sum())
```

### Finding Bottlenecks

```bash
# Extract high-latency requests
cat metrics.csv | awk -F',' '$8 > 5000' | head -20

# Count errors by type and modality
cat metrics.csv | awk -F',' '{ print $2,$14 }' | sort | uniq -c
```

---

## Troubleshooting

### "Ratios must sum to 1.0"

Ensure ratios add up to 1.0:
```bash
--text-ratio 0.7 --tts-ratio 0.2 --stt-ratio 0.1  # ✅ 0.7 + 0.2 + 0.1 = 1.0
--text-ratio 0.7 --tts-ratio 0.2 --stt-ratio 0.2  # ❌ 0.7 + 0.2 + 0.2 = 1.1
```

### "Mixed mode requires --audio-file"

Provide both CSV and audio:
```bash
python concurrency.py --mode mixed \
  --csv data.csv \
  --audio-file audio.wav \
  ...
```

### Inconsistent Results

Run multiple iterations:
```bash
for i in {1..5}; do
  python concurrency.py --mode mixed ... --output run_${i}.json
done

# Compare variance
python analyze.py run_*.json | grep "p99"
```

---

## Advanced: Custom Traffic Patterns

Edit `concurrency.py` to add custom patterns:

```python
def distribute_requests_custom(total_requests: int, duration_seconds: int) -> List[float]:
    """Your custom traffic pattern"""
    schedule = []
    # Your logic here
    return sorted(schedule)
```

Then use:
```bash
# After adding to choices in argparse
--pattern custom
```

---

## Summary

| Pattern | Best For | Characteristics |
|---------|----------|-----------------|
| **uniform** | Baseline | Steady, predictable load |
| **bursty** | Production simulation | Random spikes, realistic |
| **rampup** | Scaling tests | Gradual increase |
| **spike** | Peak capacity | Extreme bursts |

| Mode | Use Case |
|------|----------|
| **text** | Text-only workloads |
| **stt** | Voice transcription only |
| **tts** | Speech synthesis only |
| **mixed** | **Production-realistic** |

**Recommendation**: For most realistic testing, use:
```bash
--mode mixed --pattern bursty
```

---

**Last Updated**: 2026-02-23
