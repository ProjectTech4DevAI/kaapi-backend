# Implementation Summary: Multimodal Load Testing with Mixed Mode

## What Was Implemented

### ✅ Extended `concurrency.py` with:

1. **Mixed Mode Support**
   - Combines text, STT, and TTS requests in a single test
   - Configurable ratios (e.g., 70% text, 20% TTS, 10% STT)
   - Random assignment and shuffling for realistic distribution

2. **Traffic Pattern Options**
   - **uniform**: Evenly distributed (original behavior)
   - **bursty**: Random spikes and quiet periods (realistic production)
   - **rampup**: Gradual increase from low to target rate
   - **spike**: Periodic sharp bursts (simulates batch jobs)

3. **New Command-Line Arguments**
   ```bash
   --mode mixed              # New mode option
   --pattern bursty          # Traffic pattern selection
   --text-ratio 0.7          # Proportion of text requests
   --tts-ratio 0.2           # Proportion of TTS requests
   --stt-ratio 0.1           # Proportion of STT requests
   ```

### ✅ Updated `analyze.py`

- Already supports mixed results (test_type field exists)
- Analyzes metrics by modality automatically
- No changes needed - works out of the box

### ✅ Updated `callback.py`

- Enhanced logging with test_type and request_id
- Optimized for audio responses (truncates base64 data in logs)
- Preserves audio size information

### ✅ Documentation

- **LOAD_TESTING_GUIDE.md**: Original comprehensive guide
- **MIXED_MODE_GUIDE.md**: New guide for mixed mode and traffic patterns
- **IMPLEMENTATION_SUMMARY.md**: This file

### ✅ Helper Scripts

- **run_load_test.sh**: Updated with mixed mode support
- **requests_data_template.csv**: Sample CSV template

---

## Quick Start Examples

### Production-Realistic Test (Recommended)

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
  --stt-ratio 0.1 \
  --output prod_test.json
```

**Result**: 10-minute test with 600 requests (420 text, 120 TTS, 60 STT) with realistic burst patterns

### Using Helper Script

```bash
# Start callback server
python callback.py &

# Run mixed mode test
./run_load_test.sh mixed audio.wav

# Analyze results
python analyze.py mixed_load_test_*.json
```

---

## Traffic Pattern Comparison

| Pattern | Behavior | Use Case |
|---------|----------|----------|
| `uniform` | Evenly distributed over time | Baseline, steady-state testing |
| `bursty` | Random 2-3x spikes in 30s windows | **Production simulation** (recommended) |
| `rampup` | Gradual increase from low to target | Testing scaling behavior |
| `spike` | 3-5 sharp bursts during test | Peak capacity, batch job impact |

---

## Key Features Addressing Your Requirements

### ✅ Mixed Modality Support

> "I want to make sure the load testing supports real usecases where multimodal and text based usecases come randomly"

**Implementation**:
- `--mode mixed` combines all three modalities in one test
- Random shuffling ensures no predictable patterns
- Configurable ratios match your production distribution

**Example**:
```bash
# 60% text, 25% TTS, 15% STT
--mode mixed --text-ratio 0.6 --tts-ratio 0.25 --stt-ratio 0.15
```

### ✅ Bursty Traffic

> "It could be bursty as well. I think no real load consists of only text, stt and tts usecases"

**Implementation**:
- `--pattern bursty` creates realistic burst patterns
- 30% of time windows have 2-3x normal rate
- Remaining windows are evenly distributed
- Unpredictable spike timing

**Example**:
```bash
# Realistic bursty production load
--mode mixed --pattern bursty
```

### ✅ All Metrics Still Captured

- `test_type` field identifies each request's modality
- All existing metrics (CPU, memory, latency, etc.) still tracked
- Real-time factor still calculated for STT requests
- Analyze by modality or overall

---

## Changes Made to Files

| File | Changes |
|------|---------|
| `concurrency.py` | ✏️ **Modified** - Added mixed mode, traffic patterns, modality assignment |
| `analyze.py` | ✅ **No changes** - Already handles mixed results |
| `callback.py` | ✏️ **Modified** - Enhanced logging, audio optimization |
| `run_load_test.sh` | ✏️ **Modified** - Added mixed mode support |
| `MIXED_MODE_GUIDE.md` | ✨ **New** - Comprehensive guide for new features |
| `IMPLEMENTATION_SUMMARY.md` | ✨ **New** - This summary |
| `requests_data_template.csv` | ✨ **New** - Sample data file |

---

## Example Workflow: Finding ECS Saturation

### Step 1: Baseline (Uniform, Single Modality)

```bash
python concurrency.py --mode text --csv data.csv \
  --workers 10 --duration 300 --rate 50 --pattern uniform \
  --output baseline_text.json
```

### Step 2: Realistic Production (Mixed, Bursty)

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 15 --duration 600 --rate 60 --pattern bursty \
  --text-ratio 0.7 --tts-ratio 0.2 --stt-ratio 0.1 \
  --output prod_mixed.json
```

### Step 3: Stress Test (Higher Rate, Spikes)

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 25 --duration 600 --rate 100 --pattern spike \
  --text-ratio 0.7 --tts-ratio 0.2 --stt-ratio 0.1 \
  --output stress_spike.json
```

### Step 4: Analyze and Compare

```bash
python analyze.py prod_mixed.json > report_prod.txt
python analyze.py stress_spike.json > report_stress.txt

# Look for saturation indicators:
# - Memory pressure events
# - CPU p99 > 180%
# - Queue wait p99 > 1000ms
# - Failure rate increase
```

---

## Metrics Collected (Unchanged)

All original metrics still captured:

- **Latency**: `latency_ms`, `ttfb_ms`
- **Resources**: `worker_memory_mb`, `cpu_percent`
- **Queue**: `queue_wait_ms` (Celery backpressure)
- **Audio**: `audio_duration_s`, `real_time_factor` (STT)
- **Concurrency**: `concurrency_level`
- **Success**: `success`, `error_type`

**New field**: `test_type` (text/stt/tts) for mixed mode analysis

---

## Analyzing Mixed Mode Results

### Overall Statistics

```bash
python analyze.py mixed_results.json
```

Shows aggregate statistics across all modalities.

### Per-Modality Analysis

```bash
# Export to CSV
python analyze.py mixed_results.json --format csv --output metrics.csv

# Analyze in Python/pandas
import pandas as pd
df = pd.read_csv('metrics.csv')

# Latency by modality
print(df.groupby('test_type')['latency_ms'].describe())

# Memory pressure by modality
print(df.groupby('test_type')['memory_pressure'].value_counts())

# Success rate by modality
print(df.groupby('test_type')['success'].mean() * 100)
```

---

## Testing the Implementation

### 1. Single Modality (Original Behavior)

```bash
# These still work as before
python concurrency.py --mode text --csv data.csv --workers 10 --duration 300
python concurrency.py --mode stt --audio-file audio.wav --workers 5 --duration 300
python concurrency.py --mode tts --csv data.csv --workers 8 --duration 300
```

### 2. Mixed Mode (New)

```bash
python concurrency.py --mode mixed \
  --csv data.csv \
  --audio-file audio.wav \
  --workers 15 \
  --duration 300 \
  --rate 50 \
  --output mixed_test.json
```

### 3. Different Traffic Patterns

```bash
# Try all four patterns
for pattern in uniform bursty rampup spike; do
  python concurrency.py --mode mixed \
    --csv data.csv --audio-file audio.wav \
    --workers 15 --duration 300 --rate 50 \
    --pattern $pattern \
    --output test_${pattern}.json
done

# Compare results
for file in test_*.json; do
  echo "=== $file ===" python analyze.py $file | head -30
done
```

---

## Backwards Compatibility

✅ **All existing functionality preserved**:

- `--mode text/stt/tts` work exactly as before
- Default `--pattern uniform` maintains original behavior
- Existing scripts and workflows unaffected
- CSV format unchanged

**Migration**: No changes needed to existing tests

---

## Next Steps

### 1. Prepare Test Data

```bash
# Create CSV with production-like questions
cp requests_data_template.csv requests_data.csv
# Edit to add your actual test questions

# Get sample audio file
# Place in backend/ directory as audio.wav
```

### 2. Run Baseline Test

```bash
python callback.py &  # Start callback server

python concurrency.py --mode mixed \
  --csv requests_data.csv \
  --audio-file audio.wav \
  --workers 10 \
  --duration 300 \
  --rate 40 \
  --pattern bursty \
  --output baseline.json
```

### 3. Analyze Results

```bash
python analyze.py baseline.json

# Look for:
# - Memory pressure events
# - High queue wait times
# - CPU saturation
# - Failure patterns
```

### 4. Increase Load Until Saturation

```bash
# Gradually increase workers and rate
for workers in 10 15 20 25 30; do
  python concurrency.py --mode mixed \
    --csv requests_data.csv \
    --audio-file audio.wav \
    --workers $workers \
    --duration 300 \
    --rate 50 \
    --pattern bursty \
    --output saturation_${workers}w.json

  python analyze.py saturation_${workers}w.json | grep -E "Memory|CPU|Queue"
done
```

---

## Files Reference

### Core Scripts
- **concurrency.py**: Load test execution
- **callback.py**: Webhook server
- **analyze.py**: Metrics analysis

### Documentation
- **LOAD_TESTING_GUIDE.md**: Comprehensive guide (original)
- **MIXED_MODE_GUIDE.md**: Mixed mode & traffic patterns guide (new)
- **IMPLEMENTATION_SUMMARY.md**: This summary (new)

### Helper Files
- **run_load_test.sh**: Quick-start script
- **requests_data_template.csv**: Sample CSV
- **CLAUDE.md**: Project conventions (unchanged)

---

## Summary

✅ **Implemented**: Mixed mode support for realistic production workloads
✅ **Implemented**: Four traffic patterns (uniform, bursty, rampup, spike)
✅ **Implemented**: Configurable modality ratios
✅ **Implemented**: Enhanced callback logging
✅ **Maintained**: All existing functionality and metrics
✅ **Documented**: Comprehensive guides and examples

**Ready to use!** Start with:
```bash
python concurrency.py --mode mixed --csv requests_data.csv --audio-file audio.wav \
  --workers 15 --duration 600 --rate 60 --pattern bursty
```

---

**Questions or Issues?**
- See MIXED_MODE_GUIDE.md for detailed examples
- See LOAD_TESTING_GUIDE.md for comprehensive reference
- Check troubleshooting sections in both guides
