# Load Testing Quick Reference Card

## Single Modality Tests

```bash
# Text only
python concurrency.py --mode text --csv data.csv --workers 10 --duration 600

# STT only
python concurrency.py --mode stt --audio-file audio.wav --workers 5 --duration 600

# TTS only
python concurrency.py --mode tts --csv data.csv --workers 8 --duration 600
```

---

## Mixed Mode (Recommended for Production Simulation)

```bash
# Default mix: 70% text, 20% TTS, 10% STT
python concurrency.py --mode mixed \
  --csv data.csv \
  --audio-file audio.wav \
  --workers 15 \
  --duration 600 \
  --rate 60

# Custom mix: 50% text, 30% TTS, 20% STT
python concurrency.py --mode mixed \
  --csv data.csv \
  --audio-file audio.wav \
  --workers 15 \
  --duration 600 \
  --rate 60 \
  --text-ratio 0.5 \
  --tts-ratio 0.3 \
  --stt-ratio 0.2
```

---

## Traffic Patterns

```bash
# Uniform (default) - steady, predictable
--pattern uniform

# Bursty (recommended) - realistic production with random spikes
--pattern bursty

# Ramp-up - gradual increase to test scaling
--pattern rampup

# Spike - periodic sharp bursts
--pattern spike
```

---

## Complete Example: Production-Like Test

```bash
# 1. Start callback server
python callback.py &

# 2. Run mixed mode test with bursty traffic
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

# 3. Analyze results
python analyze.py prod_test.json

# 4. Export detailed metrics
python analyze.py prod_test.json --format csv --output metrics.csv
```

---

## Finding Resource Saturation

```bash
# Gradually increase load
for workers in 10 15 20 25 30; do
  python concurrency.py --mode mixed \
    --csv data.csv --audio-file audio.wav \
    --workers $workers --duration 300 --rate 50 \
    --pattern bursty --output test_${workers}w.json

  python analyze.py test_${workers}w.json | grep -A 10 "RESOURCE"
done
```

---

## Saturation Indicators

⚠️ **Memory Saturation**:
- Memory pressure > 10%
- Max memory approaching 4096 MB
- OOM kills in logs

⚠️ **CPU Saturation**:
- CPU p99 > 180% (90% per core)
- High queue wait times
- Increasing latency

⚠️ **Queue Backpressure**:
- Queue wait p99 > 1000ms
- High queue wait rate
- Celery workers overwhelmed

---

## Helper Script

```bash
# Easy mode (handles callback server automatically)
./run_load_test.sh mixed audio.wav

# Choose mode
./run_load_test.sh text      # Text only
./run_load_test.sh stt audio.wav     # STT only
./run_load_test.sh tts       # TTS only
./run_load_test.sh mixed audio.wav   # Mixed mode
```

---

## Analysis Shortcuts

```bash
# View report
python analyze.py results.json

# Save to file
python analyze.py results.json --output report.txt

# JSON format
python analyze.py results.json --format json --output report.json

# CSV export
python analyze.py results.json --format csv --output metrics.csv

# Quick summary (first 30 lines)
python analyze.py results.json | head -30

# Check resource warnings
python analyze.py results.json | grep "⚠️"

# Check saturation
python analyze.py results.json | grep -E "Memory|CPU|Queue" | grep -E "p99|rate"
```

---

## File Locations

| File | Purpose |
|------|---------|
| `concurrency.py` | Load test runner |
| `callback.py` | Webhook receiver |
| `analyze.py` | Results analyzer |
| `run_load_test.sh` | Helper script |
| `requests_data.csv` | Input data (text/TTS) |
| `audio.wav` | Input audio (STT) |
| `load_test_results.json` | Test output |
| `callback_responses.json` | Server callbacks |

---

## Documentation

| Guide | Content |
|-------|---------|
| **QUICK_REFERENCE.md** | This card |
| **MIXED_MODE_GUIDE.md** | Detailed mixed mode guide |
| **LOAD_TESTING_GUIDE.md** | Complete reference |
| **IMPLEMENTATION_SUMMARY.md** | What was implemented |

---

## Cheat Sheet

### Most Common Command

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 15 --duration 600 --rate 60 --pattern bursty
```

### Fastest Test (5 min)

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 10 --duration 300 --rate 40 --pattern uniform
```

### Stress Test

```bash
python concurrency.py --mode mixed --csv data.csv --audio-file audio.wav \
  --workers 30 --duration 600 --rate 100 --pattern spike
```

---

**Last Updated**: 2026-02-23
