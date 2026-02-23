#!/bin/bash
# Quick-start script for multimodal load testing
# Usage: ./run_load_test.sh [text|stt|tts]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODE="${1:-text}"
WORKERS=10
DURATION=300  # 5 minutes
RATE=50

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Multimodal Load Testing Framework${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if callback server is running
if ! lsof -i:8001 >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Callback server not detected on port 8001${NC}"
    echo -e "${YELLOW}Starting callback server in background...${NC}"
    python callback.py > callback.log 2>&1 &
    CALLBACK_PID=$!
    echo -e "${GREEN}✅ Callback server started (PID: $CALLBACK_PID)${NC}"
    sleep 2
else
    echo -e "${GREEN}✅ Callback server already running on port 8001${NC}"
fi

echo ""

# Run load test based on mode
case "$MODE" in
    mixed)
        echo -e "${GREEN}Running MIXED mode load test (production-like workload)...${NC}"
        if [ ! -f "requests_data.csv" ]; then
            echo -e "${YELLOW}⚠️  requests_data.csv not found, using template${NC}"
            if [ ! -f "requests_data_template.csv" ]; then
                echo -e "${RED}❌ Error: No CSV file available${NC}"
                exit 1
            fi
            cp requests_data_template.csv requests_data.csv
        fi

        AUDIO_FILE="${2:-test_audio.wav}"
        if [ ! -f "$AUDIO_FILE" ]; then
            echo -e "${RED}❌ Error: Audio file not found: $AUDIO_FILE${NC}"
            echo -e "${YELLOW}Mixed mode requires both CSV and audio file${NC}"
            echo -e "${YELLOW}Usage: $0 mixed <path-to-audio-file>${NC}"
            exit 1
        fi

        echo -e "${GREEN}Using CSV: requests_data.csv${NC}"
        echo -e "${GREEN}Using audio: $AUDIO_FILE${NC}"
        python concurrency.py \
            --mode mixed \
            --csv requests_data.csv \
            --audio-file "$AUDIO_FILE" \
            --workers 15 \
            --duration $DURATION \
            --rate $RATE \
            --pattern bursty \
            --text-ratio 0.7 \
            --tts-ratio 0.2 \
            --stt-ratio 0.1 \
            --output mixed_load_test_$(date +%Y%m%d_%H%M%S).json
        ;;

    text)
        echo -e "${GREEN}Running TEXT load test...${NC}"
        if [ ! -f "requests_data.csv" ]; then
            echo -e "${YELLOW}⚠️  requests_data.csv not found, using template${NC}"
            if [ ! -f "requests_data_template.csv" ]; then
                echo -e "${RED}❌ Error: No CSV file available${NC}"
                exit 1
            fi
            cp requests_data_template.csv requests_data.csv
        fi

        python concurrency.py \
            --mode text \
            --csv requests_data.csv \
            --workers $WORKERS \
            --duration $DURATION \
            --rate $RATE \
            --output text_load_test_$(date +%Y%m%d_%H%M%S).json
        ;;

    stt)
        echo -e "${GREEN}Running STT (Speech-to-Text) load test...${NC}"

        # Check for audio file
        AUDIO_FILE="${2:-test_audio.wav}"
        if [ ! -f "$AUDIO_FILE" ]; then
            echo -e "${RED}❌ Error: Audio file not found: $AUDIO_FILE${NC}"
            echo -e "${YELLOW}Usage: $0 stt <path-to-audio-file>${NC}"
            exit 1
        fi

        echo -e "${GREEN}Using audio file: $AUDIO_FILE${NC}"
        python concurrency.py \
            --mode stt \
            --audio-file "$AUDIO_FILE" \
            --workers 5 \
            --duration $DURATION \
            --rate 30 \
            --output stt_load_test_$(date +%Y%m%d_%H%M%S).json
        ;;

    tts)
        echo -e "${GREEN}Running TTS (Text-to-Speech) load test...${NC}"
        if [ ! -f "requests_data.csv" ]; then
            echo -e "${YELLOW}⚠️  requests_data.csv not found, using template${NC}"
            if [ ! -f "requests_data_template.csv" ]; then
                echo -e "${RED}❌ Error: No CSV file available${NC}"
                exit 1
            fi
            cp requests_data_template.csv requests_data.csv
        fi

        python concurrency.py \
            --mode tts \
            --csv requests_data.csv \
            --workers 8 \
            --duration $DURATION \
            --rate 40 \
            --output tts_load_test_$(date +%Y%m%d_%H%M%S).json
        ;;

    *)
        echo -e "${RED}❌ Error: Invalid mode '$MODE'${NC}"
        echo -e "${YELLOW}Usage: $0 [text|stt|tts|mixed] [audio-file-for-stt-or-mixed]${NC}"
        exit 1
        ;;
esac

# Find the most recent results file
LATEST_RESULTS=$(ls -t *_load_test_*.json 2>/dev/null | head -1)

if [ -n "$LATEST_RESULTS" ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Load test complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "${GREEN}Results saved to: $LATEST_RESULTS${NC}"
    echo ""
    echo -e "${YELLOW}Generating analysis report...${NC}"
    python analyze.py "$LATEST_RESULTS" --output "${LATEST_RESULTS%.json}_report.txt"
    echo ""
    echo -e "${GREEN}✅ Analysis complete!${NC}"
    echo -e "${GREEN}Report saved to: ${LATEST_RESULTS%.json}_report.txt${NC}"
    echo ""
    echo -e "${YELLOW}Quick summary:${NC}"
    python analyze.py "$LATEST_RESULTS" | head -30
else
    echo -e "${RED}❌ Error: No results file found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}To view full report:${NC}"
echo -e "${YELLOW}cat ${LATEST_RESULTS%.json}_report.txt${NC}"
echo ""
echo -e "${GREEN}To view callback responses:${NC}"
echo -e "${YELLOW}cat callback_responses.json | jq${NC}"
echo -e "${GREEN}======================================${NC}"
