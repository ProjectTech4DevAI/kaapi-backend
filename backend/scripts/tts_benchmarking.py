"""Benchmark TTS providers over a CSV column of text.

Writes audio to audio_output/<csv-stem>/<model>/row_NNN.wav and one
benchmark_result_<model>.csv per model.
"""

import argparse
import asyncio
import base64
import csv
import io
import logging
import os
import random
import re
import statistics
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from bodhi import BodhiTTSClient
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from google import genai
from sarvamai import SarvamAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = ROOT_DIR / "audio_input" / "setu_law_golden_qna.csv"
DEFAULT_TEXT_COLUMN = "answer"
OUTPUT_DIR = ROOT_DIR / "audio_output"

DEFAULT_MODELS = ["BODHAN_AI", "BODHI_AI"]
MAX_WORKERS = 4
JITTER_MIN_SECONDS = 0.5
JITTER_MAX_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 120

WAV_CHANNELS = 1
WAV_SAMPLE_WIDTH = 2  # 16-bit PCM
WAV_FRAME_RATE = 24000

GEMINI_VOICE = "Kore"
# The accent comes from this, not from the prompt: en-IN gives Indian-English
# pronunciation, while hi-IN reads English text with a Hindi voice. The lite
# model ignores accent direction in the prompt entirely, so this is the only
# lever that works across all three models.
GEMINI_LANGUAGE = "en-IN"
# Gemini takes direction as prose, so the accent is steered by a prompt wrapper
# rather than a parameter. Everything outside the transcript is instruction.
GEMINI_PROMPT_TEMPLATE = """
# AUDIO PROFILE
Character: Rahul, a professional customer support agent from Urban India.

# DIRECTOR'S NOTES
Accent: Indian English (Urban, clear Indian accent with natural Indian English rhythm and stress patterns).
Pace: Natural conversational pace.
Style: Professional, warm, and polite.

# TRANSCRIPT
"{text}"
"""
SARVAM_SPEAKER = "shubh"
SARVAM_LANGUAGE = "hi-IN"
SARVAM_PACE = 1.0
ELEVENLABS_VOICE_ID = "2cdvnKJ5TZi631y5PN1s"
ELEVENLABS_LANGUAGE = "hi"
# Both Bodh* providers break on long input: Bodhi 502s past ~400 chars, Bodhan
# silently truncates its output at 30.549s.
CHUNK_MAX_CHARS = 350
# Sent in chunks and stitched back together, so the audio covers the whole row.
CHUNKED_MODELS = {"BODHAN_AI"}
# Sent as one capped request instead, keeping latency comparable to the
# single-call providers at the cost of only voicing the start of the row.
TRUNCATED_MODELS = {"BODHI_AI"}
BODHI_VOICE = "default_female"
BODHI_LANGUAGE = "en"
BODHI_ENCODING = "pcm16"
BODHAN_URL = "https://api.bodhan.ai/v1/audio/speech"
BODHAN_MODEL = "indic-speak"
BODHAN_VOICE = "Amit"
BODHAN_LANGUAGE = "en"

PERCENTILE_P95 = 0.95
# RTF's bad tail is the LOW end (slow calls), so p5 is its equivalent of p95 latency.
PERCENTILE_P05 = 0.05
RESULT_FIELDNAMES = [
    "model",
    "row_index",
    "filename",
    "char_count",
    "sent_chars",
    "chunks",
    "audio_seconds",
    "elapsed_seconds",
    "realtime_factor",
    "status",
    "error",
]

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_AISTUDIO_API_KEY"))
sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def pcm_to_wav(pcm_bytes: bytes, frame_rate: int = WAV_FRAME_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(WAV_CHANNELS)
        wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Greedily pack sentences into chunks, so splits land on sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?।])\s+", text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def truncate_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> str:
    """Cap at a word boundary. Cutting on sentences instead would send wildly
    uneven payloads (one row's first sentence is 103 chars, another's 326),
    which makes the latency numbers incomparable between rows."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = head.rfind(" ")
    return head[:cut] if cut > 0 else head


def text_parts(text: str, model: str) -> list[str]:
    """The request bodies to send for one row — several, one capped, or one whole."""
    if model in CHUNKED_MODELS:
        return split_text(text)
    if model in TRUNCATED_MODELS:
        return [truncate_text(text)]
    return [text]


def concat_wavs(parts: list[bytes]) -> bytes:
    """Join WAVs by re-framing their samples, so only one header survives."""
    if len(parts) == 1:
        return parts[0]
    frames = []
    frame_rate = WAV_FRAME_RATE
    for part in parts:
        with wave.open(io.BytesIO(part)) as wav_file:
            frame_rate = wav_file.getframerate()
            frames.append(wav_file.readframes(wav_file.getnframes()))
    return pcm_to_wav(b"".join(frames), frame_rate)


def generate_bodhi(chunks: list[str]) -> bytes:
    async def run() -> bytes:
        client = BodhiTTSClient(api_key=os.environ["BODHI_API_KEY"])
        parts = []
        for chunk in chunks:
            speech = await client.synthesize(
                chunk,
                lang=BODHI_LANGUAGE,
                voice=BODHI_VOICE,
                sample_rate=WAV_FRAME_RATE,
                encoding=BODHI_ENCODING,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            # The server's rate is authoritative; it may not be the one we asked for.
            parts.append(pcm_to_wav(speech.audio, speech.sample_rate))
        return concat_wavs(parts)

    # Each worker thread gets its own loop, so the async client stays thread-local.
    return asyncio.run(run())


def generate_bodhan(chunks: list[str]) -> bytes:
    parts = []
    for chunk in chunks:
        response = requests.post(
            BODHAN_URL,
            headers={"Authorization": f"Bearer {os.getenv('BODHAN_API_KEY')}"},
            json={
                "model": BODHAN_MODEL,
                "input": chunk,
                "voice": BODHAN_VOICE,
                "instructions": f'{{"lang": "{BODHAN_LANGUAGE}"}}',
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        # Served as audio/mpeg but the body is really a RIFF/WAVE container.
        parts.append(response.content)
    return concat_wavs(parts)


def generate_gemini(text: str, model: str) -> bytes:
    interaction = gemini_client.interactions.create(
        model=model,
        input=GEMINI_PROMPT_TEMPLATE.format(text=text),
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [{"voice": GEMINI_VOICE, "language": GEMINI_LANGUAGE}],
        },
    )
    return pcm_to_wav(base64.b64decode(interaction.output_audio.data))


def generate_sarvam(text: str, model: str) -> bytes:
    response = sarvam_client.text_to_speech.convert(
        text=text,
        model=model,
        target_language_code=SARVAM_LANGUAGE,
        speaker=SARVAM_SPEAKER,
        pace=SARVAM_PACE,
        speech_sample_rate=WAV_FRAME_RATE,
    )
    # Long inputs come back split across several base64 chunks of one WAV stream.
    return base64.b64decode("".join(response.audios))


def generate_elevenlabs(text: str, model: str) -> bytes:
    chunks = elevenlabs_client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id=model,
        language_code=ELEVENLABS_LANGUAGE,
        output_format=f"pcm_{WAV_FRAME_RATE}",
    )
    return pcm_to_wav(b"".join(chunks))


def synthesize(text: str, model: str) -> bytes:
    """Every provider returns a complete WAV here, whatever it sends on the wire."""
    if model == "BODHI_AI":
        return generate_bodhi(text_parts(text, model))
    if model == "BODHAN_AI":
        return generate_bodhan(text_parts(text, model))
    if model.startswith("bulbul"):
        return generate_sarvam(text, model)
    if model.startswith("eleven"):
        return generate_elevenlabs(text, model)
    return generate_gemini(text, model)


def wav_duration_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile — no interpolation, so every value is a real measurement."""
    index = min(
        len(sorted_values) - 1, max(0, round(fraction * len(sorted_values) + 0.5) - 1)
    )
    return sorted_values[index]


def synthesize_row(
    model: str, row_index: int, text: str, output_dir: Path
) -> dict[str, Any]:
    # Stagger request starts so a burst of workers doesn't trip provider rate limits.
    time.sleep(random.uniform(JITTER_MIN_SECONDS, JITTER_MAX_SECONDS))

    filename = f"row_{row_index:03d}.wav"
    parts = text_parts(text, model)
    result: dict[str, Any] = {
        "model": model,
        "row_index": row_index,
        "filename": filename,
        "char_count": len(text),
        # Differs from char_count when the row was capped — keeps truncation visible.
        "sent_chars": sum(len(p) for p in parts),
        "chunks": len(parts),
        "audio_seconds": "",
        "elapsed_seconds": "",
        "realtime_factor": "",
        "status": "failure",
        "error": "",
    }

    start_time = time.perf_counter()
    try:
        audio = synthesize(text, model)
        elapsed = time.perf_counter() - start_time
        # Measure before writing: latency is the API call, not the disk write.
        audio_seconds = wav_duration_seconds(audio)
        (output_dir / model / filename).write_bytes(audio)
        result["elapsed_seconds"] = round(elapsed, 3)
        result["audio_seconds"] = round(audio_seconds, 3)
        result["realtime_factor"] = round(audio_seconds / elapsed, 3)
        result["status"] = "success"
        logger.info(
            f"[synthesize_row] Generated | model: {model} | row: {row_index} | "
            f"elapsed: {result['elapsed_seconds']}s | audio: {result['audio_seconds']}s | "
            f"rtf: {result['realtime_factor']}x"
        )
    except Exception as exc:
        result["elapsed_seconds"] = round(time.perf_counter() - start_time, 3)
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning(
            f"[synthesize_row] Failed | model: {model} | row: {row_index} | "
            f"error: {result['error']}"
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        metavar="MODEL",
        help=f"Repeat for several. Defaults to: {', '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument(
        "--gemini-language",
        default=GEMINI_LANGUAGE,
        help=f"BCP-47 tag driving Gemini's accent (default: {GEMINI_LANGUAGE}).",
    )
    parser.add_argument("--limit", type=int, help="Only the first N rows.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    global GEMINI_LANGUAGE
    args = parse_args()
    GEMINI_LANGUAGE = args.gemini_language
    models = args.models or DEFAULT_MODELS

    with args.input.open(newline="", encoding="utf-8") as f:
        texts = [row[args.column] for row in csv.DictReader(f) if row.get(args.column)]
    if args.limit:
        texts = texts[: args.limit]

    # Per-dataset tree so a run on another CSV doesn't overwrite row-indexed files.
    output_dir = OUTPUT_DIR / args.input.stem
    for model in models:
        (output_dir / model).mkdir(parents=True, exist_ok=True)

    jobs = [
        (model, index, text)
        for model in models
        for index, text in enumerate(texts, start=1)
    ]
    logger.info(
        f"[main] Starting | input: {args.input.name} | column: {args.column} | "
        f"rows: {len(texts)} | models: {', '.join(models)} | jobs: {len(jobs)} | "
        f"workers: {args.workers}"
    )

    total_start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(synthesize_row, *job, output_dir) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    total_elapsed = time.perf_counter() - total_start

    results.sort(key=lambda r: (r["model"], r["row_index"]))
    for model in models:
        results_csv = output_dir / f"benchmark_result_{model}.csv"
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(r for r in results if r["model"] == model)
        logger.info(f"[main] Wrote results | model: {model} | file: {results_csv}")

    succeeded = [r for r in results if r["status"] == "success"]
    logger.info(
        f"[main] Summary | total: {len(results)} | success: {len(succeeded)} | "
        f"failure: {len(results) - len(succeeded)}"
    )
    for model in models:
        model_ok = [r for r in succeeded if r["model"] == model]
        if not model_ok:
            logger.info(
                f"[main] Model stats | model: {model} | success: 0/{len(texts)}"
            )
            continue
        latencies = sorted(r["elapsed_seconds"] for r in model_ok)
        rtfs = sorted(r["realtime_factor"] for r in model_ok)
        logger.info(
            f"[main] Model stats | model: {model} | success: {len(model_ok)}/{len(texts)} | "
            f"avg_elapsed: {statistics.mean(latencies):.3f}s | "
            f"p95_elapsed: {percentile(latencies, PERCENTILE_P95):.3f}s | "
            f"median_rtf: {statistics.median(rtfs):.2f}x | "
            f"p5_rtf: {percentile(rtfs, PERCENTILE_P05):.2f}x"
        )
    logger.info(f"[main] Total wall time: {total_elapsed:.3f}s | output: {output_dir}")


if __name__ == "__main__":
    main()
