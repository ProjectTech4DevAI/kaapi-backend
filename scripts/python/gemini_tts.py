"""
Gemini TTS test script using gemini-2.5-flash-preview-tts via Vertex AI.

API key auth — no OAuth/service account needed.
Output: WAV file (24kHz, 16-bit, mono PCM).

Deps: pip install requests
Usage: python3 gemini_tts.py --text "Hello world" --out output.wav
"""

import argparse
import base64
import json
import os
import struct
import wave
import requests


# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "starlit-lotus-492004-k0"
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash-preview-tts"
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

VERTEX_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1"
    f"/projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/publishers/google/models/{MODEL}:generateContent"
)


# Available voices: Aoede, Charon, Fenrir, Kore, Puck
DEFAULT_VOICE = "Kore"
SAMPLE_RATE = 24000

TEST_TEXTS = [
    "Hello! This is a test of Gemini text to speech.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to Kaapi, your intelligent AI platform.",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_tts(text: str, voice: str, api_key: str) -> bytes:
    """Call Gemini TTS endpoint, return raw PCM bytes."""
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": text}]}
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            },
        },
    }

    resp = requests.post(
        VERTEX_URL,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    if not resp.ok:
        raise RuntimeError(f"[call_tts] {resp.status_code}: {resp.text}")

    data = resp.json()

    try:
        part = data["candidates"][0]["content"]["parts"][0]
        mime_type = part["inlineData"]["mimeType"]
        audio_b64 = part["inlineData"]["data"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"[call_tts] Unexpected response shape: {json.dumps(data, indent=2)}") from e

    print(f"[call_tts] mimeType={mime_type}, encoded_size={len(audio_b64)} chars")
    return base64.b64decode(audio_b64)


def save_wav(pcm_bytes: bytes, output_path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Wrap raw PCM bytes in WAV container and save."""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    size_kb = len(pcm_bytes) / 1024
    duration_s = len(pcm_bytes) / (sample_rate * 2)
    print(f"[save_wav] Saved {output_path} ({size_kb:.1f} KB, {duration_s:.2f}s)")


def run_all_test_texts(voice: str, api_key: str) -> None:
    """Run TTS on all TEST_TEXTS and save numbered WAV files."""
    for i, text in enumerate(TEST_TEXTS, 1):
        out = f"tts_test_{i}.wav"
        print(f"\n[test {i}] text={text!r}")
        pcm = call_tts(text, voice, api_key)
        save_wav(pcm, out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini TTS test via Vertex AI API key")
    parser.add_argument("--text", help="Text to synthesise (omit to run all test phrases)")
    parser.add_argument("--out", default="output.wav", help="Output WAV file (default: output.wav)")
    parser.add_argument("--voice", default=DEFAULT_VOICE,
                        help="Voice name: Aoede|Charon|Fenrir|Kore|Puck (default: Kore)")
    parser.add_argument("--api-key", default=API_KEY, help="Google API key (or set GOOGLE_API_KEY)")
    parser.add_argument("--list-voices", action="store_true", help="Print available voices and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_voices:
        voices = ["Aoede", "Charon", "Fenrir", "Kore", "Puck"]
        print("Available voices:")
        for v in voices:
            print(f"  {v}")
        return

    if not args.api_key:
        raise SystemExit("API key required. Pass --api-key or set GOOGLE_API_KEY env var.")

    if args.text:
        pcm = call_tts(args.text, args.voice, args.api_key)
        save_wav(pcm, args.out)
    else:
        print("[main] No --text provided — running all test phrases.")
        run_all_test_texts(args.voice, args.api_key)


if __name__ == "__main__":
    main()
