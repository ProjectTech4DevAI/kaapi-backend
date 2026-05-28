"""
Gemini Speech-to-Text test script via Vertex AI.

Sends base64-encoded audio to Gemini and returns transcript.
Input: base64 string (from file, stdin, or --wav to auto-encode a WAV file).

API key auth — no OAuth/service account needed.
Deps: pip install requests
Usage:
  python3 gemini_stt.py --wav output.wav
  python3 gemini_stt.py --b64 "<base64string>" --mime audio/wav
  python3 gemini_stt.py --b64-file audio.b64
"""

import argparse
import base64
import json
import os
import sys
import requests

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "starlit-lotus-492004-k0"
LOCATION = "us-central1"
MODEL = "gemini-3.1-flash-lite"  # multimodal model — handles audio input
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

VERTEX_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1"
    f"/projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/publishers/google/models/{MODEL}:generateContent"
)

DEFAULT_PROMPT = "Transcribe the audio exactly as spoken. Return only the transcript, no commentary."


# ── Helpers ───────────────────────────────────────────────────────────────────

def wav_to_b64(wav_path: str) -> tuple[str, str]:
    """Read WAV file, return (base64_string, mime_type)."""
    with open(wav_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8"), "audio/wav"


def call_stt(audio_b64: str, mime_type: str, prompt: str, api_key: str) -> str:
    """Send base64 audio to Gemini, return transcript string."""
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": audio_b64,
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,         # deterministic for transcription
            "maxOutputTokens": 2048,
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
        raise RuntimeError(f"[call_stt] {resp.status_code}: {resp.text}")

    data = resp.json()

    try:
        transcript = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"[call_stt] Unexpected response shape: {json.dumps(data, indent=2)}") from e

    return transcript.strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini STT — base64 audio input via Vertex AI API key")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", metavar="FILE", help="WAV file to encode and transcribe")
    source.add_argument("--b64", metavar="STRING", help="Raw base64 audio string")
    source.add_argument("--b64-file", metavar="FILE", help="File containing base64 audio string")

    parser.add_argument(
        "--mime", default="audio/wav",
        help="MIME type when using --b64 or --b64-file (default: audio/wav). "
             "Options: audio/wav, audio/mp3, audio/flac, audio/ogg, audio/webm",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction sent alongside audio")
    parser.add_argument("--api-key", default=API_KEY, help="Google API key (or set GOOGLE_API_KEY)")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Output raw JSON response")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("API key required. Pass --api-key or set GOOGLE_API_KEY env var.")

    # Resolve audio input
    if args.wav:
        print(f"[main] Encoding {args.wav} …")
        audio_b64, mime_type = wav_to_b64(args.wav)
        print(f"[main] Encoded {len(audio_b64)} chars, mime={mime_type}")
    elif args.b64_file:
        with open(args.b64_file) as f:
            audio_b64 = f.read().strip()
        mime_type = args.mime
        print(f"[main] Loaded b64 from file: {len(audio_b64)} chars, mime={mime_type}")
    else:
        audio_b64 = args.b64.strip()
        mime_type = args.mime
        print(f"[main] Using inline b64: {len(audio_b64)} chars, mime={mime_type}")

    transcript = call_stt(audio_b64, mime_type, args.prompt, args.api_key)

    print("\n── Transcript ───────────────────────────────────────────────")
    print(transcript)
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
