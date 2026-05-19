import os
import time

import pandas as pd
import requests

BASE_URL = ""
ENDPOINT = "/api/v1/llm/call"
CALLBACK_URL = ""
CSV_PATH = ""

API_KEY = ""


def build_payload(answer: str) -> dict:
    return {
        "query": {
            "input": {
                "type": "text",
                "content": {
                    "format": "text",
                    "value": answer,
                },
            }
        },
        "config": {
            "blob": {
                "completion": {
                    "type": "tts",
                    "provider": "sarvamai",
                    "params": {
                        "model": "bulbul:v3",
                        "voice": "simran",
                        "language": "hi-IN",
                    },
                }
            }
        },
        "callback_url": CALLBACK_URL,
        "include_provider_raw_response": False,
        "request_metadata": {
            "test_id": "creador-costing",
            "user": "prajna",
        },
    }


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    answers = df["answer"].iloc[:25].tolist()

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-KEY"] = f"{API_KEY}"

    for i, answer in enumerate(answers):
        payload = build_payload(answer)
        try:
            resp = requests.post(
                f"{BASE_URL}{ENDPOINT}",
                json=payload,
                headers=headers,
                timeout=30,
            )
            print(
                f"[{i+1}/{len(answers)}] status={resp.status_code} | {str(answer)[:60]!r}"
            )
            if resp.status_code >= 400:
                print(f"  error: {resp.text[:300]}")
        except requests.RequestException as e:
            print(f"[{i+1}/{len(answers)}] request failed: {e}")

        time.sleep(0.5)


if __name__ == "__main__":
    main()
