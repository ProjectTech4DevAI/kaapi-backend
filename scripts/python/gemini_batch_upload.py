"""
Upload Gemini batch input JSONL to GCS using service account credentials.

Auth: Service account JSON key
Deps: pip install google-cloud-storage google-auth
"""

import json
import argparse
import os
import time

from google.cloud import storage
from google.oauth2 import service_account

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "starlit-lotus-492004-k0"
BUCKET_NAME = "starlit-lotus-batch-data-us-test-1" #kaapi-audio-bucket
SA_KEY_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/prashantvasudevan/secret/starlit-lotus-492004-k0-66a7028241db.json",
)

# ── Test data ─────────────────────────────────────────────────────────────────
TEST_PROMPTS: list[dict] = [
    {"id": "q1",  "prompt": "Explain quantum computing in a hundred words."},
    {"id": "q2",  "prompt": "Summarise the history of the internet in three sentences."},
    {"id": "q3",  "prompt": "What are the top five Python libraries for data science and why?"},
    {"id": "q4",  "prompt": "Write a haiku about machine learning."},
    {"id": "q5",  "prompt": "What is the difference between supervised and unsupervised learning?"},
    {"id": "q6",  "prompt": "Explain REST vs GraphQL in plain English."},
    {"id": "q7",  "prompt": "List five best practices for securing a FastAPI application."},
    {"id": "q8",  "prompt": "What is RAG (Retrieval Augmented Generation) and when should you use it?"},
    {"id": "q9",  "prompt": "Explain database connection pooling in two paragraphs."},
    {"id": "q10", "prompt": "What are the SOLID principles in software engineering?"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_storage_client(sa_key_path: str) -> storage.Client:
    """Build GCS client from service account key file."""
    credentials = service_account.Credentials.from_service_account_file(
        sa_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return storage.Client(project=PROJECT_ID, credentials=credentials)


def build_jsonl(prompts: list[dict]) -> str:
    """Convert prompt list to Vertex AI batch JSONL string."""
    lines = []
    for item in prompts:
        record = {
            "id": item["id"],
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": item["prompt"]}]}
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024,
                },
            },
        }
        lines.append(json.dumps(record))
    return "\n".join(lines)


def upload_file(content: str, bucket_name: str, blob_name: str, sa_key_path: str) -> str:
    """Upload JSONL string to GCS, return gs:// URI."""
    storage_client = build_storage_client(sa_key_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(content, content_type="application/jsonl")

    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"[upload_file] Uploaded {len(content)} bytes → {uri}")
    return uri


def preview_jsonl(content: str, n: int = 2) -> None:
    """Print first n records for sanity check."""
    lines = content.strip().split("\n")
    print(f"\n[preview] Showing {min(n, len(lines))} of {len(lines)} records:")
    for line in lines[:n]:
        print(json.dumps(json.loads(line), indent=2))
    print("…\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Gemini batch JSONL to GCS via service account")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="GCS bucket name (must already exist)")
    parser.add_argument("--blob", default=f"batch-input-{int(time.time())}.jsonl", help="Destination blob name")
    parser.add_argument("--sa-key", default=SA_KEY_PATH, help="Path to service account JSON key")
    parser.add_argument("--preview", action="store_true", help="Print first 2 records before upload")
    parser.add_argument("--dry-run", action="store_true", help="Build JSONL but skip upload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    content = build_jsonl(TEST_PROMPTS)
    print(f"[main] Built JSONL: {len(TEST_PROMPTS)} records, {len(content)} bytes")

    if args.preview or args.dry_run:
        preview_jsonl(content)

    if args.dry_run:
        print("[main] Dry run — skipping upload.")
        return

    uri = upload_file(content, args.bucket, args.blob, args.sa_key)

    print(f"\n[main] Input URI for batch job:")
    print(f"  {uri}")
    print(f"\n[main] Run batch job:")
    print(f"  python gemini_batch.py --input {uri}")


if __name__ == "__main__":
    main()
