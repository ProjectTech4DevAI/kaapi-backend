"""
Vertex AI Gemini Batch Prediction Script

Uploads JSONL to GCS then submits batch job using the same URI.
Auth: Service account JSON key
Deps: pip install google-genai google-cloud-storage google-auth
"""

import json
import time
import argparse
import os

from google import genai
from google.cloud import storage
from google.oauth2 import service_account

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "starlit-lotus-492004-k0"
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"
BUCKET_NAME = "starlit-lotus-batch-data-us-test-1"
SA_KEY_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/prashantvasudevan/secret/starlit-lotus-492004-k0-66a7028241db.json",
)

_TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

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


# ── Auth ──────────────────────────────────────────────────────────────────────

def build_credentials(sa_key_path: str) -> service_account.Credentials:
    return service_account.Credentials.from_service_account_file(
        sa_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def build_genai_client(credentials: service_account.Credentials) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials,
    )


def build_storage_client(credentials: service_account.Credentials) -> storage.Client:
    return storage.Client(project=PROJECT_ID, credentials=credentials)


# ── Upload ────────────────────────────────────────────────────────────────────

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


def upload_jsonl(gcs_client: storage.Client, content: str, bucket_name: str) -> str:
    """Upload JSONL to GCS, return gs:// URI."""
    blob_name = f"batch-input-{int(time.time())}.jsonl"
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type="application/jsonl")
    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"[upload_jsonl] Uploaded {len(content)} bytes → {uri}")
    return uri


# ── Batch job ─────────────────────────────────────────────────────────────────

def submit_batch_job(client: genai.Client, input_uri: str, output_uri: str) -> any:
    """Submit batch job using exact input_uri from upload."""
    display_name = f"gemini-batch-{int(time.time())}"
    job = client.batches.create(
        model=MODEL,
        src=input_uri,
        config={"display_name": display_name},
    )
    state = job.state.name if job.state else "UNKNOWN"
    print(f"[submit_batch_job] Job submitted: {job.name}")
    print(f"[submit_batch_job] Input URI: {input_uri}")
    print(f"[submit_batch_job] State: {state}")
    return job


def poll_until_done(client: genai.Client, job: any, interval_sec: int = 30) -> any:
    """Block until job reaches terminal state."""
    while True:
        state = job.state.name if job.state else "UNKNOWN"
        if state in _TERMINAL_STATES:
            break
        print(f"[poll_until_done] State={state} — sleeping {interval_sec}s …")
        time.sleep(interval_sec)
        job = client.batches.get(name=job.name)
    print(f"[poll_until_done] Final state: {job.state.name}")
    return job


def print_results(job: any) -> None:
    """Print job summary after completion."""
    state = job.state.name if job.state else "UNKNOWN"
    if state == "JOB_STATE_SUCCEEDED":
        output = getattr(getattr(job, "dest", None), "gcs_uri", None)
        print(f"[print_results] Output written to: {output}")
    else:
        print(f"[print_results] Job ended with state: {state}")
        error = getattr(job, "error", None)
        if error:
            print(f"[print_results] Error code: {getattr(error, 'code', 'N/A')}")
            print(f"[print_results] Error message: {getattr(error, 'message', 'N/A')}")
        print(f"[print_results] Full job details:")
        print(job)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload JSONL + run Gemini batch job")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="GCS bucket name")
    parser.add_argument("--sa-key", default=SA_KEY_PATH, help="Path to service account JSON key")
    parser.add_argument("--no-wait", action="store_true", help="Submit and exit without polling")
    parser.add_argument("--status", metavar="JOB_NAME", help="Check status of existing job")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    credentials = build_credentials(args.sa_key)
    genai_client = build_genai_client(credentials)

    if args.status:
        job = genai_client.batches.get(name=args.status)
        print_results(job)
        return

    # Upload then immediately use returned URI for batch job
    gcs_client = build_storage_client(credentials)
    content = build_jsonl(TEST_PROMPTS)
    print(f"[main] Built JSONL: {len(TEST_PROMPTS)} records")

    input_uri = upload_jsonl(gcs_client, content, args.bucket)
    output_uri = f"gs://{args.bucket}/batch-output-{int(time.time())}/"

    job = submit_batch_job(genai_client, input_uri=input_uri, output_uri=output_uri)

    if not args.no_wait:
        job = poll_until_done(genai_client, job)
        print_results(job)
    else:
        print(f"[main] Job running: {job.name}")
        print(f"[main] Check status:")
        print(f"  python gemini_batch.py --status {job.name}")


if __name__ == "__main__":
    main()
