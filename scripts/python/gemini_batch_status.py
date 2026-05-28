"""
Check Gemini batch job status and failure reason.

Auth: Service account JSON key
Deps: pip install google-genai google-auth
Usage: python3 gemini_batch_status.py --job <job_name>
"""

import argparse
import json
import os

from google import genai
from google.oauth2 import service_account

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "starlit-lotus-492004-k0"
LOCATION = "us-central1"
SA_KEY_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/prashantvasudevan/secret/starlit-lotus-492004-k0-66a7028241db.json",
)

LAST_KNOWN_JOB = "projects/877240188032/locations/us-central1/batchPredictionJobs/7121870772582219776"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_client(sa_key_path: str) -> genai.Client:
    credentials = service_account.Credentials.from_service_account_file(
        sa_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, credentials=credentials)


def check_job(client: genai.Client, job_name: str) -> None:
    job = client.batches.get(name=job_name)
    state = job.state.name if job.state else "UNKNOWN"

    print(f"Job:   {job.name}")
    print(f"State: {state}")

    error = getattr(job, "error", None)
    if error:
        print(f"\n── Error ─────────────────────────────────────────────────────")
        print(f"Code:    {getattr(error, 'code', 'N/A')}")
        print(f"Message: {getattr(error, 'message', 'N/A')}")
        details = getattr(error, "details", None)
        if details:
            print(f"Details: {json.dumps(details, indent=2, default=str)}")

    print(f"\n── Full job object ───────────────────────────────────────────")
    try:
        print(json.dumps(json.loads(job.__repr__()), indent=2))
    except Exception:
        print(job)


def list_recent_jobs(client: genai.Client, limit: int = 5) -> None:
    print(f"── Recent batch jobs (last {limit}) ──────────────────────────")
    count = 0
    for job in client.batches.list():
        state = job.state.name if job.state else "UNKNOWN"
        error = getattr(job, "error", None)
        error_msg = getattr(error, "message", "") if error else ""
        print(f"  {job.name}")
        print(f"    state={state}" + (f" | error={error_msg}" if error_msg else ""))
        count += 1
        if count >= limit:
            break


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Gemini batch job status and failure reason")
    parser.add_argument("--job", default=LAST_KNOWN_JOB, help="Full job resource name")
    parser.add_argument("--list", action="store_true", help="List recent batch jobs")
    parser.add_argument("--sa-key", default=SA_KEY_PATH, help="Path to service account JSON key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client(args.sa_key)

    if args.list:
        list_recent_jobs(client)
        return

    check_job(client, args.job)


if __name__ == "__main__":
    main()
