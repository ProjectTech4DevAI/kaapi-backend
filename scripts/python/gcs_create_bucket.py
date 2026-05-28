"""
Create a GCS bucket using API key only (no OAuth / service account).

GCS JSON API: POST https://storage.googleapis.com/storage/v1/b?project={project}&key={key}
Deps: pip install requests
"""

import argparse
import json
import os
import requests

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "starlit-lotus-492004-k0"
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GCS_BUCKET_URL = "https://storage.googleapis.com/storage/v1/b"


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_bucket(
    bucket_name: str,
    project_id: str,
    api_key: str,
    location: str = "US",
    storage_class: str = "STANDARD",
) -> dict:
    """Create GCS bucket via JSON API using API key."""
    payload = {
        "name": bucket_name,
        "location": location,
        "storageClass": storage_class,
    }

    resp = requests.post(
        GCS_BUCKET_URL,
        params={"project": project_id, "key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        raise RuntimeError(f"[create_bucket] {resp.status_code}: {resp.text}")

    bucket = resp.json()
    print(f"[create_bucket] Created: gs://{bucket['name']} (location={bucket['location']})")
    return bucket


def bucket_exists(bucket_name: str, api_key: str) -> bool:
    """Check if bucket already exists."""
    resp = requests.get(
        f"{GCS_BUCKET_URL}/{bucket_name}",
        params={"key": api_key},
        timeout=10,
    )
    return resp.status_code == 200


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GCS bucket via API key")
    parser.add_argument("--bucket", required=True, help="Bucket name (globally unique)")
    parser.add_argument("--project", default=PROJECT_ID, help="GCP project ID")
    parser.add_argument("--location", default="US", help="Bucket location (default: US)")
    parser.add_argument("--storage-class", default="STANDARD", help="Storage class (default: STANDARD)")
    parser.add_argument("--api-key", default=API_KEY, help="Google API key (or set GOOGLE_API_KEY)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("API key required. Pass --api-key or set GOOGLE_API_KEY env var.")

    if bucket_exists(args.bucket, args.api_key):
        print(f"[main] Bucket gs://{args.bucket} already exists — nothing to do.")
        return

    bucket = create_bucket(
        bucket_name=args.bucket,
        project_id=args.project,
        api_key=args.api_key,
        location=args.location,
        storage_class=args.storage_class,
    )
    print(json.dumps(bucket, indent=2))


if __name__ == "__main__":
    main()
