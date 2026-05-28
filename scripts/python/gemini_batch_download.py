"""
Download Gemini batch output files from GCS.

Auth: Service account JSON key
Deps: pip install google-cloud-storage google-auth
Usage: python3 gemini_batch_download.py --gcs-path gs://bucket/prefix/
"""

import argparse
import os
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "starlit-lotus-492004-k0"
SA_KEY_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/prashantvasudevan/secret/starlit-lotus-492004-k0-66a7028241db.json",
)

DEFAULT_GCS_PATH = "gs://starlit-lotus-batch-data-us-test-1/batch-input-1779799704/dest"
DEFAULT_OUTPUT_DIR = "./batch_output"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Split gs://bucket/prefix into (bucket, prefix)."""
    path = gcs_path.removeprefix("gs://")
    bucket, _, prefix = path.partition("/")
    return bucket, prefix


def build_storage_client(sa_key_path: str) -> storage.Client:
    credentials = service_account.Credentials.from_service_account_file(
        sa_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return storage.Client(project=PROJECT_ID, credentials=credentials)


def download_blobs(
    gcs_client: storage.Client,
    bucket_name: str,
    prefix: str,
    output_dir: str,
) -> list[str]:
    """Download all blobs under prefix to output_dir, return local paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bucket = gcs_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"[download_blobs] No files found at gs://{bucket_name}/{prefix}")
        return []

    print(f"[download_blobs] Found {len(blobs)} file(s)")
    downloaded = []

    for blob in blobs:
        filename = Path(blob.name).name
        local_path = output_path / filename
        blob.download_to_filename(str(local_path))
        print(f"[download_blobs] ✓ {blob.name} → {local_path} ({blob.size} bytes)")
        downloaded.append(str(local_path))

    return downloaded


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Gemini batch output from GCS")
    parser.add_argument("--gcs-path", default=DEFAULT_GCS_PATH, help="GCS path (gs://bucket/prefix)")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Local output directory")
    parser.add_argument("--sa-key", default=SA_KEY_PATH, help="Path to service account JSON key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bucket_name, prefix = parse_gcs_path(args.gcs_path)
    print(f"[main] Downloading from gs://{bucket_name}/{prefix}")

    gcs_client = build_storage_client(args.sa_key)
    files = download_blobs(gcs_client, bucket_name, prefix, args.out)

    print(f"\n[main] Downloaded {len(files)} file(s) → {args.out}")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
