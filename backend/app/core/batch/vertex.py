"""Google GCP Vertex AI batch provider implementation.

Vertex batch prediction reads its input JSONL from GCS and writes results back to
GCS (no File API, unlike the AI-Studio ``GeminiBatchProvider``). Input/output
therefore ride the project's ``google-gcp`` credential (SA key + ``gcs_bucket``).
"""

import json
import logging
import time
from typing import Any
from uuid import uuid4

from google import genai
from google.genai import types
from google.cloud import storage as gcs
from google.oauth2 import service_account

from app.core.cloud.storage import GCS_SCOPES, CloudStorageError

from .base import BATCH_KEY, BatchProvider
from .gemini import BatchJobState

logger = logging.getLogger(__name__)

# Terminal Vertex job states (superset of AI-Studio: Vertex adds PAUSED).
_TERMINAL_STATES = {
    BatchJobState.SUCCEEDED.value,
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
    "JOB_STATE_PAUSED",
}
_FAILED_STATES = {
    BatchJobState.FAILED.value,
    BatchJobState.CANCELLED.value,
    BatchJobState.EXPIRED.value,
}

_DEFAULT_INPUT_PREFIX = "batch-input"
_DEFAULT_OUTPUT_PREFIX = "batch-output"


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/key`` into ``(bucket, key)``."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got '{uri}'.")
    bucket, _, key = uri[len("gs://") :].partition("/")
    return bucket, key


class VertexBatchProvider(BatchProvider):
    """Vertex AI implementation of the BatchProvider interface (GCS in/out).

    Each JSONL line is the Vertex request schema, e.g.
        {"request": {"contents": [{"parts": [...], "role": "user"}]}}
    """

    DEFAULT_MODEL = "gemini-2.5-pro"

    def __init__(
        self,
        client: genai.Client,
        storage_client: gcs.Client,
        gcs_bucket: str,
        model: str | None = None,
        input_prefix: str = _DEFAULT_INPUT_PREFIX,
        output_prefix: str = _DEFAULT_OUTPUT_PREFIX,
    ) -> None:
        self._client = client
        self._storage = storage_client
        self._bucket = gcs_bucket
        self._model = model or self.DEFAULT_MODEL
        self._input_prefix = input_prefix
        self._output_prefix = output_prefix

    @classmethod
    def from_credentials(
        cls, credentials: dict[str, Any], model: str | None = None
    ) -> "VertexBatchProvider":
        """Build a Vertex batch provider from a ``google-gcp`` credential dict."""
        project_id = credentials.get("project_id")
        location = credentials.get("location")
        gcs_bucket = credentials.get("gcs_bucket")
        sa_info = credentials.get("sa_key")
        missing = [
            name
            for name, value in (
                ("project_id", project_id),
                ("location", location),
                ("gcs_bucket", gcs_bucket),
                ("sa_key", sa_info),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Vertex batch provider missing required fields: {', '.join(missing)}"
            )

        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=list(GCS_SCOPES)
        )
        client = genai.Client(
            vertexai=True, project=project_id, location=location, credentials=creds
        )
        storage_client = gcs.Client(project=project_id, credentials=creds)
        return cls(
            client=client,
            storage_client=storage_client,
            gcs_bucket=gcs_bucket,
            model=model,
        )

    def create_batch(
        self, jsonl_data: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Upload input JSONL to GCS and start a Vertex batch prediction job."""
        model = config.get("model", self._model)
        display_name = config.get("display_name", f"batch-{int(time.time())}")

        jsonl_content = "\n".join(
            json.dumps(item, ensure_ascii=False) for item in jsonl_data
        )
        src_uri = self.upload_file(jsonl_content, purpose="batch")
        dest_uri = f"gs://{self._bucket}/{self._output_prefix}/{uuid4().hex}/"

        logger.info(
            f"[create_batch] Creating Vertex batch | items={len(jsonl_data)} | "
            f"model={model} | src={src_uri} | dest={dest_uri}"
        )

        try:
            batch_job = self._client.batches.create(
                model=model,
                src=src_uri,
                config=types.CreateBatchJobConfig(
                    dest=dest_uri, display_name=display_name
                ),
            )
            initial_state = batch_job.state.name if batch_job.state else "UNKNOWN"
            result = {
                "provider_batch_id": batch_job.name,
                "provider_file_id": src_uri,
                "provider_output_prefix": dest_uri,
                "provider_status": initial_state,
                "total_items": len(jsonl_data),
            }
            logger.info(
                f"[create_batch] Created Vertex batch | batch_id={batch_job.name} | "
                f"status={initial_state} | items={len(jsonl_data)}"
            )
            return result
        except Exception as e:
            logger.error(f"[create_batch] Failed to create Vertex batch | {e}")
            raise

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Poll Vertex for batch job status."""
        logger.info(f"[get_batch_status] Polling Vertex batch | batch_id={batch_id}")
        try:
            batch_job = self._client.batches.get(name=batch_id)
            state = batch_job.state.name if batch_job.state else "UNKNOWN"
            # Results live in the job's GCS dest; get_batch_status re-fetches it.
            output_uri = (
                batch_job.dest.gcs_uri
                if batch_job.dest and batch_job.dest.gcs_uri
                else None
            )
            result: dict[str, Any] = {
                "provider_status": state,
                "provider_output_file_id": output_uri or batch_id,
            }
            if state in _FAILED_STATES:
                message = batch_job.error.message if batch_job.error else state
                result["error_message"] = message
            logger.info(
                f"[get_batch_status] Vertex batch status | batch_id={batch_id} | "
                f"status={state}"
            )
            return result
        except Exception as e:
            logger.error(
                f"[get_batch_status] Failed to poll Vertex batch | "
                f"batch_id={batch_id} | {e}"
            )
            raise

    def download_batch_results(self, output_file_id: str) -> list[dict[str, Any]]:
        """Read prediction JSONL files from the batch job's GCS output prefix.

        Vertex echoes the input ``key`` per line; line order is only the fallback.
        """
        logger.info(
            f"[download_batch_results] Reading Vertex results | src={output_file_id}"
        )
        output_uri = output_file_id
        if not output_uri.startswith("gs://"):
            batch_job = self._client.batches.get(name=output_file_id)
            state = batch_job.state.name if batch_job.state else "UNKNOWN"
            if state != BatchJobState.SUCCEEDED.value:
                raise ValueError(f"Batch job not complete. Current state: {state}")
            if not (batch_job.dest and batch_job.dest.gcs_uri):
                raise ValueError(f"Batch job has no GCS output | id={output_file_id}")
            output_uri = batch_job.dest.gcs_uri

        try:
            bucket_name, prefix = _parse_gs_uri(output_uri)
            bucket = self._storage.bucket(bucket_name)
            results: list[dict[str, Any]] = []
            index = 0
            for blob in self._storage.list_blobs(bucket, prefix=prefix):
                if not blob.name.endswith(".jsonl"):
                    continue
                content = blob.download_as_text()
                for line in content.strip().split("\n"):
                    if not line:
                        continue
                    parsed = json.loads(line)
                    custom_id = parsed.get("key") or str(index)
                    response_obj = parsed.get("response")
                    error_obj = parsed.get("error") or parsed.get("status")
                    results.append(
                        {
                            BATCH_KEY: custom_id,
                            "response": response_obj,
                            "error": str(error_obj) if error_obj else None,
                        }
                    )
                    index += 1
            logger.info(
                f"[download_batch_results] Read Vertex results | src={output_uri} | "
                f"results={len(results)}"
            )
            return results
        except Exception as e:
            logger.error(
                f"[download_batch_results] Failed to read Vertex results | "
                f"src={output_uri} | {e}"
            )
            raise

    def upload_file(self, content: str, purpose: str = "batch") -> str:
        """Upload a JSONL string to GCS and return its ``gs://`` URI."""
        key = f"{self._input_prefix}/{int(time.time())}-{uuid4().hex}.jsonl"
        logger.info(f"[upload_file] Uploading batch input to GCS | key={key}")
        try:
            blob = self._storage.bucket(self._bucket).blob(key)
            blob.upload_from_string(content, content_type="application/jsonl")
            return f"gs://{self._bucket}/{key}"
        except Exception as e:
            logger.error(f"[upload_file] Failed to upload batch input to GCS | {e}")
            raise CloudStorageError(f"GCS upload failed: {e}") from e

    def download_file(self, file_id: str) -> str:
        """Download a ``gs://`` object's content as text."""
        logger.info(f"[download_file] Downloading from GCS | uri={file_id}")
        try:
            bucket_name, key = _parse_gs_uri(file_id)
            blob = self._storage.bucket(bucket_name).blob(key)
            return blob.download_as_text()
        except Exception as e:
            logger.error(
                f"[download_file] Failed to download from GCS | uri={file_id} | {e}"
            )
            raise CloudStorageError(f"GCS download failed: {e}") from e
