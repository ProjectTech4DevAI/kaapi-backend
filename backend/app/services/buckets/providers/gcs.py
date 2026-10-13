"""GCS bucket provider: signed-URL generation over Google Cloud Storage."""

import logging
from typing import Any
from datetime import timedelta
from urllib.parse import urlparse

from google.cloud import storage as gcs

from app.core.cloud.storage import CloudStorageError, build_gcp_sa_credentials
from app.services.buckets.providers.base import BaseBucketProvider

logger = logging.getLogger(__name__)

GCS_URI_SCHEME = "gs"


class GCSClient:
    # default_bucket carried for parity with the credential row; signing derives
    # the bucket from each URI.
    def __init__(self, storage_client: gcs.Client, default_bucket: str | None):
        self.storage_client = storage_client
        self.default_bucket = default_bucket


class GCSBucketProvider(BaseBucketProvider):
    """Bucket provider for Google Cloud Storage (``gs://`` scheme)."""

    SCHEME = GCS_URI_SCHEME

    def __init__(self, client: GCSClient):
        super().__init__(client)
        self.client = client

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> GCSClient:
        """Build a signing-capable GCS client from the given credentials."""
        credentials = credentials or {}
        sa_info = credentials.get("sa_key")
        gcs_bucket = credentials.get("gcs_bucket")

        if not sa_info:
            raise ValueError(
                "GCS bucket provider requires a service-account key (sa_key) in "
                "the credentials to sign URLs."
            )
        if not gcs_bucket:
            raise ValueError(
                "GCS bucket provider requires a gcs_bucket in the credentials."
            )

        logger.info(
            f"[GCSBucketProvider.create_client] gcs creds | bucket={gcs_bucket}"
        )

        creds = build_gcp_sa_credentials(sa_info)
        storage_client = gcs.Client(
            project=sa_info.get("project_id"), credentials=creds
        )
        return GCSClient(storage_client=storage_client, default_bucket=gcs_bucket)

    @staticmethod
    def _parse_gcs_uri(uri: str) -> tuple[str, str]:
        """Split ``gs://bucket/key`` into ``(bucket, key)``."""
        parsed = urlparse(uri)
        if parsed.scheme != GCS_URI_SCHEME or not parsed.netloc:
            raise ValueError(f"Invalid GCS URI '{uri}'; expected 'gs://bucket/key'.")
        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError(f"GCS URI '{uri}' is missing an object key.")
        return parsed.netloc, key

    def get_signed_url(self, uri: str, expires_in: int) -> str:
        if expires_in < 1:
            raise ValueError(
                f"Signed-URL expiry must be at least 1 second, got {expires_in}."
            )
        expires_in = min(expires_in, self.MAX_SIGNED_URL_EXPIRY)
        bucket_name, key = self._parse_gcs_uri(uri)

        try:
            blob = self.client.storage_client.bucket(bucket_name).blob(key)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method="GET",
            )
        except Exception as e:
            logger.error(
                f"[GCSBucketProvider.get_signed_url] GCS signing failed | "
                f"bucket={bucket_name}, key={key}, error={e}",
                exc_info=True,
            )
            raise CloudStorageError(f"GCS signing failed: {e} ({uri})") from e

        logger.info(
            f"[GCSBucketProvider.get_signed_url] Signed URL generated | "
            f"bucket={bucket_name}, key={key}, expires_in={expires_in}"
        )
        return signed_url

    def get_bulk_signed_urls(self, uris: list[str], expires_in: int) -> dict[str, str]:
        """Sign each URI reusing this provider's single client."""
        logger.info(
            f"[GCSBucketProvider.get_bulk_signed_urls] Signing batch | "
            f"count={len(uris)}, expires_in={expires_in}"
        )
        return {uri: self.get_signed_url(uri, expires_in=expires_in) for uri in uris}
