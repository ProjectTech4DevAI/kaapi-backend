"""Tests for the GCS bucket provider."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.services.buckets.providers.gcs import (
    GCSBucketProvider,
    GCSClient,
)


def _make_provider() -> tuple[GCSBucketProvider, MagicMock]:
    """Build a provider over a mock storage client; return the blob mock too."""
    blob = MagicMock()
    storage_client = MagicMock()
    storage_client.bucket.return_value.blob.return_value = blob
    provider = GCSBucketProvider(
        client=GCSClient(storage_client=storage_client, default_bucket="b")
    )
    return provider, blob


class TestCreateClient:
    def test_uses_byok_credentials(self):
        byok_sa = {"project_id": "byok-project"}

        with (
            patch(
                "app.services.buckets.providers.gcs.build_gcp_sa_credentials"
            ) as mock_build,
            patch("app.services.buckets.providers.gcs.gcs.Client") as mock_client,
        ):
            client = GCSBucketProvider.create_client(
                {"gcs_bucket": "byok-bucket", "sa_key": byok_sa}
            )

        assert client.default_bucket == "byok-bucket"
        mock_build.assert_called_once_with(byok_sa)
        mock_client.assert_called_once_with(
            project="byok-project", credentials=mock_build.return_value
        )

    def test_missing_sa_info_raises(self):
        with pytest.raises(ValueError) as exc_info:
            GCSBucketProvider.create_client({"gcs_bucket": "b"})

        assert "sa_key" in str(exc_info.value)

    def test_missing_bucket_raises(self):
        with pytest.raises(ValueError) as exc_info:
            GCSBucketProvider.create_client({"sa_key": {"project_id": "p"}})

        assert "gcs_bucket" in str(exc_info.value)


class TestParseGcsUri:
    def test_parses_bucket_and_key(self):
        assert GCSBucketProvider._parse_gcs_uri(
            "gs://my-bucket/path/to/object.wav"
        ) == ("my-bucket", "path/to/object.wav")

    def test_non_gs_scheme_raises(self):
        with pytest.raises(ValueError):
            GCSBucketProvider._parse_gcs_uri("s3://my-bucket/key")

    def test_missing_key_raises(self):
        with pytest.raises(ValueError):
            GCSBucketProvider._parse_gcs_uri("gs://my-bucket")


class TestGetSignedUrl:
    def test_signs_with_v4_and_get(self):
        provider, blob = _make_provider()
        blob.generate_signed_url.return_value = "https://signed.example/obj"

        url = provider.get_signed_url("gs://my-bucket/key.wav", expires_in=1800)

        assert url == "https://signed.example/obj"
        provider.client.storage_client.bucket.assert_called_once_with("my-bucket")
        provider.client.storage_client.bucket.return_value.blob.assert_called_once_with(
            "key.wav"
        )
        blob.generate_signed_url.assert_called_once_with(
            version="v4",
            expiration=timedelta(seconds=1800),
            method="GET",
        )

    def test_expiry_capped_at_max(self):
        provider, blob = _make_provider()
        blob.generate_signed_url.return_value = "https://signed.example/obj"

        provider.get_signed_url(
            "gs://my-bucket/key.wav",
            expires_in=provider.MAX_SIGNED_URL_EXPIRY + 10_000,
        )

        _, kwargs = blob.generate_signed_url.call_args
        assert kwargs["expiration"] == timedelta(seconds=provider.MAX_SIGNED_URL_EXPIRY)


class TestGetBulkSignedUrls:
    def test_returns_uri_to_url_map_reusing_one_client(self):
        provider, blob = _make_provider()
        blob.generate_signed_url.side_effect = [
            "https://signed.example/a",
            "https://signed.example/b",
        ]

        result = provider.get_bulk_signed_urls(
            ["gs://bucket/a.wav", "gs://bucket/b.wav"], expires_in=86400
        )

        assert result == {
            "gs://bucket/a.wav": "https://signed.example/a",
            "gs://bucket/b.wav": "https://signed.example/b",
        }
        assert blob.generate_signed_url.call_count == 2
