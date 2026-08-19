"""Tests for the bucket attachment utilities (path strategy + URL resolution)."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.buckets.attachments import (
    BucketPathStrategyEnum,
    is_gcs_uri,
    resolve_attachments,
    resolve_bucket_path_strategy,
)

_RESOLVER_PATH = "app.services.buckets.attachments.get_bucket_provider"


class TestResolveBucketPathStrategy:
    @pytest.mark.parametrize("llm_provider", ["google-gcp", "google-gcp-native"])
    def test_gcs_uri_native_provider_is_native(self, llm_provider):
        strategy = resolve_bucket_path_strategy(
            llm_provider=llm_provider,
            source_uri="gs://bucket/key.wav",
        )
        assert strategy is BucketPathStrategyEnum.NATIVE

    @pytest.mark.parametrize("llm_provider", ["openai", "anthropic", "google-aistudio"])
    def test_gcs_uri_non_native_provider_is_signed_url(self, llm_provider):
        strategy = resolve_bucket_path_strategy(
            llm_provider=llm_provider,
            source_uri="gs://bucket/key.wav",
        )
        assert strategy is BucketPathStrategyEnum.SIGNED_URL

    def test_non_gcs_uri_is_signed_url_even_for_native_provider(self):
        strategy = resolve_bucket_path_strategy(
            llm_provider="google-gcp",
            source_uri="https://example.com/key.wav",
        )
        assert strategy is BucketPathStrategyEnum.SIGNED_URL


class TestIsGcsUri:
    def test_gcs_uri(self):
        assert is_gcs_uri("gs://bucket/key.wav") is True

    def test_non_gcs_uri(self):
        assert is_gcs_uri("https://example.com/key.wav") is False


class TestResolveAttachmentsSingle:
    def test_https_returned_as_is_without_provider(self):
        with patch(_RESOLVER_PATH) as mock_get:
            url = resolve_attachments(
                session=MagicMock(),
                source="https://example.com/img.png",
                llm_provider="openai",
                project_id=1,
                organization_id=2,
            )
        assert url == "https://example.com/img.png"
        mock_get.assert_not_called()

    def test_gcs_native_passthrough(self):
        with patch(_RESOLVER_PATH) as mock_get:
            url = resolve_attachments(
                session=MagicMock(),
                source="gs://bucket/key.png",
                llm_provider="google-gcp",
                project_id=1,
                organization_id=2,
            )
        assert url == "gs://bucket/key.png"
        mock_get.assert_not_called()

    def test_gcs_signed_for_non_native_provider(self):
        provider = MagicMock()
        provider.get_bulk_signed_urls.return_value = {
            "gs://bucket/key.png": "https://signed"
        }
        with patch(_RESOLVER_PATH, return_value=provider):
            url = resolve_attachments(
                session=MagicMock(),
                source="gs://bucket/key.png",
                llm_provider="openai",
                project_id=1,
                organization_id=2,
                expires_in=1200,
            )
        assert url == "https://signed"
        provider.get_bulk_signed_urls.assert_called_once_with(
            ["gs://bucket/key.png"], expires_in=1200
        )


class TestResolveAttachmentsList:
    def test_mixed_schemes_partitioned(self):
        provider = MagicMock()
        provider.get_bulk_signed_urls.return_value = {"gs://b/2.png": "https://signed2"}
        with patch(_RESOLVER_PATH, return_value=provider):
            result = resolve_attachments(
                session=MagicMock(),
                source=["https://x/1.png", "gs://b/2.png"],
                llm_provider="anthropic",
                project_id=1,
                organization_id=2,
            )
        assert result == {
            "https://x/1.png": "https://x/1.png",
            "gs://b/2.png": "https://signed2",
        }
        provider.get_bulk_signed_urls.assert_called_once_with(
            ["gs://b/2.png"], expires_in=3600
        )

    def test_all_native_skips_provider(self):
        with patch(_RESOLVER_PATH) as mock_get:
            result = resolve_attachments(
                session=MagicMock(),
                source=["gs://b/1.wav", "gs://b/2.wav"],
                llm_provider="google-gcp",
                project_id=1,
                organization_id=2,
            )
        assert result == {
            "gs://b/1.wav": "gs://b/1.wav",
            "gs://b/2.wav": "gs://b/2.wav",
        }
        mock_get.assert_not_called()
