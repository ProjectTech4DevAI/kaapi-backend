"""Tests for the abstract bucket-provider base class."""

from typing import Any

import pytest

from app.services.buckets.providers.base import BaseBucketProvider


class _StubProvider(BaseBucketProvider):
    """Concrete provider that signs by echoing the URI, for the bulk loop."""

    @staticmethod
    def create_client(credentials: dict[str, Any]) -> Any:
        return None

    def get_signed_url(self, uri: str, expires_in: int) -> str:
        return f"{uri}?exp={expires_in}"


class TestBaseBucketProviderAbstractMethods:
    def test_create_client_body_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="create_client"):
            BaseBucketProvider.create_client({})

    def test_get_signed_url_body_raises_not_implemented(self):
        provider = _StubProvider(client=None)
        with pytest.raises(NotImplementedError, match="get_signed_url"):
            BaseBucketProvider.get_signed_url(provider, "gs://b/k", 60)


class TestGetBulkSignedUrls:
    def test_maps_each_uri_via_get_signed_url(self):
        provider = _StubProvider(client=None)
        result = provider.get_bulk_signed_urls(["gs://b/a", "gs://b/c"], expires_in=120)
        assert result == {
            "gs://b/a": "gs://b/a?exp=120",
            "gs://b/c": "gs://b/c?exp=120",
        }
