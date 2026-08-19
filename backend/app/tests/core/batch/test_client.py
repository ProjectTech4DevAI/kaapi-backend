"""Test cases for the Gemini-family batch provider resolver."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.core.batch.client import (
    get_gemini_batch_provider,
    is_vertex_batch_provider,
)
from app.core.batch.gemini import GeminiBatchProvider


class TestIsVertexBatchProvider:
    @pytest.mark.parametrize("name", ["google-gcp", "google-gcp-native"])
    def test_vertex_providers(self, name):
        assert is_vertex_batch_provider(name) is True

    @pytest.mark.parametrize("name", ["google", "google-aistudio", "openai"])
    def test_non_vertex_providers(self, name):
        assert is_vertex_batch_provider(name) is False


class TestGetGeminiBatchProvider:
    def test_google_routes_to_aistudio(self):
        gemini = MagicMock()
        gemini.client = MagicMock()
        with patch(
            "app.core.batch.client.GeminiClient.from_credentials",
            return_value=gemini,
        ) as from_cred:
            provider = get_gemini_batch_provider(
                session=MagicMock(),
                organization_id=1,
                project_id=2,
                provider_name="google",
            )
        assert isinstance(provider, GeminiBatchProvider)
        from_cred.assert_called_once()

    def test_google_gcp_routes_to_vertex(self):
        sentinel = MagicMock()
        with (
            patch(
                "app.core.batch.client.get_provider_credential",
                return_value={"gcs_bucket": "b", "sa_key": {}},
            ) as get_cred,
            patch(
                "app.core.batch.vertex.VertexBatchProvider.from_credentials",
                return_value=sentinel,
            ) as from_cred,
        ):
            provider = get_gemini_batch_provider(
                session=MagicMock(),
                organization_id=1,
                project_id=2,
                provider_name="google-gcp",
                model="gemini-2.5-pro",
            )
        assert provider is sentinel
        assert get_cred.call_args.kwargs["provider"] == "google-gcp"
        assert from_cred.call_args.kwargs["model"] == "gemini-2.5-pro"

    def test_missing_gcp_credential_raises_404(self):
        with patch(
            "app.core.batch.client.get_provider_credential", return_value=None
        ):
            with pytest.raises(HTTPException) as exc:
                get_gemini_batch_provider(
                    session=MagicMock(),
                    organization_id=1,
                    project_id=2,
                    provider_name="google-gcp",
                )
        assert exc.value.status_code == 404
