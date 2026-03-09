from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.core.exception_handlers import _filter_union_branch_errors
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_evaluation_dataset


# ---------------------------------------------------------------------------
# Unit tests for _filter_union_branch_errors
# ---------------------------------------------------------------------------


class TestFilterUnionBranchErrors:
    """Unit tests for the discriminated union branch error filter."""

    def test_no_union_errors_returned_unchanged(self) -> None:
        """Non-union errors pass through unchanged."""
        errors = [
            {"type": "missing", "loc": ("body", "name"), "msg": "Field required"},
            {
                "type": "missing",
                "loc": ("body", "config_blob"),
                "msg": "Field required",
            },
        ]
        result = _filter_union_branch_errors(errors)
        assert result == errors

    def test_single_branch_errors_passed_through(self) -> None:
        """When only one branch has errors it is included without filtering."""
        errors = [
            {
                "type": "missing",
                "loc": ("body", "completion", "KaapiCompletionConfig", "type"),
                "msg": "Field required",
            }
        ]
        result = _filter_union_branch_errors(errors)
        assert len(result) == 1
        # Branch identifier stripped from loc
        assert "KaapiCompletionConfig" not in result[0]["loc"]

    def test_picks_branch_with_fewer_literal_errors(self) -> None:
        """When multiple branches exist, the one with fewer literal_errors wins."""
        errors = [
            # NativeCompletionConfig branch — provider literal_error (wrong value)
            {
                "type": "literal_error",
                "loc": ("body", "completion", "NativeCompletionConfig", "provider"),
                "msg": "Input should be 'openai-native'",
            },
            {
                "type": "missing",
                "loc": ("body", "completion", "NativeCompletionConfig", "params"),
                "msg": "Field required",
            },
            # KaapiCompletionConfig branch — no literal_error (provider matched)
            {
                "type": "missing",
                "loc": ("body", "completion", "KaapiCompletionConfig", "type"),
                "msg": "Field required",
            },
            {
                "type": "missing",
                "loc": ("body", "completion", "KaapiCompletionConfig", "params"),
                "msg": "Field required",
            },
        ]
        result = _filter_union_branch_errors(errors)
        # Only KaapiCompletionConfig errors should remain
        assert len(result) == 2
        for err in result:
            assert "NativeCompletionConfig" not in err["loc"]
            assert "KaapiCompletionConfig" not in err["loc"]

    def test_branch_identifiers_stripped_from_loc(self) -> None:
        """Branch class names and pydantic internals are removed from loc tuples."""
        errors = [
            {
                "type": "missing",
                "loc": (
                    "body",
                    "config_blob",
                    "completion",
                    "function-after[validate_params(), KaapiCompletionConfig]",
                    "params",
                ),
                "msg": "Field required",
            }
        ]
        result = _filter_union_branch_errors(errors)
        assert len(result) == 1
        loc = result[0]["loc"]
        assert "function-after[validate_params(), KaapiCompletionConfig]" not in loc
        assert loc == ("body", "config_blob", "completion", "params")

    def test_non_union_errors_preserved_alongside_union_errors(self) -> None:
        """Top-level field errors coexist with filtered union branch errors."""
        errors = [
            # Top-level missing field (not a union branch error)
            {"type": "missing", "loc": ("body", "name"), "msg": "Field required"},
            # Union branch errors
            {
                "type": "literal_error",
                "loc": ("body", "completion", "NativeCompletionConfig", "provider"),
                "msg": "Input should be 'openai-native'",
            },
            {
                "type": "missing",
                "loc": ("body", "completion", "KaapiCompletionConfig", "type"),
                "msg": "Field required",
            },
        ]
        result = _filter_union_branch_errors(errors)
        # name error + KaapiCompletionConfig error
        assert len(result) == 2
        locs = [r["loc"] for r in result]
        assert ("body", "name") in locs

    def test_empty_errors_list(self) -> None:
        """Empty list returns empty list without raising."""
        assert _filter_union_branch_errors([]) == []

    def test_fallback_on_malformed_input(self) -> None:
        """Malformed errors are returned as-is via the try/except fallback."""
        # Passing non-dict items — should not raise, returns original list
        malformed = [None, 42]  # type: ignore[list-item]
        result = _filter_union_branch_errors(malformed)
        assert result == malformed


# ---------------------------------------------------------------------------
# Integration tests — validation error response format via API
# ---------------------------------------------------------------------------


class TestValidationErrorResponseFormat:
    """Test that the structured errors array is returned correctly by the API."""

    def test_missing_required_field_returns_structured_errors(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """Missing required field returns {field, message} structured error."""
        # config_blob is present but name is missing
        response = client.post(
            f"{settings.API_V1_STR}/configs/",
            headers={"X-API-KEY": user_api_key.key},
            json={
                "config_blob": {
                    "completion": {
                        "provider": "openai",
                        "type": "text",
                        "params": {"model": "gpt-4o-mini"},
                    }
                }
            },
        )
        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "Validation failed"
        assert body["errors"] is not None
        assert isinstance(body["errors"], list)

        fields = [e["field"] for e in body["errors"]]
        assert "name" in fields

        name_error = next(e for e in body["errors"] if e["field"] == "name")
        assert "required" in name_error["message"].lower()

    def test_union_branch_noise_not_in_response(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """NativeCompletionConfig errors must not appear when using openai provider."""
        response = client.post(
            f"{settings.API_V1_STR}/configs/",
            headers={"X-API-KEY": user_api_key.key},
            json={
                "name": "test-config",
                "config_blob": {
                    "completion": {
                        "provider": "openai",
                        # type and params are intentionally missing to trigger errors
                    }
                },
            },
        )
        assert response.status_code == 422
        body = response.json()
        assert body["errors"] is not None

        # No NativeCompletionConfig literal errors should be in the response
        for error in body["errors"]:
            assert "openai-native" not in error["message"]
            assert "NativeCompletionConfig" not in error["field"]

    def test_nested_field_path_in_error(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """Field path shows full dotted path, not just the last segment."""
        response = client.post(
            f"{settings.API_V1_STR}/configs/",
            headers={"X-API-KEY": user_api_key.key},
            json={
                "name": "test-config",
                "config_blob": {
                    "completion": {
                        "provider": "openai",
                        "type": "text",
                        # params missing — error should show config_blob.completion.params
                    }
                },
            },
        )
        assert response.status_code == 422
        body = response.json()
        fields = [e["field"] for e in body["errors"]]
        # Should show full path, not just "params"
        assert any("." in f for f in fields)
        assert any("params" in f for f in fields)

    def test_error_response_structure(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """Validation error response always has success=False, error summary, and errors array."""
        response = client.post(
            f"{settings.API_V1_STR}/configs/",
            headers={"X-API-KEY": user_api_key.key},
            json={},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["data"] is None
        assert body["error"] == "Validation failed"
        assert isinstance(body["errors"], list)
        assert len(body["errors"]) > 0
        for err in body["errors"]:
            assert "field" in err
            assert "message" in err


# ---------------------------------------------------------------------------
# Integration tests — dataset signed URL
# ---------------------------------------------------------------------------


class TestDatasetSignedUrl:
    """Test GET /evaluations/datasets/{id} signed URL feature."""

    def test_get_dataset_without_signed_url(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """By default signed_url is not included in the response."""
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        response = client.get(
            f"{settings.API_V1_STR}/evaluations/datasets/{dataset.id}",
            headers={"X-API-KEY": user_api_key.key},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["dataset_id"] == dataset.id
        assert body["data"].get("signed_url") is None

    def test_get_dataset_with_signed_url(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """include_signed_url=true returns a presigned URL."""
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        mock_signed_url = "https://s3.amazonaws.com/bucket/key?X-Amz-Signature=abc123"

        with patch(
            "app.api.routes.evaluations.dataset.get_cloud_storage"
        ) as mock_get_storage:
            mock_storage = mock_get_storage.return_value
            mock_storage.get_signed_url.return_value = mock_signed_url

            response = client.get(
                f"{settings.API_V1_STR}/evaluations/datasets/{dataset.id}",
                headers={"X-API-KEY": user_api_key.key},
                params={"include_signed_url": True},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["signed_url"] == mock_signed_url

    def test_get_dataset_signed_url_none_when_no_object_store_url(
        self,
        db: Session,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """signed_url is None when dataset has no object_store_url."""
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        # Ensure no object_store_url
        dataset.object_store_url = None
        db.add(dataset)
        db.commit()

        with patch(
            "app.api.routes.evaluations.dataset.get_cloud_storage"
        ) as mock_get_storage:
            response = client.get(
                f"{settings.API_V1_STR}/evaluations/datasets/{dataset.id}",
                headers={"X-API-KEY": user_api_key.key},
                params={"include_signed_url": True},
            )
            mock_get_storage.assert_not_called()

        assert response.status_code == 200
        body = response.json()
        assert body["data"].get("signed_url") is None

    def test_get_dataset_not_found(
        self,
        client: TestClient,
        user_api_key: TestAuthContext,
    ) -> None:
        """Non-existent dataset returns 404."""
        response = client.get(
            f"{settings.API_V1_STR}/evaluations/datasets/999999",
            headers={"X-API-KEY": user_api_key.key},
        )
        assert response.status_code == 404
        body = response.json()
        assert body["success"] is False
