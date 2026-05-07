from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.exception_handlers import _sanitize_validation_errors
from app.tests.utils.auth import TestAuthContext


class TestSanitizeValidationErrors:
    """Unit tests for _sanitize_validation_errors."""

    def test_non_union_errors_pass_through(self) -> None:
        errors = [
            {"type": "missing", "loc": ("body", "name"), "msg": "Field required"},
        ]
        assert _sanitize_validation_errors(errors) == errors

    def test_picks_branch_with_fewer_literal_errors(self) -> None:
        errors = [
            {
                "type": "literal_error",
                "loc": ("body", "c", "NativeConfig", "provider"),
                "msg": "bad",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "NativeConfig", "params"),
                "msg": "Field required",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "KaapiConfig", "type"),
                "msg": "Field required",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "KaapiConfig", "params"),
                "msg": "Field required",
            },
        ]
        result = _sanitize_validation_errors(errors)
        assert len(result) == 2
        for err in result:
            assert "NativeConfig" not in err["loc"]

    def test_tied_branches_keep_both_and_dedup(self) -> None:
        """When two branches have the same literal_error count, both are kept but duplicates removed."""
        errors = [
            {
                "type": "missing",
                "loc": ("body", "c", "BranchA", "x"),
                "msg": "Field required",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "BranchB", "x"),
                "msg": "Field required",
            },
        ]
        result = _sanitize_validation_errors(errors)
        assert len(result) == 1
        assert result[0]["loc"] == ("body", "c", "x")

    def test_strips_branch_identifiers_from_loc(self) -> None:
        errors = [
            {
                "type": "missing",
                "loc": (
                    "body",
                    "cfg",
                    "completion",
                    "function-after[validate_params(), Foo]",
                    "params",
                ),
                "msg": "Field required",
            }
        ]
        result = _sanitize_validation_errors(errors)
        assert result[0]["loc"] == ("body", "cfg", "completion", "params")

    def test_non_union_preserved_with_union(self) -> None:
        errors = [
            {"type": "missing", "loc": ("body", "name"), "msg": "Field required"},
            {
                "type": "literal_error",
                "loc": ("body", "c", "NativeConfig", "p"),
                "msg": "bad",
            },
            {
                "type": "missing",
                "loc": ("body", "c", "KaapiConfig", "t"),
                "msg": "Field required",
            },
        ]
        result = _sanitize_validation_errors(errors)
        assert len(result) == 2
        locs = [r["loc"] for r in result]
        assert ("body", "name") in locs

    def test_empty_list(self) -> None:
        assert _sanitize_validation_errors([]) == []

    def test_fallback_on_malformed_input(self) -> None:
        malformed = [None, 42]  # type: ignore[list-item]
        result = _sanitize_validation_errors(malformed)
        assert result == malformed


class TestValidationErrorResponse:
    """Integration: structured errors via configs endpoint."""

    def test_structured_error_format(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        response = client.post(
            f"{settings.API_V1_STR}/configs",
            headers={"X-API-KEY": user_api_key.key},
            json={},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "Validation failed"
        assert isinstance(body["errors"], list)
        assert all("field" in e and "message" in e for e in body["errors"])

    def test_union_noise_filtered(
        self, client: TestClient, user_api_key: TestAuthContext
    ) -> None:
        response = client.post(
            f"{settings.API_V1_STR}/configs",
            headers={"X-API-KEY": user_api_key.key},
            json={
                "name": "test-config",
                "config_blob": {"completion": {"provider": "openai"}},
            },
        )
        assert response.status_code == 422
        for error in response.json()["errors"]:
            assert "openai-native" not in error["message"]
            assert "NativeCompletionConfig" not in error["field"]
