"""Langfuse failures map to distinct status codes rather than a flat 500."""

from unittest.mock import MagicMock

import pytest
from langfuse.api import NotFoundError as LangfuseNotFoundError
from langfuse.api.core import ApiError as LangfuseApiError

from app.core.exceptions import InvalidPayloadError, NotFoundError, UpstreamError
from app.crud.evaluations.batch import fetch_dataset_items


def test_dataset_missing_raises_404() -> None:
    langfuse = MagicMock()
    langfuse.get_dataset.side_effect = LangfuseNotFoundError(
        body={"message": "Dataset not found"}
    )

    with pytest.raises(NotFoundError) as exc:
        fetch_dataset_items(langfuse=langfuse, dataset_name="missing")

    assert exc.value.status_code == 404
    assert "missing" in exc.value.detail


def test_langfuse_api_error_raises_502() -> None:
    langfuse = MagicMock()
    langfuse.get_dataset.side_effect = LangfuseApiError(status_code=500, body="boom")

    with pytest.raises(UpstreamError) as exc:
        fetch_dataset_items(langfuse=langfuse, dataset_name="ds")

    assert exc.value.status_code == 502
    assert exc.value.provider == "langfuse"


def test_langfuse_unreachable_raises_502() -> None:
    langfuse = MagicMock()
    langfuse.get_dataset.side_effect = ConnectionError("dns failure")

    with pytest.raises(UpstreamError) as exc:
        fetch_dataset_items(langfuse=langfuse, dataset_name="ds")

    assert exc.value.status_code == 502


def test_empty_dataset_raises_422() -> None:
    langfuse = MagicMock()
    langfuse.get_dataset.return_value = MagicMock(items=[])

    with pytest.raises(InvalidPayloadError) as exc:
        fetch_dataset_items(langfuse=langfuse, dataset_name="ds")

    assert exc.value.status_code == 422
