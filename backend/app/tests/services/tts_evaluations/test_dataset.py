import io
from unittest.mock import MagicMock

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlmodel import Session

from app.services.tts_evaluations.dataset import get_sample_texts_from_dataset
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import TEST_DATASET_URL, create_test_tts_dataset


def _storage(stream: object) -> MagicMock:
    storage = MagicMock()
    storage.stream.side_effect = stream
    return storage


@pytest.mark.parametrize(
    ("object_store_url", "stream", "expected"),
    [
        (
            TEST_DATASET_URL,
            lambda url: io.BytesIO(b"text\nHello world\n   \n  Good morning  \n"),
            ["Hello world", "Good morning"],
        ),
        (None, lambda url: io.BytesIO(b"text\nunused\n"), []),
        (TEST_DATASET_URL, OSError("s3 down"), []),
    ],
    ids=["reads_csv", "no_url", "stream_error"],
)
def test_get_sample_texts_from_dataset(
    db: Session,
    user_api_key: TestAuthContext,
    object_store_url: str | None,
    stream: object,
    expected: list[str],
) -> None:
    dataset = create_test_tts_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        object_store_url=object_store_url,
    )

    assert get_sample_texts_from_dataset(_storage(stream), dataset) == expected


def test_get_sample_texts_from_dataset_reraises_soft_timeout(
    db: Session, user_api_key: TestAuthContext
) -> None:
    dataset = create_test_tts_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )

    with pytest.raises(SoftTimeLimitExceeded):
        get_sample_texts_from_dataset(_storage(SoftTimeLimitExceeded()), dataset)
