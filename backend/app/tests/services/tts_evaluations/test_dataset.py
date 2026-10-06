import io
from unittest.mock import MagicMock, patch

from sqlmodel import Session

from app.services.tts_evaluations.dataset import (
    get_sample_texts_from_dataset,
    load_sample_texts,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.tts_evaluation import TEST_DATASET_URL, create_test_tts_dataset

CSV_BYTES = b"text\nHello world\n   \n  Good morning  \n"


def _storage(content: bytes = CSV_BYTES) -> MagicMock:
    def stream(url: str) -> io.BytesIO:
        if url != TEST_DATASET_URL:
            raise FileNotFoundError(url)
        return io.BytesIO(content)

    storage = MagicMock()
    storage.stream.side_effect = stream
    return storage


class TestLoadSampleTexts:
    def test_reads_non_blank_stripped_texts(self) -> None:
        storage = _storage()

        texts = load_sample_texts(
            storage=storage, object_store_url=TEST_DATASET_URL, dataset_id=1
        )

        assert texts == ["Hello world", "Good morning"]
        storage.stream.assert_called_once_with(TEST_DATASET_URL)

    def test_stream_failure_returns_empty(self) -> None:
        storage = MagicMock()
        storage.stream.side_effect = OSError("s3 down")

        assert (
            load_sample_texts(
                storage=storage, object_store_url=TEST_DATASET_URL, dataset_id=1
            )
            == []
        )

    def test_undecodable_csv_returns_empty(self) -> None:
        texts = load_sample_texts(
            storage=_storage(b"\xff\xfe\xfa"),
            object_store_url=TEST_DATASET_URL,
            dataset_id=1,
        )

        assert texts == []


class TestGetSampleTextsFromDataset:
    def test_reads_dataset_csv(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        dataset = create_test_tts_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with patch(
            "app.services.tts_evaluations.dataset.get_cloud_storage",
            return_value=_storage(),
        ):
            texts = get_sample_texts_from_dataset(db, dataset, user_api_key.project_id)

        assert texts == ["Hello world", "Good morning"]

    def test_dataset_without_url_returns_empty(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        dataset = create_test_tts_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            object_store_url=None,
        )

        with patch(
            "app.services.tts_evaluations.dataset.get_cloud_storage"
        ) as get_storage:
            assert (
                get_sample_texts_from_dataset(db, dataset, user_api_key.project_id)
                == []
            )

        get_storage.assert_not_called()

    def test_storage_init_failure_returns_empty(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        dataset = create_test_tts_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        with patch(
            "app.services.tts_evaluations.dataset.get_cloud_storage",
            side_effect=ValueError("no storage credentials"),
        ):
            assert (
                get_sample_texts_from_dataset(db, dataset, user_api_key.project_id)
                == []
            )
