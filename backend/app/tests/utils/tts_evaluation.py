"""Factories and session helpers for TTS evaluation tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from sqlmodel import Session

from app.core.util import now
from app.crud.tts_evaluations import create_tts_run
from app.models import EvaluationDataset, EvaluationRun
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType
from app.models.tts_evaluation import TTSResult
from app.services.tts_evaluations.constants import GEMINI_PRO_TTS_MODEL
from app.tests.utils.utils import random_lower_string

TEST_DATASET_URL = "s3://test-bucket/tts_datasets/test.csv"


class NonClosingSession:
    """Lets a service's `with Session(engine)` reuse the transactional test session.

    Services open their own sessions; pointing them at the conftest `db` keeps
    every write inside the per-test rollback and visible to the test's asserts.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(self, *_exc: object) -> bool:
        return False


@contextmanager
def use_test_session(module_path: str, db: Session) -> Iterator[Session]:
    with patch(
        f"{module_path}.Session", side_effect=lambda *_a, **_k: NonClosingSession(db)
    ):
        yield db


class FakeGeminiBatchProvider:
    """Stands in for the Gemini Batch API (the network boundary of a TTS batch).

    `failures` maps a model path ("models/<model>") to the exception its
    create_batch raises; every call's JSONL is recorded per model path.
    """

    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.submitted: dict[str, list[dict[str, Any]]] = {}

    def __call__(
        self, *, client: Any, model: str | None = None
    ) -> "FakeGeminiBatchProvider":
        self._model = model or ""
        return self

    def create_batch(
        self, jsonl_data: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        if self._model in self.failures:
            raise self.failures[self._model]
        self.submitted[self._model] = jsonl_data
        return {
            "provider_batch_id": f"batches/{self._model.split('/')[-1]}",
            "provider_file_id": "files/input",
            "provider_status": "JOB_STATE_PENDING",
            "total_items": len(jsonl_data),
        }


@contextmanager
def fake_gemini_batch(
    module_path: str, failures: dict[str, Exception] | None = None
) -> Iterator[FakeGeminiBatchProvider]:
    fake = FakeGeminiBatchProvider(failures)
    with (
        patch(f"{module_path}.GeminiClient") as gemini_client,
        patch(f"{module_path}.GeminiBatchProvider", side_effect=fake),
    ):
        gemini_client.from_credentials.return_value = MagicMock()
        yield fake


def create_test_tts_dataset(
    db: Session,
    *,
    organization_id: int,
    project_id: int,
    language_id: int | None = None,
    object_store_url: str | None = TEST_DATASET_URL,
    sample_count: int = 1,
) -> EvaluationDataset:
    dataset = EvaluationDataset(
        name=f"tts_dataset_{random_lower_string()}",
        type=EvaluationType.TTS.value,
        language_id=language_id,
        object_store_url=object_store_url,
        dataset_metadata={"sample_count": sample_count},
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def create_test_tts_run_with_dataset(
    db: Session,
    *,
    organization_id: int,
    project_id: int,
    models: list[str] | None = None,
    language_id: int | None = None,
    object_store_url: str | None = TEST_DATASET_URL,
) -> EvaluationRun:
    dataset = create_test_tts_dataset(
        db,
        organization_id=organization_id,
        project_id=project_id,
        language_id=language_id,
        object_store_url=object_store_url,
    )
    return create_tts_run(
        session=db,
        run_name=f"tts_run_{random_lower_string()}",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        org_id=organization_id,
        project_id=project_id,
        models=models or [GEMINI_PRO_TTS_MODEL],
    )


def create_test_tts_result_row(
    db: Session,
    *,
    run: EvaluationRun,
    provider: str = GEMINI_PRO_TTS_MODEL,
    status: JobStatus = JobStatus.PENDING,
    sample_text: str | None = None,
    object_store_url: str | None = None,
) -> TTSResult:
    result = TTSResult(
        sample_text=sample_text or f"text {random_lower_string()}",
        object_store_url=object_store_url,
        provider=provider,
        status=status.value,
        evaluation_run_id=run.id,
        organization_id=run.organization_id,
        project_id=run.project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result
