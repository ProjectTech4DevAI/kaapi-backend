from typing import Any, TypeVar

from sqlmodel import Session

from app.core.util import now
from app.models import EvaluationDataset, EvaluationRun, File, FileType
from app.models.job import JobStatus
from app.models.stt_evaluation import EvaluationType, STTResult, STTSample
from app.models.tts_evaluation import TTSResult
from app.tests.utils.utils import random_lower_string

T = TypeVar("T")


def _persist(db: Session, record: T) -> T:
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def create_test_speech_dataset(
    db: Session,
    *,
    organization_id: int,
    project_id: int,
    evaluation_type: EvaluationType,
    description: str | None = None,
    object_store_url: str | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> EvaluationDataset:
    return _persist(
        db,
        EvaluationDataset(
            name=f"{evaluation_type.value}_dataset_{random_lower_string()}",
            description=description,
            type=evaluation_type.value,
            object_store_url=object_store_url,
            dataset_metadata=dataset_metadata or {"sample_count": 0},
            organization_id=organization_id,
            project_id=project_id,
            inserted_at=now(),
            updated_at=now(),
        ),
    )


def create_test_stt_sample(
    db: Session,
    *,
    dataset_id: int,
    organization_id: int,
    project_id: int,
    ground_truth: str,
    object_store_url: str,
) -> STTSample:
    audio = _persist(
        db,
        File(
            object_store_url=object_store_url,
            filename=f"{random_lower_string()}.mp3",
            size_bytes=1024,
            content_type="audio/mpeg",
            file_type=FileType.AUDIO.value,
            organization_id=organization_id,
            project_id=project_id,
            inserted_at=now(),
            updated_at=now(),
        ),
    )
    return _persist(
        db,
        STTSample(
            file_id=audio.id,
            ground_truth=ground_truth,
            dataset_id=dataset_id,
            organization_id=organization_id,
            project_id=project_id,
            inserted_at=now(),
            updated_at=now(),
        ),
    )


def create_test_speech_run(
    db: Session,
    *,
    dataset: EvaluationDataset,
    evaluation_type: EvaluationType,
    providers: list[str],
    score: dict[str, Any] | None = None,
) -> EvaluationRun:
    return _persist(
        db,
        EvaluationRun(
            run_name=f"{evaluation_type.value}_run_{random_lower_string()}",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            type=evaluation_type.value,
            providers=providers,
            status="completed",
            total_items=1,
            score=score,
            organization_id=dataset.organization_id,
            project_id=dataset.project_id,
            inserted_at=now(),
            updated_at=now(),
        ),
    )


def create_test_stt_result(
    db: Session,
    *,
    run: EvaluationRun,
    sample: STTSample,
    transcription: str,
) -> STTResult:
    return _persist(
        db,
        STTResult(
            transcription=transcription,
            provider=run.providers[0],
            status=JobStatus.SUCCESS.value,
            stt_sample_id=sample.id,
            evaluation_run_id=run.id,
            organization_id=run.organization_id,
            project_id=run.project_id,
            inserted_at=now(),
            updated_at=now(),
        ),
    )


def create_test_tts_result(
    db: Session,
    *,
    run: EvaluationRun,
    sample_text: str,
    object_store_url: str,
) -> TTSResult:
    return _persist(
        db,
        TTSResult(
            sample_text=sample_text,
            object_store_url=object_store_url,
            provider=run.providers[0],
            status=JobStatus.SUCCESS.value,
            evaluation_run_id=run.id,
            organization_id=run.organization_id,
            project_id=run.project_id,
            inserted_at=now(),
            updated_at=now(),
        ),
    )
