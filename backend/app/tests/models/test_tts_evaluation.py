import pytest
from pydantic import ValidationError

from app.models.job import JobStatus
from app.models.tts_evaluation import TTSEvaluationRunCreate, TTSResultUpdate


class TestTTSEvaluationRunCreate:
    def test_accepts_sync_and_batch_models_together(self) -> None:
        run = TTSEvaluationRunCreate(
            run_name="mixed",
            dataset_id=1,
            models=[
                "gemini-2.5-pro-preview-tts",
                "bulbul:v3",
                "eleven_v3",
                "eleven_v4",
            ],
        )

        assert run.models == [
            "gemini-2.5-pro-preview-tts",
            "bulbul:v3",
            "eleven_v3",
            "eleven_v4",
        ]

    def test_defaults_to_gemini(self) -> None:
        run = TTSEvaluationRunCreate(run_name="default", dataset_id=1)

        assert run.models == ["gemini-2.5-pro-preview-tts"]

    def test_rejects_unknown_model(self) -> None:
        with pytest.raises(
            ValidationError, match="Unsupported model\\(s\\): eleven_v9"
        ):
            TTSEvaluationRunCreate(
                run_name="bad", dataset_id=1, models=["bulbul:v3", "eleven_v9"]
            )


class TestTTSResultUpdate:
    def test_optional_fields_default_to_none(self) -> None:
        update = TTSResultUpdate(result_id=7, status=JobStatus.FAILED)

        assert update.object_store_url is None
        assert update.metadata is None
        assert update.error_message is None

    def test_rejects_unknown_status(self) -> None:
        with pytest.raises(ValidationError):
            TTSResultUpdate(result_id=7, status="exploded")
