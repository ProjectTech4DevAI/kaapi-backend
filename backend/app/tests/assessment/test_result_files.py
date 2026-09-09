"""Tests for the durable result-file dumps (app/services/assessment/api/result_files.py).

Real assessment/execution/batch_job rows on the transactional session; object storage
and the provider client are the only seams stubbed.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.core.util import now
from app.crud.assessment import api
from app.models.assessment import AssessmentMethod, BatchRunState
from app.models.batch_job import BatchJob, BatchJobType
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.services.assessment.api.batch import ApiStage
from app.services.assessment.api.result_files import (
    build_and_upload_errors,
    build_callback_metadata,
    finalize_result_files,
    record_stage_dump,
    stage_file_kind,
)
from app.tests.utils.auth import get_user_test_auth_context
from app.tests.utils.test_data import create_test_config
from app.tests.utils.utils import random_lower_string

ONE_DAY_SECONDS = 86400

_BLOB = AssessmentConfigBlob.model_validate(
    {
        "input_schema": {"a": {"type": "text"}},
        "assessment": {
            "provider": "openai",
            "type": "text",
            "params": {"model": "gpt-4o", "submission": "assess {a}"},
        },
    }
)


def _bag(**overrides) -> BatchRunState:
    bag: dict = {
        "pipeline": [{"stage": ApiStage.ASSESSMENT.value, "kind": "ASSESSMENT"}],
        "stage": ApiStage.ASSESSMENT.value,
        "stage_status": "COMPLETED",
        "stage_batches": {},
        "stage_output_urls": {},
        "verdicts": {},
        "counters": {},
        "gate_passed": [True],
        "provider": "openai",
        "model": "gpt-4o",
        "input_schema": None,
        "callback_url": "",
        "request_metadata": None,
    }
    bag.update(overrides)
    return bag  # type: ignore[return-value]


def _seed(db, auth, *, rows: int = 1):
    assessment = api.create_assessment(
        session=db,
        method=AssessmentMethod.BATCH,
        input={"data": [{"a": str(i)} for i in range(rows)]},
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )
    config = create_test_config(
        db,
        project_id=auth.project_id,
        name=f"assess-{random_lower_string()}",
        config_blob=_BLOB,
        tag=ConfigTag.ASSESSMENT,
    )
    execution = api.create_execution(
        session=db,
        assessment_id=assessment.id,
        config_id=config.id,
        config_version=1,
        total_items=rows,
    )
    return assessment, execution


def _batch_job(db, auth, **kwargs) -> BatchJob:
    job = BatchJob(
        provider="openai",
        job_type=BatchJobType.ASSESSMENT.value,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        total_items=1,
        **kwargs,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


class _Uploads:
    """Records every errors.jsonl upload instead of writing to object storage."""

    def __init__(self, url: str | None = "s3://bucket/errors.jsonl") -> None:
        self.url = url
        self.calls: list[dict] = []

    def __call__(self, *, storage, results, filename, subdirectory) -> str | None:
        self.calls.append(
            {
                "rows": list(results),
                "filename": filename,
                "subdirectory": subdirectory,
            }
        )
        return self.url

    @property
    def rows(self) -> list[dict]:
        return self.calls[-1]["rows"]


def _storage_patch(storage: MagicMock | None = None):
    return patch(
        "app.services.assessment.api.result_files.get_cloud_storage",
        return_value=storage or MagicMock(),
    )


def _upload_patch(uploads: _Uploads):
    return patch(
        "app.services.assessment.api.result_files.upload_jsonl_to_object_store",
        new=uploads,
    )


class TestStageFileKind:
    def test_assessment_stage_is_the_runs_results(self) -> None:
        assert stage_file_kind(ApiStage.ASSESSMENT.value) == "results"

    def test_prefilter_stage_is_suffixed(self) -> None:
        assert stage_file_kind(ApiStage.TOPIC_RELEVANCE.value) == (
            "topic_relevance_results"
        )


class TestRecordStageDump:
    def test_dump_is_on_the_parent_row_before_any_terminal_state(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)

        record_stage_dump(
            session=db,
            assessment=assessment,
            stage=ApiStage.TOPIC_RELEVANCE.value,
            url="s3://bucket/batch-1170/output.jsonl",
            count=998,
        )

        db.refresh(assessment)
        assert assessment.result_files == {
            "topic_relevance_results": {
                "url": "s3://bucket/batch-1170/output.jsonl",
                "count": 998,
            }
        }

    def test_missing_url_records_nothing(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)

        record_stage_dump(
            session=db,
            assessment=assessment,
            stage=ApiStage.ASSESSMENT.value,
            url=None,
            count=0,
        )

        db.refresh(assessment)
        assert assessment.result_files == {}


class TestBuildAndUploadErrors:
    def test_clean_success_still_uploads_an_empty_file(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)
        uploads = _Uploads()

        with _storage_patch(), _upload_patch(uploads):
            url, count = build_and_upload_errors(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(),
                failure_message=None,
            )

        assert (url, count) == ("s3://bucket/errors.jsonl", 0)
        assert uploads.rows == []
        assert uploads.calls[0]["filename"] == "errors.jsonl"
        assert (
            uploads.calls[0]["subdirectory"] == f"assessment/execution-{execution.id}"
        )

    def test_row_errors_are_flattened_per_stage(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth, rows=2)
        uploads = _Uploads()
        bag = _bag(
            stage_errors={ApiStage.ASSESSMENT.value: {"1": "rate limit exceeded"}}
        )

        with _storage_patch(), _upload_patch(uploads):
            _, count = build_and_upload_errors(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=bag,
                failure_message=None,
            )

        assert count == 1
        assert uploads.rows == [
            {
                "type": "row_error",
                "stage": ApiStage.ASSESSMENT.value,
                "row_index": 1,
                "error": "rate limit exceeded",
            }
        ]

    def test_openai_error_file_lines_become_rows(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)
        job = _batch_job(db, auth, provider_error_file_id="file-err-1")
        uploads = _Uploads()
        error_file = (
            json.dumps({"custom_id": "row_0", "error": {"message": "bad request"}})
            + "\n"
            + json.dumps({"custom_id": "row_3", "error": {"message": "too long"}})
            + "\n"
        )
        provider = MagicMock()
        provider.download_file.return_value = error_file

        with (
            _storage_patch(),
            _upload_patch(uploads),
            patch(
                "app.services.assessment.api.result_files._build_batch_provider",
                return_value=provider,
            ),
        ):
            _, count = build_and_upload_errors(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(stage_batches={ApiStage.ASSESSMENT.value: job.id}),
                failure_message=None,
            )

        assert count == 2
        assert [row["type"] for row in uploads.rows] == [
            "provider_error_file",
            "provider_error_file",
        ]
        assert [row["entry"]["custom_id"] for row in uploads.rows] == [
            "row_0",
            "row_3",
        ]
        assert uploads.rows[0]["provider_error_file_id"] == "file-err-1"
        provider.download_file.assert_called_once_with("file-err-1")

    def test_unreadable_error_file_degrades_to_one_row(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)
        job = _batch_job(db, auth, provider_error_file_id="file-err-2")
        uploads = _Uploads()
        provider = MagicMock()
        provider.download_file.side_effect = RuntimeError("404 file expired")

        with (
            _storage_patch(),
            _upload_patch(uploads),
            patch(
                "app.services.assessment.api.result_files._build_batch_provider",
                return_value=provider,
            ),
        ):
            _, count = build_and_upload_errors(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(stage_batches={ApiStage.ASSESSMENT.value: job.id}),
                failure_message=None,
            )

        assert count == 1
        row = uploads.rows[0]
        assert row["type"] == "provider_error_file_unavailable"
        assert row["provider_error_file_id"] == "file-err-2"
        assert "404 file expired" in row["error"]

    def test_batch_without_an_error_file_contributes_nothing(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)
        job = _batch_job(db, auth)  # Anthropic/Gemini report errors inline, no file id
        uploads = _Uploads()

        with _storage_patch(), _upload_patch(uploads):
            _, count = build_and_upload_errors(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(stage_batches={ApiStage.ASSESSMENT.value: job.id}),
                failure_message=None,
            )

        assert count == 0

    def test_storage_outage_yields_no_url(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)

        with patch(
            "app.services.assessment.api.result_files.get_cloud_storage",
            side_effect=RuntimeError("s3 unreachable"),
        ):
            url, count = build_and_upload_errors(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(),
                failure_message="kaboom",
            )

        assert url is None
        assert count == 1


class TestFinalizeResultFiles:
    def test_pre_provider_failure_records_only_a_synthetic_execution_error(
        self, db
    ) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)
        uploads = _Uploads()

        with _storage_patch(), _upload_patch(uploads):
            finalize_result_files(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(),
                failure_message="Vertex model gemini-2.5-pro not found",
            )

        db.refresh(assessment)
        assert set(assessment.result_files) == {"errors"}
        assert assessment.result_files["errors"] == {
            "url": "s3://bucket/errors.jsonl",
            "count": 1,
        }
        assert uploads.rows == [
            {
                "type": "execution_error",
                "stage": ApiStage.ASSESSMENT.value,
                "error": "Vertex model gemini-2.5-pro not found",
            }
        ]

    def test_completed_run_carries_both_a_results_and_an_errors_record(
        self, db
    ) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth, rows=2)
        record_stage_dump(
            session=db,
            assessment=assessment,
            stage=ApiStage.ASSESSMENT.value,
            url="s3://bucket/batch-1173/output.jsonl",
            count=2,
        )
        uploads = _Uploads()

        with _storage_patch(), _upload_patch(uploads):
            finalize_result_files(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(
                    stage_output_urls={
                        ApiStage.ASSESSMENT.value: "s3://bucket/batch-1173/output.jsonl"
                    }
                ),
            )

        db.refresh(assessment)
        assert assessment.result_files == {
            "results": {"url": "s3://bucket/batch-1173/output.jsonl", "count": 2},
            "errors": {"url": "s3://bucket/errors.jsonl", "count": 0},
        }

    def test_prefilter_and_assessment_dumps_coexist(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth, rows=2)
        uploads = _Uploads()
        bag = _bag(
            stage_output_urls={
                ApiStage.TOPIC_RELEVANCE.value: "s3://bucket/batch-1170/output.jsonl",
                ApiStage.ASSESSMENT.value: "s3://bucket/batch-1173/output.jsonl",
            },
            counters={
                ApiStage.TOPIC_RELEVANCE.value: {
                    "total": 2,
                    "passed": 1,
                    "rejected": 1,
                },
                ApiStage.ASSESSMENT.value: {"total": 1, "passed": 1, "rejected": 0},
            },
        )

        with _storage_patch(), _upload_patch(uploads):
            finalize_result_files(
                session=db, execution=execution, assessment=assessment, bag=bag
            )

        db.refresh(assessment)
        assert set(assessment.result_files) == {
            "topic_relevance_results",
            "results",
            "errors",
        }
        assert assessment.result_files["topic_relevance_results"]["count"] == 2

    def test_a_second_tick_does_not_duplicate_or_lose_records(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)
        uploads = _Uploads()
        bag = _bag(
            stage_output_urls={
                ApiStage.ASSESSMENT.value: "s3://bucket/batch-1173/output.jsonl"
            }
        )

        with _storage_patch(), _upload_patch(uploads):
            finalize_result_files(
                session=db, execution=execution, assessment=assessment, bag=bag
            )
            finalize_result_files(
                session=db, execution=execution, assessment=assessment, bag=bag
            )

        db.refresh(assessment)
        assert set(assessment.result_files) == {"results", "errors"}

    def test_upload_failure_does_not_raise_into_the_terminal_path(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, execution = _seed(db, auth)

        with (
            _storage_patch(),
            patch(
                "app.services.assessment.api.result_files."
                "upload_jsonl_to_object_store",
                side_effect=RuntimeError("bucket write denied"),
            ),
        ):
            finalize_result_files(
                session=db,
                execution=execution,
                assessment=assessment,
                bag=_bag(
                    stage_output_urls={
                        ApiStage.ASSESSMENT.value: "s3://bucket/out.jsonl"
                    }
                ),
            )

        db.refresh(assessment)
        assert assessment.result_files == {}


class TestBuildCallbackMetadata:
    def _signing_storage(self, failing_url: str | None = None) -> MagicMock:
        storage = MagicMock()

        def sign(url, expires_in):
            if url == failing_url:
                raise RuntimeError("presign refused")
            return f"https://signed.example/{url}?exp={expires_in}"

        storage.get_signed_url.side_effect = sign
        return storage

    def test_every_kind_is_signed_and_keeps_its_count(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)
        api.set_result_files(
            session=db,
            assessment=assessment,
            files={
                "results": {"url": "s3://bucket/out.jsonl", "count": 998},
                "errors": {"url": "s3://bucket/errors.jsonl", "count": 389},
            },
        )

        with _storage_patch(self._signing_storage()):
            metadata = build_callback_metadata(session=db, assessment=assessment)

        assert metadata["result_files"]["results"] == {
            "url": f"https://signed.example/s3://bucket/out.jsonl?exp={ONE_DAY_SECONDS}",
            "count": 998,
        }
        assert metadata["result_files"]["errors"]["count"] == 389

    def test_expires_at_is_one_day_out(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)

        with _storage_patch(self._signing_storage()):
            metadata = build_callback_metadata(session=db, assessment=assessment)

        expires_at = datetime.fromisoformat(metadata["expires_at"])
        assert timedelta(hours=23, minutes=59) < expires_at - now() <= timedelta(days=1)

    def test_a_failing_presign_drops_only_its_own_kind(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)
        api.set_result_files(
            session=db,
            assessment=assessment,
            files={
                "results": {"url": "s3://bucket/out.jsonl", "count": 998},
                "errors": {"url": "s3://bucket/errors.jsonl", "count": 389},
            },
        )

        with _storage_patch(self._signing_storage(failing_url="s3://bucket/out.jsonl")):
            metadata = build_callback_metadata(session=db, assessment=assessment)

        assert set(metadata["result_files"]) == {"errors"}
        assert metadata["expires_at"]

    def test_storage_outage_still_returns_the_envelope_keys(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)
        api.set_result_files(
            session=db,
            assessment=assessment,
            files={"results": {"url": "s3://bucket/out.jsonl", "count": 1}},
        )

        with patch(
            "app.services.assessment.api.result_files.get_cloud_storage",
            side_effect=RuntimeError("s3 unreachable"),
        ):
            metadata = build_callback_metadata(session=db, assessment=assessment)

        assert metadata["result_files"] == {}
        assert metadata["expires_at"]
