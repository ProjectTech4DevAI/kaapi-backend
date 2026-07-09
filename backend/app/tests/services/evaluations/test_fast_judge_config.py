"""Service-layer tests for judge_config in the chunked fast-eval flow.

`judge_config` no longer rides a Celery arg. `validate_and_start_fast_evaluation`
persists it onto `EvaluationRun.judge_config` (JSONB), and
`execute_fast_evaluation_aggregate` re-parses that column back into an
`LLMCallConfig` for the judge. These tests exercise both halves against a real
DB (`db` fixture) with the Celery/OpenAI/Langfuse boundaries mocked.

- FR-3: no judge_config → the column is NULL (zero-config default).
- FR-4/FR-6: an ad-hoc blob (or stored ref) is persisted verbatim on the row.
- FR-5/FR-12: an unresolvable saved ref (unknown, or from another project) fails
  the request with 404 before any run row is created.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.evaluation import EvaluationRun, RunModeEnum
from app.models.llm.request import (
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
)
from app.services.evaluations.fast import (
    ERR_JUDGE_CONFIG_NOT_FOUND,
    execute_fast_evaluation_aggregate,
    validate_and_start_fast_evaluation,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
    create_test_project,
)


class _FakeSessionCtx:
    """Context manager yielding the test session; `__exit__` never closes it."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def __enter__(self) -> Session:
        return self._db

    def __exit__(self, *exc: object) -> bool:
        return False


def _dataset_item(item_id: str) -> dict:
    return {
        "id": item_id,
        "input": {"question": "Q"},
        "expected_output": {"answer": "A"},
        "metadata": {"question_id": item_id},
    }


@pytest.fixture
def _patch_dispatch() -> Iterator[MagicMock]:
    """Stub the in-request fan-out boundaries; yield the chunk-enqueue mock.

    `validate_and_start_fast_evaluation` fetches the dataset in-request to size
    the fan-out, so the Langfuse client + item load must be stubbed alongside the
    chunk enqueue.
    """
    with (
        patch("app.services.evaluations.fast.get_tracing_client"),
        patch(
            "app.services.evaluations.fast.load_evaluation_dataset_items",
            return_value=[_dataset_item(f"item-{i}") for i in range(3)],
        ),
        patch(
            "app.services.evaluations.fast.start_fast_evaluation_chunk",
            return_value="fake-task-id",
        ) as mock_start_chunk,
    ):
        yield mock_start_chunk


def _dataset_and_config(db: Session, user_api_key: TestAuthContext):
    dataset = create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        original_items_count=3,
    )
    config = create_test_config(
        db, project_id=user_api_key.project_id, use_kaapi_schema=True
    )
    return dataset, config


def _adhoc_judge_config() -> LLMCallConfig:
    return LLMCallConfig(
        blob=ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                type="text",
                params={"model": "gpt-4o", "temperature": 0.0},
            )
        )
    )


def _run_by_name(db: Session, run_name: str) -> EvaluationRun | None:
    return db.exec(
        select(EvaluationRun).where(EvaluationRun.run_name == run_name)
    ).first()


class TestStoredJudgeConfig:
    def test_fr5_stored_reference_resolves_and_persists(
        self, db: Session, user_api_key: TestAuthContext, _patch_dispatch
    ) -> None:
        dataset, config = _dataset_and_config(db, user_api_key)
        judge = create_test_config(
            db, project_id=user_api_key.project_id, use_kaapi_schema=True
        )
        judge_config = LLMCallConfig(id=judge.id, version=1)

        eval_run = validate_and_start_fast_evaluation(
            session=db,
            dataset_id=dataset.id,
            run_name="fr5-stored-judge",
            config_id=config.id,
            config_version=1,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            judge_config=judge_config,
        )

        assert eval_run.status == "processing"
        assert eval_run.run_mode == RunModeEnum.FAST.value
        assert eval_run.judge_config == judge_config.model_dump(mode="json")
        assert eval_run.judge_config["id"] == str(judge.id)
        assert eval_run.judge_config["version"] == 1
        _patch_dispatch.assert_called()

    def test_fr5_unknown_reference_returns_404_and_creates_no_run(
        self, db: Session, user_api_key: TestAuthContext, _patch_dispatch
    ) -> None:
        dataset, config = _dataset_and_config(db, user_api_key)
        judge_config = LLMCallConfig(id=uuid4(), version=1)

        with pytest.raises(HTTPException) as exc:
            validate_and_start_fast_evaluation(
                session=db,
                dataset_id=dataset.id,
                run_name="fr5-unknown-judge",
                config_id=config.id,
                config_version=1,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                judge_config=judge_config,
            )

        assert exc.value.status_code == 404
        assert exc.value.detail == ERR_JUDGE_CONFIG_NOT_FOUND
        # 404 fires in the pre-check, before the run row or any dispatch.
        _patch_dispatch.assert_not_called()
        assert _run_by_name(db, "fr5-unknown-judge") is None

    def test_fr12_config_from_other_project_not_resolvable(
        self, db: Session, user_api_key: TestAuthContext, _patch_dispatch
    ) -> None:
        dataset, config = _dataset_and_config(db, user_api_key)
        other_project = create_test_project(db)
        foreign_judge = create_test_config(
            db, project_id=other_project.id, use_kaapi_schema=True
        )
        judge_config = LLMCallConfig(id=foreign_judge.id, version=1)

        with pytest.raises(HTTPException) as exc:
            validate_and_start_fast_evaluation(
                session=db,
                dataset_id=dataset.id,
                run_name="fr12-tenant-isolation",
                config_id=config.id,
                config_version=1,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                judge_config=judge_config,
            )

        assert exc.value.status_code == 404
        _patch_dispatch.assert_not_called()
        assert _run_by_name(db, "fr12-tenant-isolation") is None


class TestPerRunJudgeConfig:
    def test_fr3_no_judge_config_persists_null(
        self, db: Session, user_api_key: TestAuthContext, _patch_dispatch
    ) -> None:
        dataset, config = _dataset_and_config(db, user_api_key)

        eval_run = validate_and_start_fast_evaluation(
            session=db,
            dataset_id=dataset.id,
            run_name="fr3-no-judge",
            config_id=config.id,
            config_version=1,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            judge_config=None,
        )

        assert eval_run.judge_config is None
        assert db.get(EvaluationRun, eval_run.id).judge_config is None

    def test_fr4_adhoc_blob_persisted_on_row(
        self, db: Session, user_api_key: TestAuthContext, _patch_dispatch
    ) -> None:
        dataset, config = _dataset_and_config(db, user_api_key)
        judge_config = _adhoc_judge_config()

        eval_run = validate_and_start_fast_evaluation(
            session=db,
            dataset_id=dataset.id,
            run_name="fr4-adhoc-judge",
            config_id=config.id,
            config_version=1,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            judge_config=judge_config,
        )

        persisted = db.get(EvaluationRun, eval_run.id).judge_config
        assert persisted == judge_config.model_dump(mode="json")
        assert persisted["id"] is None
        assert persisted["blob"]["completion"]["params"]["model"] == "gpt-4o"


class TestAggregateRoundTrip:
    """`execute_fast_evaluation_aggregate` re-parses the persisted column."""

    def _make_fast_run(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        judge_config: dict | None,
    ) -> EvaluationRun:
        dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            original_items_count=3,
        )
        config = create_test_config(
            db, project_id=user_api_key.project_id, use_kaapi_schema=True
        )
        run = EvaluationRun(
            run_name=f"agg-{uuid4().hex[:8]}",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config.id,
            config_version=1,
            status="processing",
            run_mode=RunModeEnum.FAST.value,
            total_items=3,
            judge_config=judge_config,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    def _run_aggregate(self, db: Session, eval_run: EvaluationRun) -> MagicMock:
        with (
            patch(
                "app.services.evaluations.fast.Session",
                lambda *a, **k: _FakeSessionCtx(db),
            ),
            patch("app.services.evaluations.fast.get_openai_client"),
            patch("app.services.evaluations.fast.get_tracing_client"),
            patch(
                "app.services.evaluations.fast.use_langfuse_client",
                return_value=MagicMock(),
            ),
            patch("app.services.evaluations.fast.run_fast_evaluation") as mock_run,
        ):
            execute_fast_evaluation_aggregate(eval_run_id=eval_run.id)
        return mock_run

    def test_persisted_blob_reparsed_into_run_fast_evaluation(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        payload = _adhoc_judge_config().model_dump(mode="json")
        eval_run = self._make_fast_run(db, user_api_key, judge_config=payload)

        mock_run = self._run_aggregate(db, eval_run)

        mock_run.assert_called_once()
        threaded = mock_run.call_args.kwargs["judge_config"]
        assert isinstance(threaded, LLMCallConfig)
        assert threaded.model_dump(mode="json") == payload

    def test_null_column_passes_none_to_run_fast_evaluation(
        self, db: Session, user_api_key: TestAuthContext
    ) -> None:
        eval_run = self._make_fast_run(db, user_api_key, judge_config=None)

        mock_run = self._run_aggregate(db, eval_run)

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["judge_config"] is None
