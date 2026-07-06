"""Service-layer tests for judge_config resolution in fast-eval dispatch.

Exercises `validate_and_start_fast_evaluation` with a real DB (`db` fixture) and
a mocked Celery dispatch. Covers:

- FR-5: a stored judge_config (id + version) resolves and the run is dispatched;
  an unknown id/version fails the request with 404 (not a run failure).
- FR-6: judge_config is per-run — it is threaded to the worker as a JSON-able
  arg, and a run without it dispatches judge_config=None.
- FR-12: a saved config from another (org, project) is never resolvable — 404.
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.models.evaluation import RunModeEnum
from app.models.llm.request import (
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
)
from app.services.evaluations.fast import (
    ERR_JUDGE_CONFIG_NOT_FOUND,
    validate_and_start_fast_evaluation,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
    create_test_project,
)
from unittest.mock import patch


@pytest.fixture
def _patch_dispatch() -> Iterator:
    with patch(
        "app.services.evaluations.fast.enqueue_fast_evaluation",
        return_value="fake-task-id",
    ) as m:
        yield m


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


class TestStoredJudgeConfig:
    def test_fr5_stored_reference_resolves_and_dispatches(
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
        # judge_config threaded to the worker as its stored reference.
        payload = _patch_dispatch.call_args.kwargs["judge_config"]
        assert payload["id"] == str(judge.id)
        assert payload["version"] == 1

    def test_fr5_unknown_reference_returns_404_not_run_failure(
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
        # The request failed before any run was dispatched.
        _patch_dispatch.assert_not_called()

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


class TestPerRunJudgeConfig:
    def test_fr6_no_judge_config_dispatches_none(
        self, db: Session, user_api_key: TestAuthContext, _patch_dispatch
    ) -> None:
        dataset, config = _dataset_and_config(db, user_api_key)

        validate_and_start_fast_evaluation(
            session=db,
            dataset_id=dataset.id,
            run_name="fr6-no-judge",
            config_id=config.id,
            config_version=1,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            judge_config=None,
        )

        assert _patch_dispatch.call_args.kwargs["judge_config"] is None

    def test_fr6_adhoc_blob_is_threaded_to_worker(
        self, db: Session, user_api_key: TestAuthContext, _patch_dispatch
    ) -> None:
        dataset, config = _dataset_and_config(db, user_api_key)

        validate_and_start_fast_evaluation(
            session=db,
            dataset_id=dataset.id,
            run_name="fr6-adhoc-judge",
            config_id=config.id,
            config_version=1,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            judge_config=_adhoc_judge_config(),
        )

        payload = _patch_dispatch.call_args.kwargs["judge_config"]
        assert payload["id"] is None
        assert payload["blob"]["completion"]["params"]["model"] == "gpt-4o"
