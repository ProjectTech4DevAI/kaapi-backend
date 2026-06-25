"""Tests for POST /evaluations/{evaluation_id}/improve-prompt.

Covers the redesigned prompt-improvement behavior: no request body, S3 trace
download, Anthropic Files API call, and ConfigVersionPublic response.

HTTP boundaries mocked:
- app.services.evaluations.prompt_improvement.Anthropic  (Files API + messages)
- app.services.evaluations.prompt_improvement.get_cloud_storage  (S3 download)

DB is real (transactional db fixture; rolls back after each test).
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.config.version import ConfigVersionCrud
from app.models import ConfigVersion, EvaluationDataset, EvaluationRun
from app.models.config.config import ConfigTag
from app.services.evaluations.prompt_improvement import AI_GENERATED_MARKER
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_evaluation_dataset
from app.tests.utils.utils import random_lower_string

# ── constants ─────────────────────────────────────────────────────────────────

IMPROVE_URL = "/api/v1/evaluations/{evaluation_id}/improve-prompt"

_IMPROVED_INSTRUCTIONS = "You are an improved assistant. Answer precisely."
_RATIONALE = "Tightened answer scoping to address weak categories."
_LLM_JSON_RESPONSE = json.dumps(
    {
        "improved_instructions": _IMPROVED_INSTRUCTIONS,
        "rationale": _RATIONALE,
    }
)

_TRACE_BYTES = json.dumps(
    [
        {
            "trace_id": "t1",
            "question": "What is the capital?",
            "llm_answer": "Lyon",
            "ground_truth_answer": "Paris",
            "category": "Geography",
            "scores": [
                {"name": "cosine_similarity", "value": 0.3, "unscoreable": False}
            ],
        }
    ]
).encode()

# Sentinel: "generate a default score_trace_url from the run id after commit"
_AUTO_URL = object()


# ── shared helpers ─────────────────────────────────────────────────────────────


def _make_anthropic_mock(text_content: str = _LLM_JSON_RESPONSE) -> MagicMock:
    """Return a mock Anthropic client whose beta.files and beta.messages look real."""
    uploaded_file = MagicMock()
    uploaded_file.id = "file-test-id-abc123"

    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text_content

    response = MagicMock()
    response.content = [content_block]
    response.id = "msg_test_id"

    client_instance = MagicMock()
    client_instance.beta.files.upload.return_value = uploaded_file
    client_instance.beta.messages.create.return_value = response
    client_instance.beta.files.delete.return_value = None

    return client_instance


def _make_storage_mock(trace_bytes: bytes = _TRACE_BYTES) -> MagicMock:
    """Return a mock cloud-storage factory whose instance's .get(url) returns trace bytes."""
    storage_instance = MagicMock()
    storage_instance.get.return_value = trace_bytes

    storage_factory = MagicMock(return_value=storage_instance)
    return storage_factory


def _make_config_with_instructions(
    db: Session,
    project_id: int,
    instructions: str = "You are a helpful assistant.",
) -> Any:
    """Create a config whose config_blob has completion.params.instructions set."""
    from app.crud.config import ConfigCrud
    from app.models.llm.request import ConfigBlob
    from app.models.llm import KaapiCompletionConfig
    from app.models.config.config import ConfigCreate

    config_blob = ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            type="text",
            params={
                "model": "gpt-4o",
                "temperature": 0.5,
                "instructions": instructions,
                "knowledge_base_ids": ["vs_abc123"],
            },
        )
    )
    config_create = ConfigCreate(
        name=f"test-config-{random_lower_string()}",
        description="Test configuration for improve-prompt",
        config_blob=config_blob,
        commit_message="Initial version",
        tag=ConfigTag.DEFAULT,
    )
    config_crud = ConfigCrud(session=db, project_id=project_id)
    config, _ = config_crud.create_or_raise(config_create)
    return config


def _make_completed_run(
    db: Session,
    config_id: Any,
    config_version: int,
    organization_id: int,
    project_id: int,
    dataset_id: int,
    status: str = "completed",
    run_name: str | None = None,
    score_trace_url: Any = _AUTO_URL,
) -> EvaluationRun:
    """Create and persist an EvaluationRun.

    score_trace_url behaviour:
      - omitted (default _AUTO_URL): a valid S3 URL is generated after commit so the
        run id is known. Use for all tests that mock the storage layer.
      - None or "": stored verbatim (use to exercise the traces_not_available path).
      - any str: stored verbatim.
    """
    if run_name is None:
        run_name = f"run-{random_lower_string()}"

    initial_url: str | None = None if score_trace_url is _AUTO_URL else score_trace_url

    run = EvaluationRun(
        run_name=run_name,
        dataset_name=f"ds-{random_lower_string()}",
        dataset_id=dataset_id,
        config_id=config_id,
        config_version=config_version,
        status=status,
        total_items=1,
        type="text",
        organization_id=organization_id,
        project_id=project_id,
        score_trace_url=initial_url,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if score_trace_url is _AUTO_URL:
        run.score_trace_url = (
            f"s3://test-bucket/evaluations/score/{run.id}/traces_{run.id}.json"
        )
        db.add(run)
        db.commit()
        db.refresh(run)

    return run


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def auth(user_api_key: TestAuthContext) -> TestAuthContext:
    return user_api_key


@pytest.fixture
def headers(user_api_key_header: dict[str, str]) -> dict[str, str]:
    return user_api_key_header


@pytest.fixture
def dataset(db: Session, auth: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )


@pytest.fixture
def config_with_instructions(db: Session, auth: TestAuthContext) -> Any:
    return _make_config_with_instructions(
        db=db,
        project_id=auth.project_id,
        instructions="You are a helpful assistant. Answer clearly.",
    )


@pytest.fixture
def anthropic_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the platform-owned Anthropic key the service reads from settings."""
    monkeypatch.setattr(
        settings,
        "ANTHROPIC_API_KEY",
        "sk-ant-test-" + random_lower_string(),
    )


@pytest.fixture
def completed_run(
    db: Session,
    auth: TestAuthContext,
    dataset: EvaluationDataset,
    config_with_instructions: Any,
    anthropic_creds: None,
) -> EvaluationRun:
    return _make_completed_run(
        db=db,
        config_id=config_with_instructions.id,
        config_version=1,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
        dataset_id=dataset.id,
    )


# ── 1. Happy path ──────────────────────────────────────────────────────────────


class TestHappyPath:
    """Completed run with score_trace_url → 201 with correct ConfigVersionPublic."""

    def test_returns_201_with_new_config_version(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        completed_run: EvaluationRun,
        config_with_instructions: Any,
    ) -> None:
        """POST with no body → 201; response contains new config version at latest+1."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert body["version"] == 2  # initial version was 1
        assert body["config_id"] == str(config_with_instructions.id)

    def test_new_version_has_improved_instructions_in_db(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        completed_run: EvaluationRun,
    ) -> None:
        """The new config version in DB has the mocked improved_instructions."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        new_version_id = resp.json()["data"]["id"]
        stmt = select(ConfigVersion).where(ConfigVersion.id == new_version_id)
        new_version = db.exec(stmt).one()
        assert (
            new_version.config_blob["completion"]["params"]["instructions"]
            == _IMPROVED_INSTRUCTIONS
        )

    def test_commit_message_starts_with_ai_generated_marker(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """commit_message starts with AI_GENERATED_MARKER."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert body["commit_message"].startswith(AI_GENERATED_MARKER)

    def test_commit_message_contains_source_evaluation_run_id(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """commit_message contains source_evaluation_run_id=<evaluation_id>."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert f"source_evaluation_run_id={completed_run.id}" in body["commit_message"]

    def test_commit_message_contains_rationale(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """commit_message contains the LLM rationale."""
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert _RATIONALE in body["commit_message"]


# ── 2. commit_message provenance: NO metric= or threshold= ────────────────────


class TestCommitMessageProvenance:
    """commit_message must NOT contain metric= or threshold= (old design gone)."""

    def test_commit_message_does_not_contain_metric_or_threshold(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        commit_message = resp.json()["data"]["commit_message"]
        assert (
            "metric=" not in commit_message
        ), f"commit_message must not contain 'metric=' but got: {commit_message!r}"
        assert (
            "threshold=" not in commit_message
        ), f"commit_message must not contain 'threshold=' but got: {commit_message!r}"


# ── 3. Files API lifecycle ─────────────────────────────────────────────────────


class TestFilesApiLifecycle:
    """Anthropic Files API sequence is called correctly on the happy path."""

    def test_files_upload_and_delete_called_and_message_uses_file_id(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """upload called → messages.create uses that file_id in a document block → delete called."""
        mock_client = _make_anthropic_mock()
        expected_file_id = mock_client.beta.files.upload.return_value.id

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=mock_client,
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text

        # upload was called
        mock_client.beta.files.upload.assert_called_once()

        # messages.create was called
        mock_client.beta.messages.create.assert_called_once()
        call_kwargs = mock_client.beta.messages.create.call_args.kwargs

        messages = call_kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        content_blocks = user_msg["content"]

        # There is at least one text block
        text_blocks = [b for b in content_blocks if b["type"] == "text"]
        assert len(text_blocks) >= 1

        # There is a document block whose file_id matches the uploaded file
        doc_blocks = [b for b in content_blocks if b["type"] == "document"]
        assert len(doc_blocks) >= 1
        assert doc_blocks[0]["source"]["file_id"] == expected_file_id

        # delete was called with the same file id
        mock_client.beta.files.delete.assert_called_once_with(expected_file_id)


# ── 4. 422 traces_not_available ───────────────────────────────────────────────


class TestTracesNotAvailable:
    """422 when score_trace_url is empty/None."""

    @pytest.mark.parametrize("trace_url", [None, ""])
    def test_missing_trace_url_returns_422(
        self,
        trace_url: str | None,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            score_trace_url=trace_url,  # explicitly None or ""
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            headers=headers,
        )

        assert resp.status_code == 422, resp.text
        assert "traces_not_available" in resp.json()["error"], resp.text


# ── 5. 502 trace_download_failed ──────────────────────────────────────────────


class TestTraceDownloadFailed:
    """502 when S3 .get() raises."""

    def test_storage_get_raises_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        failing_storage = MagicMock()
        failing_storage.get.side_effect = RuntimeError("S3 not reachable")
        failing_factory = MagicMock(return_value=failing_storage)

        with patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            failing_factory,
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "trace_download_failed" in resp.json()["error"], resp.text


# ── 6. 502 on LLM failure + file still deleted ────────────────────────────────


class TestLLMFailure:
    """502 when messages.create raises; file must still be deleted via finally."""

    def test_rate_limit_error_returns_502_and_file_deleted(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        mock_client = _make_anthropic_mock()
        expected_file_id = mock_client.beta.files.upload.return_value.id

        # Simulate a RateLimitError from the Anthropic SDK
        mock_client.beta.messages.create.side_effect = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(headers={}),
            body={},
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=mock_client,
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text

        # File must be cleaned up even when the LLM call fails
        mock_client.beta.files.delete.assert_called_once_with(expected_file_id)

    def test_api_status_error_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        mock_client = _make_anthropic_mock()
        mock_client.beta.messages.create.side_effect = anthropic.APIStatusError(
            message="Internal server error",
            response=MagicMock(status_code=500, headers={}),
            body={},
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=mock_client,
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text


# ── 7. Run not found / wrong type ─────────────────────────────────────────────


class TestRunNotFound:
    """404 when the evaluation run doesn't exist or is non-TEXT type."""

    def test_nonexistent_run_returns_404(
        self,
        client: TestClient,
        headers: dict[str, str],
        anthropic_creds: None,
    ) -> None:
        resp = client.post(
            IMPROVE_URL.format(evaluation_id=9999999),
            headers=headers,
        )

        assert resp.status_code == 404, resp.text
        assert "evaluation_not_found" in resp.json()["error"], resp.text

    def test_non_text_run_returns_404(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        """get_evaluation_run_by_id filters type=='text'; a non-text run is invisible."""
        run = EvaluationRun(
            run_name=f"run-stt-{random_lower_string()}",
            dataset_name=f"ds-{random_lower_string()}",
            dataset_id=dataset.id,
            config_id=config_with_instructions.id,
            config_version=1,
            status="completed",
            total_items=1,
            type="stt",  # non-text type
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            score_trace_url="s3://test-bucket/traces.json",
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            headers=headers,
        )

        assert resp.status_code == 404, resp.text
        assert "evaluation_not_found" in resp.json()["error"], resp.text


# ── 8. Malformed LLM JSON → 502 ───────────────────────────────────────────────


class TestMalformedLLMJson:
    """502 when the LLM returns text that _parse_llm_response cannot handle."""

    def test_non_json_response_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock("This is not JSON at all."),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text

    def test_missing_improved_instructions_key_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        bad_json = json.dumps({"rationale": "only rationale present"})

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(bad_json),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text

    def test_missing_rationale_key_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        bad_json = json.dumps({"improved_instructions": "some instructions"})

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(bad_json),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text


# ── Existing behaviors preserved ──────────────────────────────────────────────


class TestNonCompletedStatus:
    """409 when evaluation run is not in 'completed' status."""

    @pytest.mark.parametrize("status", ["pending", "processing", "failed"])
    def test_non_completed_returns_409(
        self,
        status: str,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        run = _make_completed_run(
            db=db,
            config_id=config_with_instructions.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
            status=status,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            headers=headers,
        )

        assert resp.status_code == 409, resp.text
        assert "evaluation_not_completed" in resp.json()["error"], resp.text


class TestSourceConfigUnavailable:
    """409 when the source config or config_version is missing/soft-deleted."""

    def test_soft_deleted_config_returns_409(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        anthropic_creds: None,
    ) -> None:
        from app.core.util import now
        from app.models.config.config import Config

        config = _make_config_with_instructions(db=db, project_id=auth.project_id)
        run = _make_completed_run(
            db=db,
            config_id=config.id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        stmt = select(Config).where(Config.id == config.id)
        cfg = db.exec(stmt).one()
        cfg.deleted_at = now()
        db.add(cfg)
        db.commit()

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            headers=headers,
        )

        assert resp.status_code == 409, resp.text
        assert "source_config_unavailable" in resp.json()["error"], resp.text

    def test_missing_config_version_returns_409(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        anthropic_creds: None,
    ) -> None:
        config = _make_config_with_instructions(db=db, project_id=auth.project_id)
        run = _make_completed_run(
            db=db,
            config_id=config.id,
            config_version=99,  # does not exist
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            headers=headers,
        )

        assert resp.status_code == 409, resp.text
        assert "source_config_unavailable" in resp.json()["error"], resp.text


class TestTenantIsolation:
    """404 when a run belongs to a different org/project."""

    def test_run_from_different_project_returns_404(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        superuser_api_key: TestAuthContext,
        anthropic_creds: None,
    ) -> None:
        su_dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=superuser_api_key.organization_id,
            project_id=superuser_api_key.project_id,
        )
        su_config = _make_config_with_instructions(
            db=db,
            project_id=superuser_api_key.project_id,
        )
        su_run = _make_completed_run(
            db=db,
            config_id=su_config.id,
            config_version=1,
            organization_id=superuser_api_key.organization_id,
            project_id=superuser_api_key.project_id,
            dataset_id=su_dataset.id,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=su_run.id),
            headers=headers,  # normal user headers, not superuser
        )

        assert resp.status_code == 404, resp.text
        assert "evaluation_not_found" in resp.json()["error"], resp.text


class TestRepeatableIteration:
    """Running improvement twice creates version at latest+1 each time."""

    def test_second_improvement_creates_next_version(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        config_id = config_with_instructions.id

        run1 = _make_completed_run(
            db=db,
            config_id=config_id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp1 = client.post(
                IMPROVE_URL.format(evaluation_id=run1.id),
                headers=headers,
            )
        assert resp1.status_code == 201, resp1.text
        assert resp1.json()["data"]["version"] == 2

        run2 = _make_completed_run(
            db=db,
            config_id=config_id,
            config_version=2,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp2 = client.post(
                IMPROVE_URL.format(evaluation_id=run2.id),
                headers=headers,
            )
        assert resp2.status_code == 201, resp2.text
        assert resp2.json()["data"]["version"] == 3


class TestPriorVersionsPreserved:
    """Pre-existing config_version rows are unchanged after a new improvement."""

    def test_prior_versions_still_retrievable(
        self,
        client: TestClient,
        headers: dict[str, str],
        db: Session,
        auth: TestAuthContext,
        dataset: EvaluationDataset,
        config_with_instructions: Any,
        anthropic_creds: None,
    ) -> None:
        config_id = config_with_instructions.id
        crud = ConfigVersionCrud(
            session=db, config_id=config_id, project_id=auth.project_id
        )

        original_version = crud.read_one(version_number=1)
        assert original_version is not None
        original_instructions = original_version.config_blob["completion"]["params"][
            "instructions"
        ]

        run = _make_completed_run(
            db=db,
            config_id=config_id,
            config_version=1,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset.id,
        )

        with patch(
            "app.services.evaluations.prompt_improvement.Anthropic",
            return_value=_make_anthropic_mock(),
        ), patch(
            "app.services.evaluations.prompt_improvement.get_cloud_storage",
            _make_storage_mock(),
        ):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text

        db.expire_all()
        still_v1 = crud.read_one(version_number=1)
        assert still_v1 is not None
        assert (
            still_v1.config_blob["completion"]["params"]["instructions"]
            == original_instructions
        )

        v2 = crud.read_one(version_number=2)
        assert v2 is not None
        assert v2.commit_message is not None
        assert v2.commit_message.startswith(AI_GENERATED_MARKER)
