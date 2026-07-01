"""Tests for POST /evaluations/{evaluation_id}/improve-prompt.

Covers the redesigned prompt-improvement behavior: no request body, traces
loaded from object storage via load_json_from_object_store, a single
client.messages.create call with structured outputs, and a ConfigVersionPublic
response.

HTTP boundaries mocked (patched as bound in the service module):
- app.services.evaluations.prompt_improvement.ClaudeProvider.create_client
  (returns a fake Anthropic client whose messages.create is stubbed)
- app.services.evaluations.prompt_improvement.get_cloud_storage
- app.services.evaluations.prompt_improvement.load_json_from_object_store
  (returns already-parsed trace data, or None for the download-failed path)

DB is real (transactional db fixture; rolls back after each test).
"""

import contextlib
import json
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.config.version import ConfigVersionCrud
from app.models import ConfigVersion, EvaluationDataset, EvaluationRun
from app.models.config.config import ConfigTag
from app.services.evaluations.prompt_improvement import (
    AI_GENERATED_MARKER,
    COMMIT_MESSAGE_MAX_LENGTH,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_evaluation_dataset
from app.tests.utils.utils import random_lower_string


IMPROVE_URL = "/api/v1/evaluations/{evaluation_id}/improve-prompt"

_IMPROVED_INSTRUCTIONS = "You are an improved assistant. Answer precisely."
_RATIONALE = "Tightened answer scoping to address weak categories."


def _llm_json(
    improved_instructions: str = _IMPROVED_INSTRUCTIONS,
    rationale: str = _RATIONALE,
) -> str:
    return json.dumps(
        {
            "improved_instructions": improved_instructions,
            "rationale": rationale,
        }
    )


# Already-parsed trace data — the new service path returns parsed JSON, not bytes.
_TRACES: list[dict] = [
    {
        "trace_id": "t1",
        "question": "What is the capital?",
        "llm_answer": "Lyon",
        "ground_truth_answer": "Paris",
        "category": "Geography",
        "scores": [{"name": "cosine_similarity", "value": 0.3, "unscoreable": False}],
    }
]

# Sentinel: "generate a default score_trace_url from the run id after commit"
_AUTO_URL = object()


def _make_fake_claude_client(text_content: str | None = None) -> MagicMock:
    """Return a fake Anthropic client whose messages.create yields a text block."""
    if text_content is None:
        text_content = _llm_json()

    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text_content

    response = MagicMock()
    response.content = [content_block]
    response.id = "msg_test_id"

    client = MagicMock()
    client.messages.create.return_value = response
    return client


@contextlib.contextmanager
def _patched_boundaries(
    *,
    claude_client: MagicMock | None = None,
    traces: Any = _TRACES,
) -> Iterator[MagicMock]:
    """Patch the three HTTP boundaries and yield the fake Claude client.

    - ClaudeProvider.create_client → returns ``claude_client``
    - get_cloud_storage → returns a harmless MagicMock (load_json is patched, so
      the storage object is never really exercised)
    - load_json_from_object_store → returns ``traces`` (use None to exercise the
      trace_download_failed path)
    """
    if claude_client is None:
        claude_client = _make_fake_claude_client()

    base = "app.services.evaluations.prompt_improvement"
    with patch(
        f"{base}.ClaudeProvider.create_client",
        return_value=claude_client,
    ), patch(
        f"{base}.get_cloud_storage",
        return_value=MagicMock(),
    ), patch(
        f"{base}.load_json_from_object_store",
        return_value=traces,
    ):
        yield claude_client


def _make_config_with_instructions(
    db: Session,
    project_id: int,
    instructions: str = "You are a helpful assistant.",
) -> Any:
    """Create a config whose config_blob has completion.params.instructions set."""
    from app.crud.config import ConfigCrud
    from app.models.config.config import ConfigCreate
    from app.models.llm import KaapiCompletionConfig
    from app.models.llm.request import ConfigBlob

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


class TestHappyPath:
    """Completed run with traces → 201 with correct ConfigVersionPublic."""

    def test_returns_201_with_new_config_version(
        self,
        client: TestClient,
        headers: dict[str, str],
        config_with_instructions: Any,
        completed_run: EvaluationRun,
    ) -> None:
        """POST with no body → 201; response contains new config version at latest+1."""
        with _patched_boundaries():
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
        """The new config version persisted in DB carries the improved instructions."""
        with _patched_boundaries():
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

    def test_commit_message_marker_and_rationale(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """commit_message starts with the AI marker and carries run name + rationale."""
        with _patched_boundaries():
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        commit_message = resp.json()["data"]["commit_message"]
        assert commit_message.startswith(AI_GENERATED_MARKER)
        assert f"(Evaluation: {completed_run.run_name})" in commit_message
        assert _RATIONALE in commit_message
        assert len(commit_message) <= COMMIT_MESSAGE_MAX_LENGTH

    def test_long_rationale_truncated_to_max_length(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """A rationale longer than the cap forces real truncation at the boundary."""
        long_rationale = "x" * (COMMIT_MESSAGE_MAX_LENGTH + 200)
        fake_client = _make_fake_claude_client(_llm_json(rationale=long_rationale))

        with _patched_boundaries(claude_client=fake_client):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        commit_message = resp.json()["data"]["commit_message"]
        assert len(commit_message) == COMMIT_MESSAGE_MAX_LENGTH
        assert commit_message.startswith(AI_GENERATED_MARKER)

    def test_messages_create_uses_json_schema_output_config(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """The single messages.create call requests structured json_schema output."""
        with _patched_boundaries() as fake_client:
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text

        fake_client.messages.create.assert_called_once()
        # The legacy Files API must be gone — no beta.* surface is touched.
        fake_client.beta.files.upload.assert_not_called()

        output_config = fake_client.messages.create.call_args.kwargs["output_config"]
        assert output_config["format"]["type"] == "json_schema"
        assert "schema" in output_config["format"]

    def test_commit_message_has_no_metric_or_threshold(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """The new design's provenance string must not carry metric=/threshold=."""
        with _patched_boundaries():
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 201, resp.text
        commit_message = resp.json()["data"]["commit_message"]
        assert "metric=" not in commit_message
        assert "threshold=" not in commit_message


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
            type="stt",
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


class TestTracesNotAvailable:
    """422 when score_trace_url is empty/None (precondition unmet)."""

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
            score_trace_url=trace_url,
        )

        resp = client.post(
            IMPROVE_URL.format(evaluation_id=run.id),
            headers=headers,
        )

        assert resp.status_code == 422, resp.text
        assert "traces_not_available" in resp.json()["error"], resp.text


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


class TestMissingAnthropicKey:
    """500 when the platform ANTHROPIC_API_KEY is not configured (server misconfig)."""

    def test_empty_api_key_returns_500(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Storage succeeds but the key check inside the LLM step fails with 500."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

        with _patched_boundaries():
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 500, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text


class TestTraceDownloadFailed:
    """502 when the trace file cannot be loaded from object storage."""

    def test_load_json_returns_none_returns_502(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        with _patched_boundaries(traces=None):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 502, resp.text
        assert "trace_download_failed" in resp.json()["error"], resp.text


def _anthropic_exceptions() -> list[tuple[str, Exception, int]]:
    """Build one instance of each SDK error plus the HTTP status the service maps it to.

    The mapping reflects who's at fault: rate limit -> 503 (upstream unavailable),
    timeout -> 504 (upstream timeout), bad upstream response/network -> 502, and an
    unexpected non-SDK error -> 500 (Kaapi-side bug).
    """
    return [
        (
            "authentication",
            anthropic.AuthenticationError(
                message="auth failed",
                response=MagicMock(status_code=401, headers={}),
                body={},
            ),
            502,
        ),
        (
            "rate_limit",
            anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={},
            ),
            503,
        ),
        ("timeout", anthropic.APITimeoutError(request=MagicMock()), 504),
        (
            "connection",
            anthropic.APIConnectionError(message="conn failed", request=MagicMock()),
            502,
        ),
        (
            "status",
            anthropic.APIStatusError(
                message="server error",
                response=MagicMock(status_code=500, headers={}),
                body={},
            ),
            502,
        ),
        ("generic", RuntimeError("something unexpected"), 500),
    ]


class TestLLMFailureMapping:
    """Each Anthropic/SDK failure from messages.create maps to its matching HTTP status."""

    @pytest.mark.parametrize(
        "name,exc,expected_status",
        _anthropic_exceptions(),
        ids=[name for name, _, _ in _anthropic_exceptions()],
    )
    def test_llm_error_maps_to_status(
        self,
        name: str,
        exc: Exception,
        expected_status: int,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        fake_client = _make_fake_claude_client()
        fake_client.messages.create.side_effect = exc

        with _patched_boundaries(claude_client=fake_client):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == expected_status, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text

    def test_malformed_json_response_returns_500(
        self,
        client: TestClient,
        headers: dict[str, str],
        completed_run: EvaluationRun,
    ) -> None:
        """Non-JSON text from the model falls through to the generic 500 branch."""
        fake_client = _make_fake_claude_client("this is not json")

        with _patched_boundaries(claude_client=fake_client):
            resp = client.post(
                IMPROVE_URL.format(evaluation_id=completed_run.id),
                headers=headers,
            )

        assert resp.status_code == 500, resp.text
        assert "prompt_generation_failed" in resp.json()["error"], resp.text


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
    """Running improvement twice creates a version at latest+1 each time."""

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

        with _patched_boundaries():
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

        with _patched_boundaries():
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

        with _patched_boundaries():
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
