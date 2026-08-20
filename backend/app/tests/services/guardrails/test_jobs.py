from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from sqlmodel import Session, select

from app.models import Job, JobStatus, JobType
from app.models.guardrails import GuardrailsRequest
from app.services.guardrails.jobs import (
    _build_callback_payload,
    _coerce_int,
    execute_job,
    start_job,
)
from app.services.llm.guardrails import GuardrailsOutcome
from app.tests.utils.auth import TestAuthContext


VALIDATOR_A = uuid4()
VALIDATOR_B = uuid4()


def _make_request(
    *,
    text: str = "My email is alice@example.com",
    validator_ids: list | None = None,
    callback_url: str | None = None,
    metadata: dict | None = None,
) -> GuardrailsRequest:
    ids = validator_ids if validator_ids is not None else [VALIDATOR_A]
    return GuardrailsRequest.model_validate(
        {
            "text": text,
            "config": [{"validator_config_id": str(i)} for i in ids],
            "callback_url": callback_url,
            "request_metadata": metadata,
        }
    )


# ---------- _coerce_int ----------


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, 0),
        (5, 5),
        ("7", 7),
        ("not-a-number", 0),
        ({"x": 1}, 0),
        (3.9, 3),
    ],
)
def test_coerce_int(value, expected) -> None:
    assert _coerce_int(value) == expected


# ---------- _build_callback_payload ----------


def test_build_callback_payload_extracts_usage_and_appends_warnings() -> None:
    payload = _build_callback_payload(
        response_id="resp-1",
        safe_text="clean text",
        raw={
            "data": {
                "usage": {
                    "input_tokens": "10",
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "reasoning_tokens": None,
                }
            }
        },
        request_metadata={"client_id": "abc"},
        warnings=["dup ignored"],
    )

    assert payload["success"] is True
    assert payload["data"]["response"]["response_id"] == "resp-1"
    assert payload["data"]["response"]["output"]["content"]["value"] == "clean text"
    assert payload["data"]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "reasoning_tokens": 0,
    }
    assert payload["metadata"]["client_id"] == "abc"
    assert payload["metadata"]["warnings"] == ["dup ignored"]


def test_build_callback_payload_caller_warnings_overwritten() -> None:
    payload = _build_callback_payload(
        response_id="r",
        safe_text="t",
        raw={},
        request_metadata={"warnings": ["client-supplied"]},
        warnings=["server-supplied"],
    )
    assert payload["metadata"]["warnings"] == ["server-supplied"]


def test_build_callback_payload_handles_missing_usage() -> None:
    payload = _build_callback_payload(
        response_id="r",
        safe_text="t",
        raw={"data": {}},
        request_metadata=None,
        warnings=[],
    )
    assert payload["data"]["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }


# ---------- start_job (uses real test session — rolls back cleanly) ----------


def test_start_job_happy_path_creates_row_and_enqueues(
    db: Session, user_api_key: TestAuthContext
) -> None:
    with patch("app.services.guardrails.jobs.start_guardrails_job") as mock_enqueue:
        mock_enqueue.return_value = "task-123"
        job = start_job(
            db=db,
            request=_make_request(),
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
        )

    assert job.job_type == JobType.LLM_GUARDRAILS
    assert job.status == JobStatus.PENDING
    assert job.meta is not None and "request" in job.meta
    mock_enqueue.assert_called_once()
    kwargs = mock_enqueue.call_args.kwargs
    assert kwargs["project_id"] == user_api_key.project_id
    assert kwargs["job_id"] == str(job.id)


def test_start_job_celery_failure_marks_failed_and_raises_503(
    db: Session, user_api_key: TestAuthContext
) -> None:
    with patch("app.services.guardrails.jobs.start_guardrails_job") as mock_enqueue:
        mock_enqueue.side_effect = RuntimeError("broker down")
        with pytest.raises(HTTPException) as exc:
            start_job(
                db=db,
                request=_make_request(),
                project_id=user_api_key.project_id,
                organization_id=user_api_key.organization_id,
            )

    assert exc.value.status_code == 503

    rows = db.exec(select(Job).where(Job.project_id == user_api_key.project_id)).all()
    failed = [r for r in rows if r.status == JobStatus.FAILED]
    assert failed, "expected a FAILED job row after broker failure"
    assert "broker down" in (failed[-1].error_message or "")


# ---------- execute_job ----------
# execute_job opens its own `Session(engine)` which bypasses the test
# fixture's savepoint rollback. We mock both `Session` and `JobCrud` at the
# module boundary so writes never hit the real DB and we can assert on
# update kwargs directly.


@contextmanager
def _patched_worker_db():
    """Patch Session + JobCrud inside services.guardrails.jobs; yields the
    JobCrud constructor mock so callers can assert on .update kwargs."""
    with patch("app.services.guardrails.jobs.Session") as mock_session_cls, patch(
        "app.services.guardrails.jobs.JobCrud"
    ) as mock_crud_cls:
        mock_session_cls.return_value.__enter__.return_value = MagicMock(spec=Session)
        mock_session_cls.return_value.__exit__.return_value = None
        yield mock_crud_cls


def _update_calls(mock_crud_cls) -> list:
    """All JobUpdate payloads passed to JobCrud(...).update across the run."""
    return [
        call.kwargs["job_update"]
        for call in mock_crud_cls.return_value.update.call_args_list
    ]


def _final_status(mock_crud_cls) -> JobStatus | None:
    updates = _update_calls(mock_crud_cls)
    return updates[-1].status if updates else None


def test_execute_job_success_no_callback_returns_payload_without_sending(
    user_api_key: TestAuthContext,
) -> None:
    job_id = str(uuid4())
    req = _make_request(callback_url=None)

    with _patched_worker_db() as mock_crud, patch(
        "app.services.guardrails.jobs.apply_guardrails"
    ) as mock_apply, patch(
        "app.services.guardrails.jobs.send_callback"
    ) as mock_send, patch(
        "app.services.guardrails.jobs.get_webhook_secret"
    ):
        mock_apply.return_value = GuardrailsOutcome(
            safe_text="My email is [REDACTED]",
            error=None,
            bypassed=False,
            rephrase_needed=False,
            raw={"data": {"safe_text": "My email is [REDACTED]", "usage": {}}},
        )
        result = execute_job(
            project_id=user_api_key.project_id,
            job_id=job_id,
            task_id="task-1",
            task_instance=None,
            request_data=req.model_dump(mode="json"),
            organization_id=user_api_key.organization_id,
        )

    assert result["success"] is True
    mock_send.assert_not_called()
    assert _final_status(mock_crud) == JobStatus.SUCCESS

    # Final update writes callback metadata indicating no delivery.
    final = _update_calls(mock_crud)[-1]
    assert final.meta["callback"]["delivered"] is False
    assert final.meta["callback"]["response_id"]


def test_execute_job_success_with_callback_dispatches_webhook(
    user_api_key: TestAuthContext,
) -> None:
    req = _make_request(callback_url="https://example.com/cb")

    with _patched_worker_db() as mock_crud, patch(
        "app.services.guardrails.jobs.apply_guardrails"
    ) as mock_apply, patch(
        "app.services.guardrails.jobs.send_callback"
    ) as mock_send, patch(
        "app.services.guardrails.jobs.get_webhook_secret", return_value="secret"
    ):
        mock_apply.return_value = GuardrailsOutcome(
            safe_text="ok",
            error=None,
            bypassed=False,
            rephrase_needed=False,
            raw={"data": {"safe_text": "ok"}},
        )
        execute_job(
            project_id=user_api_key.project_id,
            job_id=str(uuid4()),
            task_id="task-1",
            task_instance=None,
            request_data=req.model_dump(mode="json"),
            organization_id=user_api_key.organization_id,
        )

    mock_send.assert_called_once()
    assert mock_send.call_args.kwargs["callback_url"] == "https://example.com/cb"
    final = _update_calls(mock_crud)[-1]
    assert final.meta["callback"]["delivered"] is True


def test_execute_job_dedupes_duplicate_validators_and_warns(
    user_api_key: TestAuthContext,
) -> None:
    req = _make_request(validator_ids=[VALIDATOR_A, VALIDATOR_A, VALIDATOR_B])

    with _patched_worker_db() as mock_crud, patch(
        "app.services.guardrails.jobs.apply_guardrails"
    ) as mock_apply:
        mock_apply.return_value = GuardrailsOutcome(
            safe_text="ok",
            error=None,
            bypassed=False,
            rephrase_needed=False,
            raw={"data": {}},
        )
        execute_job(
            project_id=user_api_key.project_id,
            job_id=str(uuid4()),
            task_id="t",
            task_instance=None,
            request_data=req.model_dump(mode="json"),
            organization_id=user_api_key.organization_id,
        )

    called_validators = mock_apply.call_args.kwargs["validators"]
    assert len(called_validators) == 2

    final = _update_calls(mock_crud)[-1]
    warnings = final.meta["callback"]["warnings"]
    assert any("duplicate" in w for w in warnings)


def test_execute_job_bypassed_adds_warning(user_api_key: TestAuthContext) -> None:
    req = _make_request()

    with _patched_worker_db() as mock_crud, patch(
        "app.services.guardrails.jobs.apply_guardrails"
    ) as mock_apply:
        mock_apply.return_value = GuardrailsOutcome(
            safe_text=None,
            error=None,
            bypassed=True,
            rephrase_needed=False,
            raw={},
        )
        execute_job(
            project_id=user_api_key.project_id,
            job_id=str(uuid4()),
            task_id="t",
            task_instance=None,
            request_data=req.model_dump(mode="json"),
            organization_id=user_api_key.organization_id,
        )

    assert _final_status(mock_crud) == JobStatus.SUCCESS
    final = _update_calls(mock_crud)[-1]
    assert any("unavailable" in w for w in final.meta["callback"]["warnings"])


def test_execute_job_hard_block_marks_failed_and_sends_failure_callback(
    user_api_key: TestAuthContext,
) -> None:
    req = _make_request(callback_url="https://example.com/cb")

    with _patched_worker_db() as mock_crud, patch(
        "app.services.guardrails.jobs.apply_guardrails"
    ) as mock_apply, patch(
        "app.services.guardrails.jobs.send_callback"
    ) as mock_send, patch(
        "app.services.guardrails.jobs.get_webhook_secret", return_value="s"
    ):
        mock_apply.return_value = GuardrailsOutcome(
            safe_text=None,
            error="text rejected by validator X",
            bypassed=False,
            rephrase_needed=False,
            raw={"data": {"reason": "pii"}},
        )
        result = execute_job(
            project_id=user_api_key.project_id,
            job_id=str(uuid4()),
            task_id="t",
            task_instance=None,
            request_data=req.model_dump(mode="json"),
            organization_id=user_api_key.organization_id,
        )

    assert result["success"] is False
    assert "rejected" in result["error"]
    assert _final_status(mock_crud) == JobStatus.FAILED

    final = _update_calls(mock_crud)[-1]
    assert final.error_message == "text rejected by validator X"

    mock_send.assert_called_once()
    sent_payload = mock_send.call_args.kwargs["data"]
    assert sent_payload["success"] is False


def test_execute_job_crash_marks_failed_and_returns_error(
    user_api_key: TestAuthContext,
) -> None:
    req = _make_request(callback_url=None)

    with _patched_worker_db() as mock_crud, patch(
        "app.services.guardrails.jobs.apply_guardrails"
    ) as mock_apply:
        mock_apply.side_effect = RuntimeError("upstream exploded")
        result = execute_job(
            project_id=user_api_key.project_id,
            job_id=str(uuid4()),
            task_id="t",
            task_instance=None,
            request_data=req.model_dump(mode="json"),
            organization_id=user_api_key.organization_id,
        )

    assert result == {"success": False, "error": "upstream exploded"}
    assert _final_status(mock_crud) == JobStatus.FAILED
    final = _update_calls(mock_crud)[-1]
    assert "upstream exploded" in final.error_message


def test_execute_job_no_safe_text_adds_fallback_warning_and_echoes_original(
    user_api_key: TestAuthContext,
) -> None:
    req = _make_request(text="original text here")

    with _patched_worker_db() as mock_crud, patch(
        "app.services.guardrails.jobs.apply_guardrails"
    ) as mock_apply:
        mock_apply.return_value = GuardrailsOutcome(
            safe_text=None,
            error=None,
            bypassed=False,
            rephrase_needed=False,
            raw={"data": {}},
        )
        result = execute_job(
            project_id=user_api_key.project_id,
            job_id=str(uuid4()),
            task_id="t",
            task_instance=None,
            request_data=req.model_dump(mode="json"),
            organization_id=user_api_key.organization_id,
        )

    assert (
        result["data"]["response"]["output"]["content"]["value"] == "original text here"
    )
    final = _update_calls(mock_crud)[-1]
    assert any(
        "did not return a sanitised text" in w
        for w in final.meta["callback"]["warnings"]
    )
