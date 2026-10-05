"""Tests for webhook delivery (app/services/assessment/api/callbacks.py)."""

from unittest.mock import patch

from app.models.assessment import AssessmentBatchResult
from app.services.assessment.api.callbacks import deliver
from app.tests.assessment.test_result_files import _seed
from app.tests.utils.auth import get_user_test_auth_context


def _result() -> AssessmentBatchResult:
    return AssessmentBatchResult(total_items=1)


class TestDeliver:
    def test_sends_presigned_files_in_metadata(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)

        with (
            patch(
                "app.services.assessment.api.callbacks.get_webhook_secret",
                return_value=None,
            ),
            patch(
                "app.services.assessment.api.callbacks.presign_result_files",
                return_value=None,
            ) as presign,
            patch(
                "app.services.assessment.api.callbacks.send_callback",
                return_value=True,
            ) as send_callback,
        ):
            sent = deliver(
                session=db,
                assessment=assessment,
                result=_result(),
                callback_url="https://client.example/webhook",
                request_metadata=None,
                failure_message=None,
            )

        assert sent is True
        presign.assert_called_once_with(session=db, assessment=assessment)
        assert send_callback.call_args.args[1]["metadata"] is None

    def test_a_metadata_bug_still_delivers_the_result(self, db) -> None:
        auth = get_user_test_auth_context(db)
        assessment, _ = _seed(db, auth)

        with (
            patch(
                "app.services.assessment.api.callbacks.get_webhook_secret",
                return_value=None,
            ),
            patch(
                "app.services.assessment.api.callbacks.presign_result_files",
                side_effect=RuntimeError("s3 unreachable"),
            ),
            patch(
                "app.services.assessment.api.callbacks.send_callback",
                return_value=True,
            ) as send_callback,
        ):
            sent = deliver(
                session=db,
                assessment=assessment,
                result=_result(),
                callback_url="https://client.example/webhook",
                request_metadata=None,
                failure_message=None,
            )

        assert sent is True
        assert send_callback.call_args.args[1]["metadata"] is None
