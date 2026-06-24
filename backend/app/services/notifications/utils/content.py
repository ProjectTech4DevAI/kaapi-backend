"""Helpers for building eval-completion notification content.

Pure formatting/linking logic split out of the
`send_eval_completion_notification` task so the task module stays focused on
orchestration.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.core.config import settings
from app.models import EvaluationRun, NotificationType

IST_TZ = ZoneInfo("Asia/Kolkata")


def build_eval_results_link(eval_run: EvaluationRun) -> str:
    return f"{settings.FRONTEND_HOST}/evaluations/{eval_run.id}"


def notification_type_for_status(status: str) -> str:
    if status == "failed":
        return NotificationType.EVAL_FAILED.value
    return NotificationType.EVAL_COMPLETED.value


def format_completed_at(dt: datetime | None) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(IST_TZ)
    hour_12 = local.strftime("%I").lstrip("0") or "12"
    return local.strftime(f"%B %d, %Y at {hour_12}:%M %p")


def build_eval_completion_payload(
    *, eval_run: EvaluationRun, project_name: str
) -> dict:
    return {
        "run_name": eval_run.run_name,
        "project_name": project_name,
        "status": eval_run.status,
        "completed_at": format_completed_at(eval_run.updated_at),
        "link": build_eval_results_link(eval_run),
        "error_message": eval_run.error_message,
    }
