"""Celery task definitions for the single priority `default` queue.

All tasks share one queue (`default`, declared with `x-max-priority=10`) and are
ordered by the per-task `priority`:

    9  LLM call + LLM chain (run_llm_job, run_llm_chain_job, run_response_job)
    6  Fast evaluation (run_evaluation_fast)
    2  Everything else (doctransform, collections, STT/TTS evaluation, assessment)
    1  Notifications (send_eval_completion_notification)

Higher priority drains first; within the same priority, delivery is FIFO.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from asgi_correlation_id import correlation_id
from celery import Task, current_task
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract
from sqlmodel import Session

from app.celery.celery_app import celery_app
from app.celery.utils import gevent_timeout
from app.core.config import settings
from app.core.db import engine
from app.crud.notification import (
    create_pending_notification,
    list_pending_notifications_for_entity,
    mark_notification_failed,
    mark_notification_sent,
    notifications_exist_for_entity,
)
from app.crud.user_project import get_users_by_project
from app.models import (
    EvaluationRun,
    NotificationEntityType,
    NotificationProvider,
    NotificationType,
    Project,
)
from app.utils import generate_eval_completion_email, send_email

logger = logging.getLogger(__name__)

# Sentinel correlation id used when no trace id is propagated from the
# enqueueing request. Matches the codebase-wide "N/A" default (see
# app/core/logger.py and app/celery/utils.py).
DEFAULT_TRACE_ID = "N/A"

EVAL_COMPLETION_TEMPLATE = "eval_completion_v1"
IST_TZ = ZoneInfo("Asia/Kolkata")


def _set_trace(trace_id: str) -> None:
    correlation_id.set(trace_id)
    logger.info(f"[_set_trace] Set correlation ID: {trace_id}")


def _extract_parent_context(task_instance) -> otel_context.Context:
    """Extract OTel parent context from Celery headers if available."""
    headers = getattr(task_instance.request, "headers", None) or {}
    carrier: dict[str, str] = {}

    if isinstance(headers, dict):
        for key, value in headers.items():
            if isinstance(value, str):
                carrier[str(key)] = value

        nested = headers.get("otel", {})
        if isinstance(nested, dict):
            for key, value in nested.items():
                if isinstance(value, str):
                    carrier[str(key)] = value

    return extract(carrier)


def _run_with_otel_parent(task_instance, fn):
    """Attach extracted parent context and execute function.

    When Celery auto-instrumentation is active, there is already a current
    `run/...` span. Re-attaching extracted parent context here would make
    service spans become siblings of `run/...` instead of children.

    We only attach extracted context as a fallback when no active span exists.
    """
    current_ctx = trace.get_current_span().get_span_context()
    if current_ctx and current_ctx.is_valid:
        return fn()

    parent_ctx = _extract_parent_context(task_instance)
    token = otel_context.attach(parent_ctx)
    try:
        return fn()
    finally:
        otel_context.detach(token)


@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_llm_job")
def run_llm_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.llm.jobs import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_llm_chain_job")
def run_llm_chain_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.llm.jobs import execute_chain_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_chain_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=9)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_response_job")
def run_response_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.response.jobs import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_doctransform_job")
def run_doctransform_job(self, project_id: int, job_id: str, trace_id: str, **kwargs):
    from app.services.doctransform.job import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_collection_setup_job")
def run_collection_setup_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.create_collection import execute_setup_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_setup_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_collection_batch_job")
def run_collection_batch_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.create_collection import execute_batch_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_batch_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_delete_collection_job")
def run_delete_collection_job(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.collections.delete_collection import execute_job

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_job(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_stt_batch_submission")
def run_stt_batch_submission(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.stt_evaluations.batch_job import execute_batch_submission

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_batch_submission(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_stt_metric_computation")
def run_stt_metric_computation(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.stt_evaluations.metric_job import execute_metric_computation

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_metric_computation(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_tts_batch_submission")
def run_tts_batch_submission(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.tts_evaluations.batch_job import execute_batch_submission

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_batch_submission(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_assessment_pipeline")
def run_assessment_pipeline(
    self,
    run_id: int,
    organization_id: int,
    project_id: int,
    trace_id: str,
    **kwargs,
):
    from app.services.assessment.tasks import execute_assessment_pipeline

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_assessment_pipeline(
            run_id=run_id,
            organization_id=organization_id,
            project_id=project_id,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=2)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_tts_result_processing")
def run_tts_result_processing(
    self, project_id: int, job_id: str, trace_id: str, **kwargs
):
    from app.services.tts_evaluations.batch_result_processing import (
        execute_tts_result_processing,
    )

    _set_trace(trace_id)
    return _run_with_otel_parent(
        self,
        lambda: execute_tts_result_processing(
            project_id=project_id,
            job_id=job_id,
            task_id=current_task.request.id,
            task_instance=self,
            **kwargs,
        ),
    )


@celery_app.task(bind=True, queue="default", priority=6)
@gevent_timeout(settings.CELERY_TASK_SOFT_TIME_LIMIT, "run_evaluation_fast")
def run_evaluation_fast(
    self: Task, eval_run_id: int, trace_id: str = DEFAULT_TRACE_ID
) -> None:
    """Run the fast evaluation pipeline for one EvaluationRun.

    Idempotency: each stage is skipped on retry when its `batch_job` marker is
    already set on the EvaluationRun, so Celery redelivery never re-calls
    OpenAI for work that already succeeded.

    Args:
        eval_run_id: ID of the EvaluationRun (run_mode="fast").
        trace_id: Correlation id from the enqueueing request, propagated into
            the worker for log correlation.
    """
    from app.services.evaluations.fast import execute_fast_evaluation

    _set_trace(trace_id)
    logger.info(
        f"[run_evaluation_fast] Starting fast evaluation task | "
        f"eval_run_id={eval_run_id} | task_id={current_task.request.id}"
    )

    return _run_with_otel_parent(
        self,
        lambda: execute_fast_evaluation(eval_run_id=eval_run_id),
    )


def _build_eval_results_link(eval_run: EvaluationRun) -> str:
    return f"{settings.FRONTEND_HOST}/evaluations/{eval_run.id}"


def _notification_type_for_status(status: str) -> str:
    if status == "failed":
        return NotificationType.EVAL_FAILED.value
    return NotificationType.EVAL_COMPLETED.value


def _format_completed_at(dt: datetime | None) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(IST_TZ)
    hour_12 = local.strftime("%I").lstrip("0") or "12"
    return local.strftime(f"%B %d, %Y at {hour_12}:%M %p")


def _build_eval_completion_payload(
    *, eval_run: EvaluationRun, project_name: str
) -> dict:
    return {
        "run_name": eval_run.run_name,
        "project_name": project_name,
        "status": eval_run.status,
        "completed_at": _format_completed_at(eval_run.updated_at),
        "link": _build_eval_results_link(eval_run),
        "error_message": eval_run.error_message,
    }


@celery_app.task(bind=True, queue="default", priority=1)
@gevent_timeout(
    settings.CELERY_TASK_SOFT_TIME_LIMIT, "send_eval_completion_notification"
)
def send_eval_completion_notification(self, evaluation_id: int) -> dict:
    """
    Fan out a completion notification for an eval run to every project member.

    The flow per recipient is: insert a `pending` notification row with the
    payload snapshot, attempt SMTP delivery, then flip the row to `sent`
    (with `sent_at`) or `failed` (with `failed_reason`). The `notification`
    table itself acts as the idempotency guard — if any rows already exist
    for this (entity_type, entity_id, notification_type), the task bails
    out without sending again.
    """
    with Session(engine) as session:
        eval_run = session.get(EvaluationRun, evaluation_id)
        if not eval_run:
            logger.error(
                f"[send_eval_completion_notification] EvaluationRun not found | "
                f"evaluation_id={evaluation_id}"
            )
            return {
                "evaluation_id": evaluation_id,
                "sent": 0,
                "failed": 0,
                "not_found": True,
            }

        notification_type = _notification_type_for_status(eval_run.status)

        already_processed = notifications_exist_for_entity(
            session=session,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=eval_run.id,
            notification_type=notification_type,
        )
        if already_processed:
            logger.info(
                f"[send_eval_completion_notification] Already processed; skipping | "
                f"evaluation_id={evaluation_id} | type={notification_type}"
            )
            return {
                "evaluation_id": evaluation_id,
                "sent": 0,
                "failed": 0,
                "skipped": True,
            }

        if not settings.emails_enabled:
            logger.warning(
                f"[send_eval_completion_notification] Email not configured; skipping | "
                f"evaluation_id={evaluation_id}"
            )
            return {
                "evaluation_id": evaluation_id,
                "sent": 0,
                "failed": 0,
                "skipped": True,
            }

        project = session.get(Project, eval_run.project_id)
        project_name = project.name if project else f"Project {eval_run.project_id}"

        users = get_users_by_project(session=session, project_id=eval_run.project_id)
        recipients = [u for u in users if u.is_active and u.email]
        if not recipients:
            logger.info(
                f"[send_eval_completion_notification] No recipients for project | "
                f"evaluation_id={evaluation_id} | project_id={eval_run.project_id}"
            )
            return {"evaluation_id": evaluation_id, "sent": 0, "failed": 0}

        payload = _build_eval_completion_payload(
            eval_run=eval_run, project_name=project_name
        )
        email_data = generate_eval_completion_email(
            run_name=eval_run.run_name,
            project_name=project_name,
            status=eval_run.status,
            completed_at=payload["completed_at"],
            link=payload["link"],
            error_message=eval_run.error_message,
        )

        seen_user_ids: set[int] = set()
        for user in recipients:
            if user.user_id in seen_user_ids:
                continue
            seen_user_ids.add(user.user_id)
            create_pending_notification(
                session=session,
                notification_type=notification_type,
                provider=NotificationProvider.EMAIL.value,
                recipient_user_id=user.user_id,
                entity_type=NotificationEntityType.EVAL_RUN.value,
                entity_id=eval_run.id,
                project_id=eval_run.project_id,
                subject=email_data.subject,
                body_template=EVAL_COMPLETION_TEMPLATE,
                payload=payload,
            )
        session.commit()

        pending = list_pending_notifications_for_entity(
            session=session,
            entity_type=NotificationEntityType.EVAL_RUN.value,
            entity_id=eval_run.id,
            notification_type=notification_type,
        )

        sent_count = 0
        failed_count = 0
        for notification in pending:
            email_to = next(
                (
                    u.email
                    for u in recipients
                    if u.user_id == notification.recipient_user_id
                ),
                None,
            )
            if not email_to:
                mark_notification_failed(
                    session=session,
                    notification=notification,
                    reason="Recipient email not available",
                )
                failed_count += 1
                continue
            try:
                send_email(
                    email_to=email_to,
                    subject=email_data.subject,
                    html_content=email_data.html_content,
                )
                mark_notification_sent(session=session, notification=notification)
                sent_count += 1
                logger.info(
                    f"[send_eval_completion_notification] Sent | "
                    f"evaluation_id={evaluation_id} | "
                    f"notification_id={notification.id} | to={email_to}"
                )
            except Exception as e:
                mark_notification_failed(
                    session=session, notification=notification, reason=str(e)
                )
                failed_count += 1
                logger.error(
                    f"[send_eval_completion_notification] Send failed | "
                    f"evaluation_id={evaluation_id} | "
                    f"notification_id={notification.id} | to={email_to} | error={e}",
                    exc_info=True,
                )
        session.commit()

        logger.info(
            f"[send_eval_completion_notification] Done | "
            f"evaluation_id={evaluation_id} | project_id={eval_run.project_id} | "
            f"type={notification_type} | recipients={len(pending)} | "
            f"sent={sent_count} | failed={failed_count}"
        )
        return {
            "evaluation_id": evaluation_id,
            "notification_type": notification_type,
            "recipients": len(pending),
            "sent": sent_count,
            "failed": failed_count,
        }
