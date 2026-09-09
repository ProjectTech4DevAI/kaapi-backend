from app.core.sentry_filters import before_send_filter


def _event_with_celery_job(task_name: str, kwargs: dict) -> dict:
    return {
        "extra": {
            "celery-job": {
                "task_name": task_name,
                "args": [],
                "kwargs": kwargs,
            }
        }
    }


def test_before_send_filter_redacts_query_for_llm_job():
    event = _event_with_celery_job(
        "app.celery.tasks.job_execution.run_llm_job",
        {
            "job_id": "b9ce6621-9b5a-45c4-969f-df7613ff7dc4",
            "organization_id": 1,
            "project_id": 1,
            "request_data": {
                "callback_url": "https://webhooksite.net/some-id",
                "config": {"blob": {"completion": {"provider": "openai"}}},
                "query": {
                    "input": {
                        "type": "text",
                        "content": {"value": "Amit Gupta phone number is 919611188278"},
                    }
                },
                "request_metadata": None,
            },
        },
    )

    result = before_send_filter(event, {})

    request_data = result["extra"]["celery-job"]["kwargs"]["request_data"]
    assert request_data["query"] == "[REDACTED]"
    # callback_url can carry identifying/credential data and is redacted too.
    assert request_data["callback_url"] == "[REDACTED]"
    # Non-sensitive fields are left untouched so the trace stays useful.
    assert request_data["config"] == {"blob": {"completion": {"provider": "openai"}}}


def test_before_send_filter_redacts_query_for_chain_and_response_jobs():
    for task_name in (
        "app.celery.tasks.job_execution.run_llm_chain_job",
        "app.celery.tasks.job_execution.run_response_job",
    ):
        event = _event_with_celery_job(
            task_name,
            {"request_data": {"query": {"input": "sensitive text"}}},
        )

        result = before_send_filter(event, {})

        request_data = result["extra"]["celery-job"]["kwargs"]["request_data"]
        assert request_data["query"] == "[REDACTED]"


def test_before_send_filter_ignores_unrelated_tasks():
    event = _event_with_celery_job(
        "app.celery.tasks.job_execution.run_doctransform_job",
        {"request_data": {"query": {"input": "not an llm job"}}},
    )

    result = before_send_filter(event, {})

    request_data = result["extra"]["celery-job"]["kwargs"]["request_data"]
    assert request_data["query"] == {"input": "not an llm job"}


def test_before_send_filter_passes_through_events_without_celery_job():
    event = {"message": "some unrelated error"}

    result = before_send_filter(event, {})

    assert result == event
