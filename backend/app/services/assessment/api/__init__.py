"""Assessment API-client service: BATCH staged pipeline, result assembly, callbacks.

Separate from the legacy RUN pipeline (``services/assessment/{service,stages,tasks}``).
Entrypoint: ``submission.submit``. Async driver: ``batch.run_batch_stage`` (via the
``run_assessment_api_batch`` Celery task).
"""
