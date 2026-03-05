"""Debug script for STS endpoint and chain job execution."""

import logging
import sys
from sqlmodel import Session

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_chain_job_creation():
    """Test if chain job can be created and queued."""
    from app.core.db import engine
    from app.models.llm.request import (
        LLMChainRequest,
        QueryParams,
        AudioInput,
        AudioContent,
        ChainBlock,
        LLMCallConfig,
        ConfigBlob,
        NativeCompletionConfig,
    )
    from app.services.llm.jobs import start_chain_job

    print("\n" + "=" * 80)
    print("STEP 1: Creating test chain request")
    print("=" * 80)

    # Create a minimal valid chain request
    test_request = LLMChainRequest(
        query=QueryParams(
            input=AudioInput(
                type="audio",
                content=AudioContent(
                    format="base64",
                    value="dGVzdF9hdWRpbw==",  # base64 encoded "test_audio"
                    mime_type="audio/ogg",
                ),
            )
        ),
        blocks=[
            ChainBlock(
                config=LLMCallConfig(
                    blob=ConfigBlob(
                        completion=NativeCompletionConfig(
                            provider="sarvamai-native",
                            type="stt",
                            params={
                                "model": "saarika:v1",
                                "language_code": "unknown",
                                "mode": "transcription",
                            },
                        )
                    )
                ),
                intermediate_callback=True,
            )
        ],
    )

    print(f"✅ Test request created with {len(test_request.blocks)} block(s)")

    print("\n" + "=" * 80)
    print("STEP 2: Attempting to start chain job")
    print("=" * 80)

    try:
        with Session(engine) as session:
            job_id = start_chain_job(
                db=session,
                request=test_request,
                project_id=1,  # Use test project ID
                organization_id=1,  # Use test org ID
            )
            print(f"✅ Chain job created successfully!")
            print(f"   Job ID: {job_id}")
            print(f"   Check your Celery worker logs for task execution")
            return job_id
    except Exception as e:
        print(f"❌ Failed to create chain job: {e}")
        import traceback

        traceback.print_exc()
        return None


def check_celery_connection():
    """Check if Celery is running and can receive tasks."""
    print("\n" + "=" * 80)
    print("STEP 3: Checking Celery connection")
    print("=" * 80)

    try:
        from app.celery.celery_app import celery_app

        # Check if broker is reachable
        inspector = celery_app.control.inspect()
        active_workers = inspector.active()

        if active_workers:
            print(f"✅ Celery workers are running:")
            for worker_name, tasks in active_workers.items():
                print(f"   - {worker_name}: {len(tasks)} active tasks")
        else:
            print("⚠️  No active Celery workers found!")
            print("   Make sure to start the Celery worker with:")
            print("   celery -A app.celery.celery_app worker --loglevel=info")

        # Check registered tasks
        registered = inspector.registered()
        if registered:
            print(f"\n✅ Registered tasks:")
            for worker_name, tasks in registered.items():
                print(f"   Worker: {worker_name}")
                for task in sorted(tasks):
                    if "high_priority" in task or "chain" in task.lower():
                        print(f"      - {task}")

    except Exception as e:
        print(f"❌ Failed to check Celery: {e}")
        import traceback

        traceback.print_exc()


def check_function_import():
    """Verify execute_chain_job can be imported."""
    print("\n" + "=" * 80)
    print("STEP 4: Verifying execute_chain_job import")
    print("=" * 80)

    try:
        from app.services.llm.jobs import execute_chain_job

        print(f"✅ execute_chain_job is importable")
        print(f"   Parameters: {execute_chain_job.__code__.co_varnames[:6]}")

        # Try dynamic import (same way Celery does it)
        import importlib

        module = importlib.import_module("app.services.llm.jobs")
        func = getattr(module, "execute_chain_job")
        print(f"✅ Dynamic import successful (same as Celery)")

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STS ENDPOINT DEBUG SCRIPT")
    print("=" * 80)

    check_function_import()
    check_celery_connection()
    job_id = test_chain_job_creation()

    if job_id:
        print("\n" + "=" * 80)
        print("DEBUGGING SUMMARY")
        print("=" * 80)
        print(f"✅ Chain job was queued successfully: {job_id}")
        print(f"\nNext steps:")
        print(f"1. Check your Celery worker logs for:")
        print(
            f"   - Task app.celery.tasks.job_execution.execute_high_priority_task received"
        )
        print(f"   - Executing high_priority job {job_id}")
        print(f"   - Function path: app.services.llm.jobs.execute_chain_job")
        print(f"\n2. If you don't see the task in worker logs:")
        print(f"   - Verify Celery broker (RabbitMQ/Redis) is running")
        print(f"   - Check broker connection in Celery worker startup logs")
        print(f"   - Restart Celery worker")
        print(f"\n3. If task starts but fails:")
        print(f"   - Look for error in Celery worker logs")
        print(
            f"   - Check database for job status: SELECT * FROM job WHERE id = '{job_id}';"
        )
    else:
        print("\n" + "=" * 80)
        print("DEBUGGING SUMMARY")
        print("=" * 80)
        print(f"❌ Failed to queue chain job")
        print(f"   Check the error messages above for details")

    print("=" * 80 + "\n")
