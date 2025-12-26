from sqlmodel import Session

from app.models import OpenAIThreadCreate
from app.crud import upsert_thread_result, get_thread_result


def test_upsert_and_get_thread_result(db: Session):
    thread_id = "thread_test_123"
    prompt = "What is the capital of Spain?"
    response = "Madrid is the capital of Spain."

    # Insert
    upsert_thread_result(
        db,
        OpenAIThreadCreate(
            thread_id=thread_id,
            prompt=prompt,
            response=response,
            status="completed",
            error=None,
        ),
    )

    # Retrieve
    result = get_thread_result(db, thread_id)

    assert result is not None
    assert result.thread_id == thread_id
    assert result.prompt == prompt
    assert result.response == response

    # Update with new response
    updated_response = "Madrid."
    upsert_thread_result(
        db,
        OpenAIThreadCreate(
            thread_id=thread_id,
            prompt=prompt,
            response=updated_response,
            status="completed",
            error=None,
        ),
    )

    result_updated = get_thread_result(db, thread_id)
    assert result_updated.response == updated_response
