import uuid
from unittest.mock import MagicMock, patch

import pytest
import openai
from openai import OpenAIError
from sqlmodel import select

from app.api.routes.threads import (
    process_run,
    validate_thread,
    setup_thread,
    process_message_content,
    handle_openai_error,
    poll_run_and_prepare_response,
)
from app.models import OpenAI_Thread
from app.crud import get_thread_result
from app.core.langfuse.langfuse import LangfuseTracer


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
@patch("app.api.routes.threads.send_callback")
@patch("app.api.routes.threads.process_run")
def test_threads_endpoint(
    mock_process_run,
    mock_send_callback,
    mock_get_provider_credential,
    mock_configure_openai,
    client,
    db,
    user_api_key_header,
):
    """
    Test the /threads endpoint when creating a new thread.
    The patched configure_openai function simulates:
    - A successful assistant ID validation.
    - New thread creation with a dummy thread id.
    - No existing runs.
    The expected response should have status "processing" and include a thread_id.
    """
    # Create a dummy client to simulate OpenAI API behavior.
    dummy_client = MagicMock()
    # Simulate a valid assistant ID by ensuring retrieve doesn't raise an error.
    dummy_client.beta.assistants.retrieve.return_value = None
    # Simulate thread creation.
    dummy_thread = MagicMock()
    dummy_thread.id = "dummy_thread_id"
    dummy_client.beta.threads.create.return_value = dummy_thread
    # Simulate message creation.
    dummy_client.beta.threads.messages.create.return_value = None
    # Simulate that no active run exists.
    dummy_client.beta.threads.runs.list.return_value = MagicMock(data=[])

    # Mock get_provider_credential to return dummy credentials
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    # Mock configure_openai to return our dummy client
    mock_configure_openai.return_value = (dummy_client, True)

    request_data = {
        "question": "What is Glific?",
        "assistant_id": "assistant_123",
        "callback_url": "http://example.com/callback",
    }
    response = client.post(
        "/api/v1/threads", json=request_data, headers=user_api_key_header
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    assert response_json["data"]["status"] == "processing"
    assert response_json["data"]["message"] == "Run started"
    assert response_json["data"]["thread_id"] == "dummy_thread_id"


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
@pytest.mark.parametrize(
    "remove_citation, expected_message",
    [
        (
            True,
            "Glific is an open-source, two-way messaging platform designed for nonprofits to scale their outreach via WhatsApp",
        ),
        (
            False,
            "Glific is an open-source, two-way messaging platform designed for nonprofits to scale their outreach via WhatsApp【1:2†citation】",
        ),
    ],
)
def test_process_run_variants(
    mock_get_provider_credential,
    mock_configure_openai,
    remove_citation,
    expected_message,
):
    """
    Test process_run for both remove_citation variants:
    - Mocks the configure_openai function to simulate a completed run.
    - Verifies that send_callback is called with the expected message based on the remove_citation flag.
    """
    # Setup the mock client.
    mock_client = MagicMock()
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (mock_client, True)

    # Create the request with the variable remove_citation flag.
    request = {
        "question": "What is Glific?",
        "assistant_id": "assistant_123",
        "callback_url": "http://example.com/callback",
        "thread_id": "thread_123",
        "remove_citation": remove_citation,
    }

    # Simulate a completed run.
    mock_run = MagicMock()
    mock_run.status = "completed"
    mock_client.beta.threads.runs.create_and_poll.return_value = mock_run

    # Set up the dummy message based on the remove_citation flag.
    base_message = "Glific is an open-source, two-way messaging platform designed for nonprofits to scale their outreach via WhatsApp"
    citation_message = (
        base_message if remove_citation else f"{base_message}【1:2†citation】"
    )
    dummy_message = MagicMock()
    dummy_message.content = [MagicMock(text=MagicMock(value=citation_message))]
    mock_client.beta.threads.messages.list.return_value.data = [dummy_message]

    tracer = LangfuseTracer()
    # Patch send_callback and invoke process_run.
    with patch("app.api.routes.threads.send_callback") as mock_send_callback:
        process_run(request, mock_client, tracer)
        mock_send_callback.assert_called_once()
        callback_url, payload = mock_send_callback.call_args[0]
        assert callback_url == request["callback_url"]
        assert payload["data"]["message"] == expected_message
        assert payload["data"]["status"] == "success"
        assert payload["data"]["thread_id"] == "thread_123"
        assert payload["success"] is True


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
def test_threads_sync_endpoint_active_run(
    mock_get_provider_credential, mock_configure_openai, client, db, user_api_key_header
):
    """Test the /threads/sync endpoint when there's an active run."""
    # Setup mock client
    mock_client = MagicMock()
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (mock_client, True)

    # Simulate active run
    mock_run = MagicMock()
    mock_run.status = "in_progress"
    mock_client.beta.threads.runs.list.return_value = MagicMock(data=[mock_run])

    request_data = {
        "question": "Test question",
        "assistant_id": "assistant_123",
        "thread_id": "existing_thread",
    }

    # Expect the endpoint to raise when there's an active run
    with pytest.raises(Exception) as excinfo:
        client.post(
            "/api/v1/threads/sync", json=request_data, headers=user_api_key_header
        )
    assert "active run" in str(excinfo.value).lower()


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
def test_threads_sync_endpoint_success(
    mock_get_provider_credential, mock_configure_openai, client, db, user_api_key_header
):
    """Test the /threads/sync endpoint for successful completion."""
    # Setup mock client
    mock_client = MagicMock()
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (mock_client, True)

    # Simulate thread validation (no active runs)
    mock_client.beta.threads.runs.list.return_value = MagicMock(data=[])

    # Simulate thread creation
    dummy_thread = MagicMock()
    dummy_thread.id = "sync_thread_id"
    mock_client.beta.threads.create.return_value = dummy_thread

    # Simulate message creation
    mock_client.beta.threads.messages.create.return_value = None

    # Simulate successful run
    mock_run = MagicMock()
    mock_run.status = "completed"
    mock_run.usage.prompt_tokens = 10
    mock_run.usage.completion_tokens = 20
    mock_run.usage.total_tokens = 30
    mock_run.model = "gpt-4"
    mock_client.beta.threads.runs.create_and_poll.return_value = mock_run

    # Simulate message retrieval
    dummy_message = MagicMock()
    dummy_message.content = [MagicMock(text=MagicMock(value="Test response"))]
    mock_client.beta.threads.messages.list.return_value.data = [dummy_message]

    request_data = {
        "question": "Test question",
        "assistant_id": "assistant_123",
    }

    response = client.post(
        "/api/v1/threads/sync", json=request_data, headers=user_api_key_header
    )

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True
    assert response_json["data"]["status"] == "success"
    assert response_json["data"]["message"] == "Test response"
    assert response_json["data"]["thread_id"] == "sync_thread_id"
    assert "diagnostics" in response_json["data"]
    assert response_json["data"]["diagnostics"]["total_tokens"] == 30


def test_validate_thread_no_thread_id():
    """Test validate_thread when no thread_id is provided."""
    mock_client = MagicMock()
    is_valid, error = validate_thread(mock_client, None)
    assert is_valid is True
    assert error is None


def test_validate_thread_invalid_thread():
    """Test validate_thread with an invalid thread_id."""
    mock_client = MagicMock()
    error = openai.OpenAIError()
    error.message = "Invalid thread"
    error.response = MagicMock(status_code=404)
    error.body = {"message": "Invalid thread"}
    mock_client.beta.threads.runs.list.side_effect = error

    is_valid, error = validate_thread(mock_client, "invalid_thread")
    assert is_valid is False
    assert "Invalid thread ID" in error


def test_validate_thread_with_active_run():
    """Test validate_thread when there is an active run."""
    mock_client = MagicMock()
    mock_run = MagicMock()
    mock_run.status = "in_progress"
    mock_client.beta.threads.runs.list.return_value = MagicMock(data=[mock_run])

    is_valid, error = validate_thread(mock_client, "thread_123")
    assert is_valid is False
    assert "active run" in error.lower()
    assert "in_progress" in error


def test_validate_thread_with_queued_run():
    """Test validate_thread when there is a queued run."""
    mock_client = MagicMock()
    mock_run = MagicMock()
    mock_run.status = "queued"
    mock_client.beta.threads.runs.list.return_value = MagicMock(data=[mock_run])

    is_valid, error = validate_thread(mock_client, "thread_123")
    assert is_valid is False
    assert "active run" in error.lower()
    assert "queued" in error


def test_validate_thread_with_requires_action_run():
    """Test validate_thread when there is a run requiring action."""
    mock_client = MagicMock()
    mock_run = MagicMock()
    mock_run.status = "requires_action"
    mock_client.beta.threads.runs.list.return_value = MagicMock(data=[mock_run])

    is_valid, error = validate_thread(mock_client, "thread_123")
    assert is_valid is False
    assert "active run" in error.lower()
    assert "requires_action" in error


def test_validate_thread_with_completed_run():
    """Test validate_thread when there is a completed run."""
    mock_client = MagicMock()
    mock_run = MagicMock()
    mock_run.status = "completed"
    mock_client.beta.threads.runs.list.return_value = MagicMock(data=[mock_run])

    is_valid, error = validate_thread(mock_client, "thread_123")
    assert is_valid is True
    assert error is None


def test_validate_thread_with_no_runs():
    """Test validate_thread when there are no runs."""
    mock_client = MagicMock()
    mock_client.beta.threads.runs.list.return_value = MagicMock(data=[])

    is_valid, error = validate_thread(mock_client, "thread_123")
    assert is_valid is True
    assert error is None


def test_setup_thread_new_thread():
    """Test setup_thread for creating a new thread."""
    mock_client = MagicMock()
    mock_thread = MagicMock()
    mock_thread.id = "new_thread_id"
    mock_client.beta.threads.create.return_value = mock_thread
    mock_client.beta.threads.messages.create.return_value = None

    request = {"question": "Test question"}
    is_success, error = setup_thread(mock_client, request)

    assert is_success is True
    assert error is None
    assert request["thread_id"] == "new_thread_id"


def test_setup_thread_existing_thread():
    """Test setup_thread for using an existing thread."""
    mock_client = MagicMock()
    mock_client.beta.threads.messages.create.return_value = None

    request = {"question": "Test question", "thread_id": "existing_thread"}
    is_success, error = setup_thread(mock_client, request)

    assert is_success is True
    assert error is None


def test_process_message_content():
    """Test process_message_content with and without citation removal."""
    message = "Test message【1:2†citation】"

    # Test with citation removal
    processed = process_message_content(message, True)
    assert processed == "Test message"

    # Test without citation removal
    processed = process_message_content(message, False)
    assert processed == message


def test_handle_openai_error():
    """Test handle_openai_error with different error types."""
    # Test with error containing message in body
    error = MagicMock()
    error.body = {"message": "Test error message"}
    assert handle_openai_error(error) == "Test error message"

    # Test with error without message in body
    error = MagicMock()
    error.body = {}
    error.__str__.return_value = "Generic error"
    assert handle_openai_error(error) == "Generic error"


def test_handle_openai_error_with_message():
    """Test handle_openai_error when error has a message in its body."""
    error = MagicMock()
    error.body = {"message": "Test error message"}
    result = handle_openai_error(error)
    assert result == "Test error message"


def test_handle_openai_error_without_message():
    """Test handle_openai_error when error doesn't have a message in its body."""
    error = MagicMock()
    error.body = {"some_other_field": "value"}
    error.__str__.return_value = "Generic error message"
    result = handle_openai_error(error)
    assert result == "Generic error message"


def test_handle_openai_error_with_empty_body():
    """Test handle_openai_error when error has an empty body."""
    error = MagicMock()
    error.body = {}
    error.__str__.return_value = "Empty body error"
    result = handle_openai_error(error)
    assert result == "Empty body error"


def test_handle_openai_error_with_non_dict_body():
    """Test handle_openai_error when error body is not a dictionary."""
    error = MagicMock()
    error.body = "Not a dictionary"
    error.__str__.return_value = "Non-dict body error"
    result = handle_openai_error(error)
    assert result == "Non-dict body error"


def test_handle_openai_error_with_none_body():
    """Test handle_openai_error when error body is None."""
    error = MagicMock()
    error.body = None
    error.__str__.return_value = "None body error"
    result = handle_openai_error(error)
    assert result == "None body error"


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
def test_poll_run_and_prepare_response_completed(
    mock_get_provider_credential, mock_configure_openai, db
):
    mock_client = MagicMock()
    mock_run = MagicMock()
    mock_run.status = "completed"
    mock_client.beta.threads.runs.create_and_poll.return_value = mock_run

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MagicMock(value="Answer"))]
    mock_client.beta.threads.messages.list.return_value.data = [mock_message]
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (mock_client, True)

    request = {
        "question": "What is Glific?",
        "assistant_id": "assist_123",
        "thread_id": "test_thread_001",
        "remove_citation": True,
    }

    poll_run_and_prepare_response(request, mock_client, db)

    result = get_thread_result(db, "test_thread_001")
    assert result.response.strip() == "Answer"


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
def test_poll_run_and_prepare_response_openai_error_handling(
    mock_get_provider_credential, mock_configure_openai, db
):
    mock_client = MagicMock()
    mock_error = OpenAIError("Simulated OpenAI error")
    mock_client.beta.threads.runs.create_and_poll.side_effect = mock_error
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (mock_client, True)

    request = {
        "question": "Failing run",
        "assistant_id": "assist_123",
        "thread_id": "test_openai_error",
    }

    poll_run_and_prepare_response(request, mock_client, db)

    # Since thread_id is not the primary key, use select query
    statement = select(OpenAI_Thread).where(
        OpenAI_Thread.thread_id == "test_openai_error"
    )
    result = db.exec(statement).first()

    assert result is not None
    assert result.response is None
    assert result.status == "failed"
    assert "Simulated OpenAI error" in (result.error or "")


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
def test_poll_run_and_prepare_response_non_completed(
    mock_get_provider_credential, mock_configure_openai, db
):
    mock_client = MagicMock()
    mock_run = MagicMock(status="failed")
    mock_client.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (mock_client, True)

    request = {
        "question": "Incomplete run",
        "assistant_id": "assist_123",
        "thread_id": "test_non_complete",
    }

    poll_run_and_prepare_response(request, mock_client, db)

    # thread_id is not the primary key, so we query using SELECT
    statement = select(OpenAI_Thread).where(
        OpenAI_Thread.thread_id == "test_non_complete"
    )
    result = db.exec(statement).first()

    assert result is not None
    assert result.response is None
    assert result.status == "failed"


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
@patch("app.api.routes.threads.poll_run_and_prepare_response")
def test_threads_start_endpoint_creates_thread(
    mock_poll_run,
    mock_get_provider_credential,
    mock_configure_openai,
    client,
    db,
    user_api_key_header,
):
    """Test /threads/start creates thread and schedules background task."""
    mock_client = MagicMock()
    mock_thread = MagicMock()
    mock_thread.id = "mock_thread_001"
    mock_client.beta.threads.create.return_value = mock_thread
    mock_client.beta.threads.messages.create.return_value = None
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (mock_client, True)

    data = {"question": "What's 2+2?", "assistant_id": "assist_123"}

    response = client.post(
        "/api/v1/threads/start", json=data, headers=user_api_key_header
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"]
    assert res_json["data"]["thread_id"] == "mock_thread_001"
    assert res_json["data"]["status"] == "processing"
    assert res_json["data"]["prompt"] == "What's 2+2?"


def test_threads_result_endpoint_success(client, db, user_api_key_header):
    """Test /threads/result/{thread_id} returns completed thread."""
    thread_id = f"test_processing_{uuid.uuid4()}"
    question = "Capital of France?"
    message = "Paris."

    db.add(OpenAI_Thread(thread_id=thread_id, prompt=question, response=message))
    db.commit()

    response = client.get(
        f"/api/v1/threads/result/{thread_id}", headers=user_api_key_header
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "success"
    assert data["response"] == "Paris."
    assert data["thread_id"] == thread_id
    assert data["prompt"] == question


def test_threads_result_endpoint_processing(client, db, user_api_key_header):
    """Test /threads/result/{thread_id} returns processing status if no message yet."""
    thread_id = f"test_processing_{uuid.uuid4()}"
    question = "What is Glific?"

    db.add(OpenAI_Thread(thread_id=thread_id, prompt=question, response=None))
    db.commit()

    response = client.get(
        f"/api/v1/threads/result/{thread_id}", headers=user_api_key_header
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "processing"
    assert data["response"] is None
    assert data["thread_id"] == thread_id
    assert data["prompt"] == question


def test_threads_result_not_found(client, user_api_key_header):
    """Test /threads/result/{thread_id} returns error for nonexistent thread."""
    response = client.get(
        "/api/v1/threads/result/nonexistent_thread", headers=user_api_key_header
    )
    assert response.status_code == 404
    response_data = response.json()
    assert response_data["success"] is False
    assert "thread not found" in response_data["error"].lower()


@patch("app.api.routes.threads.configure_openai")
@patch("app.api.routes.threads.get_provider_credential")
def test_threads_start_missing_question(
    mock_get_provider_credential, mock_configure_openai, client, user_api_key_header
):
    """Test /threads/start with missing 'question' key in request."""
    mock_get_provider_credential.return_value = {"api_key": "dummy_api_key"}
    mock_configure_openai.return_value = (MagicMock(), True)

    bad_data = {"assistant_id": "assist_123"}  # no "question" key

    response = client.post(
        "/api/v1/threads/start", json=bad_data, headers=user_api_key_header
    )

    assert response.status_code == 422  # Unprocessable Entity (FastAPI will raise 422)
    error_response = response.json()
    assert error_response["success"] is False
    assert "question" in error_response["error"]
