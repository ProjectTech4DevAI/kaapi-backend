"""Unit tests for LangfuseTracer class."""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from openai import OpenAIError

from app.core.langfuse.langfuse import LangfuseTracer
from app.models import Assistant, ResponsesAPIRequest
from app.services.response.response import generate_response, process_response


@pytest.fixture
def valid_credentials() -> dict:
    def _create(**overrides) -> dict:
        defaults = {
            "public_key": "pk-test-123",
            "secret_key": "sk-test-456",
            "host": "https://langfuse.example.com",
        }
        return {**defaults, **overrides}

    return _create()


@pytest.fixture
def assistant_mock() -> Assistant:
    def _create(**overrides) -> Assistant:
        defaults = {
            "id": 123,
            "assistant_id": "asst_test123",
            "name": "Test Assistant",
            "model": "gpt-4",
            "temperature": 0.7,
            "instructions": "You are a helpful assistant.",
            "vector_store_ids": ["vs1"],
            "max_num_results": 5,
            "project_id": 1,
            "organization_id": 1,
        }
        return Assistant(**{**defaults, **overrides})

    return _create()


class TestLangfuseTracerInit:
    """Tests for LangfuseTracer initialization."""

    def test_no_credentials_sets_langfuse_to_none(self) -> None:
        tracer = LangfuseTracer(credentials=None)
        assert tracer.langfuse is None
        assert tracer.trace is None
        assert tracer.generation is None

    def test_empty_credentials_sets_langfuse_to_none(self) -> None:
        tracer = LangfuseTracer(credentials={})
        assert tracer.langfuse is None

    def test_missing_public_key_sets_langfuse_to_none(self) -> None:
        tracer = LangfuseTracer(
            credentials={"secret_key": "sk", "host": "https://x.com"}
        )
        assert tracer.langfuse is None

    def test_missing_secret_key_sets_langfuse_to_none(self) -> None:
        tracer = LangfuseTracer(
            credentials={"public_key": "pk", "host": "https://x.com"}
        )
        assert tracer.langfuse is None

    def test_missing_host_sets_langfuse_to_none(self) -> None:
        tracer = LangfuseTracer(credentials={"public_key": "pk", "secret_key": "sk"})
        assert tracer.langfuse is None

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_langfuse_exception_sets_langfuse_to_none(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        mock_langfuse_class.side_effect = Exception("Connection failed")
        tracer = LangfuseTracer(credentials=valid_credentials)
        assert tracer.langfuse is None

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_fetch_traces_failure_keeps_tracer_enabled(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.fetch_traces.side_effect = Exception("Network error")
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials, response_id="resp-123")

        assert tracer.langfuse is not None
        assert tracer.langfuse.enabled is True

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_resumes_session_from_existing_traces(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        existing_trace = MagicMock()
        existing_trace.session_id = "existing-session-456"

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.fetch_traces.return_value.data = [existing_trace]
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials, response_id="resp-123")

        assert tracer.session_id == "existing-session-456"


class TestLangfuseTracerMethodsDisabled:
    """Tests that methods are no-ops when Langfuse is disabled."""

    def test_start_trace_is_noop(self) -> None:
        tracer = LangfuseTracer(credentials=None)
        tracer.start_trace(name="test", input={"question": "hello"})
        assert tracer.trace is None

    def test_start_generation_is_noop(self) -> None:
        tracer = LangfuseTracer(credentials=None)
        tracer.start_generation(name="test", input={"question": "hello"})
        assert tracer.generation is None

    def test_end_generation_is_noop(self) -> None:
        tracer = LangfuseTracer(credentials=None)
        tracer.end_generation(output={"response": "world"})

    def test_update_trace_is_noop(self) -> None:
        tracer = LangfuseTracer(credentials=None)
        tracer.update_trace(tags=["test"], output={"status": "success"})

    def test_log_error_is_noop(self) -> None:
        tracer = LangfuseTracer(credentials=None)
        tracer.log_error(error_message="Test error", response_id="resp-123")

    def test_flush_is_noop(self) -> None:
        tracer = LangfuseTracer(credentials=None)
        tracer.flush()


class TestLangfuseTracerMethodsFailure:
    """Tests that method failures are caught and don't propagate."""

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_start_trace_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.side_effect = Exception("Trace creation failed")
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})

        assert tracer.trace is None

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_start_generation_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        mock_trace = MagicMock(id="trace-123")
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.side_effect = Exception("Generation failed")
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})

        assert tracer.generation is None

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_end_generation_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()
        mock_generation.end.side_effect = Exception("End failed")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})
        tracer.end_generation(output={"response": "world"})

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_update_trace_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        mock_trace = MagicMock(id="trace-123")
        mock_trace.update.side_effect = Exception("Update failed")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.update_trace(tags=["test"], output={"status": "success"})

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_flush_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.flush.side_effect = Exception("Flush failed")
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.flush()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_log_error_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        mock_trace = MagicMock(id="trace-123")
        mock_trace.update.side_effect = Exception("Log error failed")
        mock_generation = MagicMock()
        mock_generation.end.side_effect = Exception("End failed")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})
        tracer.log_error(error_message="Test error", response_id="resp-123")


class TestGenerateResponseWithTracer:
    """Tests that generate_response works regardless of tracer state."""

    def test_with_no_credentials(self, assistant_mock: Assistant) -> None:
        mock_client = MagicMock()
        tracer = LangfuseTracer(credentials=None)

        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="What is 2+2?"
        )
        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        mock_client.responses.create.assert_called_once()
        assert error is None

    def test_with_incomplete_credentials(self, assistant_mock: Assistant) -> None:
        mock_client = MagicMock()
        tracer = LangfuseTracer(credentials={"incomplete": True})

        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="What is 2+2?"
        )
        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        mock_client.responses.create.assert_called_once()
        assert error is None

    def test_openai_error_with_disabled_tracer(self, assistant_mock: Assistant) -> None:
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("API failed")
        tracer = LangfuseTracer(credentials=None)

        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="What is 2+2?"
        )
        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        assert response is None
        assert error is not None
        assert "API failed" in error


class TestLangfuseTracerSuccess:
    """Tests for successful tracer operations."""

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_full_tracing_flow(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})
        tracer.end_generation(
            output={"response": "world"},
            usage={"input": 10, "output": 20, "total": 30},
            model="gpt-4",
        )
        tracer.update_trace(tags=["resp-123"], output={"status": "success"})
        tracer.flush()

        enabled_mock.trace.assert_called_once()
        enabled_mock.generation.assert_called_once()
        mock_generation.end.assert_called_once()
        mock_trace.update.assert_called_once()
        enabled_mock.flush.assert_called_once()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_start_generation_without_trace_is_noop(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.side_effect = Exception("Trace failed")
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})  # Fails
        assert tracer.trace is None

        tracer.start_generation(name="gen", input={"q": "hello"})
        assert tracer.generation is None
        enabled_mock.generation.assert_not_called()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_end_generation_without_generation_is_noop(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        mock_trace = MagicMock(id="trace-123")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.side_effect = Exception("Generation failed")
        mock_langfuse_class.return_value = enabled_mock

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})  # Fails
        assert tracer.generation is None

        tracer.end_generation(output={"response": "world"})  # No exception


class TestGenerateResponseWithEnabledTracer:
    """Tests for generate_response with enabled tracer."""

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_openai_error_still_calls_tracer(
        self,
        mock_langfuse_class: MagicMock,
        valid_credentials: dict,
        assistant_mock: Assistant,
    ) -> None:
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation
        mock_langfuse_class.return_value = enabled_mock

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("API failed")

        tracer = LangfuseTracer(credentials=valid_credentials)
        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="What is 2+2?"
        )

        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        assert response is None
        assert "API failed" in error
        enabled_mock.trace.assert_called_once()
        enabled_mock.generation.assert_called_once()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_success_calls_all_tracer_methods(
        self,
        mock_langfuse_class: MagicMock,
        valid_credentials: dict,
        assistant_mock: Assistant,
    ) -> None:
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation
        mock_langfuse_class.return_value = enabled_mock

        mock_response = MagicMock()
        mock_response.id = "resp-456"
        mock_response.output_text = "The answer is 4"
        mock_response.model = "gpt-4"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response

        tracer = LangfuseTracer(credentials=valid_credentials)
        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="What is 2+2?"
        )

        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        assert response is not None
        assert error is None
        enabled_mock.trace.assert_called_once()
        enabled_mock.generation.assert_called_once()
        mock_generation.end.assert_called_once()
        mock_trace.update.assert_called_once()


class TestProcessResponseIntegration:
    """Integration tests for process_response."""

    @patch("app.services.response.response.persist_conversation")
    @patch("app.services.response.response.get_conversation_by_ancestor_id")
    @patch("app.services.response.response.Session")
    @patch("app.services.response.response.get_openai_client")
    @patch("app.services.response.response.get_assistant_by_id")
    @patch("app.services.response.response.get_provider_credential")
    @patch("app.services.response.response.JobCrud")
    def test_works_without_langfuse_credentials(
        self,
        mock_job_crud: MagicMock,
        mock_get_credential: MagicMock,
        mock_get_assistant: MagicMock,
        mock_get_client: MagicMock,
        mock_session: MagicMock,
        mock_get_conversation: MagicMock,
        mock_persist: MagicMock,
        assistant_mock: Assistant,
    ) -> None:
        mock_get_credential.return_value = None
        mock_get_assistant.return_value = assistant_mock
        mock_get_conversation.return_value = None

        mock_response = MagicMock()
        mock_response.id = "resp-123"
        mock_response.output_text = "Answer"
        mock_response.model = "gpt-4"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.previous_response_id = None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="Test question"
        )
        result = process_response(
            request=request,
            project_id=1,
            organization_id=1,
            job_id=uuid4(),
            task_id="task-123",
            task_instance=None,
        )

        mock_client.responses.create.assert_called_once()
        assert result.success is True

    @patch("app.services.response.response.persist_conversation")
    @patch("app.services.response.response.get_conversation_by_ancestor_id")
    @patch("app.services.response.response.Session")
    @patch("app.services.response.response.get_openai_client")
    @patch("app.services.response.response.get_assistant_by_id")
    @patch("app.services.response.response.get_provider_credential")
    @patch("app.services.response.response.JobCrud")
    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_works_when_langfuse_init_fails(
        self,
        mock_langfuse_class: MagicMock,
        mock_job_crud: MagicMock,
        mock_get_credential: MagicMock,
        mock_get_assistant: MagicMock,
        mock_get_client: MagicMock,
        mock_session: MagicMock,
        mock_get_conversation: MagicMock,
        mock_persist: MagicMock,
        assistant_mock: Assistant,
        valid_credentials: dict,
    ) -> None:
        mock_langfuse_class.side_effect = Exception("Langfuse connection failed")
        mock_get_credential.return_value = valid_credentials
        mock_get_assistant.return_value = assistant_mock
        mock_get_conversation.return_value = None

        mock_response = MagicMock()
        mock_response.id = "resp-123"
        mock_response.output_text = "Answer"
        mock_response.model = "gpt-4"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.previous_response_id = None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="Test question"
        )
        result = process_response(
            request=request,
            project_id=1,
            organization_id=1,
            job_id=uuid4(),
            task_id="task-123",
            task_instance=None,
        )

        mock_client.responses.create.assert_called_once()
        assert result.success is True

    @patch("app.services.response.response.Session")
    @patch("app.services.response.response.get_openai_client")
    @patch("app.services.response.response.get_assistant_by_id")
    @patch("app.services.response.response.get_provider_credential")
    @patch("app.services.response.response.JobCrud")
    @patch("app.services.response.response._fail_job")
    def test_handles_openai_error_gracefully(
        self,
        mock_fail_job: MagicMock,
        mock_job_crud: MagicMock,
        mock_get_credential: MagicMock,
        mock_get_assistant: MagicMock,
        mock_get_client: MagicMock,
        mock_session: MagicMock,
        assistant_mock: Assistant,
    ) -> None:
        mock_get_credential.return_value = None
        mock_get_assistant.return_value = assistant_mock

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("API failed")
        mock_get_client.return_value = mock_client
        mock_fail_job.return_value = MagicMock(success=False, error="API failed")

        request = ResponsesAPIRequest(
            assistant_id="asst_test123", question="Test question"
        )
        result = process_response(
            request=request,
            project_id=1,
            organization_id=1,
            job_id=uuid4(),
            task_id="task-123",
            task_instance=None,
        )

        assert result.success is False
