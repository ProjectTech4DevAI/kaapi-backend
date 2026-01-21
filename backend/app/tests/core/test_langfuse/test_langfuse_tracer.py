"""
Unit tests for LangfuseTracer class.

Tests all scenarios to ensure:
1. OpenAI responses are ALWAYS generated regardless of Langfuse state
2. Langfuse failures never block the main execution flow
3. All edge cases are handled gracefully

Scenarios covered:
- No credentials (None)
- Incomplete credentials (missing keys)
- Langfuse initialization exception
- fetch_traces failure (non-critical)
- Runtime exceptions in all methods
- Full working tracer
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from openai import OpenAIError

from app.core.langfuse.langfuse import LangfuseTracer
from app.models import Assistant, ResponsesAPIRequest
from app.services.response.response import generate_response, process_response


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_credentials() -> dict:
    """Valid Langfuse credentials."""
    return {
        "public_key": "pk-test-123",
        "secret_key": "sk-test-456",
        "host": "https://langfuse.example.com",
    }


@pytest.fixture
def assistant_mock() -> Assistant:
    """Mock assistant for generate_response tests."""
    return Assistant(
        id=123,
        assistant_id="asst_test123",
        name="Test Assistant",
        model="gpt-4",
        temperature=0.7,
        instructions="You are a helpful assistant.",
        vector_store_ids=["vs1"],
        max_num_results=5,
        project_id=1,
        organization_id=1,
    )


# =============================================================================
# Test: LangfuseTracer.__init__
# =============================================================================


class TestLangfuseTracerInit:
    """Tests for LangfuseTracer initialization scenarios."""

    def test_init_no_credentials_disables_langfuse(self) -> None:
        """Scenario #1: No credentials - tracer should be disabled."""
        tracer = LangfuseTracer(credentials=None)

        assert tracer.langfuse.enabled is False
        assert tracer.trace is None
        assert tracer.generation is None

    def test_init_empty_credentials_disables_langfuse(self) -> None:
        """Empty dict credentials - tracer should be disabled."""
        tracer = LangfuseTracer(credentials={})

        assert tracer.langfuse.enabled is False

    def test_init_missing_public_key_disables_langfuse(self) -> None:
        """Missing public_key - tracer should be disabled."""
        tracer = LangfuseTracer(
            credentials={
                "secret_key": "sk-test",
                "host": "https://example.com",
            }
        )

        assert tracer.langfuse.enabled is False

    def test_init_missing_secret_key_disables_langfuse(self) -> None:
        """Missing secret_key - tracer should be disabled."""
        tracer = LangfuseTracer(
            credentials={
                "public_key": "pk-test",
                "host": "https://example.com",
            }
        )

        assert tracer.langfuse.enabled is False

    def test_init_missing_host_disables_langfuse(self) -> None:
        """Missing host - tracer should be disabled."""
        tracer = LangfuseTracer(
            credentials={
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )

        assert tracer.langfuse.enabled is False

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_init_langfuse_exception_disables_tracer(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """Scenario #3: Langfuse init throws exception - tracer should be disabled."""
        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        # First call: default disabled, second call: raises exception, third: fallback disabled
        mock_langfuse_class.side_effect = [
            disabled_mock,
            Exception("Connection failed"),
            disabled_mock,
        ]

        tracer = LangfuseTracer(credentials=valid_credentials)

        assert tracer.langfuse.enabled is False

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_init_fetch_traces_fails_but_tracer_still_enabled(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """Scenario #4: fetch_traces fails - tracer should STILL be enabled."""
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.fetch_traces.side_effect = Exception("Network error")

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials, response_id="resp-123")

        # Key assertion: tracer is STILL enabled despite fetch_traces failure
        assert tracer.langfuse.enabled is True

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_init_success_resumes_session_from_traces(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """Successful init with existing traces - should resume session."""
        existing_trace = MagicMock()
        existing_trace.session_id = "existing-session-456"

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.fetch_traces.return_value.data = [existing_trace]

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials, response_id="resp-123")

        assert tracer.session_id == "existing-session-456"


# =============================================================================
# Test: Methods when Langfuse is disabled
# =============================================================================


class TestLangfuseTracerMethodsDisabled:
    """Tests that all methods are no-ops when Langfuse is disabled."""

    def test_start_trace_when_disabled_is_noop(self) -> None:
        """start_trace should do nothing when disabled."""
        tracer = LangfuseTracer(credentials=None)

        # Should not raise any exception
        tracer.start_trace(name="test", input={"question": "hello"})

        assert tracer.trace is None

    def test_start_generation_when_disabled_is_noop(self) -> None:
        """start_generation should do nothing when disabled."""
        tracer = LangfuseTracer(credentials=None)

        tracer.start_generation(name="test", input={"question": "hello"})

        assert tracer.generation is None

    def test_end_generation_when_disabled_is_noop(self) -> None:
        """end_generation should do nothing when disabled."""
        tracer = LangfuseTracer(credentials=None)

        # Should not raise any exception
        tracer.end_generation(output={"response": "world"})

    def test_update_trace_when_disabled_is_noop(self) -> None:
        """update_trace should do nothing when disabled."""
        tracer = LangfuseTracer(credentials=None)

        # Should not raise any exception
        tracer.update_trace(tags=["test"], output={"status": "success"})

    def test_log_error_when_disabled_is_noop(self) -> None:
        """log_error should do nothing when disabled."""
        tracer = LangfuseTracer(credentials=None)

        # Should not raise any exception
        tracer.log_error(error_message="Test error", response_id="resp-123")

    def test_flush_when_disabled_is_noop(self) -> None:
        """flush should do nothing when disabled."""
        tracer = LangfuseTracer(credentials=None)

        # Should not raise any exception
        tracer.flush()


# =============================================================================
# Test: Methods when Langfuse is enabled but operations fail
# =============================================================================


class TestLangfuseTracerMethodsFailure:
    """Tests that method failures are caught and don't propagate."""

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_start_trace_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """start_trace exception should be caught, not propagated."""
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.side_effect = Exception("Trace creation failed")

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)

        # Should NOT raise exception
        tracer.start_trace(name="test", input={"q": "hello"})

        assert tracer.trace is None

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_start_generation_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """start_generation exception should be caught, not propagated."""
        mock_trace = MagicMock(id="trace-123")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.side_effect = Exception("Generation failed")

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})

        # Should NOT raise exception
        tracer.start_generation(name="gen", input={"q": "hello"})

        assert tracer.generation is None

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_end_generation_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """end_generation exception should be caught, not propagated."""
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()
        mock_generation.end.side_effect = Exception("End failed")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})

        # Should NOT raise exception
        tracer.end_generation(output={"response": "world"})

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_update_trace_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """update_trace exception should be caught, not propagated."""
        mock_trace = MagicMock(id="trace-123")
        mock_trace.update.side_effect = Exception("Update failed")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})

        # Should NOT raise exception
        tracer.update_trace(tags=["test"], output={"status": "success"})

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_flush_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """flush exception should be caught, not propagated."""
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.flush.side_effect = Exception("Flush failed")

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)

        # Should NOT raise exception
        tracer.flush()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_log_error_exception_is_caught(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """log_error exception should be caught, not propagated."""
        mock_trace = MagicMock(id="trace-123")
        mock_trace.update.side_effect = Exception("Log error failed")
        mock_generation = MagicMock()
        mock_generation.end.side_effect = Exception("End failed")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})

        # Should NOT raise exception
        tracer.log_error(error_message="Test error", response_id="resp-123")


# =============================================================================
# Test: Integration with generate_response
# =============================================================================


class TestGenerateResponseWithTracer:
    """Tests that generate_response works in all tracer scenarios."""

    def test_generate_response_with_no_credentials(
        self, assistant_mock: Assistant
    ) -> None:
        """Scenario #1: No credentials - OpenAI should still work."""
        mock_client = MagicMock()
        tracer = LangfuseTracer(credentials=None)

        request = ResponsesAPIRequest(
            assistant_id="asst_test123",
            question="What is 2+2?",
        )

        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        # OpenAI client was called despite disabled tracer
        mock_client.responses.create.assert_called_once()
        assert error is None

    def test_generate_response_with_incomplete_credentials(
        self, assistant_mock: Assistant
    ) -> None:
        """Scenario #2: Incomplete credentials - OpenAI should still work."""
        mock_client = MagicMock()
        tracer = LangfuseTracer(credentials={"incomplete": True})

        request = ResponsesAPIRequest(
            assistant_id="asst_test123",
            question="What is 2+2?",
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

    def test_generate_response_openai_error_with_disabled_tracer(
        self, assistant_mock: Assistant
    ) -> None:
        """OpenAI error with disabled tracer - should handle gracefully."""
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("API failed")

        tracer = LangfuseTracer(credentials=None)

        request = ResponsesAPIRequest(
            assistant_id="asst_test123",
            question="What is 2+2?",
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


# =============================================================================
# Test: Successful tracing flow
# =============================================================================


class TestLangfuseTracerSuccess:
    """Tests for successful tracer operations."""

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_full_tracing_flow_success(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """Test complete tracing flow when everything works."""
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)

        # Full flow
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})
        tracer.end_generation(
            output={"response": "world"},
            usage={"input": 10, "output": 20, "total": 30},
            model="gpt-4",
        )
        tracer.update_trace(tags=["resp-123"], output={"status": "success"})
        tracer.flush()

        # Verify all methods were called
        enabled_mock.trace.assert_called_once()
        enabled_mock.generation.assert_called_once()
        mock_generation.end.assert_called_once()
        mock_trace.update.assert_called_once()
        enabled_mock.flush.assert_called_once()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_start_generation_without_trace_is_noop(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """start_generation should be no-op if trace doesn't exist."""
        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.side_effect = Exception("Trace failed")

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})  # This fails

        assert tracer.trace is None

        # This should be a no-op since trace is None
        tracer.start_generation(name="gen", input={"q": "hello"})

        assert tracer.generation is None
        # generation method should not be called since trace is None
        enabled_mock.generation.assert_not_called()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_end_generation_without_generation_is_noop(
        self, mock_langfuse_class: MagicMock, valid_credentials: dict
    ) -> None:
        """end_generation should be no-op if generation doesn't exist."""
        mock_trace = MagicMock(id="trace-123")

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.side_effect = Exception("Generation failed")

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        tracer = LangfuseTracer(credentials=valid_credentials)
        tracer.start_trace(name="test", input={"q": "hello"})
        tracer.start_generation(name="gen", input={"q": "hello"})  # This fails

        assert tracer.generation is None

        # This should be a no-op since generation is None
        tracer.end_generation(output={"response": "world"})
        # No exception should be raised


# =============================================================================
# Test: generate_response with ENABLED tracer
# =============================================================================


class TestGenerateResponseWithEnabledTracer:
    """Tests for generate_response with ENABLED tracer."""

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_generate_response_openai_error_with_enabled_tracer(
        self,
        mock_langfuse_class: MagicMock,
        valid_credentials: dict,
        assistant_mock: Assistant,
    ) -> None:
        """OpenAI error with enabled tracer - tracer.log_error should be called."""
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("API failed")

        tracer = LangfuseTracer(credentials=valid_credentials)

        request = ResponsesAPIRequest(
            assistant_id="asst_test123",
            question="What is 2+2?",
        )

        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        # OpenAI failed
        assert response is None
        assert error is not None
        assert "API failed" in error

        # Tracer methods were called before the error
        enabled_mock.trace.assert_called_once()
        enabled_mock.generation.assert_called_once()

    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_generate_response_success_with_enabled_tracer(
        self,
        mock_langfuse_class: MagicMock,
        valid_credentials: dict,
        assistant_mock: Assistant,
    ) -> None:
        """Full success flow with enabled tracer."""
        mock_trace = MagicMock(id="trace-123")
        mock_generation = MagicMock()

        enabled_mock = MagicMock()
        enabled_mock.enabled = True
        enabled_mock.trace.return_value = mock_trace
        enabled_mock.generation.return_value = mock_generation

        disabled_mock = MagicMock()
        disabled_mock.enabled = False

        mock_langfuse_class.side_effect = [disabled_mock, enabled_mock]

        # Mock successful OpenAI response
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
            assistant_id="asst_test123",
            question="What is 2+2?",
        )

        response, error = generate_response(
            tracer=tracer,
            client=mock_client,
            assistant=assistant_mock,
            request=request,
            ancestor_id=None,
        )

        # OpenAI succeeded
        assert response is not None
        assert error is None

        # All tracer methods were called
        enabled_mock.trace.assert_called_once()
        enabled_mock.generation.assert_called_once()
        mock_generation.end.assert_called_once()
        mock_trace.update.assert_called_once()


# =============================================================================
# Test: Integration tests for process_response
# =============================================================================


class TestProcessResponseIntegration:
    """Integration tests for process_response with various tracer scenarios."""

    @patch("app.services.response.response.persist_conversation")
    @patch("app.services.response.response.get_conversation_by_ancestor_id")
    @patch("app.services.response.response.Session")
    @patch("app.services.response.response.get_openai_client")
    @patch("app.services.response.response.get_assistant_by_id")
    @patch("app.services.response.response.get_provider_credential")
    @patch("app.services.response.response.JobCrud")
    def test_process_response_with_no_langfuse_credentials(
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
        """process_response works when langfuse_credentials is None."""
        # Setup mocks
        mock_get_credential.return_value = None  # No Langfuse credentials
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
            assistant_id="asst_test123",
            question="Test question",
        )

        result = process_response(
            request=request,
            project_id=1,
            organization_id=1,
            job_id=uuid4(),
            task_id="task-123",
            task_instance=None,
        )

        # OpenAI was called
        mock_client.responses.create.assert_called_once()
        # Response succeeded
        assert result.success is True

    @patch("app.services.response.response.persist_conversation")
    @patch("app.services.response.response.get_conversation_by_ancestor_id")
    @patch("app.services.response.response.Session")
    @patch("app.services.response.response.get_openai_client")
    @patch("app.services.response.response.get_assistant_by_id")
    @patch("app.services.response.response.get_provider_credential")
    @patch("app.services.response.response.JobCrud")
    @patch("app.core.langfuse.langfuse.Langfuse")
    def test_process_response_with_langfuse_init_failure(
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
        """process_response works even when Langfuse init fails."""
        # Langfuse init fails
        disabled_mock = MagicMock()
        disabled_mock.enabled = False
        mock_langfuse_class.side_effect = [
            disabled_mock,  # Default disabled
            Exception("Langfuse connection failed"),  # Init fails
            disabled_mock,  # Fallback disabled
        ]

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
            assistant_id="asst_test123",
            question="Test question",
        )

        result = process_response(
            request=request,
            project_id=1,
            organization_id=1,
            job_id=uuid4(),
            task_id="task-123",
            task_instance=None,
        )

        # OpenAI was still called despite Langfuse failure
        mock_client.responses.create.assert_called_once()
        assert result.success is True

    @patch("app.services.response.response.Session")
    @patch("app.services.response.response.get_openai_client")
    @patch("app.services.response.response.get_assistant_by_id")
    @patch("app.services.response.response.get_provider_credential")
    @patch("app.services.response.response.JobCrud")
    @patch("app.services.response.response._fail_job")
    def test_process_response_openai_error_with_disabled_tracer(
        self,
        mock_fail_job: MagicMock,
        mock_job_crud: MagicMock,
        mock_get_credential: MagicMock,
        mock_get_assistant: MagicMock,
        mock_get_client: MagicMock,
        mock_session: MagicMock,
        assistant_mock: Assistant,
    ) -> None:
        """process_response handles OpenAI error with disabled tracer."""
        mock_get_credential.return_value = None  # No Langfuse
        mock_get_assistant.return_value = assistant_mock

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("API failed")
        mock_get_client.return_value = mock_client

        # Mock _fail_job to return a failure response
        mock_fail_job.return_value = MagicMock(success=False, error="API failed")

        request = ResponsesAPIRequest(
            assistant_id="asst_test123",
            question="Test question",
        )

        result = process_response(
            request=request,
            project_id=1,
            organization_id=1,
            job_id=uuid4(),
            task_id="task-123",
            task_instance=None,
        )

        # Response failed gracefully
        assert result.success is False
