import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from openai import OpenAIError
from sqlmodel import Session

from app.core.langfuse.langfuse import LangfuseTracer
from app.models import Assistant, ResponsesAPIRequest
from app.services.response.response import generate_response, get_file_search_results


@pytest.fixture
def assistant_mock() -> Assistant:
    """Fixture to create an assistant in DB with id=123."""
    assistant = Assistant(
        id="123",
        name="Test Assistant",
        model="gpt-4",
        temperature=0.7,
        instructions="You are a helpful assistant.",
        vector_store_ids=["vs1", "vs2"],
        max_num_results=5,
    )
    return assistant


def test_generate_response_success(db: Session, assistant_mock: Assistant):
    """Test successful OpenAI response generation."""
    mock_client = MagicMock()

    request = ResponsesAPIRequest(
        assistant_id="123",
        question="What is the capital of France?",
        callback_url="http://example.com/callback",
    )

    response, error = generate_response(
        tracer=LangfuseTracer(),
        client=mock_client,
        assistant=assistant_mock,
        request=request,
        ancestor_id=None,
    )

    mock_client.responses.create.assert_called_once()
    assert error is None


def test_generate_response_openai_error(assistant_mock: Assistant):
    """Test OpenAI error handling path."""

    mock_client = MagicMock()
    mock_client.responses.create.side_effect = OpenAIError("API failed")

    request = ResponsesAPIRequest(
        assistant_id="123",
        question="What is the capital of Germany?",
    )

    response, error = generate_response(
        tracer=LangfuseTracer(),
        client=mock_client,
        assistant=assistant_mock,
        request=request,
        ancestor_id=None,
    )

    assert response is None
    assert error is not None
    assert "API failed" in error


def _file_search_call(hits: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(type="file_search_call", results=hits)


class TestGetFileSearchResults:
    """`get_file_search_results` flattens file_search hits and carries an optional filename."""

    def test_captures_filename_when_present_and_none_when_absent(self) -> None:
        # Plain stubs (not MagicMock) so getattr(hit, "filename", None) genuinely
        # returns None for the second hit instead of a truthy child mock.
        with_name = SimpleNamespace(score=0.9, text="chunk A", filename="doc.pdf")
        without_name = SimpleNamespace(score=0.4, text="chunk B")
        response = SimpleNamespace(
            output=[_file_search_call([with_name, without_name])]
        )

        chunks = get_file_search_results(response)

        assert [(c.score, c.text, c.filename) for c in chunks] == [
            (0.9, "chunk A", "doc.pdf"),
            (0.4, "chunk B", None),
        ]

    def test_ignores_non_file_search_output_items(self) -> None:
        message = SimpleNamespace(type="message")
        hit = SimpleNamespace(score=0.7, text="chunk", filename="f.pdf")
        response = SimpleNamespace(output=[message, _file_search_call([hit])])

        chunks = get_file_search_results(response)

        assert len(chunks) == 1
        assert chunks[0].filename == "f.pdf"
