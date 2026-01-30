import time
import secrets
import string
from typing import Optional
from types import SimpleNamespace
from unittest.mock import MagicMock

from openai.types.beta import Assistant as OpenAIAssistant
from openai.types.beta.assistant import ToolResources, ToolResourcesFileSearch
from openai.types.beta.assistant_tool import FileSearchTool
from openai.types.beta.file_search_tool import FileSearch


def generate_openai_id(prefix: str, length: int = 40) -> str:
    """Generate a realistic ID similar to OpenAI's format (alphanumeric only)"""
    # Generate random alphanumeric string
    chars = string.ascii_lowercase + string.digits
    random_part = "".join(secrets.choice(chars) for _ in range(length))
    return f"{prefix}{random_part}"


def mock_openai_assistant(
    assistant_id: str = "assistant_mock",
    vector_store_ids: Optional[list[str]] = ["vs_1", "vs_2"],
    max_num_results: int = 30,
) -> OpenAIAssistant:
    return OpenAIAssistant(
        id=assistant_id,
        created_at=int(time.time()),
        description="Mock description",
        instructions="Mock instructions",
        metadata={},
        model="gpt-4o",
        name="Mock Assistant",
        object="assistant",
        tools=[
            FileSearchTool(
                type="file_search",
                file_search=FileSearch(
                    max_num_results=max_num_results,
                ),
            )
        ],
        temperature=1.0,
        tool_resources=ToolResources(
            code_interpreter=None,
            file_search=ToolResourcesFileSearch(vector_store_ids=vector_store_ids),
        ),
        top_p=1.0,
        reasoning_effort=None,
    )


def mock_openai_response(
    text: str = "Hello world",
    previous_response_id: str | None = None,
    model: str = "gpt-4",
    conversation_id: str | None = None,
) -> SimpleNamespace:
    """Return a minimal mock OpenAI-like response object for testing.

    Args:
        text: The response text
        previous_response_id: Optional previous response ID
        model: Model name
        conversation_id: Optional conversation ID. If provided, adds conversation object to response.
    """

    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
    )

    output_item = SimpleNamespace(
        id=generate_openai_id("out_"),
        type="message",
        role="assistant",
        content=[{"type": "output_text", "text": text}],
    )

    conversation = None
    if conversation_id:
        conversation = SimpleNamespace(id=conversation_id)

    response = SimpleNamespace(
        id=generate_openai_id("resp_"),
        created_at=int(time.time()),
        model=model,
        object="response",
        output=[output_item],
        output_text=text,
        usage=usage,
        previous_response_id=previous_response_id,
        conversation=conversation,
        model_dump=lambda: {
            "id": response.id,
            "model": model,
            "output_text": text,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            "conversation": {"id": conversation_id} if conversation_id else None,
        },
    )
    return response


def get_mock_openai_client_with_vector_store() -> MagicMock:
    mock_client = MagicMock()

    mock_vector_store = MagicMock()
    mock_vector_store.id = "mock_vector_store_id"
    mock_client.vector_stores.create.return_value = mock_vector_store

    mock_file_batch = MagicMock()
    mock_file_batch.file_counts.completed = 2
    mock_file_batch.file_counts.total = 2
    mock_client.vector_stores.file_batches.upload_and_poll.return_value = (
        mock_file_batch
    )

    mock_client.vector_stores.files.list.return_value = {"data": []}

    mock_assistant = MagicMock()
    mock_assistant.id = "mock_assistant_id"
    mock_assistant.name = "Mock Assistant"
    mock_assistant.model = "gpt-4o"
    mock_assistant.instructions = "Mock instructions"
    mock_client.beta.assistants.create.return_value = mock_assistant

    return mock_client


def create_mock_batch(
    batch_id: str = "batch-xyz789",
    status: str = "completed",
    output_file_id: str | None = "output-file-123",
    error_file_id: str | None = None,
    total: int = 100,
    completed: int = 100,
    failed: int = 0,
) -> MagicMock:
    """
    Create a mock OpenAI batch object with configurable properties.

    Args:
        batch_id: The batch ID
        status: Batch status (completed, in_progress, failed, expired, cancelled, etc.)
        output_file_id: Output file ID (None for incomplete batches)
        error_file_id: Error file ID (None if no errors)
        total: Total number of requests in the batch
        completed: Number of completed requests
        failed: Number of failed requests

    Returns:
        MagicMock configured to represent an OpenAI batch object
    """
    mock_batch = MagicMock()
    mock_batch.id = batch_id
    mock_batch.status = status
    mock_batch.output_file_id = output_file_id
    mock_batch.error_file_id = error_file_id

    # Create request_counts mock
    mock_batch.request_counts = MagicMock()
    mock_batch.request_counts.total = total
    mock_batch.request_counts.completed = completed
    mock_batch.request_counts.failed = failed

    return mock_batch


def get_mock_provider(
    llm_service_id: str = "mock_service_id",
    llm_service_name: str = "mock_service_name",
):
    """
    Create a properly configured mock provider for tests.

    Returns a mock that mimics BaseProvider with:
    - create() method returning result with llm_service_id and llm_service_name
    - cleanup() method for cleanup on failure
    - delete() method for deletion
    """
    mock_provider = MagicMock()

    mock_result = MagicMock()
    mock_result.llm_service_id = llm_service_id
    mock_result.llm_service_name = llm_service_name

    mock_provider.create.return_value = mock_result
    mock_provider.delete = MagicMock()

    return mock_provider
