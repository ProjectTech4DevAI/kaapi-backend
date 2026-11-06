import httpx
import openai
import pytest
from langchain_core.messages import ToolMessage
from sqlmodel import Session

from app.services.chatbot.agents.rag.prompts import NO_PASSAGES_FOUND
from app.services.chatbot.agents.rag.retriever import (
    build_search_tool,
    format_passages,
    get_knowledge_base_client,
    search_knowledge_bases,
)
from app.services.chatbot.agents.rag.types import KnowledgeBasePassage
from app.services.chatbot.exceptions import ChatbotInputError, ChatbotProviderError
from app.tests.utils.chatbot import FakeOpenAIClient, FakeVectorStores, search_hit
from app.tests.utils.test_data import create_test_credential, create_test_project

_OPENAI_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/vector_stores")


def _client(**hits: list) -> FakeOpenAIClient:
    return FakeOpenAIClient(vector_stores=FakeVectorStores(hits=hits))


def _hits(prefix: str, count: int) -> list:
    return [
        search_hit(file_id=f"{prefix}-{i}", filename=f"{prefix}-{i}.pdf", score=i / 100)
        for i in range(count)
    ]


class TestSearchKnowledgeBases:
    def test_merges_knowledge_bases_and_keeps_overall_top_by_score(self) -> None:
        client = _client(
            vs_a=[
                search_hit(file_id="a-low", filename="a.pdf", score=0.2),
                search_hit(file_id="a-high", filename="a.pdf", score=0.9),
            ],
            vs_b=[
                search_hit(file_id="b-mid", filename="b.pdf", score=0.5),
                search_hit(file_id="b-high", filename="b.pdf", score=0.7),
            ],
        )

        passages = search_knowledge_bases(
            client=client,
            knowledge_base_ids=("vs_a", "vs_b"),
            query="fees",
            max_num_results=3,
        )

        assert [p.file_id for p in passages] == ["a-high", "b-high", "b-mid"]
        assert client.vector_stores.calls == [
            {"vector_store_id": "vs_a", "query": "fees", "max_num_results": 3},
            {"vector_store_id": "vs_b", "query": "fees", "max_num_results": 3},
        ]

    def test_defaults_to_five_results(self) -> None:
        client = _client(vs_a=_hits("a", 7))

        passages = search_knowledge_bases(
            client=client, knowledge_base_ids=("vs_a",), query="q", max_num_results=None
        )

        assert len(passages) == 5
        assert client.vector_stores.calls[0]["max_num_results"] == 5

    def test_caps_requested_results_at_fifty(self) -> None:
        client = _client(vs_a=_hits("a", 60))

        passages = search_knowledge_bases(
            client=client, knowledge_base_ids=("vs_a",), query="q", max_num_results=80
        )

        assert len(passages) == 50
        assert client.vector_stores.calls[0]["max_num_results"] == 50

    def test_chunks_of_a_hit_are_joined_into_one_passage(self) -> None:
        client = _client(
            vs_a=[
                search_hit(
                    file_id="f1",
                    filename="guide.pdf",
                    score=0.8,
                    texts=("first chunk", "second chunk"),
                )
            ]
        )

        passages = search_knowledge_bases(
            client=client, knowledge_base_ids=("vs_a",), query="q", max_num_results=None
        )

        assert passages == [
            KnowledgeBasePassage(
                file_id="f1",
                filename="guide.pdf",
                score=0.8,
                text="first chunk\nsecond chunk",
            )
        ]

    @pytest.mark.parametrize(
        ("error", "expected_message"),
        [
            (
                openai.APITimeoutError(request=_OPENAI_REQUEST),
                "[KAAPI] Knowledge base search timed out — retry the turn. If "
                "persistent, contact Kaapi. (code: APITimeoutError)",
            ),
            (
                openai.APIConnectionError(request=_OPENAI_REQUEST),
                "[KAAPI] Network/DNS issue reaching OpenAI for knowledge base search — "
                "check connectivity. If persistent, contact Kaapi. "
                "(code: APIConnectionError)",
            ),
            (
                openai.NotFoundError(
                    "No vector store found",
                    response=httpx.Response(
                        404,
                        request=_OPENAI_REQUEST,
                        headers={"x-request-id": "req_123"},
                    ),
                    body=None,
                ),
                "[OPENAI] Knowledge base search failed for 'vs_a': No vector store "
                "found (code: 404, request_id: req_123)",
            ),
            (
                openai.OpenAIError("unexpected payload"),
                "[OPENAI] Knowledge base search failed for 'vs_a': unexpected payload "
                "— retry the turn. If persistent, contact Kaapi. (code: OpenAIError)",
            ),
        ],
        ids=["timeout", "connection", "status", "other"],
    )
    def test_openai_errors_become_provider_errors(
        self, error: openai.OpenAIError, expected_message: str
    ) -> None:
        client = FakeOpenAIClient(vector_stores=FakeVectorStores(error=error))

        with pytest.raises(ChatbotProviderError) as exc_info:
            search_knowledge_bases(
                client=client,
                knowledge_base_ids=("vs_a",),
                query="q",
                max_num_results=None,
            )

        assert str(exc_info.value) == expected_message
        assert exc_info.value.__cause__ is error


class TestFormatPassages:
    def test_no_passages_tells_the_model_nothing_was_found(self) -> None:
        assert format_passages([]) == NO_PASSAGES_FOUND

    def test_passages_are_wrapped_with_escaped_filename(self) -> None:
        passages = [
            KnowledgeBasePassage(
                file_id="f1", filename='fees "2024".pdf', score=0.9, text="Fee is 100."
            ),
            KnowledgeBasePassage(
                file_id="f2", filename="a&b.pdf", score=0.5, text="Due in May."
            ),
        ]

        assert format_passages(passages) == (
            '<source file="fees &quot;2024&quot;.pdf">\nFee is 100.\n</source>'
            "\n\n"
            '<source file="a&amp;b.pdf">\nDue in May.\n</source>'
        )


class TestBuildSearchTool:
    def test_tool_returns_passages_as_content_and_hits_as_artifact(self) -> None:
        client = _client(
            vs_a=[search_hit(file_id="f1", filename="fees.pdf", score=0.9)]
        )
        tool = build_search_tool(
            client=client, knowledge_base_ids=("vs_a",), max_num_results=None
        )

        message = tool.invoke(
            {
                "type": "tool_call",
                "name": "search_knowledge_base",
                "args": {"query": "fees"},
                "id": "toolu_1",
            }
        )

        assert isinstance(message, ToolMessage)
        assert message.content == '<source file="fees.pdf">\ntext\n</source>'
        assert message.artifact == [
            KnowledgeBasePassage(
                file_id="f1", filename="fees.pdf", score=0.9, text="text"
            )
        ]
        assert client.vector_stores.calls[0]["query"] == "fees"

    def test_no_hits_returns_not_found_text_and_empty_artifact(self) -> None:
        tool = build_search_tool(
            client=_client(), knowledge_base_ids=("vs_a",), max_num_results=None
        )

        message = tool.invoke(
            {
                "type": "tool_call",
                "name": "search_knowledge_base",
                "args": {"query": "fees"},
                "id": "toolu_1",
            }
        )

        assert message.content == NO_PASSAGES_FOUND
        assert message.artifact == []


class TestGetKnowledgeBaseClient:
    def test_project_without_openai_credentials_rejected(self, db: Session) -> None:
        project = create_test_project(db)

        with pytest.raises(ChatbotInputError, match="OpenAI credentials"):
            get_knowledge_base_client(
                session=db,
                organization_id=project.organization_id,
                project_id=project.id,
            )

    def test_client_uses_project_api_key(self, db: Session) -> None:
        _credentials, project = create_test_credential(db)

        client = get_knowledge_base_client(
            session=db,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        assert client.api_key.startswith("sk-")
        assert client.max_retries == 1
