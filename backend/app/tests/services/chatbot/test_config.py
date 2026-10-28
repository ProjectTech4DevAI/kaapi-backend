from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.crud import CollectionCrud
from app.crud.config import ConfigVersionCrud
from app.models import Config, Project, ProviderType
from app.services.chatbot.config import (
    prepare_rag_search_client,
    resolve_chatbot_model_settings,
)
from app.services.chatbot.exceptions import ChatbotInputError
from app.services.chatbot.utils import ChatbotModelSettings
from app.tests.utils.collection import get_vector_store_collection
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
    create_test_credential,
    create_test_project,
    create_test_version,
)


def _resolve(
    db: Session,
    config: Config,
    *,
    config_version: int | None = None,
    max_tokens: int | None = None,
    history_token_limit: int | None = None,
) -> ChatbotModelSettings:
    return resolve_chatbot_model_settings(
        session=db,
        project_id=config.project_id,
        config_id=config.id,
        config_version=config_version,
        max_tokens=max_tokens,
        history_token_limit=history_token_limit,
    )


def _overwrite_latest_blob(db: Session, config: Config, blob: dict[str, Any]) -> None:
    # Bypasses ConfigVersionCrud validation to simulate a legacy/corrupt stored row.
    version = ConfigVersionCrud(
        session=db, project_id=config.project_id, config_id=config.id
    ).read_latest()
    assert version is not None
    version.config_blob = blob
    db.add(version)
    db.commit()


class TestResolveChatbotModelSettings:
    def test_latest_version_with_server_defaults(self, db: Session) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(model="claude-sonnet-4-6")
        )
        create_test_version(
            db,
            config_id=config.id,
            project_id=config.project_id,
            config_blob=build_anthropic_text_config_blob(
                model="claude-opus-5-5",
                effort="high",
                thinking={"type": "adaptive", "display": "summarized"},
            ),
        )

        model_settings = _resolve(db, config)

        assert model_settings == ChatbotModelSettings(
            model="claude-opus-5-5",
            max_tokens=1024,
            reasoning_effort="high",
            thinking={"type": "adaptive", "display": "summarized"},
            history_token_limit=4000,
        )

    def test_rag_params_are_carried_into_settings(self, db: Session) -> None:
        config = create_test_config(
            db,
            config_blob=build_anthropic_text_config_blob(
                instructions="Answer in Hindi.",
                knowledge_base_ids=["vs_alpha", "vs_beta"],
                max_num_results=8,
            ),
        )

        model_settings = _resolve(db, config)

        assert model_settings.instructions == "Answer in Hindi."
        assert model_settings.knowledge_base_ids == ("vs_alpha", "vs_beta")
        assert model_settings.max_num_results == 8

    def test_absent_rag_params_leave_empty_defaults(self, db: Session) -> None:
        config = create_test_config(db, config_blob=build_anthropic_text_config_blob())

        model_settings = _resolve(db, config)

        assert model_settings.instructions is None
        assert model_settings.knowledge_base_ids == ()
        assert model_settings.max_num_results is None

    def test_enabled_thinking_with_budget_below_max_tokens_is_accepted(
        self, db: Session
    ) -> None:
        thinking = {"type": "enabled", "budget_tokens": 2048}
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(thinking=thinking)
        )

        model_settings = _resolve(db, config, max_tokens=4096)

        assert model_settings.thinking == thinking
        assert model_settings.max_tokens == 4096

    @pytest.mark.parametrize(
        "thinking",
        [
            {"type": "enabled", "budget_tokens": 2048},
            {"type": "enabled", "budget_tokens": 1024},
            {"type": "enabled"},
        ],
        ids=["budget-above-max", "budget-equal-max", "budget-missing"],
    )
    def test_enabled_thinking_budget_not_below_max_tokens_rejected(
        self, db: Session, thinking: dict[str, Any]
    ) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(thinking=thinking)
        )

        with pytest.raises(ChatbotInputError, match="budget_tokens"):
            _resolve(db, config)

    def test_explicit_version_and_request_overrides(self, db: Session) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(model="claude-sonnet-4-6")
        )
        create_test_version(
            db,
            config_id=config.id,
            project_id=config.project_id,
            config_blob=build_anthropic_text_config_blob(model="claude-opus-5-5"),
        )

        model_settings = _resolve(
            db, config, config_version=1, max_tokens=256, history_token_limit=900
        )

        assert model_settings.model == "claude-sonnet-4-6"
        assert model_settings.max_tokens == 256
        assert model_settings.history_token_limit == 900
        assert model_settings.reasoning_effort is None
        assert model_settings.thinking == {"type": "adaptive"}

    def test_unknown_version_raises_404(self, db: Session) -> None:
        config = create_test_config(db, config_blob=build_anthropic_text_config_blob())

        with pytest.raises(HTTPException) as exc_info:
            _resolve(db, config, config_version=99)

        assert exc_info.value.status_code == 404

    def test_config_from_other_project_raises_404(self, db: Session) -> None:
        config = create_test_config(db, config_blob=build_anthropic_text_config_blob())
        other_project = create_test_project(db)

        with pytest.raises(HTTPException) as exc_info:
            resolve_chatbot_model_settings(
                session=db,
                project_id=other_project.id,
                config_id=config.id,
                config_version=None,
                max_tokens=None,
                history_token_limit=None,
            )

        assert exc_info.value.status_code == 404

    def test_unknown_config_raises_404(self, db: Session) -> None:
        project = create_test_project(db)

        with pytest.raises(HTTPException) as exc_info:
            resolve_chatbot_model_settings(
                session=db,
                project_id=project.id,
                config_id=uuid4(),
                config_version=None,
                max_tokens=None,
                history_token_limit=None,
            )

        assert exc_info.value.status_code == 404

    def test_no_live_versions_raises_404(self, db: Session) -> None:
        config = create_test_config(db, config_blob=build_anthropic_text_config_blob())
        ConfigVersionCrud(
            session=db, project_id=config.project_id, config_id=config.id
        ).delete_or_raise(1)

        with pytest.raises(HTTPException) as exc_info:
            _resolve(db, config)

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == f"No version found for config '{config.id}'"

    def test_non_anthropic_provider_rejected(self, db: Session) -> None:
        config = create_test_config(db, use_kaapi_schema=True)

        with pytest.raises(ChatbotInputError, match="only supports configs"):
            _resolve(db, config)

    def test_native_completion_config_rejected(self, db: Session) -> None:
        config = create_test_config(db)

        with pytest.raises(ChatbotInputError, match="only supports configs"):
            _resolve(db, config)

    def test_invalid_stored_blob_rejected(self, db: Session) -> None:
        config = create_test_config(db, config_blob=build_anthropic_text_config_blob())
        _overwrite_latest_blob(db, config, {"completion": {"provider": "anthropic"}})

        with pytest.raises(ChatbotInputError, match="blob is invalid"):
            _resolve(db, config)

    def test_missing_model_rejected(self, db: Session) -> None:
        config = create_test_config(db, config_blob=build_anthropic_text_config_blob())
        _overwrite_latest_blob(
            db,
            config,
            {"completion": {"provider": "anthropic", "type": "text", "params": {}}},
        )

        with pytest.raises(ChatbotInputError, match="missing completion.params.model"):
            _resolve(db, config)

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_non_anthropic_effort_rejected(self, db: Session, effort: str) -> None:
        config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob(effort=effort)
        )

        with pytest.raises(ChatbotInputError, match=f"'{effort}' is not an Anthropic"):
            _resolve(db, config)


def _prepare(project: Project, *knowledge_base_ids: str, db: Session):
    return prepare_rag_search_client(
        session=db,
        organization_id=project.organization_id,
        project_id=project.id,
        model_settings=ChatbotModelSettings(
            model="claude-sonnet-4-6", knowledge_base_ids=knowledge_base_ids
        ),
    )


@pytest.fixture
def project_with_openai_creds(db: Session) -> Project:
    _credentials, project = create_test_credential(db)
    return project


class TestPrepareRagSearchClient:
    def test_known_collections_and_credentials_return_project_client(
        self, db: Session, project_with_openai_creds: Project
    ) -> None:
        first = get_vector_store_collection(db, project_with_openai_creds)
        second = get_vector_store_collection(db, project_with_openai_creds)

        client = _prepare(
            project_with_openai_creds,
            first.knowledge_base_id,
            second.knowledge_base_id,
            db=db,
        )

        assert client.api_key.startswith("sk-")

    def test_empty_knowledge_base_ids_rejected(
        self, db: Session, project_with_openai_creds: Project
    ) -> None:
        with pytest.raises(ChatbotInputError, match="non-empty"):
            _prepare(project_with_openai_creds, db=db)

    def test_unknown_id_is_named_and_known_id_is_not(
        self, db: Session, project_with_openai_creds: Project
    ) -> None:
        known = get_vector_store_collection(db, project_with_openai_creds)

        with pytest.raises(ChatbotInputError) as exc_info:
            _prepare(
                project_with_openai_creds,
                known.knowledge_base_id,
                "vs_missing",
                db=db,
            )

        assert str(exc_info.value).endswith(
            "collections: vs_missing; use the knowledge_base_id of an existing collection."
        )
        assert known.knowledge_base_id not in str(exc_info.value)

    def test_deleted_collection_rejected(
        self, db: Session, project_with_openai_creds: Project
    ) -> None:
        collection = get_vector_store_collection(db, project_with_openai_creds)
        CollectionCrud(db, project_with_openai_creds.id).delete_by_id(collection.id)

        with pytest.raises(ChatbotInputError, match=collection.knowledge_base_id):
            _prepare(project_with_openai_creds, collection.knowledge_base_id, db=db)

    def test_non_openai_collection_rejected(
        self, db: Session, project_with_openai_creds: Project
    ) -> None:
        collection = get_vector_store_collection(
            db, project_with_openai_creds, provider=ProviderType.google_aistudio
        )

        with pytest.raises(ChatbotInputError, match=collection.knowledge_base_id):
            _prepare(project_with_openai_creds, collection.knowledge_base_id, db=db)

    def test_other_projects_collection_rejected(
        self, db: Session, project_with_openai_creds: Project
    ) -> None:
        foreign = get_vector_store_collection(db, create_test_project(db))

        with pytest.raises(ChatbotInputError, match=foreign.knowledge_base_id):
            _prepare(project_with_openai_creds, foreign.knowledge_base_id, db=db)

    def test_missing_openai_credentials_rejected(self, db: Session) -> None:
        project = create_test_project(db)
        collection = get_vector_store_collection(db, project)

        with pytest.raises(ChatbotInputError, match="OpenAI credentials"):
            _prepare(project, collection.knowledge_base_id, db=db)
