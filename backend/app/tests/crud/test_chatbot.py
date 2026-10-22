from uuid import uuid4

import pytest
from sqlmodel import Session

from app.crud.chatbot import (
    create_chatbot_session,
    create_chatbot_session_message,
    get_chatbot_session,
    list_chatbot_session_messages,
    list_chatbot_sessions_by_contact_ref,
)
from app.models import (
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionStatusEnum,
    Config,
    Project,
)
from app.tests.utils.chatbot import (
    add_test_chatbot_messages,
    create_test_chatbot_session,
)
from app.tests.utils.test_data import (
    build_anthropic_text_config_blob,
    create_test_config,
    create_test_project,
)
from app.tests.utils.utils import random_lower_string


@pytest.fixture
def config(db: Session) -> Config:
    return create_test_config(db, config_blob=build_anthropic_text_config_blob())


@pytest.fixture
def project(db: Session, config: Config) -> Project:
    project = db.get(Project, config.project_id)
    assert project is not None
    return project


class TestCreateChatbotSession:
    def test_persists_session_as_active_with_no_collected_fields(
        self, db: Session, config: Config, project: Project
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"

        created = create_chatbot_session(
            db,
            organization_id=project.organization_id,
            project_id=project.id,
            contact_ref=contact_ref,
            config_id=config.id,
            config_version=1,
            system_prompt="Collect the user's name.",
        )

        db.expire_all()
        stored = db.get(ChatbotSession, created.id)
        assert stored is not None
        assert stored.contact_ref == contact_ref
        assert stored.config_id == config.id
        assert stored.config_version == 1
        assert stored.system_prompt == "Collect the user's name."
        assert stored.status == ChatbotSessionStatusEnum.ACTIVE
        assert stored.collected_fields == {}
        assert stored.inserted_at == stored.updated_at


class TestGetChatbotSession:
    def test_returns_session_in_scope(
        self, db: Session, config: Config, project: Project
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)

        found = get_chatbot_session(
            db,
            session_id=chatbot_session.id,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        assert found is not None
        assert found.id == chatbot_session.id

    def test_other_project_cannot_see_session(
        self, db: Session, config: Config, project: Project
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)
        other_project = create_test_project(db)

        found = get_chatbot_session(
            db,
            session_id=chatbot_session.id,
            organization_id=other_project.organization_id,
            project_id=other_project.id,
        )

        assert found is None

    def test_other_organization_cannot_see_session(
        self, db: Session, config: Config, project: Project
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)
        other_project = create_test_project(db)

        found = get_chatbot_session(
            db,
            session_id=chatbot_session.id,
            organization_id=other_project.organization_id,
            project_id=project.id,
        )

        assert found is None

    def test_unknown_id_returns_none(self, db: Session, project: Project) -> None:
        assert (
            get_chatbot_session(
                db,
                session_id=uuid4(),
                organization_id=project.organization_id,
                project_id=project.id,
            )
            is None
        )


class TestChatbotSessionMessages:
    def test_create_message_persists_role_and_content(
        self, db: Session, config: Config
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)

        message = create_chatbot_session_message(
            db,
            session_id=chatbot_session.id,
            role=ChatbotMessageRoleEnum.USER,
            content="Asha",
        )

        stored = list_chatbot_session_messages(db, session_id=chatbot_session.id)
        assert [(m.id, m.role, m.content) for m in stored] == [
            (message.id, ChatbotMessageRoleEnum.USER, "Asha")
        ]

    def test_list_returns_oldest_first_and_only_this_session(
        self, db: Session, config: Config
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)
        other_session = create_test_chatbot_session(db, config=config)
        add_test_chatbot_messages(
            db,
            chatbot_session,
            [
                (ChatbotMessageRoleEnum.ASSISTANT, "Hi! What's your name?"),
                (ChatbotMessageRoleEnum.USER, "Asha"),
                (ChatbotMessageRoleEnum.ASSISTANT, "Which city?"),
            ],
        )
        add_test_chatbot_messages(
            db, other_session, [(ChatbotMessageRoleEnum.USER, "unrelated")]
        )

        stored = list_chatbot_session_messages(db, session_id=chatbot_session.id)

        assert [m.content for m in stored] == [
            "Hi! What's your name?",
            "Asha",
            "Which city?",
        ]

    def test_list_empty_session_returns_empty(
        self, db: Session, config: Config
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)

        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []


class TestListChatbotSessionsByContactRef:
    def test_returns_newest_first_scoped_to_contact_and_project(
        self, db: Session, config: Config, project: Project
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        oldest = create_test_chatbot_session(db, config=config, contact_ref=contact_ref)
        newest = create_test_chatbot_session(db, config=config, contact_ref=contact_ref)
        create_test_chatbot_session(db, config=config)
        other_config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob()
        )
        create_test_chatbot_session(db, config=other_config, contact_ref=contact_ref)

        sessions = list_chatbot_sessions_by_contact_ref(
            db,
            contact_ref=contact_ref,
            organization_id=project.organization_id,
            project_id=project.id,
        )

        assert [s.id for s in sessions] == [newest.id, oldest.id]

    def test_unknown_contact_returns_empty(
        self, db: Session, config: Config, project: Project
    ) -> None:
        create_test_chatbot_session(db, config=config)

        assert (
            list_chatbot_sessions_by_contact_ref(
                db,
                contact_ref=f"contact-{random_lower_string()}",
                organization_id=project.organization_id,
                project_id=project.id,
            )
            == []
        )
