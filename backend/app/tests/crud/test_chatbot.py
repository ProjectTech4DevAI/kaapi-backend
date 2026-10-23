from collections.abc import Iterator
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.db import engine
from app.crud.chatbot import (
    CONCURRENT_SESSION_CONFLICT_MESSAGE,
    ChatbotSessionConflictError,
    _is_active_session_conflict,
    chatbot_ref_exists,
    create_chatbot_session,
    create_chatbot_session_message,
    get_chatbot_session,
    get_latest_chatbot_session,
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.models import (
    ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX,
    ChatbotCompletionStatusEnum,
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
            name="Onboarding Bot",
            contact_ref=contact_ref,
            chatbot_ref="chatbot-onboarding",
            config_id=config.id,
            config_version=1,
            system_prompt="Collect the user's name.",
        )

        db.expire_all()
        stored = db.get(ChatbotSession, created.id)
        assert stored is not None
        assert stored.name == "Onboarding Bot"
        assert stored.contact_ref == contact_ref
        assert stored.config_id == config.id
        assert stored.config_version == 1
        assert stored.system_prompt == "Collect the user's name."
        assert stored.status == ChatbotSessionStatusEnum.ACTIVE
        assert stored.collected_fields == {}
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS
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


class TestGetLatestChatbotSession:
    def test_returns_newest_scoped_to_contact_and_project(
        self, db: Session, config: Config, project: Project
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        chatbot_ref = f"chatbot-{random_lower_string()}"
        create_test_chatbot_session(
            db, config=config, contact_ref=contact_ref, chatbot_ref=chatbot_ref
        )
        newest = create_test_chatbot_session(
            db, config=config, contact_ref=contact_ref, chatbot_ref=chatbot_ref
        )
        create_test_chatbot_session(db, config=config, chatbot_ref=chatbot_ref)
        other_config = create_test_config(
            db, config_blob=build_anthropic_text_config_blob()
        )
        create_test_chatbot_session(
            db, config=other_config, contact_ref=contact_ref, chatbot_ref=chatbot_ref
        )

        latest = get_latest_chatbot_session(
            db,
            organization_id=project.organization_id,
            project_id=project.id,
            contact_ref=contact_ref,
            chatbot_ref=chatbot_ref,
        )

        assert latest is not None
        assert latest.id == newest.id

    def test_unknown_contact_returns_none(
        self, db: Session, config: Config, project: Project
    ) -> None:
        create_test_chatbot_session(db, config=config)

        assert (
            get_latest_chatbot_session(
                db,
                organization_id=project.organization_id,
                project_id=project.id,
                contact_ref=f"contact-{random_lower_string()}",
                chatbot_ref=f"chatbot-{random_lower_string()}",
            )
            is None
        )

    def test_without_chatbot_ref_returns_newest_across_chatbots(
        self, db: Session, config: Config, project: Project
    ) -> None:
        older = create_test_chatbot_session(db, config=config)
        newest = create_test_chatbot_session(
            db, config=config, contact_ref=older.contact_ref
        )
        create_test_chatbot_session(db, config=config)

        latest = get_latest_chatbot_session(
            db,
            organization_id=project.organization_id,
            project_id=project.id,
            contact_ref=older.contact_ref,
        )

        assert latest is not None
        assert latest.id == newest.id
        assert latest.chatbot_ref != older.chatbot_ref

    def test_ignores_newer_session_of_another_chatbot(
        self, db: Session, config: Config, project: Project
    ) -> None:
        wanted = create_test_chatbot_session(db, config=config)
        create_test_chatbot_session(db, config=config, contact_ref=wanted.contact_ref)

        latest = get_latest_chatbot_session(
            db,
            organization_id=project.organization_id,
            project_id=project.id,
            contact_ref=wanted.contact_ref,
            chatbot_ref=wanted.chatbot_ref,
        )

        assert latest is not None
        assert latest.id == wanted.id

    def test_returns_newest_even_when_inactive(
        self, db: Session, config: Config, project: Project
    ) -> None:
        older = create_test_chatbot_session(db, config=config)
        newest = create_test_chatbot_session(
            db,
            config=config,
            contact_ref=older.contact_ref,
            chatbot_ref=older.chatbot_ref,
        )
        # No API path leaves the newest row inactive; set it directly so a
        # status filter would surface the older row instead.
        newest.status = ChatbotSessionStatusEnum.INACTIVE
        db.add(newest)
        db.commit()

        latest = get_latest_chatbot_session(
            db,
            organization_id=project.organization_id,
            project_id=project.id,
            contact_ref=older.contact_ref,
            chatbot_ref=older.chatbot_ref,
        )

        assert latest is not None
        assert latest.id == newest.id
        assert latest.status == ChatbotSessionStatusEnum.INACTIVE


class TestUpdateChatbotSessionProgress:
    def test_persists_fields_and_status_and_bumps_updated_at(
        self, db: Session, config: Config
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)
        inserted_at = chatbot_session.inserted_at

        returned = update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha", "city": "Pune"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )

        assert returned is not None
        db.expire_all()
        stored = db.get(ChatbotSession, chatbot_session.id)
        assert stored is not None
        assert stored.collected_fields == {"name": "Asha", "city": "Pune"}
        assert stored.completion_status == ChatbotCompletionStatusEnum.COMPLETE
        assert stored.updated_at > inserted_at
        assert stored.inserted_at == inserted_at

    def test_replaces_rather_than_merges_fields(
        self, db: Session, config: Config
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)
        update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha", "city": "Pune"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )

        update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha"},
            completion_status=ChatbotCompletionStatusEnum.IN_PROGRESS,
        )

        db.expire_all()
        stored = db.get(ChatbotSession, chatbot_session.id)
        assert stored is not None
        assert stored.collected_fields == {"name": "Asha"}
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS

    def test_unknown_id_returns_none(self, db: Session) -> None:
        assert (
            update_chatbot_session_progress(
                db,
                session_id=uuid4(),
                collected_fields={"name": "Asha"},
                completion_status=ChatbotCompletionStatusEnum.COMPLETE,
            )
            is None
        )


def _reload(db: Session, chatbot_session: ChatbotSession) -> ChatbotSession:
    db.expire_all()
    stored = db.get(ChatbotSession, chatbot_session.id)
    assert stored is not None
    return stored


class TestCreateChatbotSessionSupersedes:
    def test_in_progress_session_becomes_inactive_and_abandoned(
        self, db: Session, config: Config
    ) -> None:
        older = create_test_chatbot_session(db, config=config)
        older_updated_at = older.updated_at

        newer = create_test_chatbot_session(
            db,
            config=config,
            contact_ref=older.contact_ref,
            chatbot_ref=older.chatbot_ref,
        )

        superseded = _reload(db, older)
        assert superseded.status == ChatbotSessionStatusEnum.INACTIVE
        assert superseded.completion_status == ChatbotCompletionStatusEnum.ABANDONED
        assert superseded.updated_at > older_updated_at
        assert _reload(db, newer).status == ChatbotSessionStatusEnum.ACTIVE

    def test_complete_session_becomes_inactive_but_stays_complete(
        self, db: Session, config: Config
    ) -> None:
        older = create_test_chatbot_session(db, config=config)
        update_chatbot_session_progress(
            db,
            session_id=older.id,
            collected_fields={"name": "Asha"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )
        completed_at = _reload(db, older).updated_at

        create_test_chatbot_session(
            db,
            config=config,
            contact_ref=older.contact_ref,
            chatbot_ref=older.chatbot_ref,
        )

        superseded = _reload(db, older)
        assert superseded.status == ChatbotSessionStatusEnum.INACTIVE
        assert superseded.completion_status == ChatbotCompletionStatusEnum.COMPLETE
        assert superseded.collected_fields == {"name": "Asha"}
        assert superseded.updated_at > completed_at

    def test_other_chatbot_contact_and_project_sessions_stay_active(
        self, db: Session, config: Config, project: Project
    ) -> None:
        contact_ref = f"contact-{random_lower_string()}"
        chatbot_ref = f"chatbot-{random_lower_string()}"
        other_chatbot = create_test_chatbot_session(
            db, config=config, contact_ref=contact_ref
        )
        other_contact = create_test_chatbot_session(
            db, config=config, chatbot_ref=chatbot_ref
        )
        # Same organization, so only the project_id filter keeps it out of scope.
        sibling_project = create_test_project(
            db, organization_id=project.organization_id
        )
        other_project_config = create_test_config(
            db,
            project_id=sibling_project.id,
            config_blob=build_anthropic_text_config_blob(),
        )
        other_project = create_test_chatbot_session(
            db,
            config=other_project_config,
            contact_ref=contact_ref,
            chatbot_ref=chatbot_ref,
        )
        untouched = [other_chatbot, other_contact, other_project]
        updated_ats = [s.updated_at for s in untouched]

        create_test_chatbot_session(
            db, config=config, contact_ref=contact_ref, chatbot_ref=chatbot_ref
        )

        for chatbot_session, updated_at in zip(untouched, updated_ats, strict=True):
            stored = _reload(db, chatbot_session)
            assert stored.status == ChatbotSessionStatusEnum.ACTIVE
            assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS
            assert stored.updated_at == updated_at

    def test_already_inactive_session_is_left_unmodified(
        self, db: Session, config: Config
    ) -> None:
        first = create_test_chatbot_session(db, config=config)
        refs = {"contact_ref": first.contact_ref, "chatbot_ref": first.chatbot_ref}
        second = create_test_chatbot_session(db, config=config, **refs)
        first_after_supersede = _reload(db, first).updated_at

        third = create_test_chatbot_session(db, config=config, **refs)

        stored_first = _reload(db, first)
        assert stored_first.status == ChatbotSessionStatusEnum.INACTIVE
        assert stored_first.completion_status == ChatbotCompletionStatusEnum.ABANDONED
        assert stored_first.updated_at == first_after_supersede
        assert _reload(db, second).status == ChatbotSessionStatusEnum.INACTIVE
        assert _reload(db, third).status == ChatbotSessionStatusEnum.ACTIVE


@pytest.fixture
def savepoint_db() -> Iterator[Session]:
    # create_chatbot_session rolls back on IntegrityError. Under the shared `db`
    # fixture that rollback discards the whole test transaction (rows created
    # earlier included), so atomicity is unobservable. Joining via savepoints makes
    # its commit/rollback behave as in production while still undoing everything here.
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


class TestCreateChatbotSessionFailures:
    def test_fk_failure_propagates_and_keeps_prior_session_active(
        self, savepoint_db: Session
    ) -> None:
        config = create_test_config(
            savepoint_db, config_blob=build_anthropic_text_config_blob()
        )
        prior = create_test_chatbot_session(savepoint_db, config=config)

        with pytest.raises(IntegrityError):
            create_chatbot_session(
                savepoint_db,
                organization_id=prior.organization_id,
                project_id=prior.project_id,
                name="Onboarding Bot",
                contact_ref=prior.contact_ref,
                chatbot_ref=prior.chatbot_ref,
                config_id=uuid4(),
                config_version=None,
                system_prompt="Collect the user's name.",
            )

        stored = _reload(savepoint_db, prior)
        assert stored.status == ChatbotSessionStatusEnum.ACTIVE
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS

    def test_active_index_violation_raises_conflict_and_inserts_nothing(
        self, savepoint_db: Session
    ) -> None:
        config = create_test_config(
            savepoint_db, config_blob=build_anthropic_text_config_blob()
        )
        project = savepoint_db.get(Project, config.project_id)
        assert project is not None
        unrelated_org_id = create_test_project(savepoint_db).organization_id
        contact_ref = f"contact-{random_lower_string()}"
        chatbot_ref = f"chatbot-{random_lower_string()}"
        # Stands in for a racing request's committed row: the unique index ignores
        # organization_id but the supersede UPDATE filters on it, so this row
        # survives the UPDATE exactly like one committed after it ran.
        racing = ChatbotSession(
            organization_id=unrelated_org_id,
            project_id=project.id,
            contact_ref=contact_ref,
            chatbot_ref=chatbot_ref,
            config_id=config.id,
            system_prompt="racing",
        )
        savepoint_db.add(racing)
        savepoint_db.commit()

        with pytest.raises(ChatbotSessionConflictError) as exc_info:
            create_chatbot_session(
                savepoint_db,
                organization_id=project.organization_id,
                project_id=project.id,
                name="Onboarding Bot",
                contact_ref=contact_ref,
                chatbot_ref=chatbot_ref,
                config_id=config.id,
                config_version=None,
                system_prompt="Collect the user's name.",
            )

        assert str(exc_info.value) == CONCURRENT_SESSION_CONFLICT_MESSAGE
        savepoint_db.expire_all()
        stored = savepoint_db.exec(
            select(ChatbotSession).where(
                ChatbotSession.project_id == project.id,
                ChatbotSession.chatbot_ref == chatbot_ref,
            )
        ).all()
        assert [s.id for s in stored] == [racing.id]
        assert stored[0].status == ChatbotSessionStatusEnum.ACTIVE


class TestIsActiveSessionConflict:
    # psycopg always exposes diag.constraint_name; the message fallback guards
    # drivers/wrappers that don't, so it's exercised with stand-in errors.
    @pytest.mark.parametrize(
        ("orig", "expected"),
        [
            (
                SimpleNamespace(
                    diag=SimpleNamespace(
                        constraint_name=ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX
                    )
                ),
                True,
            ),
            (
                SimpleNamespace(
                    diag=SimpleNamespace(
                        constraint_name="chatbot_session_config_id_fkey"
                    )
                ),
                False,
            ),
            (
                Exception(
                    f'duplicate key value violates unique constraint "{ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX}"'
                ),
                True,
            ),
            (Exception("violates foreign key constraint"), False),
        ],
        ids=["diag-match", "diag-other", "message-match", "message-other"],
    )
    def test_detects_only_the_active_session_index(
        self, orig: object, expected: bool
    ) -> None:
        error = IntegrityError("INSERT ...", {}, orig)  # type: ignore[arg-type]

        assert _is_active_session_conflict(error) is expected


class TestChatbotRefExists:
    def test_true_for_active_and_inactive_sessions_in_project(
        self, db: Session, config: Config, project: Project
    ) -> None:
        first = create_test_chatbot_session(db, config=config)
        create_test_chatbot_session(
            db,
            config=config,
            contact_ref=first.contact_ref,
            chatbot_ref=first.chatbot_ref,
        )
        inactive_only = create_test_chatbot_session(db, config=config)
        create_test_chatbot_session(
            db,
            config=config,
            contact_ref=inactive_only.contact_ref,
            chatbot_ref=inactive_only.chatbot_ref,
        )

        assert chatbot_ref_exists(
            db, project_id=project.id, chatbot_ref=first.chatbot_ref
        )
        assert chatbot_ref_exists(
            db, project_id=project.id, chatbot_ref=inactive_only.chatbot_ref
        )

    def test_false_for_ref_used_only_in_another_project(
        self, db: Session, config: Config
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, config=config)
        other_project = create_test_project(db)

        assert not chatbot_ref_exists(
            db, project_id=other_project.id, chatbot_ref=chatbot_session.chatbot_ref
        )
