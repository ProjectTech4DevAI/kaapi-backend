from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import event, insert
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.core.db import engine
from app.crud.chatbot import (
    create_chatbot,
    create_chatbot_flow_version,
    create_chatbot_session,
    create_chatbot_session_message,
    delete_chatbot,
    get_chatbot,
    get_chatbot_session,
    get_latest_chatbot_flow_version,
    get_latest_chatbot_session,
    get_live_chatbot_flow_version,
    list_chatbot_session_messages,
    list_chatbots,
    publish_chatbot_flow_version,
    update_chatbot,
    update_chatbot_session_progress,
)
from app.models import (
    Chatbot,
    ChatbotCompletionStatusEnum,
    ChatbotFlowVersion,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionStatusEnum,
    ChatbotUpdate,
    Config,
    Project,
)
from app.services.chatbot.exceptions import (
    CONCURRENT_FLOW_CONFLICT_MESSAGE,
    CONCURRENT_SESSION_CONFLICT_MESSAGE,
    ChatbotFlowConflictError,
    ChatbotSessionConflictError,
)
from app.tests.utils.chatbot import (
    ONBOARDING_NODE_ID,
    add_test_chatbot_messages,
    build_onboarding_node,
    create_test_chatbot,
    create_test_chatbot_session,
    create_test_flow_version,
    create_test_anthropic_config,
    create_test_live_chatbot,
    list_stored_flow_versions,
    list_stored_sessions_for_contact,
    random_contact_ref,
)
from app.tests.utils.test_data import create_test_project


@pytest.fixture
def config(db: Session) -> Config:
    return create_test_anthropic_config(db)


@pytest.fixture
def project(db: Session, config: Config) -> Project:
    project = db.get(Project, config.project_id)
    assert project is not None
    return project


@pytest.fixture
def live_chatbot_and_flow(
    db: Session, config: Config
) -> tuple[Chatbot, ChatbotFlowVersion]:
    return create_test_live_chatbot(db, config=config)


@pytest.fixture
def chatbot(live_chatbot_and_flow: tuple[Chatbot, ChatbotFlowVersion]) -> Chatbot:
    return live_chatbot_and_flow[0]


def _reload(db: Session, chatbot_session: ChatbotSession) -> ChatbotSession:
    db.expire_all()
    stored = db.get(ChatbotSession, chatbot_session.id)
    assert stored is not None
    return stored


def _scope(project: Project) -> dict[str, int]:
    return {"organization_id": project.organization_id, "project_id": project.id}


class TestChatbotCrud:
    def test_create_persists_name_and_scope(
        self, db: Session, project: Project
    ) -> None:
        created = create_chatbot(db, **_scope(project), name="Onboarding Bot")

        db.expire_all()
        stored = db.get(Chatbot, created.id)
        assert stored is not None
        assert stored.name == "Onboarding Bot"
        assert stored.project_id == project.id
        assert stored.organization_id == project.organization_id
        assert stored.deleted_at is None

    def test_get_is_scoped_to_project(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        other_project = create_test_project(db, organization_id=project.organization_id)

        assert get_chatbot(db, chatbot_id=chatbot.id, **_scope(project)) is not None
        assert get_chatbot(db, chatbot_id=chatbot.id, **_scope(other_project)) is None

    def test_get_hides_deleted_chatbot(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        delete_chatbot(db, chatbot=chatbot)

        assert get_chatbot(db, chatbot_id=chatbot.id, **_scope(project)) is None

    def test_list_returns_projects_live_chatbots_most_recently_updated_first(
        self, db: Session, project: Project
    ) -> None:
        older = create_test_chatbot(db, project=project)
        newer = create_test_chatbot(db, project=project)
        deleted = create_test_chatbot(db, project=project)
        delete_chatbot(db, chatbot=deleted)
        create_test_chatbot(db)
        update_chatbot(db, chatbot=older, chatbot_update=ChatbotUpdate(name="renamed"))

        listed = list_chatbots(db, **_scope(project), skip=0, limit=100)

        assert [c.id for c in listed] == [older.id, newer.id]

    def test_list_paginates(self, db: Session, project: Project) -> None:
        create_test_chatbot(db, project=project)
        second = create_test_chatbot(db, project=project)
        create_test_chatbot(db, project=project)

        listed = list_chatbots(db, **_scope(project), skip=1, limit=1)

        assert [c.id for c in listed] == [second.id]

    def test_update_changes_name_and_bumps_updated_at(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        before = chatbot.updated_at

        update_chatbot(db, chatbot=chatbot, chatbot_update=ChatbotUpdate(name="New"))

        db.expire_all()
        stored = db.get(Chatbot, chatbot.id)
        assert stored is not None
        assert stored.name == "New"
        assert stored.updated_at > before

    def test_update_with_no_name_keeps_name(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        original_name = chatbot.name

        update_chatbot(db, chatbot=chatbot, chatbot_update=ChatbotUpdate())

        db.expire_all()
        stored = db.get(Chatbot, chatbot.id)
        assert stored is not None
        assert stored.name == original_name

    def test_delete_soft_deletes_and_deactivates_only_its_active_sessions(
        self, db: Session, config: Config, chatbot: Chatbot
    ) -> None:
        in_progress = create_test_chatbot_session(db, chatbot=chatbot)
        complete = create_test_chatbot_session(db, chatbot=chatbot)
        update_chatbot_session_progress(
            db,
            session_id=complete.id,
            collected_fields={"name": "Asha"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )
        other_chatbot, _ = create_test_live_chatbot(db, config=config)
        untouched = create_test_chatbot_session(
            db, chatbot=other_chatbot, contact_ref=in_progress.contact_ref
        )

        delete_chatbot(db, chatbot=chatbot)

        db.expire_all()
        stored_chatbot = db.get(Chatbot, chatbot.id)
        assert stored_chatbot is not None
        assert stored_chatbot.deleted_at is not None
        stored_in_progress = _reload(db, in_progress)
        assert stored_in_progress.status == ChatbotSessionStatusEnum.INACTIVE
        assert (
            stored_in_progress.completion_status
            == ChatbotCompletionStatusEnum.ABANDONED
        )
        stored_complete = _reload(db, complete)
        assert stored_complete.status == ChatbotSessionStatusEnum.INACTIVE
        assert stored_complete.completion_status == ChatbotCompletionStatusEnum.COMPLETE
        assert _reload(db, untouched).status == ChatbotSessionStatusEnum.ACTIVE


class TestChatbotFlowVersionCrud:
    def test_latest_is_highest_version_and_live_is_the_published_one(
        self,
        db: Session,
        config: Config,
        live_chatbot_and_flow: tuple[Chatbot, ChatbotFlowVersion],
    ) -> None:
        chatbot, live = live_chatbot_and_flow
        draft = create_test_flow_version(
            db, chatbot=chatbot, nodes=[build_onboarding_node(config_id=config.id)]
        )

        latest = get_latest_chatbot_flow_version(db, chatbot_id=chatbot.id)
        current_live = get_live_chatbot_flow_version(db, chatbot_id=chatbot.id)

        assert latest is not None and latest.id == draft.id
        assert latest.version == 2
        assert current_live is not None and current_live.id == live.id

    def test_no_versions_returns_none(self, db: Session, project: Project) -> None:
        chatbot = create_test_chatbot(db, project=project)

        assert get_latest_chatbot_flow_version(db, chatbot_id=chatbot.id) is None
        assert get_live_chatbot_flow_version(db, chatbot_id=chatbot.id) is None

    def test_publish_unsets_previous_live_and_stores_given_nodes(
        self,
        db: Session,
        config: Config,
        live_chatbot_and_flow: tuple[Chatbot, ChatbotFlowVersion],
    ) -> None:
        chatbot, previous_live = live_chatbot_and_flow
        draft = create_test_flow_version(
            db,
            chatbot=chatbot,
            nodes=[build_onboarding_node(config_id=config.id, config_version=None)],
        )
        pinned_nodes = [build_onboarding_node(config_id=config.id, config_version=1)]

        publish_chatbot_flow_version(db, flow_version=draft, nodes=pinned_nodes)

        db.expire_all()
        stored_previous = db.get(ChatbotFlowVersion, previous_live.id)
        stored_draft = db.get(ChatbotFlowVersion, draft.id)
        assert stored_previous is not None and stored_previous.is_live is False
        assert stored_draft is not None and stored_draft.is_live is True
        assert stored_draft.nodes == pinned_nodes


class TestCreateChatbotSession:
    def test_persists_session_as_active_with_no_collected_fields(
        self, db: Session, live_chatbot_and_flow: tuple[Chatbot, ChatbotFlowVersion]
    ) -> None:
        chatbot, flow_version = live_chatbot_and_flow
        contact_ref = random_contact_ref()

        created = create_chatbot_session(
            db,
            chatbot_id=chatbot.id,
            flow_version_id=flow_version.id,
            contact_ref=contact_ref,
            active_node_id=ONBOARDING_NODE_ID,
        )

        stored = _reload(db, created)
        assert stored.chatbot_id == chatbot.id
        assert stored.flow_version_id == flow_version.id
        assert stored.contact_ref == contact_ref
        assert stored.active_node_id == ONBOARDING_NODE_ID
        assert stored.status == ChatbotSessionStatusEnum.ACTIVE
        assert stored.collected_fields == {}
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS
        assert stored.inserted_at == stored.updated_at


class TestGetChatbotSession:
    def test_returns_session_in_scope(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)

        found = get_chatbot_session(
            db, session_id=chatbot_session.id, **_scope(project)
        )

        assert found is not None
        assert found.id == chatbot_session.id

    def test_other_project_cannot_see_session(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        other_project = create_test_project(db, organization_id=project.organization_id)

        found = get_chatbot_session(
            db, session_id=chatbot_session.id, **_scope(other_project)
        )

        assert found is None

    def test_other_organization_cannot_see_session(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        other_project = create_test_project(db)

        found = get_chatbot_session(
            db,
            session_id=chatbot_session.id,
            organization_id=other_project.organization_id,
            project_id=project.id,
        )

        assert found is None

    def test_deleted_chatbots_session_still_resolves(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        delete_chatbot(db, chatbot=chatbot)

        found = get_chatbot_session(
            db, session_id=chatbot_session.id, **_scope(project)
        )

        assert found is not None
        assert found.status == ChatbotSessionStatusEnum.INACTIVE

    def test_unknown_id_returns_none(self, db: Session, project: Project) -> None:
        assert get_chatbot_session(db, session_id=uuid4(), **_scope(project)) is None


class TestChatbotSessionMessages:
    def test_create_message_persists_role_content_and_node(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)

        message = create_chatbot_session_message(
            db,
            session_id=chatbot_session.id,
            role=ChatbotMessageRoleEnum.USER,
            content="Asha",
            node_id=ONBOARDING_NODE_ID,
        )

        stored = list_chatbot_session_messages(db, session_id=chatbot_session.id)
        assert [(m.id, m.role, m.content, m.node_id) for m in stored] == [
            (message.id, ChatbotMessageRoleEnum.USER, "Asha", ONBOARDING_NODE_ID)
        ]

    def test_list_returns_oldest_first_and_only_this_session(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        other_session = create_test_chatbot_session(db, chatbot=chatbot)
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
        self, db: Session, chatbot: Chatbot
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)

        assert list_chatbot_session_messages(db, session_id=chatbot_session.id) == []


class TestGetLatestChatbotSession:
    def test_returns_newest_with_its_chatbot_and_pinned_flow(
        self,
        db: Session,
        config: Config,
        live_chatbot_and_flow: tuple[Chatbot, ChatbotFlowVersion],
        project: Project,
    ) -> None:
        chatbot, flow_version = live_chatbot_and_flow
        contact_ref = random_contact_ref()
        create_test_chatbot_session(db, chatbot=chatbot, contact_ref=contact_ref)
        newest = create_test_chatbot_session(
            db, chatbot=chatbot, contact_ref=contact_ref
        )
        create_test_chatbot_session(db, chatbot=chatbot)
        foreign_chatbot, _ = create_test_live_chatbot(
            db, config=create_test_anthropic_config(db)
        )
        create_test_chatbot_session(
            db, chatbot=foreign_chatbot, contact_ref=contact_ref
        )

        row = get_latest_chatbot_session(
            db, **_scope(project), contact_ref=contact_ref, chatbot_id=chatbot.id
        )

        assert row is not None
        found_session, found_chatbot, found_flow = row
        assert found_session.id == newest.id
        assert found_chatbot.id == chatbot.id
        assert found_flow.id == flow_version.id

    def test_unknown_contact_returns_none(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        create_test_chatbot_session(db, chatbot=chatbot)

        assert (
            get_latest_chatbot_session(
                db,
                **_scope(project),
                contact_ref=random_contact_ref(),
                chatbot_id=chatbot.id,
            )
            is None
        )

    def test_without_chatbot_id_returns_newest_across_chatbots(
        self, db: Session, config: Config, chatbot: Chatbot, project: Project
    ) -> None:
        older = create_test_chatbot_session(db, chatbot=chatbot)
        other_chatbot, _ = create_test_live_chatbot(db, config=config)
        newest = create_test_chatbot_session(
            db, chatbot=other_chatbot, contact_ref=older.contact_ref
        )

        row = get_latest_chatbot_session(
            db, **_scope(project), contact_ref=older.contact_ref
        )

        assert row is not None
        assert row[0].id == newest.id
        assert row[1].id == other_chatbot.id

    def test_ignores_newer_session_of_another_chatbot(
        self, db: Session, config: Config, chatbot: Chatbot, project: Project
    ) -> None:
        wanted = create_test_chatbot_session(db, chatbot=chatbot)
        other_chatbot, _ = create_test_live_chatbot(db, config=config)
        create_test_chatbot_session(
            db, chatbot=other_chatbot, contact_ref=wanted.contact_ref
        )

        row = get_latest_chatbot_session(
            db,
            **_scope(project),
            contact_ref=wanted.contact_ref,
            chatbot_id=chatbot.id,
        )

        assert row is not None
        assert row[0].id == wanted.id

    def test_skips_sessions_of_deleted_chatbots(
        self, db: Session, config: Config, chatbot: Chatbot, project: Project
    ) -> None:
        kept = create_test_chatbot_session(db, chatbot=chatbot)
        deleted_chatbot, _ = create_test_live_chatbot(db, config=config)
        create_test_chatbot_session(
            db, chatbot=deleted_chatbot, contact_ref=kept.contact_ref
        )
        delete_chatbot(db, chatbot=deleted_chatbot)

        row = get_latest_chatbot_session(
            db, **_scope(project), contact_ref=kept.contact_ref
        )
        scoped_row = get_latest_chatbot_session(
            db,
            **_scope(project),
            contact_ref=kept.contact_ref,
            chatbot_id=deleted_chatbot.id,
        )

        assert row is not None
        assert row[0].id == kept.id
        assert scoped_row is None

    def test_returns_newest_even_when_inactive(
        self, db: Session, chatbot: Chatbot, project: Project
    ) -> None:
        older = create_test_chatbot_session(db, chatbot=chatbot)
        newest = create_test_chatbot_session(
            db, chatbot=chatbot, contact_ref=older.contact_ref
        )
        # No API path leaves the newest row inactive; set it directly so a
        # status filter would surface the older row instead.
        newest.status = ChatbotSessionStatusEnum.INACTIVE
        db.add(newest)
        db.commit()

        row = get_latest_chatbot_session(
            db,
            **_scope(project),
            contact_ref=older.contact_ref,
            chatbot_id=chatbot.id,
        )

        assert row is not None
        assert row[0].id == newest.id
        assert row[0].status == ChatbotSessionStatusEnum.INACTIVE


class TestUpdateChatbotSessionProgress:
    def test_persists_fields_and_status_and_bumps_updated_at(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
        inserted_at = chatbot_session.inserted_at

        returned = update_chatbot_session_progress(
            db,
            session_id=chatbot_session.id,
            collected_fields={"name": "Asha", "city": "Pune"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )

        assert returned is not None
        stored = _reload(db, chatbot_session)
        assert stored.collected_fields == {"name": "Asha", "city": "Pune"}
        assert stored.completion_status == ChatbotCompletionStatusEnum.COMPLETE
        assert stored.updated_at > inserted_at
        assert stored.inserted_at == inserted_at

    def test_replaces_rather_than_merges_fields(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        chatbot_session = create_test_chatbot_session(db, chatbot=chatbot)
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

        stored = _reload(db, chatbot_session)
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


class TestCreateChatbotSessionSupersedes:
    def test_in_progress_session_becomes_inactive_and_abandoned(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        older = create_test_chatbot_session(db, chatbot=chatbot)
        older_updated_at = older.updated_at

        newer = create_test_chatbot_session(
            db, chatbot=chatbot, contact_ref=older.contact_ref
        )

        superseded = _reload(db, older)
        assert superseded.status == ChatbotSessionStatusEnum.INACTIVE
        assert superseded.completion_status == ChatbotCompletionStatusEnum.ABANDONED
        assert superseded.updated_at > older_updated_at
        assert _reload(db, newer).status == ChatbotSessionStatusEnum.ACTIVE

    def test_complete_session_becomes_inactive_but_stays_complete(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        older = create_test_chatbot_session(db, chatbot=chatbot)
        update_chatbot_session_progress(
            db,
            session_id=older.id,
            collected_fields={"name": "Asha"},
            completion_status=ChatbotCompletionStatusEnum.COMPLETE,
        )
        completed_at = _reload(db, older).updated_at

        create_test_chatbot_session(db, chatbot=chatbot, contact_ref=older.contact_ref)

        superseded = _reload(db, older)
        assert superseded.status == ChatbotSessionStatusEnum.INACTIVE
        assert superseded.completion_status == ChatbotCompletionStatusEnum.COMPLETE
        assert superseded.collected_fields == {"name": "Asha"}
        assert superseded.updated_at > completed_at

    def test_other_chatbot_and_contact_sessions_stay_active(
        self, db: Session, config: Config, chatbot: Chatbot
    ) -> None:
        contact_ref = random_contact_ref()
        other_chatbot, _ = create_test_live_chatbot(db, config=config)
        same_contact_other_chatbot = create_test_chatbot_session(
            db, chatbot=other_chatbot, contact_ref=contact_ref
        )
        same_chatbot_other_contact = create_test_chatbot_session(db, chatbot=chatbot)
        untouched = [same_contact_other_chatbot, same_chatbot_other_contact]
        updated_ats = [s.updated_at for s in untouched]

        create_test_chatbot_session(db, chatbot=chatbot, contact_ref=contact_ref)

        for chatbot_session, updated_at in zip(untouched, updated_ats, strict=True):
            stored = _reload(db, chatbot_session)
            assert stored.status == ChatbotSessionStatusEnum.ACTIVE
            assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS
            assert stored.updated_at == updated_at

    def test_already_inactive_session_is_left_unmodified(
        self, db: Session, chatbot: Chatbot
    ) -> None:
        first = create_test_chatbot_session(db, chatbot=chatbot)
        second = create_test_chatbot_session(
            db, chatbot=chatbot, contact_ref=first.contact_ref
        )
        first_after_supersede = _reload(db, first).updated_at

        third = create_test_chatbot_session(
            db, chatbot=chatbot, contact_ref=first.contact_ref
        )

        stored_first = _reload(db, first)
        assert stored_first.status == ChatbotSessionStatusEnum.INACTIVE
        assert stored_first.completion_status == ChatbotCompletionStatusEnum.ABANDONED
        assert stored_first.updated_at == first_after_supersede
        assert _reload(db, second).status == ChatbotSessionStatusEnum.INACTIVE
        assert _reload(db, third).status == ChatbotSessionStatusEnum.ACTIVE


@pytest.fixture
def savepoint_db() -> Iterator[Session]:
    # The crud functions roll back on IntegrityError. Under the shared `db`
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
        config = create_test_anthropic_config(savepoint_db)
        chatbot, _ = create_test_live_chatbot(savepoint_db, config=config)
        prior = create_test_chatbot_session(savepoint_db, chatbot=chatbot)

        with pytest.raises(IntegrityError):
            create_chatbot_session(
                savepoint_db,
                chatbot_id=chatbot.id,
                flow_version_id=uuid4(),
                contact_ref=prior.contact_ref,
                active_node_id=ONBOARDING_NODE_ID,
            )

        stored = _reload(savepoint_db, prior)
        assert stored.status == ChatbotSessionStatusEnum.ACTIVE
        assert stored.completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS

    def test_active_index_violation_raises_conflict_and_rolls_back_supersede(
        self, savepoint_db: Session
    ) -> None:
        config = create_test_anthropic_config(savepoint_db)
        chatbot, flow_version = create_test_live_chatbot(savepoint_db, config=config)
        prior = create_test_chatbot_session(savepoint_db, chatbot=chatbot)
        racing_row = {
            "id": uuid4(),
            "chatbot_id": chatbot.id,
            "flow_version_id": flow_version.id,
            "contact_ref": prior.contact_ref,
            "active_node_id": ONBOARDING_NODE_ID,
            "status": ChatbotSessionStatusEnum.ACTIVE,
            "completion_status": ChatbotCompletionStatusEnum.IN_PROGRESS,
            "collected_fields": {},
            "inserted_at": prior.inserted_at,
            "updated_at": prior.inserted_at,
        }

        # Simulates a racing request committing its active row after this call's
        # supersede UPDATE ran but before its INSERT is flushed.
        raced: list[bool] = []

        def insert_racing_row(session: Session, *_args: Any) -> None:
            if raced:
                return
            raced.append(True)
            session.connection().execute(insert(ChatbotSession).values(**racing_row))

        event.listen(savepoint_db, "before_flush", insert_racing_row)
        try:
            with pytest.raises(ChatbotSessionConflictError) as exc_info:
                create_chatbot_session(
                    savepoint_db,
                    chatbot_id=chatbot.id,
                    flow_version_id=flow_version.id,
                    contact_ref=prior.contact_ref,
                    active_node_id=ONBOARDING_NODE_ID,
                )
        finally:
            event.remove(savepoint_db, "before_flush", insert_racing_row)

        assert str(exc_info.value) == CONCURRENT_SESSION_CONFLICT_MESSAGE
        stored = list_stored_sessions_for_contact(savepoint_db, prior.contact_ref)
        assert [s.id for s in stored] == [prior.id]
        assert stored[0].status == ChatbotSessionStatusEnum.ACTIVE
        assert stored[0].completion_status == ChatbotCompletionStatusEnum.IN_PROGRESS


class TestCreateChatbotFlowVersionFailures:
    def test_duplicate_version_raises_flow_conflict(
        self, savepoint_db: Session
    ) -> None:
        project = create_test_project(savepoint_db)
        chatbot = create_test_chatbot(savepoint_db, project=project)
        create_chatbot_flow_version(
            savepoint_db, chatbot_id=chatbot.id, version=1, nodes=[], edges=[]
        )

        with pytest.raises(ChatbotFlowConflictError) as exc_info:
            create_chatbot_flow_version(
                savepoint_db, chatbot_id=chatbot.id, version=1, nodes=[], edges=[]
            )

        assert str(exc_info.value) == CONCURRENT_FLOW_CONFLICT_MESSAGE
        versions = list_stored_flow_versions(savepoint_db, chatbot)
        assert [v.version for v in versions] == [1]
