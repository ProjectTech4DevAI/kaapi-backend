from uuid import uuid4

import pytest
from sqlmodel import Session

from app.crud.openai_conversation import (
    get_conversation_by_id,
    get_conversation_by_response_id,
    get_conversation_by_ancestor_id,
    get_conversations_by_project,
    get_ancestor_id_from_response,
    get_conversations_count_by_project,
    create_conversation,
    delete_conversation,
)
from app.models import OpenAIConversationCreate, Project
from app.tests.utils.utils import get_project, get_organization
from app.tests.utils.llm_provider import generate_openai_id


def test_get_conversation_by_id_success(db: Session) -> None:
    """Test successful conversation retrieval by ID."""
    project = get_project(db)
    organization = get_organization(db)

    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of Japan?",
        response="The capital of Japan is Tokyo.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    retrieved_conversation = get_conversation_by_id(
        session=db,
        conversation_id=conversation.id,
        project_id=project.id,
    )

    assert retrieved_conversation is not None
    assert retrieved_conversation.id == conversation.id
    assert retrieved_conversation.response_id == conversation.response_id


def test_get_conversation_by_id_not_found(db: Session) -> None:
    """Test conversation retrieval by non-existent ID."""
    project = get_project(db)

    retrieved_conversation = get_conversation_by_id(
        session=db,
        conversation_id=99999,
        project_id=project.id,
    )

    assert retrieved_conversation is None


def test_get_conversation_by_response_id_success(db: Session) -> None:
    """Test successful conversation retrieval by response ID."""
    project = get_project(db)
    organization = get_organization(db)

    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of Japan?",
        response="The capital of Japan is Tokyo.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )
    retrieved_conversation = get_conversation_by_response_id(
        session=db,
        response_id=conversation.response_id,
        project_id=project.id,
    )

    assert retrieved_conversation is not None
    assert retrieved_conversation.id == conversation.id
    assert retrieved_conversation.response_id == conversation.response_id


def test_get_conversation_by_response_id_not_found(db: Session) -> None:
    """Test conversation retrieval by non-existent response ID."""
    project = get_project(db)

    retrieved_conversation = get_conversation_by_response_id(
        session=db,
        response_id="nonexistent_response_id",
        project_id=project.id,
    )

    assert retrieved_conversation is None


def test_get_conversation_by_ancestor_id_success(db: Session) -> None:
    """Test successful conversation retrieval by ancestor ID."""
    project = get_project(db)
    organization = get_organization(db)

    ancestor_response_id = generate_openai_id("resp_", 40)
    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=ancestor_response_id,
        previous_response_id=None,
        user_question="What is the capital of France?",
        response="The capital of France is Paris.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    retrieved_conversation = get_conversation_by_ancestor_id(
        session=db,
        ancestor_response_id=ancestor_response_id,
        project_id=project.id,
    )

    assert retrieved_conversation is not None
    assert retrieved_conversation.id == conversation.id
    assert retrieved_conversation.ancestor_response_id == ancestor_response_id


def test_get_conversation_by_ancestor_id_not_found(db: Session) -> None:
    """Test conversation retrieval by non-existent ancestor ID."""
    project = get_project(db)

    retrieved_conversation = get_conversation_by_ancestor_id(
        session=db,
        ancestor_response_id="nonexistent_ancestor_id",
        project_id=project.id,
    )

    assert retrieved_conversation is None


def test_get_conversations_by_project_success(db: Session) -> None:
    """Test successful conversation listing by project."""
    project = get_project(db)
    organization = get_organization(db)

    # Create multiple conversations directly
    for i in range(3):
        conversation_data = OpenAIConversationCreate(
            response_id=generate_openai_id("resp_", 40),
            ancestor_response_id=generate_openai_id("resp_", 40),
            previous_response_id=None,
            user_question=f"Test question {i}",
            response=f"Test response {i}",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )
        create_conversation(
            session=db,
            conversation=conversation_data,
            project_id=project.id,
            organization_id=organization.id,
        )

    conversations = get_conversations_by_project(
        session=db,
        project_id=project.id,
    )

    assert len(conversations) >= 3
    for conversation in conversations:
        assert conversation.project_id == project.id
        assert conversation.is_deleted is False


def test_get_conversations_by_project_with_pagination(db: Session) -> None:
    """Test conversation listing by project with pagination."""
    project = get_project(db)
    organization = get_organization(db)

    # Create multiple conversations
    for i in range(5):
        conversation_data = OpenAIConversationCreate(
            response_id=generate_openai_id("resp_", 40),
            ancestor_response_id=generate_openai_id("resp_", 40),
            previous_response_id=None,
            user_question=f"Test question {i}",
            response=f"Test response {i}",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )
        create_conversation(
            session=db,
            conversation=conversation_data,
            project_id=project.id,
            organization_id=organization.id,
        )

    conversations = get_conversations_by_project(
        session=db,
        project_id=project.id,
        skip=1,
        limit=2,
    )

    assert len(conversations) <= 2


def test_delete_conversation_success(db: Session) -> None:
    """Test successful conversation deletion."""
    project = get_project(db)
    organization = get_organization(db)

    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of Japan?",
        response="The capital of Japan is Tokyo.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    deleted_conversation = delete_conversation(
        session=db,
        conversation_id=conversation.id,
        project_id=project.id,
    )

    assert deleted_conversation is not None
    assert deleted_conversation.id == conversation.id
    assert deleted_conversation.is_deleted is True
    assert deleted_conversation.deleted_at is not None


def test_delete_conversation_not_found(db: Session) -> None:
    """Test conversation deletion with non-existent ID."""
    project = get_project(db)

    deleted_conversation = delete_conversation(
        session=db,
        conversation_id=99999,
        project_id=project.id,
    )

    assert deleted_conversation is None


def test_conversation_soft_delete_behavior(db: Session) -> None:
    """Test that deleted conversations are not returned by get functions."""
    project = get_project(db)
    organization = get_organization(db)

    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="What is the capital of Japan?",
        response="The capital of Japan is Tokyo.",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    delete_conversation(
        session=db,
        conversation_id=conversation.id,
        project_id=project.id,
    )

    retrieved_conversation = get_conversation_by_id(
        session=db,
        conversation_id=conversation.id,
        project_id=project.id,
    )
    assert retrieved_conversation is None

    retrieved_conversation = get_conversation_by_response_id(
        session=db,
        response_id=conversation.response_id,
        project_id=project.id,
    )
    assert retrieved_conversation is None

    conversations = get_conversations_by_project(
        session=db,
        project_id=project.id,
    )
    assert conversation.id not in [c.id for c in conversations]


def test_get_ancestor_id_from_response_no_previous_response(db: Session) -> None:
    """Test get_ancestor_id_from_response when previous_response_id is None."""
    project = get_project(db)
    current_response_id = generate_openai_id("resp_", 40)

    ancestor_id = get_ancestor_id_from_response(
        session=db,
        current_response_id=current_response_id,
        previous_response_id=None,
        project_id=project.id,
    )

    assert ancestor_id == current_response_id


def test_get_ancestor_id_from_response_previous_not_found(db: Session) -> None:
    """Test get_ancestor_id_from_response when previous_response_id is not found in DB."""
    project = get_project(db)
    current_response_id = generate_openai_id("resp_", 40)
    previous_response_id = generate_openai_id("resp_", 40)

    ancestor_id = get_ancestor_id_from_response(
        session=db,
        current_response_id=current_response_id,
        previous_response_id=previous_response_id,
        project_id=project.id,
    )

    # When previous_response_id is not found, should return previous_response_id
    assert ancestor_id == previous_response_id


def test_get_ancestor_id_from_response_previous_found_with_ancestor(
    db: Session,
) -> None:
    """Test get_ancestor_id_from_response when previous_response_id is found and has an ancestor."""
    project = get_project(db)
    organization = get_organization(db)

    # Create a conversation chain: ancestor -> previous -> current
    ancestor_response_id = generate_openai_id("resp_", 40)

    ancestor_conversation_data = OpenAIConversationCreate(
        response_id=ancestor_response_id,
        ancestor_response_id=ancestor_response_id,  # Self-referencing
        previous_response_id=None,
        user_question="Original question",
        response="Original response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    ancestor_conversation = create_conversation(
        session=db,
        conversation=ancestor_conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    previous_response_id = generate_openai_id("resp_", 40)
    previous_conversation_data = OpenAIConversationCreate(
        response_id=previous_response_id,
        ancestor_response_id=ancestor_response_id,
        previous_response_id=ancestor_response_id,
        user_question="Previous question",
        response="Previous response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    create_conversation(
        session=db,
        conversation=previous_conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    # Test the current conversation
    current_response_id = generate_openai_id("resp_", 40)
    ancestor_id = get_ancestor_id_from_response(
        session=db,
        current_response_id=current_response_id,
        previous_response_id=previous_response_id,
        project_id=project.id,
    )

    # Should return the ancestor_response_id from the previous conversation
    assert ancestor_id == ancestor_response_id


def test_get_conversations_count_by_project_success(db: Session) -> None:
    """Test successful conversation count retrieval by project."""
    project = get_project(db)
    organization = get_organization(db)

    initial_count = get_conversations_count_by_project(
        session=db,
        project_id=project.id,
    )

    for i in range(3):
        conversation_data = OpenAIConversationCreate(
            response_id=generate_openai_id("resp_", 40),
            ancestor_response_id=generate_openai_id("resp_", 40),
            previous_response_id=None,
            user_question=f"Test question {i}",
            response=f"Test response {i}",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )
        create_conversation(
            session=db,
            conversation=conversation_data,
            project_id=project.id,
            organization_id=organization.id,
        )

    updated_count = get_conversations_count_by_project(
        session=db,
        project_id=project.id,
    )

    assert updated_count == initial_count + 3


def test_get_ancestor_id_from_response_previous_found_without_ancestor(
    db: Session,
) -> None:
    """Test get_ancestor_id_from_response when previous_response_id is found but has no ancestor."""
    project = get_project(db)
    organization = get_organization(db)

    # Create a previous conversation that is self-referencing
    previous_response_id = generate_openai_id("resp_", 40)
    previous_conversation_data = OpenAIConversationCreate(
        response_id=previous_response_id,
        ancestor_response_id=previous_response_id,  # Self-referencing for root
        previous_response_id=None,
        user_question="Previous question",
        response="Previous response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    create_conversation(
        session=db,
        conversation=previous_conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    current_response_id = generate_openai_id("resp_", 40)
    ancestor_id = get_ancestor_id_from_response(
        session=db,
        current_response_id=current_response_id,
        previous_response_id=previous_response_id,
        project_id=project.id,
    )

    # When previous conversation is root (self-referencing), should return the previous_response_id
    assert ancestor_id == previous_response_id


def test_get_ancestor_id_from_response_different_project(db: Session) -> None:
    """Test get_ancestor_id_from_response respects project scoping."""
    project1 = get_project(db)
    organization = get_organization(db)

    project2 = Project(
        name=f"test_project_{uuid4()}",
        description="Test project for scoping",
        is_active=True,
        organization_id=organization.id,
    )
    db.add(project2)
    db.commit()
    db.refresh(project2)

    # Create a conversation in project1
    previous_response_id = generate_openai_id("resp_", 40)
    previous_conversation_data = OpenAIConversationCreate(
        response_id=previous_response_id,
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="Previous question",
        response="Previous response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    create_conversation(
        session=db,
        conversation=previous_conversation_data,
        project_id=project1.id,
        organization_id=organization.id,
    )

    # Test looking for it in project2 (should not find it)
    current_response_id = generate_openai_id("resp_", 40)
    ancestor_id = get_ancestor_id_from_response(
        session=db,
        current_response_id=current_response_id,
        previous_response_id=previous_response_id,
        project_id=project2.id,
    )

    # Should return previous_response_id since it's not found in project2
    assert ancestor_id == previous_response_id


def test_get_ancestor_id_from_response_complex_chain(db: Session) -> None:
    """Test get_ancestor_id_from_response with a complex conversation chain."""
    project = get_project(db)
    organization = get_organization(db)

    # Create a complex chain: A -> B -> C -> D
    # A is the root ancestor
    response_a = generate_openai_id("resp_", 40)
    conversation_a_data = OpenAIConversationCreate(
        response_id=response_a,
        ancestor_response_id=response_a,  # Self-referencing
        previous_response_id=None,
        user_question="Question A",
        response="Response A",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    create_conversation(
        session=db,
        conversation=conversation_a_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    # B references A
    response_b = generate_openai_id("resp_", 40)
    conversation_b_data = OpenAIConversationCreate(
        response_id=response_b,
        ancestor_response_id=response_a,
        previous_response_id=response_a,
        user_question="Question B",
        response="Response B",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    create_conversation(
        session=db,
        conversation=conversation_b_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    # C references B
    response_c = generate_openai_id("resp_", 40)
    conversation_c_data = OpenAIConversationCreate(
        response_id=response_c,
        ancestor_response_id=response_a,  # Should inherit from B
        previous_response_id=response_b,
        user_question="Question C",
        response="Response C",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    create_conversation(
        session=db,
        conversation=conversation_c_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    # Test D referencing C
    response_d = generate_openai_id("resp_", 40)
    ancestor_id = get_ancestor_id_from_response(
        session=db,
        current_response_id=response_d,
        previous_response_id=response_c,
        project_id=project.id,
    )

    # Should return response_a (the root ancestor)
    assert ancestor_id == response_a


def test_create_conversation_success(db: Session) -> None:
    """Test successful conversation creation."""
    project = get_project(db)
    organization = get_organization(db)

    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="Test question",
        response="Test response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    assert conversation is not None
    assert conversation.response_id == conversation_data.response_id
    assert conversation.user_question == conversation_data.user_question
    assert conversation.response == conversation_data.response
    assert conversation.model == conversation_data.model
    assert conversation.project_id == project.id
    assert conversation.organization_id == organization.id


def test_delete_conversation_excludes_from_count(db: Session) -> None:
    """Test that deleted conversations are excluded from count."""
    project = get_project(db)
    organization = get_organization(db)

    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=generate_openai_id("resp_", 40),
        previous_response_id=None,
        user_question="Test question",
        response="Test response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    count_before = get_conversations_count_by_project(
        session=db,
        project_id=project.id,
    )

    delete_conversation(
        session=db,
        conversation_id=conversation.id,
        project_id=project.id,
    )

    count_after = get_conversations_count_by_project(
        session=db,
        project_id=project.id,
    )

    assert count_after == count_before - 1


def test_get_conversations_count_by_project_different_projects(db: Session) -> None:
    """Test that count is isolated by project."""
    project1 = get_project(db)
    organization = get_organization(db)

    # Get another project (assuming there are multiple projects in test data)
    project2 = (
        get_project(db, "Dalgo")
        if get_project(db, "Dalgo") is not None
        else get_project(db)
    )

    # Create conversations in project1
    for i in range(2):
        conversation_data = OpenAIConversationCreate(
            response_id=generate_openai_id("resp_", 40),
            ancestor_response_id=generate_openai_id("resp_", 40),
            previous_response_id=None,
            user_question=f"Test question {i}",
            response=f"Test response {i}",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )
        create_conversation(
            session=db,
            conversation=conversation_data,
            project_id=project1.id,
            organization_id=organization.id,
        )

    # Create conversations in project2
    for i in range(3):
        conversation_data = OpenAIConversationCreate(
            response_id=generate_openai_id("resp_", 40),
            ancestor_response_id=generate_openai_id("resp_", 40),
            previous_response_id=None,
            user_question=f"Test question {i}",
            response=f"Test response {i}",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )
        create_conversation(
            session=db,
            conversation=conversation_data,
            project_id=project2.id,
            organization_id=organization.id,
        )

    # Check counts are isolated
    count1 = get_conversations_count_by_project(session=db, project_id=project1.id)
    count2 = get_conversations_count_by_project(session=db, project_id=project2.id)

    assert count1 >= 2
    assert count2 >= 3


def test_response_id_validation_pattern(db: Session) -> None:
    """Test that response ID validation pattern is enforced."""
    project = get_project(db)
    organization = get_organization(db)

    # Test valid response ID
    valid_response_id = "resp_1234567890abcdef"
    conversation_data = OpenAIConversationCreate(
        response_id=valid_response_id,
        ancestor_response_id="resp_abcdef1234567890",
        previous_response_id=None,
        user_question="Test question",
        response="Test response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )
    assert conversation is not None
    assert conversation.response_id == conversation_data.response_id
    assert conversation.user_question == conversation_data.user_question
    assert conversation.response == conversation_data.response
    assert conversation.model == conversation_data.model
    assert conversation.assistant_id == conversation_data.assistant_id
    assert conversation.project_id == project.id
    assert conversation.organization_id == organization.id
    assert conversation.is_deleted is False
    assert conversation.deleted_at is None


def test_create_conversation_with_ancestor(db: Session) -> None:
    """Test conversation creation with ancestor and previous response IDs."""
    project = get_project(db)
    organization = get_organization(db)

    ancestor_response_id = generate_openai_id("resp_", 40)
    previous_response_id = generate_openai_id("resp_", 40)

    conversation_data = OpenAIConversationCreate(
        response_id=generate_openai_id("resp_", 40),
        ancestor_response_id=ancestor_response_id,
        previous_response_id=previous_response_id,
        user_question="Follow-up question",
        response="Follow-up response",
        model="gpt-4o",
        assistant_id=generate_openai_id("asst_", 20),
    )

    conversation = create_conversation(
        session=db,
        conversation=conversation_data,
        project_id=project.id,
        organization_id=organization.id,
    )

    assert conversation is not None
    assert conversation.ancestor_response_id == ancestor_response_id
    assert conversation.previous_response_id == previous_response_id
    assert conversation.response_id == conversation_data.response_id

    invalid_response_id = "resp_123"
    with pytest.raises(ValueError, match="String should have at least 10 characters"):
        OpenAIConversationCreate(
            response_id=invalid_response_id,
            ancestor_response_id="resp_abcdef1234567890",
            previous_response_id=None,
            user_question="Test question",
            response="Test response",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )

    invalid_response_id2 = "msg_1234567890abcdef"
    with pytest.raises(ValueError, match="response_id fields must follow pattern"):
        OpenAIConversationCreate(
            response_id=invalid_response_id2,
            ancestor_response_id="resp_abcdef1234567890",
            previous_response_id=None,
            user_question="Test question",
            response="Test response",
            model="gpt-4o",
            assistant_id=generate_openai_id("asst_", 20),
        )
