from uuid import uuid4

import pytest
from sqlmodel import Session, select

from app.crud import JobCrud
from app.crud.llm import (
    create_llm_call,
    get_llm_call_by_id,
    get_llm_calls_by_job_id,
    update_llm_call_response,
)
from app.models import JobType, Project, Organization
from app.models.llm import (
    ConfigBlob,
    LLMCallRequest,
    LlmCall,
    QueryParams,
)
from app.models.llm.request import (
    KaapiCompletionConfig,
    LLMCallConfig,
)


@pytest.fixture
def test_project(db: Session) -> Project:
    """Get the first available test project."""
    project = db.exec(select(Project).limit(1)).first()
    assert project is not None, "No test project found in seed data"
    return project


@pytest.fixture
def test_organization(db: Session, test_project: Project) -> Organization:
    """Get the organization for the test project."""
    org = db.get(Organization, test_project.organization_id)
    assert org is not None, "No organization found for test project"
    return org


@pytest.fixture
def test_job(db: Session):
    """Create a test job for LLM call tests."""
    crud = JobCrud(db)
    return crud.create(job_type=JobType.LLM_API, trace_id="test-llm-trace")


@pytest.fixture
def text_config_blob() -> ConfigBlob:
    """Create a text completion config blob."""
    return ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            params={
                "model": "gpt-4o",
                "instructions": "You are a helpful assistant",
                "temperature": 0.7,
            },
            type="text",
        )
    )


@pytest.fixture
def stt_config_blob() -> ConfigBlob:
    """Create a speech-to-text config blob."""
    return ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            params={
                "model": "whisper-1",
                "instructions": "Transcribe",
                "input_language": "en",
            },
            type="stt",
        )
    )


@pytest.fixture
def tts_config_blob() -> ConfigBlob:
    """Create a text-to-speech config blob."""
    return ConfigBlob(
        completion=KaapiCompletionConfig(
            provider="openai",
            params={
                "model": "tts-1",
                "voice": "alloy",
                "language": "en",
            },
            type="tts",
        )
    )


def test_create_llm_call_text(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    text_config_blob: ConfigBlob,
) -> None:
    """Test creating a text completion LLM call."""
    request = LLMCallRequest(
        query=QueryParams(input="Hello, how are you?"),
        config=LLMCallConfig(blob=text_config_blob),
    )

    llm_call = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=text_config_blob,
        original_provider="openai",
    )

    assert llm_call.id is not None
    assert llm_call.job_id == test_job.id
    assert llm_call.project_id == test_project.id
    assert llm_call.organization_id == test_organization.id
    assert llm_call.input == "Hello, how are you?"
    assert llm_call.input_type == "text"
    assert llm_call.output_type == "text"
    assert llm_call.provider == "openai"
    assert llm_call.model == "gpt-4o"
    assert llm_call.config is not None
    assert "config_blob" in llm_call.config


def test_create_llm_call_stt(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    stt_config_blob: ConfigBlob,
) -> None:
    """Test creating a speech-to-text LLM call."""
    request = LLMCallRequest(
        query=QueryParams(input="/path/to/audio.wav"),
        config=LLMCallConfig(blob=stt_config_blob),
    )

    llm_call = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=stt_config_blob,
        original_provider="openai",
    )

    assert llm_call.input_type == "audio"
    assert llm_call.output_type == "text"
    assert llm_call.model == "whisper-1"


def test_create_llm_call_tts(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    tts_config_blob: ConfigBlob,
) -> None:
    """Test creating a text-to-speech LLM call."""
    request = LLMCallRequest(
        query=QueryParams(input="Hello world"),
        config=LLMCallConfig(blob=tts_config_blob),
    )

    llm_call = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=tts_config_blob,
        original_provider="openai",
    )

    assert llm_call.input_type == "text"
    assert llm_call.output_type == "audio"
    assert llm_call.model == "tts-1"


def test_create_llm_call_with_stored_config(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    text_config_blob: ConfigBlob,
) -> None:
    """Test creating an LLM call with a stored config reference."""
    config_id = uuid4()
    request = LLMCallRequest(
        query=QueryParams(input="Test input"),
        config=LLMCallConfig(id=config_id, version=1),
    )

    llm_call = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=text_config_blob,
        original_provider="openai",
    )

    assert llm_call.config is not None
    assert "config_id" in llm_call.config
    assert llm_call.config["config_id"] == str(config_id)
    assert llm_call.config["config_version"] == 1


def test_get_llm_call_by_id(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    text_config_blob: ConfigBlob,
) -> None:
    """Test fetching an LLM call by ID."""
    request = LLMCallRequest(
        query=QueryParams(input="Test input"),
        config=LLMCallConfig(blob=text_config_blob),
    )

    created = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=text_config_blob,
        original_provider="openai",
    )

    fetched = get_llm_call_by_id(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.input == "Test input"


def test_get_llm_call_by_id_with_project_scope(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    text_config_blob: ConfigBlob,
) -> None:
    """Test fetching an LLM call with project scoping."""
    request = LLMCallRequest(
        query=QueryParams(input="Test input"),
        config=LLMCallConfig(blob=text_config_blob),
    )

    created = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=text_config_blob,
        original_provider="openai",
    )

    # Should find with correct project
    fetched = get_llm_call_by_id(db, created.id, project_id=test_project.id)
    assert fetched is not None

    # Should not find with wrong project
    fetched_wrong = get_llm_call_by_id(db, created.id, project_id=99999)
    assert fetched_wrong is None


def test_get_llm_call_by_id_not_found(db: Session) -> None:
    """Test fetching a non-existent LLM call."""
    fake_id = uuid4()
    result = get_llm_call_by_id(db, fake_id)
    assert result is None


def test_get_llm_calls_by_job_id(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    text_config_blob: ConfigBlob,
) -> None:
    """Test fetching all LLM calls for a job."""
    # Create multiple LLM calls for the same job
    for i in range(3):
        request = LLMCallRequest(
            query=QueryParams(input=f"Test input {i}"),
            config=LLMCallConfig(blob=text_config_blob),
        )
        create_llm_call(
            db,
            request=request,
            job_id=test_job.id,
            project_id=test_project.id,
            organization_id=test_organization.id,
            resolved_config=text_config_blob,
            original_provider="openai",
        )

    llm_calls = get_llm_calls_by_job_id(db, test_job.id)
    assert len(llm_calls) == 3


def test_get_llm_calls_by_job_id_empty(db: Session) -> None:
    """Test fetching LLM calls for a job with no calls."""
    fake_job_id = uuid4()
    llm_calls = get_llm_calls_by_job_id(db, fake_job_id)
    assert llm_calls == []


def test_update_llm_call_response(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    text_config_blob: ConfigBlob,
) -> None:
    """Test updating an LLM call with response data."""
    request = LLMCallRequest(
        query=QueryParams(input="Test input"),
        config=LLMCallConfig(blob=text_config_blob),
    )

    created = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=text_config_blob,
        original_provider="openai",
    )

    # Update with response data
    content = {"text": "This is the response"}
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "reasoning_tokens": None,
    }

    updated = update_llm_call_response(
        db,
        llm_call_id=created.id,
        provider_response_id="resp_123456",
        content=content,
        usage=usage,
        conversation_id="conv_abc",
    )

    assert updated.provider_response_id == "resp_123456"
    assert updated.content == content
    assert updated.usage == usage
    assert updated.conversation_id == "conv_abc"


def test_update_llm_call_response_partial(
    db: Session,
    test_job,
    test_project: Project,
    test_organization: Organization,
    text_config_blob: ConfigBlob,
) -> None:
    """Test partial update of an LLM call response."""
    request = LLMCallRequest(
        query=QueryParams(input="Test input"),
        config=LLMCallConfig(blob=text_config_blob),
    )

    created = create_llm_call(
        db,
        request=request,
        job_id=test_job.id,
        project_id=test_project.id,
        organization_id=test_organization.id,
        resolved_config=text_config_blob,
        original_provider="openai",
    )

    # Only update provider_response_id
    updated = update_llm_call_response(
        db,
        llm_call_id=created.id,
        provider_response_id="resp_partial",
    )

    assert updated.provider_response_id == "resp_partial"
    assert updated.content is None  # Should remain None
    assert updated.usage is None  # Should remain None


def test_update_llm_call_response_not_found(db: Session) -> None:
    """Test updating a non-existent LLM call."""
    fake_id = uuid4()

    with pytest.raises(ValueError, match=str(fake_id)):
        update_llm_call_response(
            db,
            llm_call_id=fake_id,
            provider_response_id="resp_123",
        )
