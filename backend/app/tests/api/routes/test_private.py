import base64
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.crud import JobCrud
from app.crud.llm import create_llm_call, update_llm_call_response
from app.models import JobType, LlmCall, User
from app.models.llm.request import (
    ConfigBlob,
    KaapiCompletionConfig,
    LLMCallConfig,
    QueryParams,
)
from app.models.llm import LLMCallRequest
from app.tests.utils.auth import get_user_test_auth_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_B64 = base64.b64encode(b"\x00\x01\x02\x03audio-bytes").decode()

TTS_CONFIG = ConfigBlob(
    completion=KaapiCompletionConfig(
        provider="openai",
        params={"model": "gpt-4o-mini-tts", "temperature": 0.7},
        type="tts",
    )
)


def _make_tts_call(
    db: Session,
    *,
    project_id: int,
    organization_id: int,
    content: dict | None = None,
) -> LlmCall:
    """Create an LlmCall with input_type=text, output_type=audio."""
    job = JobCrud(db).create(
        job_type=JobType.LLM_API,
        trace_id=f"test-tts-{uuid4().hex[:8]}",
        project_id=project_id,
    )
    call = create_llm_call(
        db,
        request=LLMCallRequest(
            query=QueryParams(input="Say hello"),
            config=LLMCallConfig(blob=TTS_CONFIG),
        ),
        job_id=job.id,
        project_id=project_id,
        organization_id=organization_id,
        resolved_config=TTS_CONFIG,
        original_provider="openai",
    )
    if content is not None:
        update_llm_call_response(
            db,
            llm_call_id=call.id,
            provider_response_id=f"resp_{uuid4().hex[:8]}",
            content=content,
        )
        db.refresh(call)
    return call


def _base64_content(mime_type: str = "audio/mp3") -> dict:
    return {
        "type": "audio",
        "content": {
            "format": "base64",
            "value": FAKE_B64,
            "mime_type": mime_type,
        },
    }


def _uri_content() -> dict:
    """Content that has already been migrated."""
    return {
        "type": "audio",
        "content": {
            "format": "uri",
            "value": "s3://bucket/audio/existing.mp3",
            "mime_type": "audio/mp3",
        },
    }


MIGRATE_URL = f"{settings.API_V1_STR}/private/migrate/tts-base64-to-s3"
UPLOAD_PATH = "app.api.routes.private.upload_audio_bytes_to_s3"
STORAGE_PATH = "app.api.routes.private.get_cloud_storage"


# ---------------------------------------------------------------------------
# Existing user test
# ---------------------------------------------------------------------------


def test_create_user(client: TestClient, db: Session) -> None:
    r = client.post(
        f"{settings.API_V1_STR}/private/users",
        json={
            "email": "pollo@listo.com",
            "password": "password123",
            "full_name": "Pollo Listo",
        },
    )

    assert r.status_code == 200

    data = r.json()

    user = db.exec(select(User).where(User.id == data["id"])).first()

    assert user
    assert user.email == "pollo@listo.com"
    assert user.full_name == "Pollo Listo"


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value="s3://bucket/orgs/1/1/audio/tts/migrated.mp3")
def test_migrate_processes_base64_rows(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """Rows with base64 content are uploaded and rewritten to URI format."""
    auth = get_user_test_auth_context(db)
    call = _make_tts_call(
        db,
        project_id=auth.project_id,
        organization_id=auth.organization_id,
        content=_base64_content(),
    )

    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    data = r.json()
    assert data["processed"] >= 1
    assert data["failed"] == 0

    db.refresh(call)
    assert call.content["content"]["format"] == "uri"
    assert call.content["content"]["value"].startswith("s3://")
    mock_upload.assert_called()


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value="s3://bucket/audio.mp3")
def test_migrate_skips_already_migrated_rows(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """Rows already in URI format are skipped, not re-uploaded."""
    auth = get_user_test_auth_context(db)
    _make_tts_call(
        db,
        project_id=auth.project_id,
        organization_id=auth.organization_id,
        content=_uri_content(),
    )

    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    data = r.json()
    assert data["skipped"] >= 1
    # upload should not be called for already-migrated rows
    mock_upload.assert_not_called()


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value="s3://bucket/audio.mp3")
def test_migrate_skips_rows_with_no_content(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """Rows with NULL content are skipped."""
    auth = get_user_test_auth_context(db)
    _make_tts_call(
        db,
        project_id=auth.project_id,
        organization_id=auth.organization_id,
        content=None,
    )

    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    data = r.json()
    assert data["skipped"] >= 1
    mock_upload.assert_not_called()


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value=None)
def test_migrate_records_failure_when_upload_returns_none(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """When upload_audio_bytes_to_s3 returns None, the row is counted as failed."""
    auth = get_user_test_auth_context(db)
    call = _make_tts_call(
        db,
        project_id=auth.project_id,
        organization_id=auth.organization_id,
        content=_base64_content(),
    )

    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    data = r.json()
    assert data["failed"] >= 1
    assert any(e["call_id"] == str(call.id) for e in data["errors"])

    # Original content should remain unchanged
    db.refresh(call)
    assert call.content["content"]["format"] == "base64"


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, side_effect=RuntimeError("S3 connection timeout"))
def test_migrate_records_failure_on_upload_exception(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """An exception during upload is caught, logged, and reported in errors."""
    auth = get_user_test_auth_context(db)
    call = _make_tts_call(
        db,
        project_id=auth.project_id,
        organization_id=auth.organization_id,
        content=_base64_content(),
    )

    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    data = r.json()
    assert data["failed"] >= 1
    error_entry = next(e for e in data["errors"] if e["call_id"] == str(call.id))
    assert "S3 connection timeout" in error_entry["error"]

    # Original content should remain unchanged
    db.refresh(call)
    assert call.content["content"]["format"] == "base64"


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value="s3://bucket/audio.mp3")
def test_migrate_uses_correct_s3_prefix(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """The S3 prefix follows orgs/{org_id}/{project_id}/audio/tts."""
    auth = get_user_test_auth_context(db)
    call = _make_tts_call(
        db,
        project_id=auth.project_id,
        organization_id=auth.organization_id,
        content=_base64_content(),
    )

    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    # Verify the prefix passed to upload_audio_bytes_to_s3
    _, kwargs = mock_upload.call_args
    # positional args: (storage, audio_bytes, call_id, mime_type, prefix)
    args = mock_upload.call_args[0]
    expected_prefix = f"orgs/{auth.organization_id}/{auth.project_id}/audio/tts"
    assert args[4] == expected_prefix


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value="s3://bucket/audio.mp3")
def test_migrate_preserves_mime_type(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """The migrated content retains the original mime_type."""
    auth = get_user_test_auth_context(db)
    call = _make_tts_call(
        db,
        project_id=auth.project_id,
        organization_id=auth.organization_id,
        content=_base64_content(mime_type="audio/wav"),
    )

    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    db.refresh(call)
    assert call.content["content"]["mime_type"] == "audio/wav"


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value="s3://bucket/audio.mp3")
def test_migrate_returns_summary_fields(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """The response includes all expected summary fields."""
    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    data = r.json()
    for key in [
        "processed",
        "committed",
        "skipped",
        "failed",
        "total_candidates",
        "elapsed_seconds",
        "errors",
    ]:
        assert key in data, f"Missing key: {key}"

    assert isinstance(data["elapsed_seconds"], (int, float))
    assert isinstance(data["errors"], list)
    assert data["total_candidates"] >= 0


@patch(STORAGE_PATH, return_value=MagicMock())
@patch(UPLOAD_PATH, return_value="s3://bucket/audio.mp3")
def test_migrate_no_candidates(
    mock_upload: MagicMock,
    mock_storage: MagicMock,
    client: TestClient,
    db: Session,
) -> None:
    """When there are no matching rows, migration completes with all zeros."""
    # Don't create any TTS LlmCall rows — the endpoint should still succeed
    r = client.post(MIGRATE_URL)
    assert r.status_code == 200

    data = r.json()
    assert data["processed"] == 0
    assert data["failed"] == 0
    assert data["committed"] == 0
    mock_upload.assert_not_called()
