import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.models import User


# ---------------------------------------------------------------------------
# Existing user test (unchanged)
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
# Unit tests for migrate_tts_base64_to_s3
# ---------------------------------------------------------------------------

MODULE = "app.api.routes.private"
FAKE_AUDIO = b"\x00\x01\x02\x03audio-bytes"
FAKE_B64 = base64.b64encode(FAKE_AUDIO).decode()


def _fake_call(
    content: dict | None = None,
    project_id: int = 1,
    organization_id: int = 10,
) -> SimpleNamespace:
    """Lightweight stand-in for an LlmCall row."""
    return SimpleNamespace(
        id=uuid4(),
        project_id=project_id,
        organization_id=organization_id,
        content=content,
        updated_at=None,
    )


def _b64_content(mime_type: str = "audio/mp3") -> dict:
    return {
        "type": "audio",
        "content": {"format": "base64", "value": FAKE_B64, "mime_type": mime_type},
    }


def _uri_content() -> dict:
    return {
        "type": "audio",
        "content": {
            "format": "uri",
            "value": "s3://bucket/existing.mp3",
            "mime_type": "audio/mp3",
        },
    }


def _mock_session(rows: list) -> MagicMock:
    """Build a mock session whose .exec() returns count then rows."""
    session = MagicMock()
    count_result = MagicMock()
    count_result.one.return_value = len(rows)
    # First exec call → count, second → row iterator
    session.exec.side_effect = [count_result, iter(rows)]
    return session


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/migrated.mp3")
def test_processes_base64_row(mock_upload: MagicMock, mock_storage: MagicMock) -> None:
    """A row with base64 content is uploaded and rewritten to URI format."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    row = _fake_call(content=_b64_content())
    session = _mock_session([row])

    result = migrate_tts_base64_to_s3(session)

    assert result["processed"] == 1
    assert result["failed"] == 0
    assert row.content["content"]["format"] == "uri"
    assert row.content["content"]["value"] == "s3://bucket/migrated.mp3"
    session.add.assert_called_once_with(row)
    mock_upload.assert_called_once()


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/migrated.mp3")
def test_skips_already_migrated_uri(
    mock_upload: MagicMock, mock_storage: MagicMock
) -> None:
    """Rows already in URI format are skipped."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    row = _fake_call(content=_uri_content())
    session = _mock_session([row])

    result = migrate_tts_base64_to_s3(session)

    assert result["skipped"] == 1
    assert result["processed"] == 0
    mock_upload.assert_not_called()


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/migrated.mp3")
def test_skips_null_content(mock_upload: MagicMock, mock_storage: MagicMock) -> None:
    """Rows with None content are skipped."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    row = _fake_call(content=None)
    session = _mock_session([row])

    result = migrate_tts_base64_to_s3(session)

    assert result["skipped"] == 1
    assert result["processed"] == 0
    mock_upload.assert_not_called()


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value=None)
def test_fails_when_upload_returns_none(
    mock_upload: MagicMock, mock_storage: MagicMock
) -> None:
    """upload returning None is recorded as a failure; original content is unchanged."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    original_content = _b64_content()
    row = _fake_call(content=original_content)
    session = _mock_session([row])

    result = migrate_tts_base64_to_s3(session)

    assert result["failed"] == 1
    assert result["processed"] == 0
    assert any(e["call_id"] == str(row.id) for e in result["errors"])
    session.expunge.assert_called_once_with(row)


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", side_effect=RuntimeError("S3 timeout"))
def test_fails_on_upload_exception(
    mock_upload: MagicMock, mock_storage: MagicMock
) -> None:
    """An upload exception is caught and recorded in errors."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    row = _fake_call(content=_b64_content())
    session = _mock_session([row])

    result = migrate_tts_base64_to_s3(session)

    assert result["failed"] == 1
    error = next(e for e in result["errors"] if e["call_id"] == str(row.id))
    assert "S3 timeout" in error["error"]
    session.expunge.assert_called_once_with(row)


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/out.mp3")
def test_uses_correct_s3_prefix(
    mock_upload: MagicMock, mock_storage: MagicMock
) -> None:
    """The prefix follows orgs/{org_id}/{project_id}/audio/tts."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    row = _fake_call(content=_b64_content(), project_id=42, organization_id=7)
    session = _mock_session([row])

    migrate_tts_base64_to_s3(session)

    args = mock_upload.call_args[0]
    assert args[4] == "orgs/7/42/audio/tts"


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/out.mp3")
def test_preserves_mime_type(mock_upload: MagicMock, mock_storage: MagicMock) -> None:
    """The migrated content retains the original mime_type."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    row = _fake_call(content=_b64_content(mime_type="audio/wav"))
    session = _mock_session([row])

    migrate_tts_base64_to_s3(session)

    assert row.content["content"]["mime_type"] == "audio/wav"


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/out.mp3")
def test_no_candidates(mock_upload: MagicMock, mock_storage: MagicMock) -> None:
    """Zero rows means all counters are zero and no uploads happen."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    session = _mock_session([])

    result = migrate_tts_base64_to_s3(session)

    assert result["processed"] == 0
    assert result["failed"] == 0
    assert result["committed"] == 0
    assert result["total_candidates"] == 0
    mock_upload.assert_not_called()
    session.commit.assert_not_called()


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/out.mp3")
def test_returns_all_summary_fields(
    mock_upload: MagicMock, mock_storage: MagicMock
) -> None:
    """The response dict contains every expected key."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    session = _mock_session([])

    result = migrate_tts_base64_to_s3(session)

    for key in [
        "processed",
        "committed",
        "skipped",
        "failed",
        "total_candidates",
        "elapsed_seconds",
        "errors",
    ]:
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["elapsed_seconds"], (int, float))
    assert isinstance(result["errors"], list)


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/out.mp3")
def test_mixed_rows(mock_upload: MagicMock, mock_storage: MagicMock) -> None:
    """A mix of base64, URI, and null-content rows are handled correctly."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    rows = [
        _fake_call(content=_b64_content()),
        _fake_call(content=_uri_content()),
        _fake_call(content=None),
        _fake_call(content=_b64_content(mime_type="audio/wav")),
    ]
    session = _mock_session(rows)

    result = migrate_tts_base64_to_s3(session)

    assert result["processed"] == 2
    assert result["skipped"] == 2
    assert result["failed"] == 0
    assert mock_upload.call_count == 2


@patch(f"{MODULE}.get_cloud_storage", return_value=MagicMock())
@patch(f"{MODULE}.upload_audio_bytes_to_s3", return_value="s3://bucket/out.mp3")
def test_caches_storage_per_project(
    mock_upload: MagicMock, mock_storage: MagicMock
) -> None:
    """get_cloud_storage is called once per unique project_id, not per row."""
    from app.api.routes.private import migrate_tts_base64_to_s3

    rows = [
        _fake_call(content=_b64_content(), project_id=1),
        _fake_call(content=_b64_content(), project_id=1),
        _fake_call(content=_b64_content(), project_id=2),
    ]
    session = _mock_session(rows)

    migrate_tts_base64_to_s3(session)

    # Only 2 distinct project_ids → 2 calls
    assert mock_storage.call_count == 2
