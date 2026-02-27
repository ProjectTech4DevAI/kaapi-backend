import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import File, FileType
from app.crud.file import create_file
from app.tests.utils.auth import TestAuthContext
from app.core.util import now
from app.core.config import settings
from app.main import app


client = TestClient(app)


@pytest.fixture
def audio_file_1(db: Session, user_api_key: TestAuthContext) -> File:
    """Create a test audio file record."""
    return create_file(
        session=db,
        object_store_url="s3://bucket/audio/test1.mp3",
        filename="test_audio_1.mp3",
        size_bytes=1024000,
        content_type="audio/mp3",
        file_type=FileType.AUDIO.value,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


@pytest.fixture
def audio_file_2(db: Session, user_api_key: TestAuthContext) -> File:
    """Create a second test audio file record."""
    return create_file(
        session=db,
        object_store_url="s3://bucket/audio/test2.wav",
        filename="test_audio_2.wav",
        size_bytes=2048000,
        content_type="audio/wav",
        file_type=FileType.AUDIO.value,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


@pytest.fixture
def audio_file_3(db: Session, user_api_key: TestAuthContext) -> File:
    """Create a third test audio file record."""
    return create_file(
        session=db,
        object_store_url="s3://bucket/audio/test3.flac",
        filename="test_audio_3.flac",
        size_bytes=3072000,
        content_type="audio/flac",
        file_type=FileType.AUDIO.value,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


class TestListAudioFiles:
    """Test cases for POST /stt-evaluations/files/list endpoint."""

    @patch("app.api.routes.stt_evaluations.files.get_cloud_storage")
    def test_list_all_audio_files_without_signed_url(
        self,
        mock_storage: MagicMock,
        db: Session,
        audio_file_1: File,
        audio_file_2: File,
        audio_file_3: File,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing all audio files without signed URLs."""
        response = client.post(
            f"{settings.API_V1_STR}/stt-evaluations/files/list",
            json={"file_ids": None},
            params={"include_url": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 3

        # Verify signed_url is not included
        for file_data in data["data"]:
            assert "signed_url" in file_data
            assert file_data["signed_url"] is None

        # Verify storage was not called
        mock_storage.assert_not_called()

    @patch("app.api.routes.stt_evaluations.files.get_cloud_storage")
    def test_list_all_audio_files_with_signed_url(
        self,
        mock_storage: MagicMock,
        db: Session,
        audio_file_1: File,
        audio_file_2: File,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing all audio files with signed URLs."""
        # Mock the storage get_signed_url method
        mock_storage_instance = MagicMock()
        mock_storage_instance.get_signed_url.return_value = (
            "https://signed.url/audio/test.mp3"
        )
        mock_storage.return_value = mock_storage_instance

        response = client.post(
            f"{settings.API_V1_STR}/stt-evaluations/files/list",
            json={},
            params={"include_url": True},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) >= 2

        # Verify signed_url is included
        for file_data in data["data"]:
            assert "signed_url" in file_data
            assert file_data["signed_url"] == "https://signed.url/audio/test.mp3"

        # Verify storage was called
        mock_storage.assert_called_once()
        assert mock_storage_instance.get_signed_url.call_count >= 2

    @patch("app.api.routes.stt_evaluations.files.get_cloud_storage")
    def test_list_specific_audio_files_by_ids(
        self,
        mock_storage: MagicMock,
        db: Session,
        audio_file_1: File,
        audio_file_2: File,
        audio_file_3: File,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing specific audio files by their IDs."""
        file_ids = [audio_file_1.id, audio_file_2.id]

        response = client.post(
            f"{settings.API_V1_STR}/stt-evaluations/files/list",
            json={"file_ids": file_ids},
            params={"include_url": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 2

        returned_ids = {file_data["id"] for file_data in data["data"]}
        assert returned_ids == {audio_file_1.id, audio_file_2.id}

    @patch("app.api.routes.stt_evaluations.files.get_cloud_storage")
    def test_list_specific_audio_files_with_signed_urls(
        self,
        mock_storage: MagicMock,
        db: Session,
        audio_file_1: File,
        audio_file_2: File,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing specific audio files with signed URLs."""
        mock_storage_instance = MagicMock()
        mock_storage_instance.get_signed_url.side_effect = (
            lambda url: f"https://signed.url/{url}"
        )
        mock_storage.return_value = mock_storage_instance

        file_ids = [audio_file_1.id, audio_file_2.id]

        response = client.post(
            f"{settings.API_V1_STR}/stt-evaluations/files/list",
            json={"file_ids": file_ids},
            params={"include_url": True},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 2

        # Verify each file has a signed URL
        for file_data in data["data"]:
            assert "signed_url" in file_data
            assert file_data["signed_url"] is not None
            assert file_data["signed_url"].startswith("https://signed.url/")

    def test_list_audio_files_empty_list(
        self,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test listing audio files with empty file_ids list."""
        response = client.post(
            f"{settings.API_V1_STR}/stt-evaluations/files/list",
            json={"file_ids": []},
            params={"include_url": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        # Should return all files when file_ids is empty
        assert isinstance(data["data"], list)

    def test_list_audio_files_requires_authentication(self) -> None:
        """Test that listing audio files requires authentication."""
        response = client.post(
            f"{settings.API_V1_STR}/stt-evaluations/files/list",
            json={"file_ids": None},
        )

        assert response.status_code == 403


class TestGetAudioFile:
    """Test cases for GET /stt-evaluations/files/{file_id} endpoint."""

    @patch("app.api.routes.stt_evaluations.files.get_cloud_storage")
    def test_get_audio_file_without_signed_url(
        self,
        mock_storage: MagicMock,
        db: Session,
        audio_file_1: File,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a single audio file without signed URL."""
        response = client.get(
            f"{settings.API_V1_STR}/stt-evaluations/files/{audio_file_1.id}",
            params={"include_url": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["id"] == audio_file_1.id
        assert data["data"]["filename"] == audio_file_1.filename
        assert data["data"]["object_store_url"] == audio_file_1.object_store_url
        assert data["data"]["signed_url"] is None

        # Verify storage was not called
        mock_storage.assert_not_called()

    @patch("app.api.routes.stt_evaluations.files.get_cloud_storage")
    def test_get_audio_file_with_signed_url(
        self,
        mock_storage: MagicMock,
        db: Session,
        audio_file_1: File,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a single audio file with signed URL."""
        mock_storage_instance = MagicMock()
        mock_storage_instance.get_signed_url.return_value = (
            "https://signed.url/audio/test1.mp3"
        )
        mock_storage.return_value = mock_storage_instance

        response = client.get(
            f"{settings.API_V1_STR}/stt-evaluations/files/{audio_file_1.id}",
            params={"include_url": True},
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["id"] == audio_file_1.id
        assert data["data"]["signed_url"] == "https://signed.url/audio/test1.mp3"

        # Verify storage was called
        mock_storage.assert_called_once()
        mock_storage_instance.get_signed_url.assert_called_once_with(
            audio_file_1.object_store_url
        )

    def test_get_audio_file_not_found(
        self,
        db: Session,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test getting a non-existent audio file returns 404."""
        non_existent_id = 999999

        response = client.get(
            f"{settings.API_V1_STR}/stt-evaluations/files/{non_existent_id}",
            params={"include_url": False},
            headers=user_api_key_header,
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_get_audio_file_requires_authentication(self, audio_file_1: File) -> None:
        """Test that getting an audio file requires authentication."""
        response = client.get(
            f"{settings.API_V1_STR}/stt-evaluations/files/{audio_file_1.id}",
        )

        assert response.status_code == 403

    @patch("app.api.routes.stt_evaluations.files.get_cloud_storage")
    def test_get_audio_file_default_include_url_is_true(
        self,
        mock_storage: MagicMock,
        db: Session,
        audio_file_1: File,
        user_api_key_header: dict[str, str],
    ) -> None:
        """Test that include_url defaults to True."""
        mock_storage_instance = MagicMock()
        mock_storage_instance.get_signed_url.return_value = (
            "https://signed.url/audio/test1.mp3"
        )
        mock_storage.return_value = mock_storage_instance

        # Don't specify include_url parameter
        response = client.get(
            f"{settings.API_V1_STR}/stt-evaluations/files/{audio_file_1.id}",
            headers=user_api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        # With default True, signed_url should be included
        assert data["data"]["signed_url"] is not None

    def test_get_audio_file_validates_project_ownership(
        self,
        db: Session,
        audio_file_1: File,
        superuser_api_key_header: dict[str, str],
    ) -> None:
        """Test that users can only access files from their own projects."""
        # Superuser trying to access user's file from different project
        response = client.get(
            f"{settings.API_V1_STR}/stt-evaluations/files/{audio_file_1.id}",
            params={"include_url": False},
            headers=superuser_api_key_header,
        )

        # Should return 404 as the file doesn't belong to superuser's project
        assert response.status_code == 404
