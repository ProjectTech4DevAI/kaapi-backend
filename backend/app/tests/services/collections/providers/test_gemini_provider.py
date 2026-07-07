"""Tests for GeminiAIStudioProvider — upload/import/create/delete against a mocked
genai client, with document persistence exercised on the real transactional db.
"""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai.types import FileState
from sqlmodel import Session

from app.crud import DocumentCrud
from app.models import Collection
from app.services.collections.helpers import get_service_name
from app.services.collections.providers.gemini import (
    GOOGLE_AISTUDIO_PROVIDER,
    GeminiAIStudioProvider,
)
from app.tests.utils.document import DocumentStore
from app.tests.utils.utils import get_project

STORE_NAME = "fileSearchStores/store-abc"
UPLOADED_NAME = "files/uploaded-123"


def _fast_time() -> MagicMock:
    mock_time = MagicMock()
    mock_time.monotonic.return_value = 0.0
    mock_time.sleep.return_value = None
    return mock_time


def _storage(content: bytes = b"file body") -> MagicMock:
    storage = MagicMock()
    storage.stream.return_value = BytesIO(content)
    return storage


def _uploaded(state=FileState.ACTIVE, name: str = UPLOADED_NAME, error=None):
    return SimpleNamespace(state=state, name=name, error=error)


def _patch_session(db: Session):
    patcher = patch("app.services.collections.providers.gemini.Session")
    mock_ctor = patcher.start()
    mock_ctor.return_value.__enter__.return_value = db
    mock_ctor.return_value.__exit__.return_value = False
    return patcher


class TestUploadFiles:
    def test_persists_file_id_and_backfills_size_on_real_db(self, db: Session) -> None:
        project = get_project(db)
        store = DocumentStore(db=db, project_id=project.id)
        doc = store.put()
        doc.fname = "report.pdf"
        doc.file_size_kb = None

        client = MagicMock()
        client.files.upload.return_value = _uploaded()
        provider = GeminiAIStudioProvider(client=client)

        content = b"x" * 3072
        patcher = _patch_session(db)
        try:
            provider.upload_files(_storage(content), [doc], project_id=project.id)
        finally:
            patcher.stop()

        _, kwargs = client.files.upload.call_args
        cfg = kwargs["config"]
        assert cfg.mime_type == "application/pdf"
        assert cfg.display_name == "report.pdf"

        persisted = DocumentCrud(db, project.id).read_one(doc.id)
        assert persisted.file_id[GOOGLE_AISTUDIO_PROVIDER] == UPLOADED_NAME
        assert persisted.file_size_kb == round(len(content) / 1024, 2)

    def test_extensionless_fname_raises_before_upload(self, db: Session) -> None:
        project = get_project(db)
        store = DocumentStore(db=db, project_id=project.id)
        doc = store.put()
        doc.fname = "no_extension"

        client = MagicMock()
        provider = GeminiAIStudioProvider(client=client)

        with pytest.raises(ValueError, match="MIME type"):
            provider.upload_files(_storage(), [doc], project_id=project.id)

        client.files.upload.assert_not_called()

    def test_db_persist_failure_rolls_back_gemini_file(self, db: Session) -> None:
        project = get_project(db)
        store = DocumentStore(db=db, project_id=project.id)
        doc = store.put()
        doc.fname = "notes.txt"

        client = MagicMock()
        client.files.upload.return_value = _uploaded()
        provider = GeminiAIStudioProvider(client=client)

        patcher = _patch_session(db)
        crud_patcher = patch(
            "app.services.collections.providers.gemini.DocumentCrud",
            side_effect=RuntimeError("DB down"),
        )
        try:
            with crud_patcher:
                with pytest.raises(RuntimeError, match="DB down"):
                    provider.upload_files(_storage(), [doc], project_id=project.id)
        finally:
            patcher.stop()

        client.files.delete.assert_called_once_with(name=UPLOADED_NAME)
        assert GOOGLE_AISTUDIO_PROVIDER not in doc.file_id


class TestWaitUntilActive:
    def test_processing_then_active_completes(self) -> None:
        client = MagicMock()
        client.files.get.return_value = _uploaded(state=FileState.ACTIVE)
        provider = GeminiAIStudioProvider(client=client)

        with patch("app.services.collections.providers.gemini.time", _fast_time()):
            provider._wait_until_active(
                _uploaded(state=FileState.PROCESSING), MagicMock(id="doc-1")
            )

        client.files.get.assert_called_once_with(name=UPLOADED_NAME)

    def test_failed_state_raises(self) -> None:
        provider = GeminiAIStudioProvider(client=MagicMock())

        with patch("app.services.collections.providers.gemini.time", _fast_time()):
            with pytest.raises(InterruptedError, match="processing failed"):
                provider._wait_until_active(
                    _uploaded(state=FileState.FAILED, error="corrupt"),
                    MagicMock(id="doc-1"),
                )

    def test_timeout_raises(self) -> None:
        client = MagicMock()
        client.files.get.return_value = _uploaded(state=FileState.PROCESSING)
        provider = GeminiAIStudioProvider(client=client)

        mock_time = MagicMock()
        mock_time.monotonic.side_effect = [0.0, 10_000.0]
        mock_time.sleep.return_value = None
        with patch("app.services.collections.providers.gemini.time", mock_time):
            with pytest.raises(InterruptedError, match="file-active-timeout"):
                provider._wait_until_active(
                    _uploaded(state=FileState.PROCESSING), MagicMock(id="doc-1")
                )


class TestCreateVectorStore:
    def test_returns_store_name(self) -> None:
        provider = GeminiAIStudioProvider(client=MagicMock())

        with patch(
            "app.services.collections.providers.gemini.GeminiFileSearchStoreCrud"
        ) as crud_cls:
            crud_cls.return_value.create.return_value = STORE_NAME
            assert provider.create_vector_store() == STORE_NAME


class TestCreate:
    def test_imports_each_doc_and_returns_collection(self) -> None:
        provider = GeminiAIStudioProvider(client=MagicMock())
        docs = [
            SimpleNamespace(file_id={GOOGLE_AISTUDIO_PROVIDER: "files/a"}),
            SimpleNamespace(file_id={GOOGLE_AISTUDIO_PROVIDER: "files/b"}),
        ]

        with patch(
            "app.services.collections.providers.gemini.GeminiFileSearchStoreCrud"
        ) as crud_cls:
            store_crud = crud_cls.return_value
            collection = provider.create(docs, vector_store_id=STORE_NAME)

        assert store_crud.import_document.call_count == 2
        store_crud.import_document.assert_any_call(STORE_NAME, "files/a")
        store_crud.import_document.assert_any_call(STORE_NAME, "files/b")
        assert isinstance(collection, Collection)
        assert collection.knowledge_base_id == STORE_NAME
        assert collection.knowledge_base_provider == get_service_name(
            GOOGLE_AISTUDIO_PROVIDER
        )


class TestDelete:
    def test_delegates_to_store_crud(self) -> None:
        provider = GeminiAIStudioProvider(client=MagicMock())
        collection = Collection(
            knowledge_base_id=STORE_NAME,
            knowledge_base_provider=get_service_name(GOOGLE_AISTUDIO_PROVIDER),
        )

        with patch(
            "app.services.collections.providers.gemini.GeminiFileSearchStoreCrud"
        ) as crud_cls:
            provider.delete(collection)

        crud_cls.return_value.delete.assert_called_once_with(STORE_NAME)
