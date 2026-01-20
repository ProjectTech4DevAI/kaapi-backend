from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlmodel import Session
from openai import OpenAI

from app.models.project import Project
from app.models import AssistantCreate, AssistantUpdate
from app.crud.assistants import (
    sync_assistant,
    create_assistant,
    update_assistant,
    delete_assistant,
    get_assistant_by_id,
    get_assistants_by_project,
)
from app.tests.utils.llm_provider import mock_openai_assistant
from app.tests.utils.utils import (
    get_project,
    get_assistant,
    get_non_existent_id,
)


class TestSyncAssistant:
    def test_sync_assistant_success(self, db: Session) -> None:
        project = get_project(db)
        openai_assistant = mock_openai_assistant(
            assistant_id="asst_success",
            vector_store_ids=["vs_1", "vs_2"],
            max_num_results=20,
        )

        result = sync_assistant(
            db, project.organization_id, project.id, openai_assistant
        )

        assert result.assistant_id == openai_assistant.id
        assert result.project_id == project.id
        assert result.organization_id == project.organization_id
        assert result.name == openai_assistant.name
        assert result.instructions == openai_assistant.instructions
        assert result.model == openai_assistant.model
        assert result.vector_store_ids == ["vs_1", "vs_2"]
        assert result.temperature == openai_assistant.temperature
        assert result.max_num_results == 20

    def test_sync_assistant_already_exists(self, db: Session) -> None:
        project = get_project(db)
        openai_assistant = mock_openai_assistant(
            assistant_id="asst_exists",
        )

        sync_assistant(db, project.organization_id, project.id, openai_assistant)

        with pytest.raises(HTTPException) as exc_info:
            sync_assistant(db, project.organization_id, project.id, openai_assistant)

        assert exc_info.value.status_code == 409
        assert "already exists" in exc_info.value.detail

    def test_sync_assistant_no_instructions(self, db: Session) -> None:
        project = get_project(db)
        openai_assistant = mock_openai_assistant(
            assistant_id="asst_no_instructions",
        )
        openai_assistant.instructions = None

        with pytest.raises(HTTPException) as exc_info:
            sync_assistant(db, project.organization_id, project.id, openai_assistant)

        assert exc_info.value.status_code == 400
        assert "no instruction" in exc_info.value.detail

    def test_sync_assistant_no_name(self, db: Session) -> None:
        project = get_project(db)
        openai_assistant = mock_openai_assistant(
            assistant_id="asst_no_name",
        )
        openai_assistant.name = None

        result = sync_assistant(
            db, project.organization_id, project.id, openai_assistant
        )

        assert result.name == openai_assistant.id
        assert result.assistant_id == openai_assistant.id
        assert result.project_id == project.id

    def test_sync_assistant_no_vector_stores(self, db: Session) -> None:
        project = get_project(db)
        openai_assistant = mock_openai_assistant(
            assistant_id="asst_no_vectors", vector_store_ids=None
        )

        result = sync_assistant(
            db, project.organization_id, project.id, openai_assistant
        )

        assert result.vector_store_ids == []
        assert result.assistant_id == openai_assistant.id
        assert result.project_id == project.id

    def test_sync_assistant_no_tools(self, db: Session) -> None:
        project = get_project(db)
        openai_assistant = mock_openai_assistant(assistant_id="asst_no_tools")

        openai_assistant.tool_resources = None
        result = sync_assistant(
            db, project.organization_id, project.id, openai_assistant
        )

        assert result.vector_store_ids == []  # Default value
        assert result.assistant_id == openai_assistant.id
        assert result.project_id == project.id

    def test_sync_assistant_no_tool_resources(self, db: Session) -> None:
        project = get_project(db)
        openai_assistant = mock_openai_assistant(
            assistant_id="asst_no_tool_resources",
        )
        openai_assistant.tools = None

        result = sync_assistant(
            db, project.organization_id, project.id, openai_assistant
        )

        assert result.max_num_results == 20
        assert result.assistant_id == openai_assistant.id
        assert result.project_id == project.id


class TestAssistantCrud:
    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_create_assistant_success(
        self, mock_vector_store_ids_exist, db: Session
    ) -> None:
        """Assistant is created when vector store IDs are valid"""
        project = get_project(db)
        assistant_create = AssistantCreate(
            name="Test Assistant",
            instructions="Test instructions",
            model="gpt-4o",
            vector_store_ids=["vs_1", "vs_2"],
            temperature=0.7,
            max_num_results=10,
        )
        client = OpenAI(api_key="test_key")
        mock_vector_store_ids_exist.return_value = None
        result = create_assistant(
            db, client, assistant_create, project.id, project.organization_id
        )

        assert result.name == assistant_create.name
        assert result.instructions == assistant_create.instructions
        assert result.model == assistant_create.model
        assert result.vector_store_ids == assistant_create.vector_store_ids
        assert result.temperature == assistant_create.temperature
        assert result.max_num_results == assistant_create.max_num_results

    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_create_assistant_with_id_success(
        self, mock_vector_store_ids_exist, db: Session
    ) -> None:
        """Assistant is created with a specific ID when vector store IDs are valid"""
        project = get_project(db)
        assistant_create = AssistantCreate(
            name="Test Assistant",
            instructions="Test instructions",
            model="gpt-4o",
            vector_store_ids=["vs_1", "vs_2"],
            temperature=0.7,
            max_num_results=10,
            assistant_id="test_assistant_id",
        )
        client = OpenAI(api_key="test_key")
        mock_vector_store_ids_exist.return_value = None
        result = create_assistant(
            db, client, assistant_create, project.id, project.organization_id
        )

        assert result.name == assistant_create.name
        assert result.instructions == assistant_create.instructions
        assert result.model == assistant_create.model
        assert result.vector_store_ids == assistant_create.vector_store_ids
        assert result.temperature == assistant_create.temperature
        assert result.max_num_results == assistant_create.max_num_results
        assert result.assistant_id == assistant_create.assistant_id

    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_create_assistant_duplicate_assistant_id(
        self, mock_vector_store_ids_exist, db: Session
    ) -> None:
        """Creating an assistant with a duplicate assistant_id should raise 409 Conflict"""
        project = get_project(db)

        assistant_id = "duplicate_id"
        assistant_create_1 = AssistantCreate(
            name="Assistant One",
            instructions="First assistant instructions",
            model="gpt-4o",
            vector_store_ids=[],
            assistant_id=assistant_id,
        )
        client = OpenAI(api_key="test_key")
        mock_vector_store_ids_exist.return_value = None
        create_assistant(
            db, client, assistant_create_1, project.id, project.organization_id
        )

        assistant_create_2 = AssistantCreate(
            name="Assistant Two",
            instructions="Second assistant instructions",
            model="gpt-4o",
            vector_store_ids=[],
            assistant_id=assistant_id,
        )

        with pytest.raises(HTTPException) as exc_info:
            create_assistant(
                db, client, assistant_create_2, project.id, project.organization_id
            )

        assert exc_info.value.status_code == 409
        assert f"Assistant with ID {assistant_id} already exists." in str(
            exc_info.value.detail
        )

    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_create_assistant_vector_store_invalid(
        self, mock_vector_store_ids_exist, db: Session
    ) -> None:
        """Should raise HTTPException when vector store IDs are invalid"""
        project = get_project(db)
        assistant_create = AssistantCreate(
            name="Invalid VS Assistant",
            instructions="Some instructions",
            model="gpt-4o",
            vector_store_ids=["invalid_vs"],
            temperature=0.5,
            max_num_results=5,
        )
        client = OpenAI(api_key="test_key")
        error_message = "Vector store ID vs_test_999 not found in OpenAI."
        mock_vector_store_ids_exist.side_effect = HTTPException(
            status_code=400, detail=error_message
        )

        with pytest.raises(HTTPException) as exc_info:
            create_assistant(
                db, client, assistant_create, project.id, project.organization_id
            )

        assert exc_info.value.status_code == 400
        assert error_message in exc_info.value.detail

    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_update_assistant_success(
        self, mock_verify_vector_store_ids, db: Session
    ) -> None:
        """Assistant is updated successfully with valid data"""
        assistant = get_assistant(db)

        assistant_update = AssistantUpdate(
            name="Updated Assistant",
            instructions="Updated instructions",
            temperature=0.9,
            max_num_results=15,
            vector_store_ids_add=["vs_2"],
        )
        mock_verify_vector_store_ids.return_value = None

        client = OpenAI(api_key="test_key")
        result = update_assistant(
            db, client, assistant.assistant_id, assistant.project_id, assistant_update
        )

        assert result.name == assistant_update.name
        assert result.instructions == assistant_update.instructions
        assert result.temperature == assistant_update.temperature
        assert result.max_num_results == assistant_update.max_num_results
        assert "vs_2" in result.vector_store_ids
        assert mock_verify_vector_store_ids.called

    def test_update_assistant_not_found(self, db: Session) -> None:
        """Should raise HTTPException when assistant is not found"""
        project = get_project(db)
        assistant_id = "non_existent_assistant_id"
        assistant_update = AssistantUpdate(name="Updated Assistant")

        client = OpenAI(api_key="test_key")
        with pytest.raises(HTTPException) as exc_info:
            update_assistant(db, client, assistant_id, project.id, assistant_update)

        assert exc_info.value.status_code == 404
        assert "Assistant not found" in exc_info.value.detail

    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_update_assistant_invalid_vector_store(
        self, mock_verify_vector_store_ids, db: Session
    ) -> None:
        """Should raise HTTPException when vector store IDs are invalid"""
        assistant = get_assistant(db)
        error_message = "Vector store ID vs_invalid not found in OpenAI."
        mock_verify_vector_store_ids.side_effect = HTTPException(
            status_code=400, detail=error_message
        )
        assistant_update = AssistantUpdate(vector_store_ids_add=["vs_invalid"])

        client = OpenAI(api_key="test_key")
        with pytest.raises(HTTPException) as exc_info:
            update_assistant(
                db,
                client,
                assistant.assistant_id,
                assistant.project_id,
                assistant_update,
            )

        assert exc_info.value.status_code == 400
        assert error_message in exc_info.value.detail

    def test_update_assistant_conflicting_vector_store_ids(self, db: Session) -> None:
        """Should raise HTTPException with 400 when vector store IDs are both added and removed"""
        assistant = get_assistant(db)
        conflicting_id = "vs_1"
        assistant_update = AssistantUpdate(
            vector_store_ids_add=[conflicting_id],
            vector_store_ids_remove=[conflicting_id],
        )

        client = OpenAI(api_key="test_key")

        with pytest.raises(HTTPException) as exc_info:
            update_assistant(
                session=db,
                openai_client=client,
                assistant_id=assistant.assistant_id,
                project_id=assistant.project_id,
                assistant_update=assistant_update,
            )

        expected_error = (
            f"Conflicting vector store IDs in add/remove: {{{conflicting_id!r}}}."
        )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == expected_error

    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_update_assistant_partial_update(
        self, mock_verify_vector_store_ids, db: Session
    ) -> None:
        """Assistant is updated successfully with partial fields"""
        assistant = get_assistant(db)
        mock_verify_vector_store_ids.return_value = None
        assistant_update = AssistantUpdate(
            name="Partially Updated Assistant", vector_store_ids_add=["vs_3"]
        )

        client = OpenAI(api_key="test_key")
        result = update_assistant(
            db, client, assistant.assistant_id, assistant.project_id, assistant_update
        )

        assert result.name == assistant_update.name
        assert result.instructions == assistant.instructions
        assert result.temperature == assistant.temperature
        assert result.max_num_results == assistant.max_num_results
        assert "vs_3" in result.vector_store_ids
        assert mock_verify_vector_store_ids.called

    def test_delete_assistant_success(self, db: Session) -> None:
        """Assistant is soft deleted successfully"""
        assistant = get_assistant(db)

        result = delete_assistant(db, assistant.assistant_id, assistant.project_id)

        assert result.is_deleted is True
        assert result.deleted_at is not None
        with pytest.raises(ValueError) as exc_info:
            get_assistant(db, name=assistant.name)
        assert "No active assistants found" in str(exc_info.value)

    def test_delete_assistant_not_found(self, db: Session) -> None:
        """Should raise HTTPException when assistant is not found"""
        project = get_project(db)
        assistant_id = "non_existent_assistant_id"

        with pytest.raises(HTTPException) as exc_info:
            delete_assistant(db, assistant_id, project.id)

        assert exc_info.value.status_code == 404
        assert "Assistant not found" in exc_info.value.detail

    def test_get_assistant_by_id_success(self, db: Session) -> None:
        """Assistant is retrieved successfully by ID and project ID"""
        assistant = get_assistant(db)

        result = get_assistant_by_id(db, assistant.assistant_id, assistant.project_id)

        assert result is not None
        assert result.assistant_id == assistant.assistant_id
        assert result.project_id == assistant.project_id
        assert result.is_deleted is False

    def test_get_assistant_by_id_not_found(self, db: Session) -> None:
        """Returns None when assistant is not found"""
        project = get_project(db)
        non_existent_id = "NonExistentAssistantId"

        result = get_assistant_by_id(db, non_existent_id, project.id)

        assert result is None

    def test_get_assistant_by_id_deleted(self, db: Session) -> None:
        """Returns None when assistant is soft deleted"""
        assistant = get_assistant(db)
        # Soft delete the assistant
        delete_assistant(db, assistant.assistant_id, assistant.project_id)

        result = get_assistant_by_id(db, assistant.assistant_id, assistant.project_id)

        assert result is None

    @patch("app.crud.assistants.verify_vector_store_ids_exist")
    def test_get_assistants_by_project_success(
        self, mock_vector_store_ids_exist, db: Session
    ) -> None:
        """Returns all assistants for a project"""
        project = get_project(db)
        client = OpenAI(api_key="test_key")
        mock_vector_store_ids_exist.return_value = None

        assistant_create1 = AssistantCreate(
            name="Assistant 1",
            instructions="Instructions 1",
            model="gpt-4o",
            vector_store_ids=["vs_1", "vs_2"],
            temperature=0.5,
            max_num_results=10,
        )
        assistant_create2 = AssistantCreate(
            name="Assistant 2",
            instructions="Instructions 2",
            model="gpt-4o",
            vector_store_ids=["vs_3"],
            temperature=0.6,
            max_num_results=20,
        )

        assistant1 = create_assistant(
            db, client, assistant_create1, project.id, project.organization_id
        )
        assistant2 = create_assistant(
            db, client, assistant_create2, project.id, project.organization_id
        )

        result = get_assistants_by_project(db, project.id)

        assert len(result) >= 2
        assistant_ids = [a.assistant_id for a in result]
        assert assistant1.assistant_id in assistant_ids
        assert assistant2.assistant_id in assistant_ids
        for assistant in result:
            assert assistant.project_id == project.id
            assert assistant.is_deleted is False

    def test_get_assistants_by_project_empty(self, db: Session) -> None:
        """Returns empty list when project has no assistants"""
        project = get_project(db)
        non_existent_project_id = get_non_existent_id(db, Project)

        result = get_assistants_by_project(db, non_existent_project_id)

        assert result == []
