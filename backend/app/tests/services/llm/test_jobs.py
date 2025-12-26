import pytest
from unittest.mock import patch, MagicMock

from fastapi import HTTPException
from sqlmodel import Session, select

from app.crud import JobCrud
from app.crud.config import ConfigVersionCrud
from app.utils import APIResponse
from app.models import ConfigVersion, JobStatus, JobType
from app.models.llm import (
    LLMCallRequest,
    NativeCompletionConfig,
    QueryParams,
    LLMCallResponse,
    LLMResponse,
    LLMOutput,
    Usage,
    KaapiLLMParams,
    KaapiCompletionConfig,
)
from app.models.llm.request import ConfigBlob, LLMCallConfig
from app.services.llm.jobs import (
    start_job,
    handle_job_error,
    execute_job,
    resolve_config_blob,
)
from app.tests.utils.utils import get_project
from app.tests.utils.test_data import create_test_config


class TestStartJob:
    """Test cases for the start_job function."""

    @pytest.fixture
    def llm_call_request(self):
        return LLMCallRequest(
            query=QueryParams(input="Test query"),
            config=LLMCallConfig(
                blob=ConfigBlob(
                    completion=NativeCompletionConfig(
                        provider="openai-native",
                        params={"model": "gpt-4"},
                    )
                )
            ),
        )

    def test_start_job_success(self, db: Session, llm_call_request: LLMCallRequest):
        """Test successful job creation and Celery task scheduling."""

        request = llm_call_request
        project = get_project(db)

        with patch("app.services.llm.jobs.start_high_priority_job") as mock_schedule:
            mock_schedule.return_value = "fake-task-id-123"

            job_id = start_job(db, request, project.id, project.organization_id)

            job_crud = JobCrud(session=db)
            job = job_crud.get(job_id)
            assert job is not None
            assert job.job_type == JobType.LLM_API
            assert job.status == JobStatus.PENDING
            assert job.trace_id is not None

            mock_schedule.assert_called_once()
            _, kwargs = mock_schedule.call_args
            assert kwargs["function_path"] == "app.services.llm.jobs.execute_job"
            assert kwargs["project_id"] == project.id
            assert kwargs["organization_id"] == project.organization_id
            assert kwargs["job_id"] == str(job_id)
            assert "request_data" in kwargs

    def test_start_job_celery_scheduling_fails(
        self, db: Session, llm_call_request: LLMCallRequest
    ):
        """Test start_job when Celery task scheduling fails."""
        project = get_project(db)

        with patch("app.services.llm.jobs.start_high_priority_job") as mock_schedule:
            mock_schedule.side_effect = Exception("Celery connection failed")

            with pytest.raises(HTTPException) as exc_info:
                start_job(db, llm_call_request, project.id, project.organization_id)

            assert exc_info.value.status_code == 500
            assert "Internal server error while executing LLM call" in str(
                exc_info.value.detail
            )

    def test_start_job_exception_during_job_creation(
        self, db: Session, llm_call_request: LLMCallRequest
    ):
        """Test handling of exceptions during job creation in database."""
        project = get_project(db)

        with patch("app.services.llm.jobs.JobCrud") as mock_job_crud:
            mock_crud_instance = MagicMock()
            mock_crud_instance.create.side_effect = Exception(
                "Database connection failed"
            )
            mock_job_crud.return_value = mock_crud_instance

            with pytest.raises(Exception) as exc_info:
                start_job(db, llm_call_request, project.id, project.organization_id)

            assert "Database connection failed" in str(exc_info.value)


class TestHandleJobError:
    """Test cases for the handle_job_error function."""

    def test_handle_job_error(self, db: Session):
        """Test handle_job_error successfully sends callback and updates job status."""
        job_crud = JobCrud(session=db)
        job = job_crud.create(job_type=JobType.LLM_API, trace_id="test-trace")
        db.commit()

        callback_url = "https://example.com/callback"
        callback_response = APIResponse.failure_response(error="Test error occurred")

        with patch("app.services.llm.jobs.Session") as mock_session_class, patch(
            "app.services.llm.jobs.send_callback"
        ) as mock_send_callback:
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            result = handle_job_error(
                job_id=job.id,
                callback_url=callback_url,
                callback_response=callback_response,
            )

            mock_send_callback.assert_called_once()
            call_args = mock_send_callback.call_args
            assert call_args[1]["callback_url"] == callback_url

            callback_data = call_args[1]["data"]
            assert callback_data["success"] is False
            assert callback_data["error"] == callback_response.error
            assert callback_data["data"] is None

            db.refresh(job)
            assert job.status == JobStatus.FAILED
            assert job.error_message == callback_response.error

            assert result["success"] is False
            assert result["error"] == callback_response.error
            assert result["data"] is None

    def test_handle_job_error_without_callback_url(self, db: Session):
        """Test handle_job_error updates job when no callback URL provided."""
        job_crud = JobCrud(session=db)
        job = job_crud.create(job_type=JobType.LLM_API, trace_id="test-trace")
        db.commit()

        callback_response = APIResponse.failure_response(error="Test error occurred")

        with patch("app.services.llm.jobs.Session") as mock_session_class, patch(
            "app.services.llm.jobs.send_callback"
        ) as mock_send_callback:
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            result = handle_job_error(
                job_id=job.id, callback_url=None, callback_response=callback_response
            )

            mock_send_callback.assert_not_called()

            db.refresh(job)
            assert job.status == JobStatus.FAILED
            assert job.error_message == callback_response.error

            # Verify return value structure
            assert result["success"] is False
            assert result["error"] == callback_response.error

    def test_handle_job_error_callback_failure_still_updates_job(self, db: Session):
        """Test that job is updated even if callback sending fails."""
        job_crud = JobCrud(session=db)
        job = job_crud.create(job_type=JobType.LLM_API, trace_id="test-trace")
        db.commit()

        callback_url = "https://example.com/callback"
        callback_response = APIResponse.failure_response(
            error="Test error with callback failure"
        )

        with patch("app.services.llm.jobs.Session") as mock_session_class, patch(
            "app.services.llm.jobs.send_callback"
        ) as mock_send_callback:
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            mock_send_callback.side_effect = Exception("Callback service unavailable")

            with pytest.raises(Exception) as exc_info:
                handle_job_error(
                    job_id=job.id,
                    callback_url=callback_url,
                    callback_response=callback_response,
                )

            assert "Callback service unavailable" in str(exc_info.value)


class TestExecuteJob:
    """Test suite for execute_job."""

    @pytest.fixture
    def job_for_execution(self, db: Session):
        job = JobCrud(session=db).create(
            job_type=JobType.LLM_API, trace_id="test-trace"
        )
        db.commit()
        return job

    @pytest.fixture
    def request_data(self):
        return {
            "query": {"input": "Test query"},
            "config": {
                "blob": {
                    "completion": {
                        "provider": "openai-native",
                        "params": {"model": "gpt-4"},
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

    @pytest.fixture
    def mock_llm_response(self):
        return LLMCallResponse(
            response=LLMResponse(
                provider_response_id="resp-123",
                conversation_id=None,
                model="gpt-4",
                provider="openai",
                output=LLMOutput(text="Test response"),
            ),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            provider_raw_response=None,
        )

    @pytest.fixture
    def job_env(self, db, mock_llm_response):
        """Set up common environment with patched Session, provider, and callback."""
        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.get_llm_provider") as mock_get_provider,
            patch("app.services.llm.jobs.send_callback") as mock_send_callback,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            # Mock LLM provider
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider

            # Provide everything needed to tests
            yield {
                "session": mock_session_class,
                "get_provider": mock_get_provider,
                "provider": mock_provider,
                "send_callback": mock_send_callback,
                "mock_llm_response": mock_llm_response,
            }

    def _execute_job(self, job, db, request_data):
        project = get_project(db)
        return execute_job(
            request_data=request_data,
            project_id=project.id,
            organization_id=project.organization_id,
            job_id=str(job.id),
            task_id="task-123",
            task_instance=None,
        )

    def test_success_with_callback(self, db, job_env, job_for_execution, request_data):
        """Successful execution with callback."""
        env = job_env
        request_data["callback_url"] = "https://example.com/callback"

        env["provider"].execute.return_value = (env["mock_llm_response"], None)
        result = self._execute_job(job_for_execution, db, request_data)

        env["get_provider"].assert_called_once()
        env["send_callback"].assert_called_once()
        assert result["success"]
        db.refresh(job_for_execution)
        assert job_for_execution.status == JobStatus.SUCCESS

    def test_success_without_callback(
        self, db, job_env, job_for_execution, request_data
    ):
        """Successful execution without callback."""
        env = job_env
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        result = self._execute_job(job_for_execution, db, request_data)

        env["send_callback"].assert_not_called()
        assert result["success"]
        db.refresh(job_for_execution)
        assert job_for_execution.status == JobStatus.SUCCESS

    def test_provider_returns_error(self, db, job_env, job_for_execution, request_data):
        """Provider returns error (no callback)."""
        env = job_env
        env["provider"].execute.return_value = (None, "API rate limit exceeded")

        result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "rate limit" in result["error"]
        db.refresh(job_for_execution)
        assert job_for_execution.status == JobStatus.FAILED

    def test_provider_error_with_callback(
        self, db, job_env, job_for_execution, request_data
    ):
        """Provider returns error (with callback)."""
        env = job_env
        request_data["callback_url"] = "https://example.com/callback"
        env["provider"].execute.return_value = (None, "Invalid API key")

        result = self._execute_job(job_for_execution, db, request_data)

        env["send_callback"].assert_called_once()
        assert not result["success"]

    def test_exception_during_execution(
        self, db, job_env, job_for_execution, request_data
    ):
        """Unhandled exception in provider execution."""
        env = job_env
        env["provider"].execute.side_effect = Exception("Network timeout")

        result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "Unexpected error occurred" in result["error"]

    def test_exception_during_provider_retrieval(
        self, db, job_env, job_for_execution, request_data
    ):
        """Provider not configured exception."""
        env = job_env
        env["get_provider"].side_effect = ValueError("Provider not configured")

        result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "Provider not configured" in result["error"]

    def test_metadata_in_callback_response(
        self, db, job_env, job_for_execution, request_data
    ):
        """Test that request metadata is included in callback response."""
        env = job_env
        request_data["callback_url"] = "https://example.com/callback"
        request_data["request_metadata"] = {"tracking_id": "track-123"}

        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        self._execute_job(job_for_execution, db, request_data)

        env["send_callback"].assert_called_once()
        callback_data = env["send_callback"].call_args[1]["data"]
        assert callback_data["metadata"] == {"tracking_id": "track-123"}

    def test_metadata_in_error_callback(
        self, db, job_env, job_for_execution, request_data
    ):
        """Test that request metadata is included in error callback response."""
        env = job_env
        request_data["callback_url"] = "https://example.com/callback"
        request_data["request_metadata"] = {"tracking_id": "track-456"}

        env["provider"].execute.return_value = (None, "Some provider error")

        self._execute_job(job_for_execution, db, request_data)

        env["send_callback"].assert_called_once()
        callback_data = env["send_callback"].call_args[1]["data"]
        assert callback_data["metadata"] == {"tracking_id": "track-456"}

    def test_stored_config_success(self, db, job_for_execution, mock_llm_response):
        """Test successful execution with stored config (id + version)."""
        project = get_project(db)

        # Create a real config in the database
        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-4", "temperature": 0.7},
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        # Build request data with stored config
        stored_request_data = {
            "query": {"input": "Test query"},
            "config": {
                "id": str(config.id),
                "version": 1,
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.get_llm_provider") as mock_get_provider,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            # Mock LLM provider
            mock_provider = MagicMock()
            mock_provider.execute.return_value = (mock_llm_response, None)
            mock_get_provider.return_value = mock_provider

            result = self._execute_job(job_for_execution, db, stored_request_data)

            # Verify provider was called
            mock_get_provider.assert_called_once()
            mock_provider.execute.assert_called_once()

            # Verify success
            assert result["success"]
            db.refresh(job_for_execution)
            assert job_for_execution.status == JobStatus.SUCCESS

    def test_stored_config_with_callback(
        self, db, job_for_execution, mock_llm_response
    ):
        """Test stored config with callback URL."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-3.5-turbo", "temperature": 0.5},
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        stored_request_data = {
            "query": {"input": "Test query with callback"},
            "config": {
                "id": str(config.id),
                "version": 1,
            },
            "include_provider_raw_response": False,
            "callback_url": "https://example.com/callback",
        }

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.get_llm_provider") as mock_get_provider,
            patch("app.services.llm.jobs.send_callback") as mock_send_callback,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            # Mock LLM provider
            mock_provider = MagicMock()
            mock_provider.execute.return_value = (mock_llm_response, None)
            mock_get_provider.return_value = mock_provider

            result = self._execute_job(job_for_execution, db, stored_request_data)

            # Verify callback was sent
            mock_send_callback.assert_called_once()
            callback_data = mock_send_callback.call_args[1]["data"]
            assert callback_data["success"]

            # Verify success
            assert result["success"]
            db.refresh(job_for_execution)
            assert job_for_execution.status == JobStatus.SUCCESS

    def test_stored_config_version_not_found(self, db, job_for_execution):
        """Test stored config when version doesn't exist."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-4"},
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        stored_request_data = {
            "query": {"input": "Test query"},
            "config": {
                "id": str(config.id),
                "version": 999,
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        with patch("app.services.llm.jobs.Session") as mock_session_class:
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            result = self._execute_job(job_for_execution, db, stored_request_data)

            # Verify failure
            assert not result["success"]
            assert "Failed to retrieve stored configuration" in result["error"]
            db.refresh(job_for_execution)
            assert job_for_execution.status == JobStatus.FAILED

    def test_kaapi_config_success(self, db, job_for_execution, mock_llm_response):
        """Test successful execution with Kaapi abstracted config."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                params=KaapiLLMParams(
                    model="gpt-4",
                    temperature=0.7,
                    instructions="You are a helpful assistant",
                ),
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        kaapi_request_data = {
            "query": {"input": "Test query with Kaapi config"},
            "config": {
                "id": str(config.id),
                "version": 1,
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.get_llm_provider") as mock_get_provider,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            mock_provider = MagicMock()
            mock_provider.execute.return_value = (mock_llm_response, None)
            mock_get_provider.return_value = mock_provider

            result = self._execute_job(job_for_execution, db, kaapi_request_data)

            mock_get_provider.assert_called_once()
            mock_provider.execute.assert_called_once()
            assert result["success"]
            db.refresh(job_for_execution)
            assert job_for_execution.status == JobStatus.SUCCESS

    def test_kaapi_config_with_callback(self, db, job_for_execution, mock_llm_response):
        """Test successful execution with Kaapi config and callback."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                params=KaapiLLMParams(
                    model="gpt-3.5-turbo",
                    temperature=0.5,
                ),
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        kaapi_request_data = {
            "query": {"input": "Test query with Kaapi config and callback"},
            "config": {
                "id": str(config.id),
                "version": 1,
            },
            "include_provider_raw_response": False,
            "callback_url": "https://example.com/callback",
        }

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.get_llm_provider") as mock_get_provider,
            patch("app.services.llm.jobs.send_callback") as mock_send_callback,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            mock_provider = MagicMock()
            mock_provider.execute.return_value = (mock_llm_response, None)
            mock_get_provider.return_value = mock_provider

            result = self._execute_job(job_for_execution, db, kaapi_request_data)

            mock_send_callback.assert_called_once()
            callback_data = mock_send_callback.call_args[1]["data"]
            assert callback_data["success"]
            assert result["success"]
            db.refresh(job_for_execution)
            assert job_for_execution.status == JobStatus.SUCCESS

    def test_kaapi_config_warnings_passed_through_metadata(
        self, db, job_for_execution, mock_llm_response
    ):
        """Test that warnings from Kaapi config transformation are passed through in metadata."""
        project = get_project(db)

        # Use a config that will generate warnings (temperature on reasoning model)
        config_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                params=KaapiLLMParams(
                    model="o1",  # Reasoning model
                    temperature=0.7,  # This will be suppressed with warning
                ),
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        kaapi_request_data = {
            "query": {"input": "Test query"},
            "config": {
                "id": str(config.id),
                "version": 1,
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.get_llm_provider") as mock_get_provider,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            mock_provider = MagicMock()
            mock_provider.execute.return_value = (mock_llm_response, None)
            mock_get_provider.return_value = mock_provider

            result = self._execute_job(job_for_execution, db, kaapi_request_data)

            # Verify the result includes warnings in metadata
            assert result["success"]
            assert "metadata" in result
            assert "warnings" in result["metadata"]
            assert len(result["metadata"]["warnings"]) == 1
            assert "temperature" in result["metadata"]["warnings"][0].lower()
            assert "suppressed" in result["metadata"]["warnings"][0]

    def test_kaapi_config_warnings_merged_with_existing_metadata(
        self, db, job_for_execution, mock_llm_response
    ):
        """Test that warnings are merged with existing request metadata."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                params=KaapiLLMParams(
                    model="gpt-4",  # Non-reasoning model
                    reasoning="high",  # This will be suppressed with warning
                ),
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        kaapi_request_data = {
            "query": {"input": "Test query"},
            "config": {
                "id": str(config.id),
                "version": 1,
            },
            "include_provider_raw_response": False,
            "callback_url": None,
            "request_metadata": {"tracking_id": "test-123"},
        }

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.get_llm_provider") as mock_get_provider,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            mock_provider = MagicMock()
            mock_provider.execute.return_value = (mock_llm_response, None)
            mock_get_provider.return_value = mock_provider

            result = self._execute_job(job_for_execution, db, kaapi_request_data)

            # Verify warnings are added to existing metadata
            assert result["success"]
            assert "metadata" in result
            assert result["metadata"]["tracking_id"] == "test-123"
            assert "warnings" in result["metadata"]
            assert len(result["metadata"]["warnings"]) == 1
            assert "reasoning" in result["metadata"]["warnings"][0].lower()
            assert "does not support reasoning" in result["metadata"]["warnings"][0]


class TestResolveConfigBlob:
    """Test suite for resolve_config_blob function."""

    def test_resolve_config_blob_success(self, db: Session):
        """Test successful resolution of stored config blob."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-4", "temperature": 0.8},
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        config_crud = ConfigVersionCrud(
            session=db, project_id=project.id, config_id=config.id
        )
        llm_call_config = LLMCallConfig(id=str(config.id), version=1)

        resolved_blob, error = resolve_config_blob(config_crud, llm_call_config)

        assert error is None
        assert resolved_blob is not None
        assert resolved_blob.completion.provider == "openai-native"
        assert resolved_blob.completion.params["model"] == "gpt-4"
        assert resolved_blob.completion.params["temperature"] == 0.8

    def test_resolve_config_blob_version_not_found(self, db: Session):
        """Test resolve_config_blob when version doesn't exist."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-4"},
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        config_crud = ConfigVersionCrud(
            session=db, project_id=project.id, config_id=config.id
        )
        llm_call_config = LLMCallConfig(id=str(config.id), version=999)

        resolved_blob, error = resolve_config_blob(config_crud, llm_call_config)

        assert resolved_blob is None
        assert error is not None
        assert "Failed to retrieve stored configuration" in error

    def test_resolve_config_blob_invalid_blob_data(self, db: Session):
        """Test resolve_config_blob when config blob is malformed."""

        project = get_project(db)

        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-4"},
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        # Query the config version directly from the database
        statement = select(ConfigVersion).where(ConfigVersion.config_id == config.id)
        config_version = db.exec(statement).first()

        # Manually corrupt the config_blob in the database
        # Set invalid data that can't be parsed as ConfigBlob
        config_version.config_blob = {"invalid": "structure", "missing": "completion"}
        db.add(config_version)
        db.commit()

        config_crud = ConfigVersionCrud(
            session=db, project_id=project.id, config_id=config.id
        )
        llm_call_config = LLMCallConfig(id=str(config.id), version=1)

        resolved_blob, error = resolve_config_blob(config_crud, llm_call_config)

        assert resolved_blob is None
        assert error is not None
        assert "Stored configuration blob is invalid" in error

    def test_resolve_config_blob_with_multiple_versions(self, db: Session):
        """Test resolving specific version when multiple versions exist."""
        from app.models.config import ConfigVersionCreate

        project = get_project(db)

        # Create a config with version 1
        config_blob_v1 = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-3.5-turbo", "temperature": 0.5},
            )
        )
        config = create_test_config(
            db, project_id=project.id, config_blob=config_blob_v1
        )
        db.commit()

        # Create version 2 using ConfigVersionCrud
        config_version_crud = ConfigVersionCrud(
            session=db, project_id=project.id, config_id=config.id
        )
        config_blob_v2 = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-4", "temperature": 0.9},
            )
        )
        version_create = ConfigVersionCreate(
            config_blob=config_blob_v2,
            commit_message="Updated to gpt-4",
        )
        config_version_crud.create_or_raise(version_create)
        db.commit()

        # Test resolving version 1
        llm_call_config_v1 = LLMCallConfig(id=str(config.id), version=1)
        resolved_blob_v1, error_v1 = resolve_config_blob(
            config_version_crud, llm_call_config_v1
        )

        assert error_v1 is None
        assert resolved_blob_v1 is not None
        assert resolved_blob_v1.completion.params["model"] == "gpt-3.5-turbo"
        assert resolved_blob_v1.completion.params["temperature"] == 0.5

        # Test resolving version 2
        llm_call_config_v2 = LLMCallConfig(id=str(config.id), version=2)
        resolved_blob_v2, error_v2 = resolve_config_blob(
            config_version_crud, llm_call_config_v2
        )

        assert error_v2 is None
        assert resolved_blob_v2 is not None
        assert resolved_blob_v2.completion.params["model"] == "gpt-4"
        assert resolved_blob_v2.completion.params["temperature"] == 0.9

    def test_resolve_kaapi_config_blob_success(self, db: Session):
        """Test successful resolution of stored Kaapi config blob."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                params=KaapiLLMParams(
                    model="gpt-4",
                    temperature=0.8,
                    instructions="You are a helpful assistant",
                ),
            )
        )
        config = create_test_config(db, project_id=project.id, config_blob=config_blob)
        db.commit()

        config_crud = ConfigVersionCrud(
            session=db, project_id=project.id, config_id=config.id
        )
        llm_call_config = LLMCallConfig(id=str(config.id), version=1)

        resolved_blob, error = resolve_config_blob(config_crud, llm_call_config)

        assert error is None
        assert resolved_blob is not None
        assert isinstance(resolved_blob.completion, KaapiCompletionConfig)
        assert resolved_blob.completion.provider == "openai"
        assert resolved_blob.completion.params.model == "gpt-4"
        assert resolved_blob.completion.params.temperature == 0.8
        assert (
            resolved_blob.completion.params.instructions
            == "You are a helpful assistant"
        )

    def test_resolve_both_native_and_kaapi_configs(self, db: Session):
        """Test that both native and Kaapi configs can be resolved correctly."""
        project = get_project(db)

        # Create native config
        native_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={"model": "gpt-3.5-turbo", "temperature": 0.5},
            )
        )
        native_config = create_test_config(
            db, project_id=project.id, config_blob=native_blob, use_kaapi_schema=False
        )

        # Create Kaapi config
        kaapi_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                params=KaapiLLMParams(
                    model="gpt-4",
                    temperature=0.7,
                ),
            )
        )
        kaapi_config = create_test_config(
            db, project_id=project.id, config_blob=kaapi_blob, use_kaapi_schema=True
        )
        db.commit()

        # Test native config resolution
        native_crud = ConfigVersionCrud(
            session=db, project_id=project.id, config_id=native_config.id
        )
        native_call_config = LLMCallConfig(id=str(native_config.id), version=1)
        resolved_native, error_native = resolve_config_blob(
            native_crud, native_call_config
        )

        assert error_native is None
        assert isinstance(resolved_native.completion, NativeCompletionConfig)
        assert resolved_native.completion.provider == "openai-native"

        # Test Kaapi config resolution
        kaapi_crud = ConfigVersionCrud(
            session=db, project_id=project.id, config_id=kaapi_config.id
        )
        kaapi_call_config = LLMCallConfig(id=str(kaapi_config.id), version=1)
        resolved_kaapi, error_kaapi = resolve_config_blob(kaapi_crud, kaapi_call_config)

        assert error_kaapi is None
        assert isinstance(resolved_kaapi.completion, KaapiCompletionConfig)
        assert resolved_kaapi.completion.provider == "openai"
