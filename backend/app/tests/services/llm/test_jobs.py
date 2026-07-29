import pytest
from fastapi import HTTPException
from unittest.mock import patch, MagicMock
from uuid import UUID, uuid4

from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout


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
    Usage,
    TextOutput,
    TextContent,
    AudioOutput,
    AudioContent,
    # KaapiLLMParams,
    KaapiCompletionConfig,
)
from app.models.llm.request import ConfigBlob, LLMCallConfig, LLMChainRequest
from app.models.llm.request import ChainBlock as ChainBlockModel
from app.services.llm.jobs import (
    start_job,
    start_chain_job,
    handle_job_error,
    execute_job,
    execute_chain_job,
    resolve_config_blob,
)
from app.tests.utils.utils import get_project
from app.tests.utils.test_data import create_test_config

VALIDATOR_CONFIG_ID_1 = "00000000-0000-0000-0000-000000000001"
VALIDATOR_CONFIG_ID_2 = "00000000-0000-0000-0000-000000000002"


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
                        type="text",
                        params={"model": "gpt-4o"},
                    )
                )
            ),
        )

    def test_start_job_success(self, db: Session, llm_call_request: LLMCallRequest):
        """Test successful job creation and Celery task scheduling."""

        request = llm_call_request
        project = get_project(db)

        with patch("app.services.llm.jobs.start_llm_job") as mock_schedule:
            mock_schedule.return_value = "fake-task-id-123"

            job_id = start_job(db, request, project.id, project.organization_id)

            job_crud = JobCrud(session=db)
            job = job_crud.get(job_id=job_id, project_id=project.id)
            assert job is not None
            assert job.job_type == JobType.LLM_API
            assert job.status == JobStatus.PENDING
            assert job.trace_id is not None

            mock_schedule.assert_called_once()
            _, kwargs = mock_schedule.call_args
            assert kwargs["project_id"] == project.id
            assert kwargs["organization_id"] == project.organization_id
            assert kwargs["job_id"] == str(job_id)
            assert "request_data" in kwargs

    def test_start_job_celery_scheduling_fails(
        self, db: Session, llm_call_request: LLMCallRequest
    ):
        """Test start_job when Celery task scheduling fails."""
        project = get_project(db)

        with patch("app.services.llm.jobs.start_llm_job") as mock_schedule:
            mock_schedule.side_effect = Exception("Celery connection failed")

            with pytest.raises(HTTPException) as exc_info:
                start_job(db, llm_call_request, project.id, project.organization_id)

            assert exc_info.value.status_code == 503
            assert "Could not queue the LLM call" in str(exc_info.value.detail)

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

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.send_callback") as mock_send_callback,
        ):
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

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.send_callback") as mock_send_callback,
        ):
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

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.send_callback") as mock_send_callback,
        ):
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

    def test_handle_job_error_with_chain_id_updates_chain_status(self, db: Session):
        """Test that chain status is updated to FAILED when chain_id is provided."""
        from app.models.llm.request import ChainStatus

        job = JobCrud(session=db).create(
            job_type=JobType.LLM_API, trace_id="test-trace"
        )
        db.commit()

        chain_id = uuid4()
        callback_response = APIResponse.failure_response(error="Chain failed")

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.send_callback"),
            patch("app.services.llm.jobs.update_llm_chain_status") as mock_update_chain,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            handle_job_error(
                job_id=job.id,
                callback_url=None,
                callback_response=callback_response,
                chain_id=chain_id,
            )

        mock_update_chain.assert_called_once_with(
            db,
            chain_id=chain_id,
            status=ChainStatus.FAILED,
            error="Chain failed",
        )

    def test_handle_job_error_without_chain_id_skips_chain_update(self, db: Session):
        """Test that update_llm_chain_status is not called when chain_id is None."""
        job = JobCrud(session=db).create(
            job_type=JobType.LLM_API, trace_id="test-trace"
        )
        db.commit()

        callback_response = APIResponse.failure_response(error="Job failed")

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.send_callback"),
            patch("app.services.llm.jobs.update_llm_chain_status") as mock_update_chain,
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            handle_job_error(
                job_id=job.id,
                callback_url=None,
                callback_response=callback_response,
                chain_id=None,
            )

        mock_update_chain.assert_not_called()

    def test_handle_job_error_chain_update_failure_is_swallowed(self, db: Session):
        """Test that an exception from update_llm_chain_status doesn't propagate."""
        job = JobCrud(session=db).create(
            job_type=JobType.LLM_API, trace_id="test-trace"
        )
        db.commit()

        chain_id = uuid4()
        callback_response = APIResponse.failure_response(error="Chain failed")

        with (
            patch("app.services.llm.jobs.Session") as mock_session_class,
            patch("app.services.llm.jobs.send_callback"),
            patch(
                "app.services.llm.jobs.update_llm_chain_status",
                side_effect=Exception("DB error updating chain"),
            ),
        ):
            mock_session_class.return_value.__enter__.return_value = db
            mock_session_class.return_value.__exit__.return_value = None

            # Should not raise — the exception is caught and logged
            result = handle_job_error(
                job_id=job.id,
                callback_url=None,
                callback_response=callback_response,
                chain_id=chain_id,
            )

        assert result["success"] is False
        assert result["error"] == "Chain failed"


class TestExecuteJob:
    """Test suite for execute_job."""

    @pytest.fixture(autouse=True)
    def mock_llm_call_crud(self):
        with (
            patch("app.services.llm.jobs.create_llm_call") as mock_create_llm_call,
            patch("app.services.llm.jobs.update_llm_call_response"),
        ):
            mock_create_llm_call.return_value = MagicMock(id=uuid4())
            yield

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
                        "type": "text",
                        "provider": "openai-native",
                        "params": {"model": "gpt-4o"},
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
                model="gpt-4o",
                provider="openai",
                output=TextOutput(content=TextContent(value="Test response")),
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

    def test_proxy_config_success(self, db, job_env, job_for_execution):
        """Proxy completion forwards the (guardrail-sanitised) input to
        client_llm_url and extracts output[0].content[0].text."""
        env = job_env
        request_data = {
            "query": {"input": "Brief overview of land rights."},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "id": "resp_abc",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Safe answer from TAP."}
                    ],
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        }

        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = None
        fake_client.post.return_value = fake_resp

        with (
            patch(
                "app.services.llm.jobs.get_provider_credential",
                return_value={"api_key": "tap-token"},
            ),
            patch("app.services.llm.jobs.httpx.Client", return_value=fake_client),
        ):
            result = self._execute_job(job_for_execution, db, request_data)

        env["get_provider"].assert_not_called()
        fake_client.post.assert_called_once()
        _, kwargs = fake_client.post.call_args
        assert kwargs["json"] == {"input": "Brief overview of land rights."}
        assert kwargs["headers"]["Authorization"] == "Bearer tap-token"

        assert result["success"]
        assert (
            result["data"]["response"]["output"]["content"]["value"]
            == "Safe answer from TAP."
        )
        assert result["data"]["response"]["provider"] == "proxy"
        assert result["data"]["response"]["model"] == "gpt-5"
        assert result["data"]["usage"]["total_tokens"] == 18

    def test_proxy_config_missing_credentials_fails(
        self, db, job_env, job_for_execution
    ):
        """Proxy flow returns a failure when no 'proxy' credential is registered."""
        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        with patch("app.services.llm.jobs.get_provider_credential", return_value=None):
            result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "Proxy credentials not registered" in result["error"]

    def test_proxy_config_http_error_fails(self, db, job_env, job_for_execution):
        """HTTP error during proxy call surfaces as BlockResult.error."""
        import httpx as _httpx

        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = None
        fake_client.post.side_effect = _httpx.ConnectError("boom")

        with (
            patch(
                "app.services.llm.jobs.get_provider_credential",
                return_value={"api_key": "tap-token"},
            ),
            patch("app.services.llm.jobs.httpx.Client", return_value=fake_client),
        ):
            result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "Proxy call failed" in result["error"]

    def test_proxy_config_malformed_response_fails(
        self, db, job_env, job_for_execution
    ):
        """Proxy returns unexpected JSON structure — missing output key."""
        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {"unexpected": "structure"}

        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = None
        fake_client.post.return_value = fake_resp

        with (
            patch(
                "app.services.llm.jobs.get_provider_credential",
                return_value={"api_key": "tap-token"},
            ),
            patch("app.services.llm.jobs.httpx.Client", return_value=fake_client),
        ):
            result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "unexpected response structure" in result["error"]

    def test_proxy_ssrf_dns_failure_allows_request(
        self, db, job_env, job_for_execution
    ):
        """SSRF guard: DNS resolution failure is logged but request proceeds (jobs.py:634)."""
        import socket

        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        ssrf_err = ValueError("DNS lookup failed")
        ssrf_err.__cause__ = socket.gaierror(-2, "Name or service not known")

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "id": "resp_abc",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = None
        fake_client.post.return_value = fake_resp

        with (
            patch(
                "app.services.llm.jobs.validate_callback_url", side_effect=ssrf_err
            ) as mock_validate,
            patch(
                "app.services.llm.jobs.get_provider_credential",
                return_value={"api_key": "tap-token"},
            ),
            patch("app.services.llm.jobs.httpx.Client", return_value=fake_client),
        ):
            result = self._execute_job(job_for_execution, db, request_data)

        mock_validate.assert_called_once()
        fake_client.post.assert_called_once()
        assert result["success"]

    def test_proxy_ssrf_non_dns_failure_rejects_request(
        self, db, job_env, job_for_execution
    ):
        """SSRF guard: non-DNS ValueError from validate_callback_url rejects the request (jobs.py:639-645)."""
        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://internal.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        with (
            patch(
                "app.services.llm.jobs.validate_callback_url",
                side_effect=ValueError("private IP not allowed"),
            ),
            patch("app.services.llm.jobs.httpx.Client") as mock_httpx,
        ):
            result = self._execute_job(job_for_execution, db, request_data)

        mock_httpx.assert_not_called()
        assert not result["success"]
        assert "Proxy URL rejected" in result["error"]

    def test_proxy_create_llm_call_record_failure(self, db, job_env, job_for_execution):
        """create_llm_call failure in proxy branch returns error (jobs.py:675-682)."""
        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        with (
            patch(
                "app.services.llm.jobs.get_provider_credential",
                return_value={"api_key": "tap-token"},
            ),
            patch(
                "app.services.llm.jobs.create_llm_call",
                side_effect=Exception("DB insert failed"),
            ),
            patch("app.services.llm.jobs.httpx.Client") as mock_httpx,
        ):
            result = self._execute_job(job_for_execution, db, request_data)

        mock_httpx.assert_not_called()
        assert not result["success"]
        assert "Failed to create LLM call record" in result["error"]
        assert "DB insert failed" in result["error"]

    def test_proxy_update_llm_call_record_failure_is_swallowed(
        self, db, job_env, job_for_execution
    ):
        """update_llm_call_response failure in proxy branch is logged but call still succeeds (jobs.py:783-787)."""
        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    }
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "id": "resp_abc",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Safe answer."}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = None
        fake_client.post.return_value = fake_resp

        with (
            patch(
                "app.services.llm.jobs.get_provider_credential",
                return_value={"api_key": "tap-token"},
            ),
            patch("app.services.llm.jobs.httpx.Client", return_value=fake_client),
            patch(
                "app.services.llm.jobs.update_llm_call_response",
                side_effect=Exception("DB update failed"),
            ) as mock_update,
        ):
            result = self._execute_job(job_for_execution, db, request_data)

        mock_update.assert_called_once()
        # Failure is swallowed — proxy call still returns success.
        assert result["success"]
        assert (
            result["data"]["response"]["output"]["content"]["value"] == "Safe answer."
        )

    def test_proxy_output_guardrails_failure_returns_error(
        self, db, job_env, job_for_execution
    ):
        """Output guardrails failure in proxy branch surfaces as BlockResult.error (jobs.py:829-833)."""
        request_data = {
            "query": {"input": "hi"},
            "config": {
                "blob": {
                    "completion": {
                        "type": "proxy",
                        "provider": None,
                        "params": {
                            "client_llm_url": "https://api.tap.example/v1/predictions"
                        },
                    },
                    "input_guardrails": [],
                    "output_guardrails": [
                        {"validator_config_id": VALIDATOR_CONFIG_ID_2}
                    ],
                }
            },
            "include_provider_raw_response": False,
            "callback_url": None,
        }

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = {
            "id": "resp_abc",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Unsafe leaked output."}
                    ],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = None
        fake_client.post.return_value = fake_resp

        with (
            patch(
                "app.services.llm.jobs.get_provider_credential",
                return_value={"api_key": "tap-token"},
            ),
            patch("app.services.llm.jobs.httpx.Client", return_value=fake_client),
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
        ):
            mock_fetch_configs.return_value = (
                [],
                [{"type": "pii_remover", "stage": "output"}],
            )
            mock_guardrails.return_value = {
                "success": False,
                "bypassed": False,
                "error": "Output blocked by guardrails",
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "Output blocked by guardrails" in result["error"]

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
                type="text",
                params={"model": "gpt-4o", "temperature": 0.7},
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
                type="text",
                params={"model": "gpt-4o", "temperature": 0.5},
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
                type="text",
                params={"model": "gpt-4o"},
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
                type="text",
                params={
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "instructions": "You are a helpful assistant",
                },
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
                type="text",
                params={
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "instructions": "You are a helpful assistant",
                },
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
                type="text",
                params={
                    "model": "gpt-5",  # Reasoning model
                    "temperature": 0.7,  # This will be suppressed with warning
                },
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
                type="text",
                params={
                    "model": "gpt-4o",  # Non-reasoning model
                    "reasoning": "high",  # This will be suppressed with warning
                },
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

    def test_guardrails_sanitize_input_before_provider(
        self, db, job_env, job_for_execution
    ):
        """
        Input guardrails should sanitize the text BEFORE provider.execute is called.
        """

        env = job_env

        env["provider"].execute.return_value = (
            env["mock_llm_response"],
            None,
        )

        unsafe_input = "My credit card is 4111 1111 1111 1111"

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": True,
                "bypassed": False,
                "data": {
                    "safe_text": "My credit card is [REDACTED]",
                    "rephrase_needed": False,
                },
            }
            mock_fetch_configs.return_value = (
                [{"type": "pii_remover", "stage": "input"}],
                [],
            )

            request_data = {
                "query": {"input": unsafe_input},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [],
                    }
                },
                "include_provider_raw_response": False,
                "callback_url": None,
            }
            result = self._execute_job(job_for_execution, db, request_data)

        provider_query = env["provider"].execute.call_args[0][1]
        assert "[REDACTED]" in provider_query.input.content.value
        assert "4111" not in provider_query.input.content.value

        assert result["success"]

    def test_guardrails_skip_input_validation_for_audio_input(
        self, db, job_env, job_for_execution
    ):
        env = job_env
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_fetch_configs.return_value = (
                [{"type": "pii_remover", "stage": "input"}],
                [],
            )

            request_data = {
                "query": {
                    "input": {
                        "type": "audio",
                        "content": {
                            "format": "base64",
                            "value": "UklGRiQAAABXQVZFZm10IA==",
                            "mime_type": "audio/wav",
                        },
                    }
                },
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert result["success"] is True
        env["provider"].execute.assert_called_once()
        mock_guardrails.assert_not_called()

    def test_guardrails_sanitize_output_after_provider(
        self, db, job_env, job_for_execution
    ):
        env = job_env

        env["mock_llm_response"].response.output.content.value = "Aadhar no 123-45-6789"
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": True,
                "bypassed": False,
                "data": {
                    "safe_text": "Aadhar [REDACTED]",
                    "rephrase_needed": False,
                },
            }
            mock_fetch_configs.return_value = (
                [],
                [{"type": "pii_remover", "stage": "output"}],
            )

            request_data = {
                "query": {"input": "hello"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [],
                        "output_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_2}
                        ],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert "REDACTED" in result["data"]["response"]["output"]["content"]["value"]

    def test_guardrails_output_validation_sends_input_output_pair(
        self, db, job_env, job_for_execution
    ):
        env = job_env
        user_query = "What is my Aadhar number?"
        llm_output = "Your Aadhar number is 1234-5678-9012"

        env["mock_llm_response"].response.output.content.value = llm_output
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": True,
                "bypassed": False,
                "data": {
                    "safe_text": "Your Aadhar number is [REDACTED]",
                    "rephrase_needed": False,
                },
            }
            mock_fetch_configs.return_value = (
                [],
                [{"type": "pii_remover", "stage": "output"}],
            )

            request_data = {
                "query": {"input": user_query},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [],
                        "output_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_2}
                        ],
                    }
                },
            }
            self._execute_job(job_for_execution, db, request_data)

        mock_guardrails.assert_called_once()
        _, kwargs = mock_guardrails.call_args
        assert kwargs.get("output_text") == llm_output
        assert mock_guardrails.call_args[0][0] == user_query

    def test_guardrails_bypass_does_not_modify_output(
        self, db, job_env, job_for_execution
    ):
        env = job_env
        original_llm_output = "Your Aadhar number is 1234-5678-9012"

        env["mock_llm_response"].response.output.content.value = original_llm_output
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": False,
                "bypassed": True,
                "data": {"safe_text": original_llm_output, "rephrase_needed": False},
            }
            mock_fetch_configs.return_value = (
                [],
                [{"type": "pii_remover", "stage": "output"}],
            )

            request_data = {
                "query": {"input": "some question"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [],
                        "output_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_2}
                        ],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert result["success"] is True
        assert (
            result["data"]["response"]["output"]["content"]["value"]
            == original_llm_output
        )

    def test_guardrails_skip_output_validation_for_audio_output(
        self, db, job_env, job_for_execution
    ):
        env = job_env

        env["mock_llm_response"].response.output = AudioOutput(
            content=AudioContent(
                value="UklGRiQAAABXQVZFZm10IA==",
                mime_type="audio/wav",
            )
        )
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_fetch_configs.return_value = (
                [],
                [{"type": "safety_filter", "stage": "output"}],
            )

            request_data = {
                "query": {"input": "hello"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [],
                        "output_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_2}
                        ],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert result["success"] is True
        assert result["data"]["response"]["output"]["type"] == "audio"
        env["provider"].execute.assert_called_once()
        mock_guardrails.assert_not_called()

    def test_guardrails_bypass_does_not_modify_input(
        self, db, job_env, job_for_execution
    ):
        env = job_env

        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        unsafe_input = "4111 1111 1111 1111"

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": True,
                "bypassed": True,
                "data": {
                    "safe_text": unsafe_input,
                    "rephrase_needed": False,
                },
            }
            mock_fetch_configs.return_value = (
                [{"type": "pii_remover", "stage": "input"}],
                [],
            )

            request_data = {
                "query": {"input": unsafe_input},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [],
                    }
                },
            }
            self._execute_job(job_for_execution, db, request_data)

        provider_query = env["provider"].execute.call_args[0][1]
        assert provider_query.input.content.value == unsafe_input

    def test_guardrails_validation_failure_blocks_job(
        self, db, job_env, job_for_execution
    ):
        env = job_env

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": False,
                "error": "Unsafe content detected",
            }
            mock_fetch_configs.return_value = (
                [{"type": "uli_slur_match", "stage": "input"}],
                [],
            )

            request_data = {
                "query": {"input": "bad input"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert not result["success"]
        assert "Unsafe content" in result["error"]
        env["provider"].execute.assert_not_called()

    def test_guardrails_rephrase_needed_allows_job_with_sanitized_input(
        self, db, job_env, job_for_execution
    ):
        env = job_env
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": True,
                "bypassed": False,
                "data": {
                    "safe_text": "Rephrased text",
                    "rephrase_needed": True,
                },
            }
            mock_fetch_configs.return_value = (
                [{"type": "policy", "stage": "input"}],
                [],
            )

            request_data = {
                "query": {"input": "unsafe text"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert result["success"] is True
        env["provider"].execute.assert_not_called()
        assert (
            result["data"]["response"]["output"]["content"]["value"] == "Rephrased text"
        )

    def test_guardrails_rephrase_saves_original_input_and_safe_text_in_db(
        self, db, job_env, job_for_execution
    ):
        from app.models.llm.request import LlmCall

        env = job_env

        with (
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
        ):
            mock_guardrails.return_value = {
                "success": True,
                "bypassed": False,
                "data": {
                    "safe_text": "Please rephrase the query without unsafe content. Input is outside the allowed topic scope.",
                    "rephrase_needed": True,
                },
            }
            mock_fetch_configs.return_value = (
                [{"type": "policy", "stage": "input"}],
                [],
            )

            request_data = {
                "query": {"input": "unsafe user query"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [],
                    }
                },
            }
            self._execute_job(job_for_execution, db, request_data)

        llm_call = db.exec(
            select(LlmCall).where(LlmCall.job_id == job_for_execution.id)
        ).first()

        assert llm_call is not None
        assert llm_call.input == "unsafe user query"
        assert llm_call.content == {
            "type": "text",
            "content": {
                "format": "text",
                "value": "Please rephrase the query without unsafe content. Input is outside the allowed topic scope.",
            },
        }

    def test_execute_job_fetches_validator_configs_from_blob_refs(
        self, db, job_env, job_for_execution
    ):
        env = job_env
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with patch(
            "app.services.llm.guardrails.list_validators_config"
        ) as mock_fetch_configs:
            mock_fetch_configs.return_value = ([], [])

            request_data = {
                "query": {"input": "hello"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_2}
                        ],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert result["success"]
        assert mock_fetch_configs.call_count == 2

        # First call: input guardrails
        _, input_kwargs = mock_fetch_configs.call_args_list[0]
        assert [
            v.validator_config_id for v in input_kwargs["input_validator_configs"]
        ] == [UUID(VALIDATOR_CONFIG_ID_1)]
        assert input_kwargs["output_validator_configs"] is None

        # Second call: output guardrails
        _, output_kwargs = mock_fetch_configs.call_args_list[1]
        assert output_kwargs["input_validator_configs"] is None
        assert [
            v.validator_config_id for v in output_kwargs["output_validator_configs"]
        ] == [UUID(VALIDATOR_CONFIG_ID_2)]

    def test_execute_job_continues_when_no_validator_configs_resolved(
        self, db, job_env, job_for_execution
    ):
        env = job_env
        env["provider"].execute.return_value = (env["mock_llm_response"], None)

        with (
            patch(
                "app.services.llm.guardrails.list_validators_config"
            ) as mock_fetch_configs,
            patch(
                "app.services.llm.guardrails.run_guardrails_validation"
            ) as mock_guardrails,
        ):
            mock_fetch_configs.return_value = ([], [])

            request_data = {
                "query": {"input": "hello"},
                "config": {
                    "blob": {
                        "completion": {
                            "provider": "openai-native",
                            "type": "text",
                            "params": {"model": "gpt-4o"},
                        },
                        "input_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_1}
                        ],
                        "output_guardrails": [
                            {"validator_config_id": VALIDATOR_CONFIG_ID_2}
                        ],
                    }
                },
            }
            result = self._execute_job(job_for_execution, db, request_data)

        assert result["success"] is True
        env["provider"].execute.assert_called_once()
        mock_guardrails.assert_not_called()

    def test_timeout_reraises_and_marks_job_failed(
        self, db, job_env, job_for_execution, request_data
    ):
        env = job_env
        env["provider"].execute.side_effect = Timeout(300)

        with pytest.raises(Timeout):
            self._execute_job(job_for_execution, db, request_data)

        db.refresh(job_for_execution)
        assert job_for_execution.status == JobStatus.FAILED

    def test_timeout_with_callback_sends_failure_payload(
        self, db, job_env, job_for_execution, request_data
    ):
        env = job_env
        request_data["callback_url"] = "https://example.com/callback"
        env["provider"].execute.side_effect = Timeout(300)

        with pytest.raises(Timeout):
            self._execute_job(job_for_execution, db, request_data)

        env["send_callback"].assert_called_once()
        payload = env["send_callback"].call_args.kwargs["data"]
        assert payload["success"] is False
        assert "soft time limit" in (payload["error"] or "")

        db.refresh(job_for_execution)
        assert job_for_execution.status == JobStatus.FAILED

    def test_soft_time_limit_reraises_and_marks_job_failed(
        self, db, job_env, job_for_execution, request_data
    ):
        env = job_env
        env["provider"].execute.side_effect = SoftTimeLimitExceeded()

        with pytest.raises(SoftTimeLimitExceeded):
            self._execute_job(job_for_execution, db, request_data)

        db.refresh(job_for_execution)
        assert job_for_execution.status == JobStatus.FAILED

    def test_soft_time_limit_with_callback_sends_failure_payload(
        self, db, job_env, job_for_execution, request_data
    ):
        env = job_env
        request_data["callback_url"] = "https://example.com/callback"
        env["provider"].execute.side_effect = SoftTimeLimitExceeded()

        with pytest.raises(SoftTimeLimitExceeded):
            self._execute_job(job_for_execution, db, request_data)

        env["send_callback"].assert_called_once()
        payload = env["send_callback"].call_args.kwargs["data"]
        assert payload["success"] is False
        assert "soft time limit" in (payload["error"] or "")

        db.refresh(job_for_execution)
        assert job_for_execution.status == JobStatus.FAILED


class TestStartChainJob:
    """Test cases for the start_chain_job function."""

    @pytest.fixture
    def chain_request(self):
        return LLMChainRequest(
            query=QueryParams(input="Test query"),
            blocks=[
                ChainBlockModel(
                    config=LLMCallConfig(
                        blob=ConfigBlob(
                            completion=NativeCompletionConfig(
                                provider="openai-native",
                                type="text",
                                params={"model": "gpt-4o"},
                            )
                        )
                    )
                )
            ],
        )

    def test_start_chain_job_success(self, db: Session, chain_request):
        project = get_project(db)

        with (
            patch("app.services.llm.jobs.start_llm_chain_job") as mock_schedule,
            patch("app.services.llm.jobs.JobCrud") as mock_job_crud_class,
        ):
            mock_schedule.return_value = "fake-task-id"
            mock_job = MagicMock()
            mock_job.id = uuid4()
            mock_job.job_type = JobType.LLM_CHAIN
            mock_job.status = JobStatus.PENDING
            mock_job_crud_class.return_value.create.return_value = mock_job

            job_id = start_chain_job(
                db, chain_request, project.id, project.organization_id
            )

            assert job_id == mock_job.id
            mock_schedule.assert_called_once()

    def test_start_chain_job_celery_failure(self, db: Session, chain_request):
        project = get_project(db)

        with (
            patch("app.services.llm.jobs.start_llm_chain_job") as mock_schedule,
            patch("app.services.llm.jobs.JobCrud") as mock_job_crud_class,
        ):
            mock_schedule.side_effect = Exception("Celery connection failed")
            mock_job = MagicMock()
            mock_job.id = uuid4()
            mock_job_crud_class.return_value.create.return_value = mock_job

            with pytest.raises(HTTPException) as exc_info:
                start_chain_job(db, chain_request, project.id, project.organization_id)

            assert exc_info.value.status_code == 503
            assert "Could not queue the LLM chain job" in str(exc_info.value.detail)


class TestExecuteChainJob:
    """Test suite for execute_chain_job."""

    @pytest.fixture
    def chain_request_data(self):
        return {
            "query": {"input": "Test query"},
            "blocks": [
                {
                    "config": {
                        "blob": {
                            "completion": {
                                "provider": "openai-native",
                                "type": "text",
                                "params": {"model": "gpt-4o"},
                            }
                        }
                    },
                }
            ],
        }

    @pytest.fixture
    def mock_llm_response(self):
        return LLMCallResponse(
            response=LLMResponse(
                provider_response_id="resp-123",
                conversation_id=None,
                model="gpt-4o",
                provider="openai",
                output=TextOutput(content=TextContent(value="Test response")),
            ),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            provider_raw_response=None,
        )

    def _execute_chain_job(self, request_data):
        return execute_chain_job(
            request_data=request_data,
            project_id=1,
            organization_id=1,
            job_id=str(uuid4()),
            task_id="task-123",
            task_instance=None,
        )

    def test_success_flow(self, chain_request_data, mock_llm_response):
        from app.services.llm.chain.types import BlockResult

        with (
            patch("app.services.llm.jobs.Session") as mock_session,
            patch("app.services.llm.jobs.create_llm_chain") as mock_create_chain,
            patch("app.services.llm.jobs.get_provider_credential") as mock_creds,
            patch("app.services.llm.chain.executor.Session") as mock_executor_session,
            patch("app.services.llm.chain.executor.send_callback"),
            patch("app.services.llm.chain.executor.update_llm_chain_status"),
            patch("app.services.llm.chain.chain.execute_llm_call") as mock_execute_llm,
            patch("app.services.llm.chain.executor.update_llm_chain_block_completed"),
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            mock_session.return_value.__exit__.return_value = None
            mock_executor_session.return_value.__enter__.return_value = MagicMock()
            mock_executor_session.return_value.__exit__.return_value = None

            mock_chain_record = MagicMock()
            mock_chain_record.id = uuid4()
            mock_create_chain.return_value = mock_chain_record
            mock_creds.return_value = None

            mock_execute_llm.return_value = BlockResult(
                response=mock_llm_response,
                llm_call_id=uuid4(),
                usage=mock_llm_response.usage,
            )

            result = self._execute_chain_job(chain_request_data)

            assert result["success"] is True

    def test_exception_during_chain_creation(self, chain_request_data):
        with (
            patch("app.services.llm.jobs.Session") as mock_session,
            patch(
                "app.services.llm.jobs.create_llm_chain",
                side_effect=Exception("DB error"),
            ),
            patch("app.services.llm.jobs.handle_job_error") as mock_handle_error,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            mock_session.return_value.__exit__.return_value = None
            mock_handle_error.return_value = {
                "success": False,
                "error": "Unexpected error occurred",
            }

            result = self._execute_chain_job(chain_request_data)

            assert result["success"] is False

    def test_chain_status_updated_to_failed_on_error(self, chain_request_data):
        chain_id = uuid4()

        with (
            patch("app.services.llm.jobs.Session") as mock_session,
            patch("app.services.llm.jobs.create_llm_chain") as mock_create_chain,
            patch("app.services.llm.jobs.get_provider_credential") as mock_creds,
            patch("app.services.llm.jobs.handle_job_error") as mock_handle_error,
            patch(
                "app.services.llm.chain.chain.LLMChain",
                side_effect=Exception("Chain init error"),
            ),
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            mock_session.return_value.__exit__.return_value = None

            mock_chain_record = MagicMock()
            mock_chain_record.id = chain_id
            mock_create_chain.return_value = mock_chain_record
            mock_creds.return_value = None
            mock_handle_error.return_value = {
                "success": False,
                "error": "Unexpected error occurred",
            }

            result = self._execute_chain_job(chain_request_data)

            mock_handle_error.assert_called_once()
            _, kwargs = mock_handle_error.call_args
            assert kwargs["chain_id"] == chain_id

    def test_timeout_reraises_and_calls_handle_job_error(self, chain_request_data):
        chain_id = uuid4()

        with (
            patch("app.services.llm.jobs.Session") as mock_session,
            patch("app.services.llm.jobs.create_llm_chain") as mock_create_chain,
            patch("app.services.llm.jobs.get_provider_credential") as mock_creds,
            patch("app.services.llm.jobs.handle_job_error") as mock_handle_error,
            patch("app.services.llm.chain.chain.LLMChain"),
            patch(
                "app.services.llm.chain.executor.ChainExecutor"
            ) as mock_executor_class,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            mock_session.return_value.__exit__.return_value = None

            mock_chain_record = MagicMock()
            mock_chain_record.id = chain_id
            mock_create_chain.return_value = mock_chain_record
            mock_creds.return_value = None
            mock_executor_class.return_value.run.side_effect = Timeout(300)
            mock_handle_error.return_value = {
                "success": False,
                "error": "Task exceeded soft time limit",
            }

            with pytest.raises(Timeout):
                self._execute_chain_job(chain_request_data)

        mock_handle_error.assert_called_once()
        _, kwargs = mock_handle_error.call_args
        assert kwargs["chain_id"] == chain_id
        callback_response = mock_handle_error.call_args.args[2]
        assert callback_response.error == "Task exceeded soft time limit"

    def test_timeout_with_callback_sends_failure_payload(self, chain_request_data):
        chain_id = uuid4()
        chain_request_data["callback_url"] = "https://example.com/chain-callback"

        with (
            patch("app.services.llm.jobs.Session") as mock_session,
            patch("app.services.llm.jobs.create_llm_chain") as mock_create_chain,
            patch("app.services.llm.jobs.get_provider_credential") as mock_creds,
            patch("app.services.llm.jobs.handle_job_error") as mock_handle_error,
            patch("app.services.llm.chain.chain.LLMChain"),
            patch(
                "app.services.llm.chain.executor.ChainExecutor"
            ) as mock_executor_class,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            mock_session.return_value.__exit__.return_value = None

            mock_chain_record = MagicMock()
            mock_chain_record.id = chain_id
            mock_create_chain.return_value = mock_chain_record
            mock_creds.return_value = None
            mock_executor_class.return_value.run.side_effect = Timeout(300)
            mock_handle_error.return_value = {
                "success": False,
                "error": "Task exceeded soft time limit",
            }

            with pytest.raises(Timeout):
                self._execute_chain_job(chain_request_data)

        mock_handle_error.assert_called_once()
        call_args = mock_handle_error.call_args
        assert call_args.args[1] == "https://example.com/chain-callback"
        assert call_args.args[2].error == "Task exceeded soft time limit"
        assert call_args.kwargs["chain_id"] == chain_id

    def test_soft_time_limit_reraises_and_calls_handle_job_error(
        self, chain_request_data
    ):
        chain_id = uuid4()

        with (
            patch("app.services.llm.jobs.Session") as mock_session,
            patch("app.services.llm.jobs.create_llm_chain") as mock_create_chain,
            patch("app.services.llm.jobs.get_provider_credential") as mock_creds,
            patch("app.services.llm.jobs.handle_job_error") as mock_handle_error,
            patch("app.services.llm.chain.chain.LLMChain"),
            patch(
                "app.services.llm.chain.executor.ChainExecutor"
            ) as mock_executor_class,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            mock_session.return_value.__exit__.return_value = None

            mock_chain_record = MagicMock()
            mock_chain_record.id = chain_id
            mock_create_chain.return_value = mock_chain_record
            mock_creds.return_value = None
            mock_executor_class.return_value.run.side_effect = SoftTimeLimitExceeded()
            mock_handle_error.return_value = {
                "success": False,
                "error": "Task exceeded soft time limit",
            }

            with pytest.raises(SoftTimeLimitExceeded):
                self._execute_chain_job(chain_request_data)

        mock_handle_error.assert_called_once()
        _, kwargs = mock_handle_error.call_args
        assert kwargs["chain_id"] == chain_id
        callback_response = mock_handle_error.call_args.args[2]
        assert callback_response.error == "Task exceeded soft time limit"

    def test_soft_time_limit_with_callback_sends_failure_payload(
        self, chain_request_data
    ):
        chain_id = uuid4()
        chain_request_data["callback_url"] = "https://example.com/chain-callback"

        with (
            patch("app.services.llm.jobs.Session") as mock_session,
            patch("app.services.llm.jobs.create_llm_chain") as mock_create_chain,
            patch("app.services.llm.jobs.get_provider_credential") as mock_creds,
            patch("app.services.llm.jobs.handle_job_error") as mock_handle_error,
            patch("app.services.llm.chain.chain.LLMChain"),
            patch(
                "app.services.llm.chain.executor.ChainExecutor"
            ) as mock_executor_class,
        ):
            mock_session.return_value.__enter__.return_value = MagicMock()
            mock_session.return_value.__exit__.return_value = None

            mock_chain_record = MagicMock()
            mock_chain_record.id = chain_id
            mock_create_chain.return_value = mock_chain_record
            mock_creds.return_value = None
            mock_executor_class.return_value.run.side_effect = SoftTimeLimitExceeded()
            mock_handle_error.return_value = {
                "success": False,
                "error": "Task exceeded soft time limit",
            }

            with pytest.raises(SoftTimeLimitExceeded):
                self._execute_chain_job(chain_request_data)

        mock_handle_error.assert_called_once()
        call_args = mock_handle_error.call_args
        assert call_args.args[1] == "https://example.com/chain-callback"
        assert call_args.args[2].error == "Task exceeded soft time limit"
        assert call_args.kwargs["chain_id"] == chain_id


class TestResolveConfigBlob:
    """Test suite for resolve_config_blob function."""

    def test_resolve_config_blob_success(self, db: Session):
        """Test successful resolution of stored config blob."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                type="text",
                params={"model": "gpt-4o", "temperature": 0.8},
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
        assert resolved_blob.completion.params["model"] == "gpt-4o"
        assert resolved_blob.completion.params["temperature"] == 0.8

    def test_resolve_config_blob_keeps_validator_refs(self, db: Session):
        project = get_project(db)
        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                type="text",
                params={"model": "gpt-4o"},
            ),
            input_guardrails=[{"validator_config_id": VALIDATOR_CONFIG_ID_1}],
            output_guardrails=[{"validator_config_id": VALIDATOR_CONFIG_ID_2}],
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
        assert [v.model_dump() for v in (resolved_blob.input_guardrails or [])] == [
            {"validator_config_id": UUID(VALIDATOR_CONFIG_ID_1)}
        ]
        assert [v.model_dump() for v in (resolved_blob.output_guardrails or [])] == [
            {"validator_config_id": UUID(VALIDATOR_CONFIG_ID_2)}
        ]

    def test_resolve_config_blob_version_not_found(self, db: Session):
        """Test resolve_config_blob when version doesn't exist."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                type="text",
                params={"model": "gpt-4o"},
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
                type="text",
                params={"model": "gpt-4o"},
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
        from app.models.config import ConfigVersionUpdate

        project = get_project(db)

        # Create a config with version 1
        config_blob_v1 = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                type="text",
                params={"model": "gpt-4o", "temperature": 0.5},
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
                type="text",
                params={"model": "gpt-4o", "temperature": 0.9},
            )
        )
        version_update = ConfigVersionUpdate(
            config_blob=config_blob_v2.model_dump(),
            commit_message="Updated to gpt-4",
        )
        config_version_crud.create_or_raise(version_update)
        db.commit()

        # Test resolving version 1
        llm_call_config_v1 = LLMCallConfig(id=str(config.id), version=1)
        resolved_blob_v1, error_v1 = resolve_config_blob(
            config_version_crud, llm_call_config_v1
        )

        assert error_v1 is None
        assert resolved_blob_v1 is not None
        assert resolved_blob_v1.completion.params["model"] == "gpt-4o"
        assert resolved_blob_v1.completion.params["temperature"] == 0.5

        # Test resolving version 2
        llm_call_config_v2 = LLMCallConfig(id=str(config.id), version=2)
        resolved_blob_v2, error_v2 = resolve_config_blob(
            config_version_crud, llm_call_config_v2
        )

        assert error_v2 is None
        assert resolved_blob_v2 is not None
        assert resolved_blob_v2.completion.params["model"] == "gpt-4o"
        assert resolved_blob_v2.completion.params["temperature"] == 0.9

    def test_resolve_kaapi_config_blob_success(self, db: Session):
        """Test successful resolution of stored Kaapi config blob."""
        project = get_project(db)

        config_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                type="text",
                params={
                    "model": "gpt-4o",
                    "temperature": 0.8,
                    "instructions": "You are a helpful assistant",
                },
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
        assert resolved_blob.completion.params["model"] == "gpt-4o"
        assert resolved_blob.completion.params["temperature"] == 0.8
        assert (
            resolved_blob.completion.params["instructions"]
            == "You are a helpful assistant"
        )

    def test_resolve_both_native_and_kaapi_configs(self, db: Session):
        """Test that both native and Kaapi configs can be resolved correctly."""
        project = get_project(db)

        # Create native config
        native_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                type="text",
                params={"model": "gpt-4o", "temperature": 0.5},
            )
        )
        native_config = create_test_config(
            db, project_id=project.id, config_blob=native_blob, use_kaapi_schema=False
        )

        # Create Kaapi config
        kaapi_blob = ConfigBlob(
            completion=KaapiCompletionConfig(
                provider="openai",
                type="text",
                params={
                    "model": "gpt-4o",
                    "temperature": 0.7,
                },
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
