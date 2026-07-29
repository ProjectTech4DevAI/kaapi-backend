from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
from sqlmodel import Session

from app.core.audio_utils import AudioRef
from app.core.config import settings
from app.models.llm import NativeCompletionConfig, QueryParams
from app.services import health_probes
from app.services.health_probes import (
    HEALTH_PROBES_MONITOR_CONFIG,
    HEALTH_PROBES_MONITOR_SLUG,
    _PROBES,
    execute_health_probes,
    run_probes,
)


class _NonClosingSession:
    """Stands in for `Session(engine)` inside run_probes, yielding the
    conftest transactional session without closing it on __exit__."""

    def __init__(self, session: Session):
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


class _FakeProvider:
    def __init__(
        self,
        response: Any = None,
        error: str | None = None,
        raise_exc: Exception | None = None,
    ):
        self._response = response
        self._error = error
        self._raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        *,
        completion_config: NativeCompletionConfig,
        query: QueryParams,
        resolved_input: str | AudioRef,
    ) -> tuple[Any, str | None]:
        self.calls.append(
            {
                "completion_config": completion_config,
                "query": query,
                "resolved_input": resolved_input,
            }
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response, self._error


@pytest.fixture
def configured_probe_settings(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "HEALTH_PROBE_ORG_ID", 1)
    monkeypatch.setattr(settings, "HEALTH_PROBE_PROJECT_ID", 1)
    # run_probes opens its own Session(engine); point it at the conftest
    # transactional session so nothing leaks past the test.
    monkeypatch.setattr(
        health_probes, "Session", lambda _engine: _NonClosingSession(db)
    )


class TestSkipWhenNotConfigured:
    def test_returns_skipped_when_org_id_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "HEALTH_PROBE_ORG_ID", None)
        monkeypatch.setattr(settings, "HEALTH_PROBE_PROJECT_ID", 1)
        monkeypatch.setattr(
            health_probes,
            "Session",
            lambda _engine: pytest.fail("no session must be opened when skipping"),
        )

        result = run_probes()

        assert result == {
            "skipped": True,
            "reason": "health_probe_org_or_project_not_set",
        }

    def test_returns_skipped_when_project_id_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "HEALTH_PROBE_ORG_ID", 1)
        monkeypatch.setattr(settings, "HEALTH_PROBE_PROJECT_ID", None)
        monkeypatch.setattr(
            health_probes,
            "get_llm_provider",
            lambda **_k: pytest.fail("provider must not be built"),
        )

        result = run_probes()

        assert result["skipped"] is True
        assert result["reason"] == "health_probe_org_or_project_not_set"


class TestRunProbesShape:
    def test_all_probes_ok_returns_expected_structure(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        fake = _FakeProvider(response={"data": "pong"}, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes()

        assert set(result.keys()) == {
            "elapsed_ms",
            "total",
            "ok",
            "failed",
            "results",
        }
        assert isinstance(result["elapsed_ms"], int)
        assert result["total"] == len(_PROBES)
        assert result["ok"] == len(_PROBES)
        assert result["failed"] == 0
        assert len(result["results"]) == len(_PROBES)

        for r in result["results"]:
            assert set(r.keys()) == {
                "endpoint",
                "provider",
                "modality",
                "model",
                "ok",
                "latency_ms",
                "error",
            }
            assert r["endpoint"] == "llm/call"
            assert r["ok"] is True
            assert r["error"] is None
            assert isinstance(r["latency_ms"], int)

        # the config build must hand the provider a native config, not the Kaapi one
        assert all(
            isinstance(c["completion_config"], NativeCompletionConfig)
            for c in fake.calls
        )


class TestProviderInitFailure:
    def test_get_llm_provider_value_error_marks_client_init_failed(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        def _boom(**_kwargs: Any) -> _FakeProvider:
            raise ValueError("no creds")

        monkeypatch.setattr(health_probes, "get_llm_provider", _boom)

        result = run_probes()

        assert result["ok"] == 0
        assert result["failed"] == len(_PROBES)
        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"] == "client_init_failed"
            assert r["latency_ms"] is None

    def test_get_llm_provider_runtime_error_marks_client_init_failed(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        monkeypatch.setattr(
            health_probes,
            "get_llm_provider",
            lambda **_k: (_ for _ in ()).throw(RuntimeError("bad")),
        )

        result = run_probes()

        assert result["failed"] == len(_PROBES)
        assert all(r["error"] == "client_init_failed" for r in result["results"])
        assert all(r["latency_ms"] is None for r in result["results"])


class TestExecuteBehaviors:
    def test_execute_raises_marks_error_with_exception_class_name(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        fake = _FakeProvider(raise_exc=TimeoutError("slow"))
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes()

        assert result["ok"] == 0
        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"].startswith("TimeoutError")
            assert "slow" in r["error"]
            assert isinstance(r["latency_ms"], int)

    def test_provider_returns_none_response_uses_provided_error(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        fake = _FakeProvider(response=None, error="some_error")
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes()

        assert result["ok"] == 0
        assert result["failed"] == len(_PROBES)
        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"] == "some_error"
            assert isinstance(r["latency_ms"], int)

    def test_provider_returns_none_response_and_no_error_defaults_no_response(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        fake = _FakeProvider(response=None, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes()

        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"] == "no_response"

    def test_provider_returns_response_marks_ok(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        fake = _FakeProvider(response={"ok": 1}, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes()

        assert result["ok"] == len(_PROBES)
        for r in result["results"]:
            assert r["ok"] is True
            assert r["error"] is None


class TestSttAudioNotConfigured:
    def test_missing_stt_audio_file_short_circuits_stt_probes(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        monkeypatch.setattr(
            health_probes, "_STT_AUDIO_PATH", Path("/nonexistent/probe.ogg")
        )
        # reset the lazy-load cache so the missing path is actually read
        monkeypatch.setattr(health_probes, "_stt_audio_bytes", None)
        fake = _FakeProvider(response={"ok": 1}, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes()

        stt_results = [r for r in result["results"] if r["modality"] == "stt"]
        assert stt_results, "expected at least one STT probe"
        for r in stt_results:
            assert r["ok"] is False
            assert r["error"] == "stt_audio_not_configured"
            # short-circuit happens before execute → no latency recorded
            assert r["latency_ms"] is None

        non_stt = [r for r in result["results"] if r["modality"] != "stt"]
        for r in non_stt:
            assert r["ok"] is True


class TestProviderCaching:
    def test_get_llm_provider_called_once_per_unique_provider(
        self, monkeypatch: pytest.MonkeyPatch, configured_probe_settings: None
    ) -> None:
        seen_providers: list[str] = []

        def _factory(**kwargs: Any) -> _FakeProvider:
            seen_providers.append(kwargs["provider_type"])
            return _FakeProvider(response={"ok": 1}, error=None)

        monkeypatch.setattr(health_probes, "get_llm_provider", _factory)

        run_probes()

        unique_probe_providers = {p.provider for p in _PROBES}
        # One factory call per unique provider name, not per probe
        assert sorted(seen_providers) == sorted(unique_probe_providers)
        assert len(seen_providers) < len(_PROBES)


class TestExecuteHealthProbes:
    @staticmethod
    def _patch_checkin(
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[dict[str, Any]]:
        checkins: list[dict[str, Any]] = []

        def _capture(**kwargs: Any) -> str:
            checkins.append(kwargs)
            return "checkin-123"

        monkeypatch.setattr(health_probes, "capture_checkin", _capture)
        return checkins

    def test_ok_result_closes_checkin_with_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkins = self._patch_checkin(monkeypatch)
        canned = {"elapsed_ms": 1, "total": 2, "ok": 2, "failed": 0, "results": []}
        monkeypatch.setattr(health_probes, "run_probes", lambda: canned)

        result = execute_health_probes()

        assert result == canned
        assert len(checkins) == 2
        assert checkins[0] == {
            "monitor_slug": HEALTH_PROBES_MONITOR_SLUG,
            "status": health_probes.MonitorStatus.IN_PROGRESS,
            "monitor_config": HEALTH_PROBES_MONITOR_CONFIG,
        }
        assert checkins[1] == {
            "check_in_id": "checkin-123",
            "monitor_slug": HEALTH_PROBES_MONITOR_SLUG,
            "status": health_probes.MonitorStatus.OK,
        }

    def test_failed_probes_close_checkin_with_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkins = self._patch_checkin(monkeypatch)
        canned = {"elapsed_ms": 1, "total": 2, "ok": 1, "failed": 1, "results": []}
        monkeypatch.setattr(health_probes, "run_probes", lambda: canned)

        result = execute_health_probes()

        assert result == canned
        assert checkins[1]["status"] == health_probes.MonitorStatus.ERROR
        assert checkins[1]["check_in_id"] == "checkin-123"

    def test_skipped_result_closes_checkin_with_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkins = self._patch_checkin(monkeypatch)
        skipped = {"skipped": True, "reason": "health_probe_org_or_project_not_set"}
        monkeypatch.setattr(health_probes, "run_probes", lambda: skipped)

        result = execute_health_probes()

        assert result == skipped
        assert checkins[1]["status"] == health_probes.MonitorStatus.ERROR

    def test_run_probes_raising_sends_error_checkin_and_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkins = self._patch_checkin(monkeypatch)

        def _boom() -> dict[str, Any]:
            raise RuntimeError("probe explosion")

        monkeypatch.setattr(health_probes, "run_probes", _boom)

        with pytest.raises(RuntimeError, match="probe explosion"):
            execute_health_probes()

        assert len(checkins) == 2
        assert checkins[0]["status"] == health_probes.MonitorStatus.IN_PROGRESS
        assert checkins[1] == {
            "check_in_id": "checkin-123",
            "monitor_slug": HEALTH_PROBES_MONITOR_SLUG,
            "status": health_probes.MonitorStatus.ERROR,
        }
