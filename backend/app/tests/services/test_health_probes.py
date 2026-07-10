from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.services import health_probes
from app.services.health_probes import _PROBES, run_probes


class _FakeProvider:
    """Minimal BaseProvider stand-in whose execute() returns a canned tuple."""

    def __init__(self, response=None, error=None, raise_exc: Exception | None = None):
        self._response = response
        self._error = error
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    def execute(self, *, completion_config, query, resolved_input):
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
def configured_probe_settings(monkeypatch):
    monkeypatch.setattr(settings, "HEALTH_PROBE_ORG_ID", 1)
    monkeypatch.setattr(settings, "HEALTH_PROBE_PROJECT_ID", 1)


class TestSkipWhenNotConfigured:
    def test_returns_skipped_when_org_id_missing(self, db: Session, monkeypatch):
        monkeypatch.setattr(settings, "HEALTH_PROBE_ORG_ID", None)
        monkeypatch.setattr(settings, "HEALTH_PROBE_PROJECT_ID", 1)

        called = {"n": 0}

        def _tracker(**_kwargs):
            called["n"] += 1
            return _FakeProvider(response={"ok": True})

        monkeypatch.setattr(health_probes, "get_llm_provider", _tracker)

        result = run_probes(session=db)

        assert result == {
            "skipped": True,
            "reason": "health_probe_org_or_project_not_set",
        }
        assert called["n"] == 0

    def test_returns_skipped_when_project_id_missing(self, db: Session, monkeypatch):
        monkeypatch.setattr(settings, "HEALTH_PROBE_ORG_ID", 1)
        monkeypatch.setattr(settings, "HEALTH_PROBE_PROJECT_ID", None)
        monkeypatch.setattr(
            health_probes,
            "get_llm_provider",
            lambda **_k: pytest.fail("provider must not be built"),
        )

        result = run_probes(session=db)

        assert result["skipped"] is True
        assert result["reason"] == "health_probe_org_or_project_not_set"


class TestRunProbesShape:
    def test_all_probes_ok_returns_expected_structure(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        fake = _FakeProvider(response={"data": "pong"}, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes(session=db)

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


class TestProviderInitFailure:
    def test_get_llm_provider_value_error_marks_client_init_failed(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        def _boom(**_kwargs):
            raise ValueError("no creds")

        monkeypatch.setattr(health_probes, "get_llm_provider", _boom)

        result = run_probes(session=db)

        assert result["ok"] == 0
        assert result["failed"] == len(_PROBES)
        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"] == "client_init_failed"
            assert r["latency_ms"] is None

    def test_get_llm_provider_runtime_error_marks_client_init_failed(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        monkeypatch.setattr(
            health_probes,
            "get_llm_provider",
            lambda **_k: (_ for _ in ()).throw(RuntimeError("bad")),
        )

        result = run_probes(session=db)

        assert result["failed"] == len(_PROBES)
        assert all(r["error"] == "client_init_failed" for r in result["results"])
        assert all(r["latency_ms"] is None for r in result["results"])


class TestExecuteBehaviors:
    def test_execute_raises_marks_error_with_exception_class_name(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        fake = _FakeProvider(raise_exc=TimeoutError("slow"))
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes(session=db)

        assert result["ok"] == 0
        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"].startswith("TimeoutError")
            assert "slow" in r["error"]
            assert isinstance(r["latency_ms"], int)

    def test_provider_returns_none_response_uses_provided_error(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        fake = _FakeProvider(response=None, error="some_error")
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes(session=db)

        assert result["ok"] == 0
        assert result["failed"] == len(_PROBES)
        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"] == "some_error"
            assert isinstance(r["latency_ms"], int)

    def test_provider_returns_none_response_and_no_error_defaults_no_response(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        fake = _FakeProvider(response=None, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes(session=db)

        for r in result["results"]:
            assert r["ok"] is False
            assert r["error"] == "no_response"

    def test_provider_returns_response_marks_ok(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        fake = _FakeProvider(response={"ok": 1}, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes(session=db)

        assert result["ok"] == len(_PROBES)
        for r in result["results"]:
            assert r["ok"] is True
            assert r["error"] is None


class TestSttAudioNotConfigured:
    def test_empty_stt_audio_short_circuits_stt_probes(
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        monkeypatch.setattr(health_probes, "_STT_AUDIO_B64", "")
        fake = _FakeProvider(response={"ok": 1}, error=None)
        monkeypatch.setattr(health_probes, "get_llm_provider", lambda **_k: fake)

        result = run_probes(session=db)

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
        self, db: Session, monkeypatch, configured_probe_settings
    ):
        seen_providers: list[str] = []

        def _factory(**kwargs):
            seen_providers.append(kwargs["provider_type"])
            return _FakeProvider(response={"ok": 1}, error=None)

        monkeypatch.setattr(health_probes, "get_llm_provider", _factory)

        run_probes(session=db)

        unique_probe_providers = {p.provider for p in _PROBES}
        # One factory call per unique provider name, not per probe
        assert sorted(seen_providers) == sorted(unique_probe_providers)
        assert len(seen_providers) < len(_PROBES)
