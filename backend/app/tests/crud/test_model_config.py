from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from app.crud import model_config as model_config_crud


def _patch_model(
    monkeypatch: pytest.MonkeyPatch,
    pricing: Any,
) -> None:
    model = SimpleNamespace(pricing=pricing)
    monkeypatch.setattr(
        model_config_crud,
        "get_model_config",
        lambda session, provider, model_name: model,
    )


def test_estimate_model_cost_response_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model(
        monkeypatch,
        pricing={
            "response": {"input_token_cost": 2.5, "output_token_cost": 10.0},
            "batch": {"input_token_cost": 1.25, "output_token_cost": 5.0},
        },
    )

    result = model_config_crud.estimate_model_cost(
        session=None,  # type: ignore[arg-type]
        provider="openai",
        model_name="gpt-4o",
        input_tokens=1_000_000,
        output_tokens=500_000,
        usage_type="response",
    )

    assert result is not None
    assert result["usage_type"] == "response"
    assert result["input_cost"] == 2.5
    assert result["output_cost"] == 5.0
    assert result["total_cost"] == 7.5


def test_estimate_model_cost_batch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model(
        monkeypatch,
        pricing={
            "response": {"input_token_cost": 2.5, "output_token_cost": 10.0},
            "batch": {"input_token_cost": 1.25, "output_token_cost": 5.0},
        },
    )

    result = model_config_crud.estimate_model_cost(
        session=None,  # type: ignore[arg-type]
        provider="openai",
        model_name="gpt-4o",
        input_tokens=1_000_000,
        output_tokens=500_000,
        usage_type="batch",
    )

    assert result is not None
    assert result["usage_type"] == "batch"
    assert result["input_cost"] == 1.25
    assert result["output_cost"] == 2.5
    assert result["total_cost"] == 3.75


def test_estimate_model_cost_returns_none_for_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_config_crud,
        "get_model_config",
        lambda session, provider, model_name: None,
    )

    result = model_config_crud.estimate_model_cost(
        session=None,  # type: ignore[arg-type]
        provider="openai",
        model_name="does-not-exist",
        input_tokens=1000,
        output_tokens=1000,
    )

    assert result is None


def test_estimate_model_cost_returns_none_for_null_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch, pricing=None)

    result = model_config_crud.estimate_model_cost(
        session=None,  # type: ignore[arg-type]
        provider="openai",
        model_name="gpt-4o",
        input_tokens=1000,
        output_tokens=1000,
    )

    assert result is None


def test_estimate_model_cost_returns_none_for_non_dict_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch, pricing=["invalid"])

    result = model_config_crud.estimate_model_cost(
        session=None,  # type: ignore[arg-type]
        provider="openai",
        model_name="gpt-4o",
        input_tokens=1000,
        output_tokens=1000,
    )

    assert result is None


def test_estimate_model_cost_returns_none_for_missing_usage_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(
        monkeypatch,
        pricing={"response": {"input_token_cost": 1.0, "output_token_cost": 2.0}},
    )

    result = model_config_crud.estimate_model_cost(
        session=None,  # type: ignore[arg-type]
        provider="openai",
        model_name="gpt-4o",
        input_tokens=1000,
        output_tokens=1000,
        usage_type="batch",
    )

    assert result is None


def test_estimate_model_cost_returns_none_for_non_numeric_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(
        monkeypatch,
        pricing={
            "response": {"input_token_cost": "cheap", "output_token_cost": "expensive"}
        },
    )

    result = model_config_crud.estimate_model_cost(
        session=None,  # type: ignore[arg-type]
        provider="openai",
        model_name="gpt-4o",
        input_tokens=1000,
        output_tokens=1000,
        usage_type="response",
    )

    assert result is None


def _make_blob(
    provider: str | None, completion_type: str, params: Mapping[str, Any]
) -> SimpleNamespace:
    completion = SimpleNamespace(provider=provider, type=completion_type, params=params)
    return SimpleNamespace(completion=completion)


def _patch_validators(
    monkeypatch: pytest.MonkeyPatch,
    *,
    row: Any | None,
    supported: bool,
    allowed: list[str] | None = None,
) -> None:
    monkeypatch.setattr(
        model_config_crud,
        "get_model_config",
        lambda session, provider, model_name: row,
    )
    monkeypatch.setattr(
        model_config_crud,
        "is_model_supported",
        lambda session, provider, completion_type, model_name: supported,
    )
    monkeypatch.setattr(
        model_config_crud,
        "list_supported_models",
        lambda session, provider, completion_type: allowed or [],
    )


def test_validate_blob_native_provider_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native pass-through never hits DB."""
    called = {"hit": False}

    def boom(*_args: Any, **_kwargs: Any) -> None:
        called["hit"] = True

    monkeypatch.setattr(model_config_crud, "get_model_config", boom)
    monkeypatch.setattr(model_config_crud, "is_model_supported", boom)

    blob = _make_blob("openai-native", "text", {"model": "anything"})
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]

    assert called["hit"] is False


def test_validate_blob_none_provider_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = _make_blob(None, "text", {"model": "gpt-4o"})
    # No patches — should never reach helpers
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]


def test_validate_blob_missing_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = _make_blob("openai", "text", {"temperature": 0.5})
    with pytest.raises(HTTPException) as exc:
        model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    assert "model is required" in exc.value.detail


def test_validate_blob_model_not_found_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing model logs a warning and lets the request proceed."""
    _patch_validators(monkeypatch, row=None, supported=False)
    blob = _make_blob("openai", "text", {"model": "gpt-4-turbo"})
    with caplog.at_level("WARNING"):
        model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]
    assert "gpt-4-turbo" in caplog.text
    assert "not found" in caplog.text


def test_validate_blob_wrong_type_for_model_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong completion type no longer validated — request proceeds silently."""
    row = SimpleNamespace(config={})
    _patch_validators(
        monkeypatch,
        row=row,
        supported=False,
        allowed=["gpt-4o", "gpt-4o-mini"],
    )
    blob = _make_blob("openai", "text", {"model": "some-audio-model"})
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]


def test_validate_blob_supported_text_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    row = SimpleNamespace(config={})
    _patch_validators(monkeypatch, row=row, supported=True)
    blob = _make_blob("openai", "text", {"model": "gpt-4o"})
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]


def test_validate_blob_tts_invalid_voice_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid TTS voice logs a warning but does not raise."""
    row = SimpleNamespace(
        config={"voice": {"type": "enum", "options": ["Kore", "Orus"]}}
    )
    _patch_validators(monkeypatch, row=row, supported=True)
    blob = _make_blob(
        "google",
        "tts",
        {"model": "gemini-2.5-flash-preview-tts", "voice": "Sarah"},
    )
    with caplog.at_level("WARNING"):
        model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]
    assert "Sarah" in caplog.text
    assert "Kore" in caplog.text


def test_validate_blob_tts_valid_voice_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = SimpleNamespace(
        config={"voice": {"type": "enum", "options": ["Kore", "Orus"]}}
    )
    _patch_validators(monkeypatch, row=row, supported=True)
    blob = _make_blob(
        "google",
        "tts",
        {"model": "gemini-2.5-flash-preview-tts", "voice": "Kore"},
    )
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]


def test_validate_blob_tts_no_voice_spec_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If model_config row has no voice schema, voice value is not enforced."""
    row = SimpleNamespace(config={})
    _patch_validators(monkeypatch, row=row, supported=True)
    blob = _make_blob("sarvamai", "tts", {"model": "bulbul:v3", "voice": "anything"})
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]


def test_validate_blob_stt_model_passes_for_text_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion-type mismatch no longer enforced — STT model passes for type=text."""
    row = SimpleNamespace(config={})
    _patch_validators(
        monkeypatch,
        row=row,
        supported=False,
        allowed=["gpt-4o", "gpt-4o-mini"],
    )
    blob = _make_blob("google", "text", {"model": "gemini-2.5-pro"})
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]


def test_validate_blob_text_model_accepted_for_text_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid text model passes for type=text."""
    row = SimpleNamespace(config={})
    _patch_validators(monkeypatch, row=row, supported=True)
    blob = _make_blob("openai", "text", {"model": "gpt-4o"})
    model_config_crud.validate_blob_model_or_raise(session=None, blob=blob)  # type: ignore[arg-type]
