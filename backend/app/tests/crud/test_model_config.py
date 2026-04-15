from types import SimpleNamespace
from typing import Any

import pytest

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
