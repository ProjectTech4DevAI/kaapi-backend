import pytest

from app.core.providers import Provider
from app.services.tts_evaluations.constants import (
    SUPPORTED_TTS_MODELS,
    TTSExecutionModeEnum,
    get_tts_model_spec,
    split_models_by_execution_mode,
)


class TestGetTTSModelSpec:
    @pytest.mark.parametrize(
        ("model", "provider", "mode"),
        [
            (
                "gemini-2.5-pro-preview-tts",
                Provider.GOOGLE_AISTUDIO,
                TTSExecutionModeEnum.BATCH,
            ),
            ("bulbul:v3", Provider.SARVAMAI, TTSExecutionModeEnum.SYNC),
            ("eleven_v3", Provider.ELEVENLABS, TTSExecutionModeEnum.SYNC),
            ("eleven_v4", Provider.ELEVENLABS, TTSExecutionModeEnum.SYNC),
        ],
    )
    def test_registry_routes_models(
        self, model: str, provider: Provider, mode: TTSExecutionModeEnum
    ) -> None:
        spec = get_tts_model_spec(model)

        assert spec.provider == provider
        assert spec.execution_mode == mode
        assert spec.default_voice

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_tts_model_spec("unknown-tts")

    def test_supported_models_match_registry(self) -> None:
        assert set(SUPPORTED_TTS_MODELS) == {
            "gemini-2.5-pro-preview-tts",
            "bulbul:v3",
            "eleven_v3",
            "eleven_v4",
        }


class TestSplitModelsByExecutionMode:
    def test_mixed_models_split_preserving_order(self) -> None:
        batch, sync = split_models_by_execution_mode(
            ["eleven_v3", "gemini-2.5-pro-preview-tts", "bulbul:v3"]
        )

        assert batch == ["gemini-2.5-pro-preview-tts"]
        assert sync == ["eleven_v3", "bulbul:v3"]

    def test_empty_list(self) -> None:
        assert split_models_by_execution_mode([]) == ([], [])

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            split_models_by_execution_mode(["nope"])
