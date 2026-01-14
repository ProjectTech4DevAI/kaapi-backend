"""
Unit tests for LLM parameter mapping functions.

Tests the transformation of Kaapi-abstracted parameters to provider-native formats.
"""
import pytest

from app.models.llm import KaapiLLMParams, KaapiCompletionConfig, NativeCompletionConfig
from app.services.llm.mappers import (
    map_kaapi_to_openai_params,
    transform_kaapi_config_to_native,
)


class TestMapKaapiToOpenAIParams:
    """Test cases for map_kaapi_to_openai_params function."""

    def test_basic_model_mapping(self):
        """Test basic model parameter mapping."""
        kaapi_params = KaapiLLMParams(model="gpt-4o")

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result == {"model": "gpt-4o"}
        assert warnings == []

    def test_instructions_mapping(self):
        """Test instructions parameter mapping."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4",
            instructions="You are a helpful assistant.",
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "gpt-4"
        assert result["instructions"] == "You are a helpful assistant."
        assert warnings == []

    def test_temperature_mapping(self):
        """Test temperature parameter mapping for non-reasoning models."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4",
            temperature=0.7,
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert warnings == []

    def test_temperature_zero_mapping(self):
        """Test that temperature=0 is correctly mapped (edge case)."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4",
            temperature=0.0,
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["temperature"] == 0.0
        assert warnings == []

    def test_reasoning_mapping_for_reasoning_models(self):
        """Test reasoning parameter mapping to OpenAI format for reasoning-capable models."""
        kaapi_params = KaapiLLMParams(
            model="o1",
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "o1"
        assert result["reasoning"] == {"effort": "high"}
        assert warnings == []

    def test_knowledge_base_ids_mapping(self):
        """Test knowledge_base_ids mapping to OpenAI tools format."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4",
            knowledge_base_ids=["vs_abc123", "vs_def456"],
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "gpt-4"
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "file_search"
        assert result["tools"][0]["vector_store_ids"] == ["vs_abc123", "vs_def456"]
        assert result["tools"][0]["max_num_results"] == 20  # default
        assert warnings == []

    def test_knowledge_base_with_max_num_results(self):
        """Test knowledge_base_ids with custom max_num_results."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4",
            knowledge_base_ids=["vs_abc123"],
            max_num_results=50,
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["tools"][0]["max_num_results"] == 50
        assert warnings == []

    def test_complete_parameter_mapping(self):
        """Test mapping all compatible parameters together."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4o",
            instructions="You are an expert assistant.",
            temperature=0.8,
            knowledge_base_ids=["vs_123"],
            max_num_results=30,
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "gpt-4o"
        assert result["instructions"] == "You are an expert assistant."
        assert result["temperature"] == 0.8
        assert result["tools"][0]["type"] == "file_search"
        assert result["tools"][0]["vector_store_ids"] == ["vs_123"]
        assert result["tools"][0]["max_num_results"] == 30
        assert warnings == []

    def test_reasoning_suppressed_for_non_reasoning_models(self):
        """Test that reasoning is suppressed with warning for non-reasoning models."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4",
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "gpt-4"
        assert "reasoning" not in result
        assert len(warnings) == 1
        assert "reasoning" in warnings[0].lower()
        assert "does not support reasoning" in warnings[0]

    def test_temperature_suppressed_for_reasoning_models(self):
        """Test that temperature is suppressed with warning for reasoning models when reasoning is set."""
        kaapi_params = KaapiLLMParams(
            model="o1",
            temperature=0.7,
            reasoning="high",
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "o1"
        assert result["reasoning"] == {"effort": "high"}
        assert "temperature" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "suppressed" in warnings[0]

    def test_temperature_without_reasoning_for_reasoning_models(self):
        """Test that temperature is suppressed for reasoning models even without explicit reasoning parameter."""
        kaapi_params = KaapiLLMParams(
            model="o1",
            temperature=0.7,
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "o1"
        assert "temperature" not in result
        assert "reasoning" not in result
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "suppressed" in warnings[0]

    def test_minimal_params(self):
        """Test mapping with minimal parameters (only model)."""
        kaapi_params = KaapiLLMParams(model="gpt-4")

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result == {"model": "gpt-4"}
        assert warnings == []

    def test_only_knowledge_base_ids(self):
        """Test mapping with only knowledge_base_ids and model."""
        kaapi_params = KaapiLLMParams(
            model="gpt-4",
            knowledge_base_ids=["vs_xyz"],
        )

        result, warnings = map_kaapi_to_openai_params(kaapi_params)

        assert result["model"] == "gpt-4"
        assert "tools" in result
        assert result["tools"][0]["vector_store_ids"] == ["vs_xyz"]
        assert warnings == []


class TestTransformKaapiConfigToNative:
    """Test cases for transform_kaapi_config_to_native function."""

    def test_transform_openai_config(self):
        """Test transformation of Kaapi OpenAI config to native format."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            params=KaapiLLMParams(
                model="gpt-4",
                temperature=0.7,
            ),
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert isinstance(result, NativeCompletionConfig)
        assert result.provider == "openai-native"
        assert result.params["model"] == "gpt-4"
        assert result.params["temperature"] == 0.7
        assert warnings == []

    def test_transform_with_all_params(self):
        """Test transformation with all Kaapi parameters."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            params=KaapiLLMParams(
                model="gpt-4o",
                instructions="System prompt here",
                temperature=0.5,
                knowledge_base_ids=["vs_abc"],
                max_num_results=25,
            ),
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "openai-native"
        assert result.params["model"] == "gpt-4o"
        assert result.params["instructions"] == "System prompt here"
        assert result.params["temperature"] == 0.5
        assert result.params["tools"][0]["type"] == "file_search"
        assert result.params["tools"][0]["max_num_results"] == 25
        assert warnings == []

    def test_transform_with_reasoning(self):
        """Test transformation with reasoning parameter for reasoning-capable models."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            params=KaapiLLMParams(
                model="o1",
                reasoning="medium",
            ),
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "openai-native"
        assert result.params["model"] == "o1"
        assert result.params["reasoning"] == {"effort": "medium"}
        assert warnings == []

    def test_transform_with_both_temperature_and_reasoning(self):
        """Test that transformation handles temperature + reasoning intelligently for reasoning models."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            params=KaapiLLMParams(
                model="o1",
                temperature=0.7,
                reasoning="high",
            ),
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        assert result.provider == "openai-native"
        assert result.params["model"] == "o1"
        assert result.params["reasoning"] == {"effort": "high"}
        assert "temperature" not in result.params
        assert len(warnings) == 1
        assert "temperature" in warnings[0].lower()
        assert "suppressed" in warnings[0]

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported providers raise ValueError."""
        # Note: This would require modifying KaapiCompletionConfig to accept other providers
        # For now, this tests the error handling in the mapper
        # We'll create a mock config that bypasses validation
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.provider = "unsupported-provider"
        mock_config.params = KaapiLLMParams(model="some-model")

        with pytest.raises(ValueError) as exc_info:
            transform_kaapi_config_to_native(mock_config)

        assert "Unsupported provider" in str(exc_info.value)

    def test_transform_preserves_param_structure(self):
        """Test that transformation correctly structures nested parameters."""
        kaapi_config = KaapiCompletionConfig(
            provider="openai",
            params=KaapiLLMParams(
                model="gpt-4",
                knowledge_base_ids=["vs_1", "vs_2", "vs_3"],
                max_num_results=15,
            ),
        )

        result, warnings = transform_kaapi_config_to_native(kaapi_config)

        # Verify the nested structure is correct
        assert isinstance(result.params["tools"], list)
        assert isinstance(result.params["tools"][0], dict)
        assert isinstance(result.params["tools"][0]["vector_store_ids"], list)
        assert len(result.params["tools"][0]["vector_store_ids"]) == 3
        assert warnings == []
