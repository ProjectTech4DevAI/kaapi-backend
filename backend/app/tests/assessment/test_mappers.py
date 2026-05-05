"""Tests for assessment/mappers.py."""

from unittest.mock import MagicMock, patch

from app.services.assessment.mappers import (
    _ensure_openai_strict_schema,
    _strip_additional_properties,
    map_kaapi_to_google_assessment_params,
    map_kaapi_to_openai_assessment_params,
    normalize_llm_text,
)


class TestNormalizeLlmText:
    def test_non_string_returns_as_is(self) -> None:
        assert normalize_llm_text(None) is None  # type: ignore[arg-type]
        assert normalize_llm_text(42) == 42  # type: ignore[arg-type]

    def test_empty_string_returns_as_is(self) -> None:
        assert normalize_llm_text("") == ""

    def test_escaped_newline_replaced(self) -> None:
        assert normalize_llm_text("line1\\nline2") == "line1\nline2"

    def test_escaped_tab_replaced(self) -> None:
        assert normalize_llm_text("col1\\tcol2") == "col1\tcol2"

    def test_escaped_quote_replaced(self) -> None:
        assert normalize_llm_text('\\"quoted\\"') == '"quoted"'

    def test_double_backslash_collapsed(self) -> None:
        assert normalize_llm_text("a\\\\b") == "a\\b"

    def test_nfc_normalization_applied(self) -> None:
        # Combining character sequence → precomposed form
        import unicodedata

        text = "é"  # e + combining acute accent
        result = normalize_llm_text(text)
        assert result == unicodedata.normalize("NFC", text)


class TestEnsureOpenAIStrictSchema:
    def test_object_type_gets_additional_properties_false(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _ensure_openai_strict_schema(schema)
        assert result["additionalProperties"] is False

    def test_nested_object_also_gets_flag(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
        }
        result = _ensure_openai_strict_schema(schema)
        assert result["properties"]["address"]["additionalProperties"] is False

    def test_array_items_processed_recursively(self) -> None:
        schema = {
            "type": "array",
            "items": {"type": "object", "properties": {"x": {"type": "number"}}},
        }
        result = _ensure_openai_strict_schema(schema)
        assert result["items"]["additionalProperties"] is False

    def test_non_object_type_not_modified(self) -> None:
        schema = {"type": "string"}
        result = _ensure_openai_strict_schema(schema)
        assert "additionalProperties" not in result


class TestStripAdditionalProperties:
    def test_removes_additional_properties(self) -> None:
        schema = {"type": "object", "additionalProperties": False, "properties": {}}
        result = _strip_additional_properties(schema)
        assert "additionalProperties" not in result

    def test_nested_removal(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"child": {"type": "object", "additionalProperties": False}},
        }
        result = _strip_additional_properties(schema)
        assert "additionalProperties" not in result["properties"]["child"]

    def test_array_items_processed(self) -> None:
        schema = {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False},
        }
        result = _strip_additional_properties(schema)
        assert "additionalProperties" not in result["items"]


class TestMapKaapiToOpenAIParams:
    def _call(self, params: dict, supports_reasoning: bool = False):
        with patch(
            "app.services.assessment.mappers.is_reasoning_model",
            return_value=supports_reasoning,
        ):
            return map_kaapi_to_openai_assessment_params(session=MagicMock(), kaapi_params=params)

    def test_basic_model_passed_through(self) -> None:
        result, warnings = self._call({"model": "gpt-4o"})
        assert result["model"] == "gpt-4o"
        assert warnings == []

    def test_instructions_normalized_and_set(self) -> None:
        result, _ = self._call({"model": "gpt-4o", "instructions": "Be helpful\\n"})
        assert result["instructions"] == "Be helpful\n"

    def test_temperature_set_for_non_reasoning_model(self) -> None:
        result, _ = self._call({"model": "gpt-4o", "temperature": 0.7})
        assert result["temperature"] == 0.7

    def test_temperature_suppressed_for_reasoning_model(self) -> None:
        result, warnings = self._call(
            {"model": "o1", "temperature": 0.5}, supports_reasoning=True
        )
        assert "temperature" not in result
        assert any("temperature" in w for w in warnings)

    def test_top_p_suppressed_for_reasoning_model(self) -> None:
        result, warnings = self._call(
            {"model": "o1", "top_p": 0.9}, supports_reasoning=True
        )
        assert "top_p" not in result
        assert any("top_p" in w for w in warnings)

    def test_effort_set_for_reasoning_model(self) -> None:
        result, _ = self._call(
            {"model": "o1", "effort": "high"}, supports_reasoning=True
        )
        assert result["reasoning"]["effort"] == "high"

    def test_effort_suppressed_for_non_reasoning_model(self) -> None:
        result, warnings = self._call({"model": "gpt-4o", "effort": "high"})
        assert "reasoning" not in result
        assert any("effort" in w for w in warnings)

    def test_output_schema_sets_text_format(self) -> None:
        schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
        result, _ = self._call({"model": "gpt-4o", "output_schema": schema})
        assert result["text"]["format"]["type"] == "json_schema"
        assert result["text"]["format"]["strict"] is True

    def test_response_format_text_not_set(self) -> None:
        result, _ = self._call({"model": "gpt-4o", "response_format": "text"})
        assert "text" not in result

    def test_knowledge_base_ids_sets_tools(self) -> None:
        result, _ = self._call(
            {"model": "gpt-4o", "knowledge_base_ids": ["vs_123"], "max_num_results": 10}
        )
        assert result["tools"][0]["type"] == "file_search"
        assert result["tools"][0]["max_num_results"] == 10

    def test_summary_null_string_sets_none(self) -> None:
        result, _ = self._call(
            {"model": "o1", "summary": "null"}, supports_reasoning=True
        )
        assert result["reasoning"]["summary"] is None

    def test_top_p_set_for_non_reasoning_model(self) -> None:
        result, _ = self._call({"model": "gpt-4o", "top_p": 0.85})
        assert result["top_p"] == 0.85


class TestMapKaapiToGoogleParams:
    def _call(self, params: dict):
        mock_schema = MagicMock()
        mock_schema.model_dump.return_value = {}
        with patch(
            "app.services.assessment.mappers.genai_transformers.t_schema",
            return_value=mock_schema,
        ):
            return map_kaapi_to_google_assessment_params(params)

    def test_missing_model_returns_warning(self) -> None:
        result, warnings = map_kaapi_to_google_assessment_params({})
        assert result == {}
        assert any("model" in w for w in warnings)

    def test_basic_model_set(self) -> None:
        result, _ = self._call({"model": "gemini-1.5-pro"})
        assert result["model"] == "gemini-1.5-pro"

    def test_temperature_set(self) -> None:
        result, _ = self._call({"model": "gemini-1.5-pro", "temperature": 0.3})
        assert result["temperature"] == 0.3

    def test_top_p_set(self) -> None:
        result, _ = self._call({"model": "gemini-1.5-pro", "top_p": 0.8})
        assert result["top_p"] == 0.8

    def test_thinking_level_set(self) -> None:
        result, _ = self._call(
            {"model": "gemini-2.0-flash-thinking", "thinking_level": "high"}
        )
        assert result["thinking_config"] == {"thinking_level": "high"}

    def test_knowledge_base_ids_warns(self) -> None:
        result, warnings = self._call(
            {"model": "gemini-1.5-pro", "knowledge_base_ids": ["kb_1"]}
        )
        assert any("knowledge_base_ids" in w for w in warnings)

    def test_output_schema_set(self) -> None:
        schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
        result, _ = self._call({"model": "gemini-1.5-pro", "output_schema": schema})
        assert "output_schema" in result

    def test_instructions_normalized(self) -> None:
        result, _ = self._call(
            {"model": "gemini-1.5-pro", "instructions": "Be kind\\n"}
        )
        assert result["instructions"] == "Be kind\n"

    def test_max_output_tokens_set(self) -> None:
        result, _ = self._call({"model": "gemini-1.5-pro", "max_output_tokens": 512})
        assert result["max_output_tokens"] == 512

    def test_reasoning_set(self) -> None:
        result, _ = self._call(
            {"model": "gemini-2.0-flash-thinking", "reasoning": "high"}
        )
        assert result["reasoning"] == "high"


class TestConvertJsonSchemaToGoogle:
    def _call(self, schema: dict) -> dict:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"properties": {"score": {}}}
        with patch(
            "app.services.assessment.mappers.genai_transformers.t_schema",
            return_value=mock_result,
        ):
            from app.services.assessment.mappers import _convert_json_schema_to_google

            return _convert_json_schema_to_google(schema)

    def test_property_ordering_added_from_required(self) -> None:
        schema = {
            "type": "object",
            "required": ["score", "reason"],
            "properties": {"score": {}, "reason": {}},
        }
        result = self._call(schema)
        assert result["propertyOrdering"] == ["score", "reason"]

    def test_property_ordering_falls_back_to_keys(self) -> None:
        schema = {"type": "object", "properties": {"a": {}, "b": {}}}
        result = self._call(schema)
        assert "propertyOrdering" in result


class TestOpenAIResponseFormat:
    def _call(self, params: dict):
        with patch(
            "app.services.assessment.mappers.is_reasoning_model",
            return_value=False,
        ):
            return map_kaapi_to_openai_assessment_params(session=MagicMock(), kaapi_params=params)

    def test_non_text_response_format_sets_text_field(self) -> None:
        result, _ = self._call({"model": "gpt-4o", "response_format": "json_object"})
        assert result["text"]["format"]["type"] == "json_object"
