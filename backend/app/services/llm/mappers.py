"""Parameter mappers for converting Kaapi-abstracted parameters to provider-specific formats."""

import litellm
from app.models.llm import KaapiLLMParams, KaapiCompletionConfig, NativeCompletionConfig


def map_kaapi_to_openai_params(kaapi_params: KaapiLLMParams) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to OpenAI API parameters.

    This mapper transforms standardized Kaapi parameters into OpenAI-specific
    parameter format, enabling provider-agnostic interface design.

    Args:
        kaapi_params: KaapiLLMParams instance with standardized parameters

    Supported Mapping:
        - model → model
        - instructions → instructions
        - knowledge_base_ids → tools[file_search].vector_store_ids
        - max_num_results → tools[file_search].max_num_results (fallback default)
        - reasoning → reasoning.effort (if reasoning supported by model else suppressed)
        - temperature → temperature (if reasoning not supported by model else suppressed)

    Returns:
        Tuple of:
        - Dictionary of OpenAI API parameters ready to be passed to the API
        - List of warnings describing suppressed or ignored parameters
    """
    openai_params = {}
    warnings = []

    support_reasoning = litellm.supports_reasoning(
        model="openai/" + f"{kaapi_params.model}"
    )

    # Handle reasoning vs temperature mutual exclusivity
    if support_reasoning:
        if kaapi_params.reasoning is not None:
            openai_params["reasoning"] = {"effort": kaapi_params.reasoning}

        if kaapi_params.temperature is not None:
            warnings.append(
                "Parameter 'temperature' was suppressed because the selected model "
                "supports reasoning, and temperature is ignored when reasoning is enabled."
            )
    else:
        if kaapi_params.reasoning is not None:
            warnings.append(
                "Parameter 'reasoning' was suppressed because the selected model "
                "does not support reasoning."
            )

        if kaapi_params.temperature is not None:
            openai_params["temperature"] = kaapi_params.temperature

    if kaapi_params.model:
        openai_params["model"] = kaapi_params.model

    if kaapi_params.instructions:
        openai_params["instructions"] = kaapi_params.instructions

    if kaapi_params.knowledge_base_ids:
        openai_params["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": kaapi_params.knowledge_base_ids,
                "max_num_results": kaapi_params.max_num_results or 20,
            }
        ]

    return openai_params, warnings


def map_kaapi_to_google_params(kaapi_params: KaapiLLMParams) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to Google AI (Gemini) API parameters.

    This mapper transforms standardized Kaapi parameters into Google-specific
    parameter format for the Gemini API.

    Args:
        kaapi_params: KaapiLLMParams instance with standardized parameters

    Supported Mapping:
        - model → model
        - instructions → instructions (for STT prompts, if available)

    Returns:
        Tuple of:
        - Dictionary of Google AI API parameters ready to be passed to the API
        - List of warnings describing suppressed or ignored parameters
    """
    google_params = {}
    warnings = []

    # Model is present in all param types
    google_params["model"] = kaapi_params.model

    # Instructions only exists in TextLLMParams, use getattr for optional access
    instructions = getattr(kaapi_params, "instructions", None)
    if instructions:
        google_params["instructions"] = instructions

    # Warn about unsupported parameters that may be present in TextLLMParams
    if getattr(kaapi_params, "knowledge_base_ids", None):
        warnings.append(
            "Parameter 'knowledge_base_ids' is not supported by Google AI and was ignored."
        )

    if getattr(kaapi_params, "temperature", None) is not None:
        warnings.append(
            "Parameter 'temperature' is not applicable for Google AI STT and was ignored."
        )

    if getattr(kaapi_params, "reasoning", None) is not None:
        warnings.append(
            "Parameter 'reasoning' is not applicable for Google AI and was ignored."
        )

    return google_params, warnings


def transform_kaapi_config_to_native(
    kaapi_config: KaapiCompletionConfig,
) -> tuple[NativeCompletionConfig, list[str]]:
    """Transform Kaapi completion config to native provider config with mapped parameters.

    Supports OpenAI and Google AI providers.

    Args:
        kaapi_config: KaapiCompletionConfig with abstracted parameters

    Returns:
        Tuple of:
        - NativeCompletionConfig with provider-native parameters ready for API
        - List of warnings for suppressed/ignored parameters
    """
    if kaapi_config.provider == "openai":
        mapped_params, warnings = map_kaapi_to_openai_params(kaapi_config.params)
        return (
            NativeCompletionConfig(
                provider="openai-native", params=mapped_params, type=kaapi_config.type
            ),
            warnings,
        )

    if kaapi_config.provider == "google":
        mapped_params, warnings = map_kaapi_to_google_params(kaapi_config.params)
        return (
            NativeCompletionConfig(
                provider="google-native", params=mapped_params, type=kaapi_config.type
            ),
            warnings,
        )

    raise ValueError(f"Unsupported provider: {kaapi_config.provider}")
