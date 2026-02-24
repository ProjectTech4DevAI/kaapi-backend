"""Parameter mappers for converting Kaapi-abstracted parameters to provider-specific formats."""

import litellm
from app.models.llm import KaapiCompletionConfig, NativeCompletionConfig


def map_kaapi_to_openai_params(kaapi_params: dict) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to OpenAI API parameters.

    This mapper transforms standardized Kaapi parameters into OpenAI-specific
    parameter format, enabling provider-agnostic interface design.

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters

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

    model = kaapi_params.get("model")
    reasoning = kaapi_params.get("reasoning")
    temperature = kaapi_params.get("temperature")
    instructions = kaapi_params.get("instructions")
    knowledge_base_ids = kaapi_params.get("knowledge_base_ids")
    max_num_results = kaapi_params.get("max_num_results")

    support_reasoning = litellm.supports_reasoning(model=f"openai/{model}")

    # Handle reasoning vs temperature mutual exclusivity
    if support_reasoning:
        if reasoning is not None:
            openai_params["reasoning"] = {"effort": reasoning}

        if temperature is not None:
            warnings.append(
                "Parameter 'temperature' was suppressed because the selected model "
                "supports reasoning, and temperature is ignored when reasoning is enabled."
            )
    else:
        if reasoning is not None:
            warnings.append(
                "Parameter 'reasoning' was suppressed because the selected model "
                "does not support reasoning."
            )

        if temperature is not None:
            openai_params["temperature"] = temperature

    if model:
        openai_params["model"] = model

    if instructions:
        openai_params["instructions"] = instructions

    if knowledge_base_ids:
        openai_params["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": knowledge_base_ids,
                "max_num_results": max_num_results or 20,
            }
        ]

    return openai_params, warnings


def map_kaapi_to_google_params(kaapi_params: dict) -> tuple[dict, list[str]]:
    """Map Kaapi-abstracted parameters to Google AI (Gemini) API parameters.

    This mapper transforms standardized Kaapi parameters into Google-specific
    parameter format for the Gemini API.

    Args:
        kaapi_params: Dictionary with standardized Kaapi parameters

    Supported Mapping:
        - model → model
        - instructions → instructions (for STT prompts, if available)
        - temperature -> temperature parameter (0-2)

    Returns:
        Tuple of:
        - Dictionary of Google AI API parameters ready to be passed to the API
        - List of warnings describing suppressed or ignored parameters
    """
    google_params = {}
    warnings = []

    # Model is present in all param types
    model = kaapi_params.get("model")
    if not model:
        return {}, ["Missing required 'model' parameter"]

    google_params["model"] = kaapi_params.get("model")

    # Instructions for STT prompts
    instructions = kaapi_params.get("instructions")
    if instructions:
        google_params["instructions"] = instructions

    temperature = kaapi_params.get("temperature")

    if temperature is not None:
        google_params["temperature"] = temperature

    # TTS Config
    voice = kaapi_params.get("voice")
    if voice:
        google_params["voice"] = voice

    language = kaapi_params.get("language")
    if language:
        google_params["language"] = language

    response_format = kaapi_params.get("response_format")
    if response_format:
        google_params["response_format"] = response_format
    # Warn about unsupported parameters
    if kaapi_params.get("knowledge_base_ids"):
        warnings.append(
            "Parameter 'knowledge_base_ids' is not supported by Google AI and was ignored."
        )

    if kaapi_params.get("reasoning") is not None:
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
