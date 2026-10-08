class AgentNotConfiguredError(Exception):
    """The platform Anthropic key is missing, so the agent cannot run at all."""


class AgentLLMError(Exception):
    """The Anthropic call behind an agent turn failed; the message is caller-safe."""
