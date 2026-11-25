"""Request-body examples for the LLM routes.


"""

from app.api.openapi_docs import Examples

LLM_CALL_EXAMPLES: Examples = {
    "stored_config": (
        "Call with a stored config",
        {
            "query": {
                "input": "What are the early signs of dehydration in children?",
                "conversation": {"auto_create": True},
            },
            "config": {
                "id": "3f2b8c1e-7d4a-4b6e-9c2f-1a5d8e7b9c40",
                "version": 1,
            },
            "callback_url": "https://example.com/webhooks/llm-call",
            "request_metadata": {"request_id": "req-20261116-0001"},
        },
    ),
    "inline_config": (
        "Call with an inline config blob",
        {
            "query": {
                "input": "Summarise the benefits of handwashing in two sentences."
            },
            "config": {
                "blob": {
                    "completion": {
                        "provider": "openai",
                        "type": "text",
                        "params": {
                            "model": "gpt-4o-mini",
                            "instructions": "You are a concise public-health assistant.",
                            "temperature": 0.2,
                        },
                    },
                    "input_guardrails": [{"type": "pii_remover", "on_fail": "fix"}],
                }
            },
            "callback_url": "https://example.com/webhooks/llm-call",
        },
    ),
}

LLM_CHAIN_EXAMPLES: Examples = {
    "answer_then_translate": (
        "Answer with a stored config, then translate",
        {
            "query": {
                "input": "My child has had a fever for three days, what should I do?"
            },
            "blocks": [
                {
                    "config": {
                        "id": "3f2b8c1e-7d4a-4b6e-9c2f-1a5d8e7b9c40",
                        "version": 1,
                    },
                    "intermediate_callback": True,
                },
                {
                    "config": {
                        "blob": {
                            "completion": {
                                "provider": "openai",
                                "type": "text",
                                "params": {
                                    "model": "gpt-4o-mini",
                                    "instructions": "Translate the text into simple Hindi.",
                                },
                            },
                            "prompt_template": {
                                "template": "Translate this answer: {{input}}"
                            },
                        }
                    }
                },
            ],
            "callback_url": "https://example.com/webhooks/llm-chain",
            "request_metadata": {"request_id": "chain-20261116-0001"},
        },
    ),
}

SPEECH_TO_SPEECH_EXAMPLES: Examples = {
    "voice_note": (
        "Answer a Hindi voice note from a knowledge base",
        {
            "query": {
                "type": "audio",
                "content": {
                    "format": "url",
                    "value": "https://example.com/media/voice-note.ogg",
                    "mime_type": "audio/ogg",
                },
            },
            "knowledge_base_ids": ["vs_abc123"],
            "input_language": "hi-IN",
            "output_language": "hi-IN",
            "rag": {
                "params": {
                    "model": "gpt-4o-mini",
                    "instructions": "Answer briefly using only the knowledge base.",
                }
            },
            "rag_provider": "openai",
            "callback_url": "https://example.com/webhooks/sts",
            "request_metadata": {"session_id": "wa-session-42"},
        },
    ),
}
