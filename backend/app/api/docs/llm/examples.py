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
    "text_to_speech": (
        "Speak text with Sarvam (voice omitted: provider default speaker)",
        {
            "query": {"input": "Your appointment is confirmed for Monday at 10 AM."},
            "config": {
                "blob": {
                    "completion": {
                        "provider": "sarvamai",
                        "type": "tts",
                        "params": {
                            "model": "bulbul:v3",
                            "language": "hi-IN",
                            "response_format": "mp3",
                        },
                    }
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
                    "conversation": {"auto_create": True},
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
                "input": {
                    "type": "audio",
                    "content": {
                        "format": "url",
                        "value": "https://example.com/media/voice-note.ogg",
                        "mime_type": "audio/ogg",
                    },
                }
            },
            "config": {
                "stt": {
                    "provider": "sarvamai",
                    "params": {"model": "saaras:v3", "input_language": "hi-IN"},
                },
                "llm": {
                    "provider": "openai",
                    "params": {
                        "model": "gpt-4o-mini",
                        "instructions": "Answer briefly using only the knowledge base.",
                        "knowledge_base_ids": ["vs_abc123"],
                    },
                },
                "tts": {
                    "provider": "sarvamai",
                    "params": {"model": "bulbul:v3", "voice": "shubh"},
                },
            },
            "callback_url": "https://example.com/webhooks/sts",
            "request_metadata": {"session_id": "wa-session-42"},
        },
    ),
    "continue_conversation": (
        "Continue an OpenAI conversation with stored STT/TTS configs",
        {
            "query": {
                "input": {
                    "type": "audio",
                    "content": {
                        "format": "url",
                        "value": "https://example.com/media/follow-up.ogg",
                        "mime_type": "audio/ogg",
                    },
                }
            },
            "config": {
                "stt": {"id": "8d1e4f2a-3b5c-4d6e-9f70-1a2b3c4d5e6f", "version": 2},
                "llm": {
                    "provider": "openai",
                    "params": {"model": "gpt-5.4"},
                    "conversation": {"id": "conv_6b0512565bdc8195"},
                },
                "tts": {"id": "9e2f5a3b-4c6d-4e7f-8a91-2b3c4d5e6f70", "version": 1},
            },
            "callback_url": "https://example.com/webhooks/sts",
        },
    ),
}
