# Speech-to-Speech (STS)

Runs a voice note through a 3-block chain as a background job: **STT → LLM → TTS**. Returns the job immediately (poll it, or receive results via `callback_url`).

## Request

```json
{
  "query": {"input": {"type": "audio", "content": {"format": "url", "value": "https://...", "mime_type": "audio/ogg"}}},
  "config": {
    "stt": {"provider": "sarvamai", "params": {"model": "saaras:v3", "input_language": "auto"}},
    "llm": {"provider": "openai", "params": {"model": "gpt-4o", "knowledge_base_ids": ["vs_abc123"]}, "conversation": {"id": "conv_abc123"}},
    "tts": {"provider": "sarvamai", "params": {"model": "bulbul:v3", "voice": "anushka", "language": "hi-IN"}}
  },
  "callback_url": "https://client.app/webhook",
  "include_provider_raw_response": false,
  "request_metadata": {"user_id": "42"}
}
```

- `query.input` must be a single audio input.
- `config` and each of `config.stt`, `config.llm`, `config.tts` are optional. Each step is **either** a stored config `{"id": "<uuid>", "version": 1}` **or** inline `{"provider": ..., "params": {...}}` — not both. `params` use the same schema as the `stt`, `text` and `tts` completion params of `/llm/call`.
- An inline step is used as written; missing `params` fields fall back to that params model's defaults (not the endpoint defaults below). An inline `llm` step without `provider` uses `openai`; inline `stt`/`tts` without `provider` use the configured Google credential.
- `knowledge_base_ids` in `llm.params` enables file-search retrieval. Not supported with provider `google-gcp`.
- `llm.conversation` continues a conversation (`{"id": "conv_..."}`) or starts one (`{"auto_create": true}`), as `query.conversation` does on `/llm/call`. OpenAI only. Read the id from the LLM callback's `response.conversation_id` and send it with the next voice note. Setting `query.conversation` is rejected.
- `request_metadata` is passed through unchanged to every callback.

## Defaults for omitted steps

- **stt**: Google, model `gemini-3.8-flash`, `input_language: "auto"`.
- **llm**: `openai`, model `gpt-4o`, `temperature: 0.1`, short concise-answer instructions, no knowledge base.
- **tts**: Google, model `gemini-3.8-flash-tts`, voice `Kore`, `response_format: "ogg"` (WhatsApp-compatible opus).

## TTS language

When the tts step is inline or omitted and `params.language` is not set:
- if the stt step's `input_language` is a concrete code (e.g. `hi-IN`), TTS speaks that language;
- otherwise (`auto`/`unknown`, or a stored stt config), TTS uses the language STT detected, falling back to `en-IN`.

A stored tts config is used untouched.

## Callbacks

With `callback_url` set you receive 3 callbacks: STT transcript (intermediate), LLM answer (intermediate), and the final TTS audio. If STT returns an empty transcript, the chain fails and later blocks do not run; any block failure stops the chain and is reported in the callback.
