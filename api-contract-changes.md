# API Contract Changes — v0.8.0 (v0.7.0 → v0.8.0)

## New Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/llm/chain` | POST | Sequential chain execution of LLM calls |
| `/evaluations/stt/files` | GET | List audio files |
| `/evaluations/stt/files/{file_id}` | GET | Get single audio file |
| `/evaluations/stt/samples/{sample_id}` | PATCH | Update STT sample (language, ground truth) |
| `/evaluations/tts/datasets` | POST | Create TTS evaluation dataset |
| `/evaluations/tts/datasets` | GET | List TTS datasets |
| `/evaluations/tts/datasets/{dataset_id}` | GET | Get single TTS dataset |
| `/evaluations/tts/runs` | POST | Start a TTS evaluation run |
| `/evaluations/tts/runs` | GET | List TTS evaluation runs |
| `/evaluations/tts/runs/{run_id}` | GET | Get TTS run with results |
| `/evaluations/tts/results/{result_id}` | PATCH | Update human feedback on TTS result |
| `/evaluations/tts/results/{result_id}` | GET | Get a TTS result |

## Breaking Changes

### Validation Error Response Format

The 422 response body now returns a structured `errors` array instead of a concatenated string. `APIResponse` has a new field:

```
errors: Optional[list[ValidationErrorDetail]]  # each item: { field, message }
```

**Clients parsing validation error responses will need to update.**

### LLM Request Model Changes

| Field | Old | New |
|---|---|---|
| `KaapiCompletionConfig.provider` | `Literal["openai", "google"]` | `Literal["openai", "google", "sarvamai", "elevenlabs"] \| None` (optional, auto-resolves) |
| `NativeCompletionConfig.provider` | `Literal["openai-native", "google-native"]` | `Literal["openai-native", "google-native", "sarvamai-native", "elevenlabs-native"]` |
| `QueryInput` union | `TextInput \| AudioInput` | `TextInput \| AudioInput \| ImageInput \| PDFInput` |
| `QueryParams.input` | `str \| QueryInput` | `str \| QueryInput \| list[QueryInput]` (list enables mixing modalities) |
| `LlmCall.input_type` | `Literal["text", "audio", "image"]` | `Literal["text", "audio", "image", "pdf", "multimodal"]` |

## Default Value Changes

| Field | Old Default | New Default |
|---|---|---|
| `TextLLMParams.temperature` | `None` | `0.1` |
| `STTLLMParams.model` | required | `"gemini-2.5-pro"` |
| `STTLLMParams.instructions` | required `str` | `None` (optional) |
| `STTLLMParams.input_language` | `None` | `"auto"` |
| `STTLLMParams.temperature` min | `0.0` | `0.01` |
| `TTSLLMParams.model` | required | `"gemini-2.5-flash-preview-tts"` |
| `TTSLLMParams.voice` | required | `"Kore"` |
| `TTSLLMParams.language` | required `str` | `None` (optional) |
| `CollectionOptions.batch_size` | `1` | `10` |

## Other Contract Changes

- **`STTFeedbackUpdate`**: `is_correct` and `comment` changed from required → optional
- **`ConfigBlob`**: added optional `prompt_template` field for chain `{{input}}` interpolation
- **`DatasetUploadResponse`**: added `description` and `signed_url` fields
- **`FilePublic`**: added `signed_url` field
- **`STTSamplePublic`**: added `signed_url` field
- **`STTSampleCreate`**: added optional `language_id` field (per-sample language override)
- **Signed URL support**: added `include_signed_url` query param to GET endpoints for datasets, STT datasets/runs, and TTS endpoints
- **New enums**: `BatchJobType.LLM_CHAIN`, `JobType.LLM_CHAIN`, `ChainStatus`

## New Providers

- **Sarvam AI** (`sarvamai`) — requires `api_key`
- **ElevenLabs** (`elevenlabs`) — requires `api_key`

## Database Migrations

- **048**: `llm_chain` table + `chain_id` FK on `llm_call` + `LLM_CHAIN` job type enum
- **049**: `tts_result` table with indexes (`ix_tts_result_run_id`, `idx_tts_result_feedback`, `idx_tts_result_status`)

## Key Architectural Additions

1. **Multimodal inputs** — image & PDF support in `/llm/call`
2. **LLM Chain** — sequential multi-block execution with intermediate callbacks
3. **TTS Evaluation** — complete evaluation suite parallel to STT evaluation
4. **ElevenLabs + Sarvam** — two new STT/TTS providers
5. **Smart defaults** — auto provider/model selection for STT/TTS
