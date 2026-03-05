# Speech-to-Speech (STS) with RAG

Execute a complete speech-to-speech workflow with knowledge base retrieval.

## Endpoint

```
POST /llm/sts
```

## Flow

```
Voice Input → STT (auto language) → RAG (Knowledge Base) → TTS → Voice Output
```

## Input

- **Voice note**: WhatsApp-compatible audio format (required)
- **Knowledge base IDs**: One or more knowledge bases for RAG (required)
- **Languages**: Input and output languages (optional, defaults to Hindi)
- **Models**: STT, LLM, and TTS model selection (optional, defaults to Sarvam)

## Output

You will receive **3 callbacks** to your webhook URL:

1. **STT Callback** (Intermediate): Transcribed text from audio
2. **LLM Callback** (Intermediate): RAG-enhanced response text
3. **TTS Callback** (Final): Audio output + response text

Each callback includes:
- Output from that step
- Token usage
- Latency information (check timestamps)

## Supported Languages

### Primary Indian Languages
- English, Hindi, Hinglish (code-switching)
- Bengali, Kannada, Malayalam, Marathi
- Odia, Punjabi, Tamil, Telugu, Gujarati

### Additional Languages (Sarvam Saaras V3)
- Assamese, Urdu, Nepali
- Konkani, Kashmiri, Sindhi
- Sanskrit, Santali, Manipuri
- Bodo, Maithili, Dogri

**Total: 25 languages** with automatic language detection

## Available Models

### STT (Speech-to-Text)
- `saaras:v3` - Sarvam Saaras V3 (**default**, fast, auto language detection, optimized for Indian languages)
- `gemini-2.5-pro` - Google Gemini 2.5 Pro

**Note:** Sarvam STT uses automatic language detection. No need to specify input language.

### LLM (RAG)
- `gpt-4o` - OpenAI GPT-4o (**default**, best quality)
- `gpt-4o-mini` - OpenAI GPT-4o Mini (faster, lower cost)

### TTS (Text-to-Speech)
- `bulbul:v3` - Sarvam Bulbul V3 (**default**, natural Indian voices, MP3 output)
- `gemini-2.5-pro-preview-tts` - Google Gemini 2.5 Pro (OGG OPUS output)

## Edge Cases & Error Handling

### Empty STT Output
If speech-to-text returns empty/blank:
- Chain fails immediately
- Error message: "STT returned no transcription"
- No subsequent blocks are executed

### Audio Size Limit
WhatsApp limit: 16MB
- TTS providers may fail if output exceeds limit
- Error is caught and reported in callback
- Consider using shorter responses or compression

### Invalid Audio Format
If input audio format is unsupported:
- STT provider fails with format error
- Error reported in callback
- Supported: MP3, WAV, OGG, OPUS, M4A

### Provider Failures
Each block has independent error handling:
- STT fails → Chain stops, STT error reported
- LLM fails → Chain stops, RAG error reported
- TTS fails → Chain stops, TTS error reported

## Example Request

```bash
curl -X POST https://api.kaapi.ai/llm/sts \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "query": {
    "type": "audio",
    "content": {
      "format": "base64",
      "value": "base64_encoded_audio_data",
      "mime_type": "audio/ogg"
    }
  },
  "knowledge_base_ids": ["kb_abc123"],
  "input_language": "hindi",
  "output_language": "english",
  "callback_url": "https://your-app.com/webhook"
}
EOF
```

**Note:** `stt_model`, `llm_model`, and `tts_model` are optional and will use defaults if not specified.

## Example Callbacks

### Callback 1: STT Output (Intermediate)
```json
{
  "success": true,
  "data": {
    "block_index": 1,
    "total_blocks": 3,
    "response": {
      "provider_response_id": "stt_xyz789",
      "provider": "sarvamai-native",
      "model": "saarika:v1",
      "output": {
        "type": "text",
        "content": {
          "value": "नमस्ते, मुझे अपने अकाउंट के बारे में जानकारी चाहिए"
        }
      }
    },
    "usage": {
      "input_tokens": 0,
      "output_tokens": 12,
      "total_tokens": 12
    }
  },
  "metadata": {
    "speech_to_speech": true,
    "input_language": "hi-IN"
  }
}
```

### Callback 2: LLM Output (Intermediate)
```json
{
  "success": true,
  "data": {
    "block_index": 2,
    "total_blocks": 3,
    "response": {
      "provider_response_id": "chatcmpl_abc123",
      "provider": "openai",
      "model": "gpt-4o",
      "output": {
        "type": "text",
        "content": {
          "value": "आपके अकाउंट में कुल बैलेंस ₹5,000 है। पिछले महीने में 3 ट्रांजैक्शन हुए हैं।"
        }
      }
    },
    "usage": {
      "input_tokens": 150,
      "output_tokens": 45,
      "total_tokens": 195
    }
  },
  "metadata": {
    "speech_to_speech": true
  }
}
```

### Callback 3: TTS Output (Final)
```json
{
  "success": true,
  "data": {
    "response": {
      "provider_response_id": "tts_def456",
      "provider": "sarvamai-native",
      "model": "bulbul:v1",
      "output": {
        "type": "audio",
        "content": {
          "format": "base64",
          "value": "base64_encoded_audio_output",
          "mime_type": "audio/ogg"
        }
      }
    },
    "usage": {
      "input_tokens": 15,
      "output_tokens": 0,
      "total_tokens": 15
    }
  },
  "metadata": {
    "speech_to_speech": true,
    "output_language": "hi-IN"
  }
}
```

## Latency Tracking

Calculate latency from callback timestamps:
- **STT latency**: Time from request to first callback
- **LLM latency**: Time between first and second callback
- **TTS latency**: Time between second and third callback
- **Total latency**: Time from request to final callback

## Best Practices

1. **Language Consistency**: If not translating, keep input_language = output_language
2. **Model Selection**: Use Sarvam models for Indian languages (faster, better quality)
3. **Knowledge Base**: Ensure KB is properly indexed and relevant to expected queries
4. **Error Handling**: Implement retry logic for transient provider failures
5. **Webhook Security**: Validate webhook signatures and use HTTPS
