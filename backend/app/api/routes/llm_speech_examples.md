# Speech-to-Speech (STS) API Examples

## Endpoint

```
POST /llm/sts
```

## Quick Start

### Minimal Request (All Defaults)
```bash
curl -X POST https://api.kaapi.ai/llm/sts \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "audio": {
      "type": "audio",
      "content": {
        "format": "base64",
        "value": "BASE64_AUDIO_DATA",
        "mime_type": "audio/ogg"
      }
    },
    "knowledge_base_ids": ["kb_abc123"],
    "callback_url": "https://your-app.com/webhook"
  }'
```

**Defaults Used:**
- Input language: Auto-detect (Sarvam STT)
- Output language: Hindi (same as detected input)
- STT: Sarvam Saaras V3 (auto language detection)
- LLM: OpenAI GPT-4o
- TTS: Sarvam Bulbul V3

---

## Full Configuration Example

### Request with All Options
```json
{
  "audio": {
    "type": "audio",
    "content": {
      "format": "base64",
      "value": "UklGRiQAAABXQVZFZm10...",
      "mime_type": "audio/ogg"
    }
  },
  "knowledge_base_ids": ["kb_customer_support", "kb_product_info"],
  "input_language": "hindi",
  "output_language": "english",
  "stt_model": "saaras:v3",
  "llm_model": "gpt-4o",
  "tts_model": "bulbul:v3",
  "callback_url": "https://api.yourapp.com/webhooks/speech-response",
  "request_metadata": {
    "user_id": "user_123",
    "session_id": "session_456",
    "source": "whatsapp"
  }
}
```

### Response (Immediate)
```json
{
  "success": true,
  "data": {
    "message": "Speech-to-speech processing initiated. You will receive intermediate callbacks for STT and LLM outputs, followed by the final callback with audio and text."
  }
}
```

---

## Callback Sequence

You'll receive **3 callbacks** to your webhook URL:

### 1. STT Callback (Intermediate)
Sent after audio transcription completes.

```json
{
  "success": true,
  "data": {
    "block_index": 1,
    "total_blocks": 3,
    "response": {
      "provider_response_id": "stt_xyz789",
      "provider": "google-native",
      "model": "gemini-2.5-pro",
      "output": {
        "type": "text",
        "content": {
          "value": "मेरा अकाउंट बैलेंस क्या है?"
        }
      }
    },
    "usage": {
      "input_tokens": 0,
      "output_tokens": 8,
      "total_tokens": 8
    }
  },
  "metadata": {
    "speech_to_speech": true,
    "input_language": "hi-IN",
    "output_language": "en-IN",
    "stt_model": "gemini-2.5-pro",
    "llm_model": "gpt-4o",
    "tts_model": "bulbul-v3",
    "user_id": "user_123",
    "session_id": "session_456",
    "source": "whatsapp"
  }
}
```

**Latency Calculation:**
```
STT_latency = callback_1_timestamp - request_timestamp
```

---

### 2. LLM/RAG Callback (Intermediate)
Sent after knowledge base retrieval and response generation.

```json
{
  "success": true,
  "data": {
    "block_index": 2,
    "total_blocks": 3,
    "response": {
      "provider_response_id": "chatcmpl_abc123",
      "conversation_id": null,
      "provider": "openai",
      "model": "gpt-4o",
      "output": {
        "type": "text",
        "content": {
          "value": "Your current account balance is ₹5,000. You have 3 transactions in the last month."
        }
      }
    },
    "usage": {
      "input_tokens": 250,
      "output_tokens": 22,
      "total_tokens": 272
    }
  },
  "metadata": {
    "speech_to_speech": true,
    "user_id": "user_123",
    "session_id": "session_456",
    "source": "whatsapp"
  }
}
```

**Latency Calculation:**
```
LLM_latency = callback_2_timestamp - callback_1_timestamp
```

---

### 3. TTS Callback (Final)
Sent after text-to-speech conversion completes. This is your final output.

```json
{
  "success": true,
  "data": {
    "response": {
      "provider_response_id": "tts_def456",
      "conversation_id": null,
      "provider": "sarvamai-native",
      "model": "bulbul:v1",
      "output": {
        "type": "audio",
        "content": {
          "format": "base64",
          "value": "T2dnUwACAAAAAAAAAAAEBQ...",
          "mime_type": "audio/ogg"
        }
      }
    },
    "usage": {
      "input_tokens": 22,
      "output_tokens": 0,
      "total_tokens": 22
    }
  },
  "metadata": {
    "speech_to_speech": true,
    "output_language": "en-IN",
    "user_id": "user_123",
    "session_id": "session_456",
    "source": "whatsapp"
  }
}
```

**Latency Calculation:**
```
TTS_latency = callback_3_timestamp - callback_2_timestamp
Total_latency = callback_3_timestamp - request_timestamp
```

---

## Error Handling Examples

### Empty STT Output
If the audio contains no speech or is unintelligible:

```json
{
  "success": false,
  "error": "STT returned no transcription. The audio may be empty or unintelligible.",
  "metadata": {
    "speech_to_speech": true,
    "user_id": "user_123"
  }
}
```

### Invalid Audio Format
If the audio format is not supported:

```json
{
  "success": false,
  "error": "SarvamAI STT transcription failed: Invalid audio format. Supported formats: mp3, wav, ogg, opus, m4a",
  "metadata": {
    "speech_to_speech": true,
    "user_id": "user_123"
  }
}
```

### Audio Size Exceeds Limit
If TTS generates audio > 16MB (rare):

```json
{
  "success": false,
  "error": "TTS audio output exceeds WhatsApp size limit (16MB). Try reducing response length.",
  "metadata": {
    "speech_to_speech": true,
    "user_id": "user_123"
  }
}
```

### Knowledge Base Not Found
If specified knowledge base doesn't exist:

```json
{
  "success": false,
  "error": "Knowledge base 'kb_invalid' not found or not accessible.",
  "metadata": {
    "speech_to_speech": true,
    "user_id": "user_123"
  }
}
```

---

## Language-Specific Examples

### English → English
```json
{
  "audio": {...},
  "knowledge_base_ids": ["kb_123"],
  "input_language": "english",
  "output_language": "english",
  "callback_url": "..."
}
```

### Hindi → English (Translation)
```json
{
  "audio": {...},
  "knowledge_base_ids": ["kb_123"],
  "input_language": "hindi",
  "output_language": "english",
  "callback_url": "..."
}
```

### Hinglish (Code-Switching)
```json
{
  "audio": {...},
  "knowledge_base_ids": ["kb_123"],
  "input_language": "hinglish",
  "output_language": "hinglish",
  "callback_url": "..."
}
```
**Note:** Hinglish is treated as Hindi for model selection.

### Regional Indian Languages
```json
{
  "audio": {...},
  "knowledge_base_ids": ["kb_123"],
  "input_language": "auto",  // Auto-detect
  "output_language": "odia",  // Odia, Bengali, Punjabi, etc.
  "callback_url": "..."
}
```

**Supported Regional Languages:**
- Bengali, Malayalam, Punjabi, Odia
- Assamese, Urdu, Nepali
- Konkani, Kashmiri, Sindhi, Sanskrit
- Santali, Manipuri, Bodo, Maithili, Dogri

---

## Model Selection Guide

### For Indian Languages (Recommended - Default)
```json
{
  "stt_model": "saaras:v3",
  "llm_model": "gpt-4o",
  "tts_model": "bulbul-v3"
}
```
**Benefits:**
- Auto language detection (no need to specify language)
- Fastest processing
- Best accent handling for Indian languages
- Natural voice quality
- MP3 output (WhatsApp compatible)

### For Maximum Accuracy
```json
{
  "stt_model": "gemini-2.5-pro",
  "llm_model": "gpt-4o",
  "tts_model": "gemini-2.5-pro-preview-tts"
}
```
**Benefits:** Highest accuracy, best for complex queries, OGG OPUS output

### For Cost Optimization
```json
{
  "stt_model": "saaras:v3",
  "llm_model": "gpt-4o-mini",
  "tts_model": "bulbul-v3"
}
```
**Benefits:** Lower cost, still good quality, faster response

---

## Integration Patterns

### WhatsApp Bot Integration
```python
import base64
import requests

def handle_whatsapp_voice_message(audio_url, user_id):
    # Download audio from WhatsApp
    audio_response = requests.get(audio_url)
    audio_base64 = base64.b64encode(audio_response.content).decode()

    # Send to Kaapi STS
    response = requests.post(
        "https://api.kaapi.ai/llm/sts",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "audio": {
                "type": "audio",
                "content": {
                    "format": "base64",
                    "value": audio_base64,
                    "mime_type": "audio/ogg"
                }
            },
            "knowledge_base_ids": ["kb_customer_support"],
            "callback_url": f"https://yourapp.com/webhook?user={user_id}",
            "request_metadata": {"user_id": user_id}
        }
    )

    return response.json()

def handle_s2s_callback(callback_data):
    """Handle the final TTS callback."""
    if not callback_data["success"]:
        # Handle error
        return

    # Extract final audio
    audio_base64 = callback_data["data"]["response"]["output"]["content"]["value"]
    audio_bytes = base64.b64decode(audio_base64)

    # Send back to WhatsApp user
    send_whatsapp_voice(audio_bytes, user_id)
```

---

## Performance Benchmarks

**Typical Latencies** (with Sarvam models, Hindi):
- STT: 1-2 seconds
- RAG: 2-4 seconds
- TTS: 1-2 seconds
- **Total: 4-8 seconds**

**With Gemini models**:
- STT: 2-3 seconds
- RAG: 2-4 seconds
- TTS: 2-3 seconds
- **Total: 6-10 seconds**

---

## Testing Tips

1. **Test with Silent Audio**: Verify error handling for empty STT
2. **Test Different Formats**: OGG, MP3, WAV, M4A
3. **Test Language Mixing**: Hinglish, code-switching
4. **Test Long Audio**: >1 minute clips
5. **Load Test**: Multiple concurrent requests
6. **Monitor Latencies**: Track each block's timing
7. **Validate Audio Output**: Ensure < 16MB for WhatsApp

---

## Troubleshooting

### High Latency
- Check knowledge base size (larger = slower retrieval)
- Consider using faster models (gemini-flash, gpt-4o-mini)
- Verify callback URL response time

### Poor Transcription Quality
- Ensure audio quality is good (no background noise)
- Try different STT models
- Check if language setting matches audio

### Unnatural TTS Voice
- Try different TTS models
- Sarvam Bulbul is best for Indian accents
- Gemini is good for neutral accents

### Callback Not Received
- Verify callback URL is publicly accessible
- Check for HTTPS (required)
- Ensure webhook can handle POST requests
- Check firewall settings
