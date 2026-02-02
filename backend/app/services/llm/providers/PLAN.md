This specification sheet is designed for implementing **Gemini 2.5 Pro TTS** via the `google-genai` Python SDK (AI Studio / API Key method).

---

### **1. Core Identity & Endpoints**

* **Model ID:** `gemini-2.5-pro-tts` (High-fidelity) or `gemini-2.5-flash-tts` (Low-latency).
* **Base URL:** `https://generativelanguage.googleapis.com/v1beta`
* **Auth Type:** API Key (`x-goog-api-key` header).

### **2. Technical Specification Table**

| Component | SDK Property / Path | Type | Constraints |
| --- | --- | --- | --- |
| **Response Modality** | `response_modalities` | `list[str]` | Must be **`["AUDIO"]`** |
| **Voice Selection** | `speech_config.voice_config.prebuilt_voice_config.voice_name` | `string` | e.g., `Aoede`, `Kore`, `Fenrir` (See Section 3) |
| **Language** | `speech_config.language_code` | `string` | BCP-47 code (e.g., `en-US`, `hi-IN`) |
| **Speed (Rate)** | `audio_config.speaking_rate` | `float` | Range: `0.25` to `4.0` (Default: `1.0`) |
| **Pitch** | `audio_config.pitch` | `float` | Range: `-20.0` to `20.0` |
| **Volume Gain** | `audio_config.volume_gain_db` | `float` | Range: `-96.0` to `16.0` |
| **Output Format** | `audio_config.audio_encoding` | `enum` | `MP3`, `LINEAR16` (WAV), `OGG_OPUS` |
| **Director Notes** | `system_instruction` | `string` | Natural language (e.g., "Speak sadly", "Professional") |

---

### **3. Voice Catalog (Common Personas)**

Gemini 2.5 voices are "Instruction-Aware." Use these IDs in the `voice_name` field:

* **`Aoede`**: Neutral, Breezy (Best for general narration).
* **`Kore`**: Firm, Professional (Best for corporate/assistants).
* **`Fenrir`**: Excitable, High-energy (Best for gaming/ads).
* **`Leda`**: Youthful, Bright.
* **`Charon`**: Informative, Mature.

---

### **4. Implementation Pattern (Python SDK)**

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

# 1:1 Mapping to your JSON schema requirements
config = types.GenerateContentConfig(
    response_modalities=["AUDIO"],
    # Maps your 'director_notes'
    system_instruction="Speak with a professional, calm tone. Pause for 1 second between sentences.",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Kore"  # Mapping your 'voice'
            )
        ),
        language_code="en-US"
    ),
    audio_config=types.AudioConfig(
        audio_encoding="MP3",      # Mapping your 'response_format'
        speaking_rate=1.0          # Mapping your 'speed'
    )
)

response = client.models.generate_content(
    model="gemini-2.5-pro-tts",
    contents="Hello world. This is a technical test of the Gemini TTS pipeline.",
    config=config
)

# Extract binary data
audio_bytes = response.candidates[0].content.parts[0].inline_data.data

with open("output.mp3", "wb") as f:
    f.write(audio_bytes)

```

### **5. Important Usage Notes**

1. **Instruction Priority:** If you set `speaking_rate=2.0` and also put "Speak very slowly" in the `system_instruction`, the model may produce erratic results. Use **natural language** for "tone" and **programmatic fields** for "fixed pacing."
2. **Streaming:** The `google-genai` SDK supports streaming audio bytes for real-time applications via the `models.generate_content_stream` method, but `audio_encoding` must be `LINEAR16` or `PCM` for minimum latency.
3. **SynthID:** Note that all output audio is watermarked with Google's SynthID for safety tracking.
4. You are required to go thorugh the implementation of _execute_stt function inside the app/services/llm/providers.gai.py and relevant models inside app/models/llm as well.
5. Make sure to follow the celery task queue structure and pydantic models idioms and database schema.
6. The generated audio is not saved in the database, so store some metadata for now.
7. Make sure to follow the TTS config types, (KaapiLLMConfig and Native LLM Configs and relevant models for config and versions).
8. Do not write over abstracted code. Focus on readability than pristine JAVA-like code.
