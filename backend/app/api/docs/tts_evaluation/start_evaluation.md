Start a TTS evaluation run on a dataset.

Required fields:
- **run_name**: Name for this evaluation run
- **dataset_id**: ID of the TTS dataset to evaluate

Optional fields:
- **models**: TTS models to use (default: `["gemini-3.8-flash-tts"]`). Each item is either a plain model name or an object `{"model": ..., "voice": ...}`; both forms can be mixed. A plain name (the original request shape, still accepted) or an object without `voice` uses the model's default voice. Objects accept no other keys. Each model may appear only once per run.

```json
{
  "run_name": "voices_compare",
  "dataset_id": 1,
  "models": [
    {"model": "gemini-3.8-flash-tts", "voice": "Puck"},
    {"model": "bulbul:v3", "voice": "priya"},
    {"model": "eleven_v3", "voice": "Rahul"},
    "eleven_v4"
  ]
}
```

**Voices per provider** (any other voice is rejected with 422 listing the allowed voices for that model):
- Gemini (`gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`), case-sensitive, default `Kore`: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat
- Sarvam (`bulbul:v3`), any casing (stored lowercase), default `shubh`: shubh, aditya, ritu, priya, neha, rahul, pooja, rohan, simran, kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir, aayan, ashutosh, advait, anand, tanya, tarun, sunny, mani, gokul, vijay, shruti, suhani, mohit, kavitha, rehan, soham, rupali
- ElevenLabs (`eleven_v3`, `eleven_v4`), case-sensitive voice names (mapped to ElevenLabs voice IDs server-side), default `Rahul`: Rahul, Sarah, George, Callum, Liam. Sarah, George, Callum and Liam are ElevenLabs default voices that ElevenLabs retires on 2026-12-31.

The evaluation will:
1. Process each text sample through the specified TTS models
2. Generate speech audio: Gemini models via the Gemini Batch API; Sarvam and ElevenLabs models via direct API calls in background workers
3. Store 24 kHz mono WAV audio files in S3 for human review

Synthesis uses the dataset's language; datasets without a language are synthesized as `en-IN`.
The response's `models` is the list of model names; `run_metadata.voices` maps each model to the voice it is synthesized with, and each result carries its `voice`. `run_metadata.voice_name` is the legacy Gemini default voice, kept for existing clients.

**Supported models:** `gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`, `bulbul:v3` (Sarvam), `eleven_v3`, `eleven_v4` (ElevenLabs)
