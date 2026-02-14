Start a TTS evaluation run on a dataset.

The evaluation will:
1. Process each text sample through the specified TTS providers
2. Generate speech audio using Gemini Batch API
3. Store WAV audio files in S3 for human review

**Supported providers:** gemini-2.5-pro-preview-tts
