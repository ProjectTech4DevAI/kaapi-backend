Start a TTS evaluation run on a dataset.

Required fields:
- **run_name**: Name for this evaluation run
- **dataset_id**: ID of the TTS dataset to evaluate

Optional fields:
- **models**: List of TTS models to use (default: `["gemini-2.5-pro-preview-tts"]`)

The evaluation will:
1. Process each text sample through the specified TTS models
2. Generate speech audio using Gemini Batch API
3. Store WAV audio files in S3 for human review

**Supported models:** `gemini-2.5-pro-preview-tts`
