Create a new STT evaluation dataset with audio samples.

Upload audio files first via `POST /evaluations/stt/files` to get `file_id` values.

**Request Body:**
- **name** (required, string): Dataset name (min 1 character)
- **description** (optional, string): Dataset description
- **language_id** (optional, integer): ID of the language from the global languages table
- **samples** (required, array, min 1 item): List of audio samples
  - **file_id** (required, integer): ID of the uploaded audio file (from `POST /evaluations/stt/files`)
  - **ground_truth** (optional, string): Reference transcription for WER/CER metrics

**Example:**

```json
{
  "name": "Hindi call center dataset",
  "description": "100 samples from call center recordings",
  "language_id": 5,
  "samples": [
    { "file_id": 12, "ground_truth": "hello how can I help you" },
  ]
}
```
