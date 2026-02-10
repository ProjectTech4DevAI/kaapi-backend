Create a new STT evaluation dataset with audio samples.

Each sample requires:
- **object_store_url**: S3 URL of the audio file (from /evaluations/stt/files endpoint)
- **ground_truth**: Reference transcription (optional, for WER/CER metrics)
