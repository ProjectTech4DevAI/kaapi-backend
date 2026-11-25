Get a single TTS synthesis result by ID.

Returns the result including audio URL, metadata, usage, and human feedback (`is_correct`, `comment`, `score`, `issues`).

`usage` is in the provider's own billing unit, null when not recorded:
- Gemini: `{"unit": "tokens", "input_tokens", "output_tokens", "total_tokens"}` from the batch response; set on every row whose response came back, including audio-processing failures.
- Sarvam: `{"unit": "characters", "characters"}` (input text length; successful rows only).
- ElevenLabs: `{"unit": "credits", "characters", "credits", "credit_multiplier"}` with credits = characters × the model's cost multiplier (computed, successful rows only); `credits`/`credit_multiplier` are null if the multiplier couldn't be fetched.
