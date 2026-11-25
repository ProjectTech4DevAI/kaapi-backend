Get a TTS evaluation run by ID with optional results.

Query parameters:
- `include_results`: Include synthesis results (default: true)

Each result carries `usage` (provider-native, see "Get TTS result") and `issues`.

`usage_by_model` is always returned and sums usage over **all** of the run's results, per model, in that model's own unit (no cross-provider normalisation, no cost). Models with no recorded usage are absent.
```json
{
  "gemini-3.8-flash-tts": {"unit": "tokens", "input_tokens": 120, "output_tokens": 9800, "total_tokens": 9920, "results_with_usage": 10},
  "bulbul:v3": {"unit": "characters", "characters": 1450, "results_with_usage": 10},
  "eleven_v3": {"unit": "credits", "characters": 1450, "credits": 1450.0, "results_with_usage": 10}
}
```
`credits` is null when no ElevenLabs result had a known cost multiplier.
