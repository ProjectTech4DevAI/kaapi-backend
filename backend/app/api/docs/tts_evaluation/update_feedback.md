Update human feedback and score on a TTS synthesis result.

Only the provided fields will be updated. Fields omitted from the request will not modify the existing value. Sending a field as `null` will clear its value.

Fields:
- **is_correct**: Whether the synthesized audio quality is acceptable (null to clear)
- **comment**: Optional feedback comment
- **score**: Evaluation metrics for the synthesized audio

**Example request:**
```json
{
  "is_correct": true,
  "comment": "string",
  "score": {
    "Speech Naturalness": "low | medium | high",
    "Pronunciation Accuracy": "low | medium | high"
  }
}
```
