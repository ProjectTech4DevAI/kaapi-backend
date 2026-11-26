Update human feedback and score on a TTS synthesis result.

Only the provided fields will be updated. Fields omitted from the request will not modify the existing value.

Fields:
- **is_correct**: Whether the synthesized audio quality is acceptable (null to clear)
- **comment**: Short optional note, max 500 characters (null to clear)
- **score**: Evaluation metrics, **merged per key** into the stored ones so metrics can be rated one at a time; a `null` value removes that metric. Removing the last metric clears the score.
- **issues**: Issue tags, replacing the stored list (`[]` or null clears; duplicates are dropped). Allowed: `mispronunciation`, `robotic`, `wrong_language`, `cut_off`, `long_pause`, `background_noise`

**Example request:**
```json
{
  "is_correct": false,
  "comment": "string",
  "score": {
    "Speech Naturalness": "low | medium | high",
    "Pronunciation Accuracy": null
  },
  "issues": ["mispronunciation", "long_pause"]
}
```
