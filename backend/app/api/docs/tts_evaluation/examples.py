"""Request-body examples for the TTS evaluation routes.


"""

from app.api.openapi_docs import Examples

TTS_DATASET_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Create a Hindi TTS dataset",
        {
            "name": "hindi-ivr-prompts",
            "description": "IVR prompts for the Hindi helpline",
            "language_id": 2,
            "samples": [
                {"text": "नमस्ते, हमारी हेल्पलाइन पर आपका स्वागत है।"},
                {"text": "कृपया अपना पंजीकृत मोबाइल नंबर दर्ज करें।"},
            ],
        },
    ),
}

TTS_RUN_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Start a TTS evaluation run",
        {
            "run_name": "hindi-ivr-gemini-tts",
            "dataset_id": 15,
            "models": ["gemini-3.8-flash-tts"],
        },
    ),
    "voices": (
        "Compare providers with a chosen voice per model",
        {
            "run_name": "hindi-ivr-voices",
            "dataset_id": 15,
            "models": [
                {"model": "gemini-3.8-flash-tts", "voice": "Puck"},
                {"model": "bulbul:v3", "voice": "priya"},
                {"model": "eleven_v3", "voice": "Rahul"},
            ],
        },
    ),
}

TTS_FEEDBACK_UPDATE_EXAMPLES: Examples = {
    "basic": (
        "Rate synthesized audio",
        {
            "is_correct": True,
            "comment": "Clear audio, slightly robotic intonation",
            "score": {
                "Speech Naturalness": "medium",
                "Pronunciation Accuracy": "high",
            },
        },
    ),
}
