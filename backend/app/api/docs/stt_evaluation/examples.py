"""Request-body examples for the STT evaluation routes.


"""

from app.api.openapi_docs import Examples

STT_DATASET_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Create a Hindi STT dataset",
        {
            "name": "hindi-banking-queries",
            "description": "Customer support voice notes in Hindi",
            "language_id": 2,
            "samples": [
                {
                    "file_id": 101,
                    "ground_truth": "मुझे अपने खाते की शेष राशि जाननी है",
                },
                {"file_id": 102},
            ],
        },
    ),
}

STT_SAMPLE_UPDATE_EXAMPLES: Examples = {
    "basic": (
        "Update language and ground truth",
        {
            "language_id": 2,
            "ground_truth": "मुझे अपने खाते की शेष राशि जाननी है",
        },
    ),
}

STT_RUN_CREATE_EXAMPLES: Examples = {
    "basic": (
        "Start an STT evaluation run",
        {
            "run_name": "hindi-banking-gemini-baseline",
            "dataset_id": 12,
            "models": ["gemini-2.5-pro"],
        },
    ),
}

STT_FEEDBACK_UPDATE_EXAMPLES: Examples = {
    "basic": (
        "Mark a transcription as incorrect",
        {
            "is_correct": False,
            "comment": "Missed the word 'balance' in the second sentence",
        },
    ),
}
