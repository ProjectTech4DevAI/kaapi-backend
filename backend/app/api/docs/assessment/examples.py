"""Request-body examples for the Assessment routes.
"""

from app.api.openapi_docs import Examples

ASSESSMENT_RUN_CREATE_EXAMPLES: Examples = {
    "idea_scoring": (
        "Score ideas with a stored config",
        {
            "experiment_name": "Idea scoring - Nov cohort",
            "submission_id": "5b9e2c7a-1f3d-4e8b-a6c2-9d0f7e3b1a24",
            "input_binding": {
                "prompt": "Assess this idea.\nTitle: {title}\nDescription: {description}",
                "text_columns": ["title", "description"],
                "attachments": [
                    {"column": "pitch_deck", "type": "pdf", "format": "url"}
                ],
            },
            "configs": [{"id": "7a3c9e1b-2d4f-4b8a-9c6e-1f0d3b5a7e92", "version": 2}],
            "post_processing_config": {
                "computed_columns": [
                    {
                        "name": "Total_Score",
                        "formula": "@Novelty_score + @Usefulness_score",
                    }
                ],
                "sort": [{"column": "Total_Score", "direction": "desc"}],
            },
        },
    ),
}

ASSESSMENT_RUN_POST_PROCESSING_EXAMPLES: Examples = {
    "computed_sort_filter": (
        "Total score column, filtered and sorted",
        {
            "computed_columns": [
                {
                    "name": "Total_Score",
                    "formula": "@Novelty_score + @Usefulness_score",
                }
            ],
            "filter": [{"column": "Total_Score", "op": "gt", "value": 5}],
            "sort": [{"column": "Total_Score", "direction": "desc"}],
        },
    ),
}

ASSESSMENT_CREATE_EXAMPLES: Examples = {
    "batch": (
        "Assess a batch of items",
        {
            "config": {
                "id": "7a3c9e1b-2d4f-4b8a-9c6e-1f0d3b5a7e92",
                "version": 2,
            },
            "input": {
                "data": [
                    {
                        "title": "Solar-powered cold storage",
                        "description": "Low-cost cold rooms for smallholder farmers.",
                    },
                    {
                        "title": "SMS crop advisory",
                        "description": "Weekly weather-based tips sent over SMS.",
                    },
                ]
            },
            "experiment_name": "Idea scoring - Nov cohort",
            "callback_url": "https://example.com/webhooks/assessments",
            "request_metadata": {"cohort": "2026-11"},
        },
    ),
    "single": (
        "Assess a single item",
        {
            "config": {
                "id": "7a3c9e1b-2d4f-4b8a-9c6e-1f0d3b5a7e92",
                "version": 2,
            },
            "input": {
                "data": {
                    "title": "Solar-powered cold storage",
                    "description": "Low-cost cold rooms for smallholder farmers.",
                }
            },
            "callback_url": "https://example.com/webhooks/assessments",
        },
    ),
}

DUPLICATE_DETECTION_CREATE_EXAMPLES: Examples = {
    "check_submission": (
        "Check a submission against both corpora",
        {
            "query": "A mobile app that sends SMS reminders to pregnant women "
            "for antenatal check-ups in rural districts.",
            "config": {
                "strict": {
                    "id": "5d1c7a2e-8b3f-4e6a-9c1d-2f4b6a8e0c13",
                    "version": 1,
                },
                "lenient": {
                    "id": "9a7e3b51-2c4d-4f8e-b6a0-7d5c1e9f3b24",
                    "version": 1,
                },
            },
            "callback_url": "https://example.com/webhooks/duplicates",
            "request_metadata": {"submission_id": "sub-20261120-0042"},
        },
    ),
}
