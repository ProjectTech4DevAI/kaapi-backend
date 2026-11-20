"""Request-body examples for the Model Evaluation routes.

Each constant maps a name to `(summary, body)` and is wired onto its route
with `openapi_extra=request_examples(...)`. The first entry is what the synced
Postman collection uses as the request body.
"""

from app.api.openapi_docs import Examples

MODEL_EVALUATION_CREATE_EXAMPLES: Examples = {
    "basic": ("Evaluate fine-tuned models", {"fine_tuning_ids": [7, 8]}),
}
