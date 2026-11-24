"""Request-body examples for the Model Evaluation routes.


"""

from app.api.openapi_docs import Examples

MODEL_EVALUATION_CREATE_EXAMPLES: Examples = {
    "basic": ("Evaluate fine-tuned models", {"fine_tuning_ids": [7, 8]}),
}
