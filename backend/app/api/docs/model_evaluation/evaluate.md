Start evaluations for one or more fine-tuned models.

For each fine-tuning job ID provided, this endpoint fetches the fine-tuned model and test data, then queues a background task that runs predictions on the test set and computes evaluation scores (Matthews correlation coefficient). Returns created or active evaluation records.
