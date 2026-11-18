Get a duplicate-detection job by its `id`: the current `status` and, once finished, the same `strict`, `lenient` and `errors` that were sent to the webhook.

Status is one of `PENDING`, `PROCESSING`, `COMPLETED`, `COMPLETED_WITH_ERRORS` or `FAILED`. Results are null until the job finishes.

Returns `404` when the job does not exist or belongs to another project.
