Upload a CSV of golden Q&A pairs for evaluation, stored natively by Kaapi with no
Langfuse dataset (v2).

Same multipart shape as the v1 dataset upload (`file`, `dataset_name`,
`description`, `duplication_factor`), but the dataset is stored in object storage
(S3) only: the row is created with `langfuse_dataset_id` null and the CSV holds
exactly the **original** rows — no physical duplication. The `duplication_factor`
is recorded in `dataset_metadata` (with a run-time-duplication marker) and applied
when the run executes, so an 8-row CSV with factor 5 evaluates 40 rows.

**Response:** `APIResponse[DatasetUploadResponse]`, same shape as v1 with
`langfuse_dataset_id` null. `original_items` is the stored row count and
`total_items` is `original_items × duplication_factor` (the count the run will
produce, not stored rows). `eligible_for_fast` reflects only the unique-row size
cap (`EVAL_FAST_MAX_UNIQUE_ROWS`).

**CSV format:** required columns `question`, `answer`; optional `category`. Rows
with a missing question or answer are skipped. Maximum upload size is 1 MB.
