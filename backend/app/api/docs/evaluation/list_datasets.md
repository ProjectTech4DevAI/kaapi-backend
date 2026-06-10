List all datasets for the current organization and project.

Returns a paginated list of datasets ordered by most recent first. Each dataset includes metadata (ID, name, item counts, duplication factor), Langfuse integration details, object store URL, and an `eligible_for_fast` flag that is `true` when the dataset's unique-row count is within `EVAL_FAST_MAX_UNIQUE_ROWS` (and so can be used with `run_mode="fast"` on `POST /evaluations`).

## Query parameters

| Parameter | Description |
| --- | --- |
| `limit` / `offset` | Pagination (default 50 / 0; max limit 100) |
| `eligible_for` | If set to `fast`, the response is filtered to only datasets where `eligible_for_fast` is `true` |
