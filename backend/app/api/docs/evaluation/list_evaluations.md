List all evaluation runs for the current organization and project.

Returns a paginated list of evaluation runs ordered by most recent first. Each run includes metadata (ID, name, dataset info, timestamps), configuration details, batch job ID, status tracking (pending/running/completed/failed), progress metrics (total/completed items), and results when available.

Optional query param `dataset_id` restricts the list to runs of a single evaluation dataset. Omit it to return runs across all datasets in the project.
