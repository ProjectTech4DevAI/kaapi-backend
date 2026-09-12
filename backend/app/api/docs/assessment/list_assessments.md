List this project's assessments, newest first.

Each row carries where the run is and what it was pinned to — status, stage, the config
id and version, and the row count. It deliberately carries **no per-row counts or
results**: those need the provider dump streamed back from object storage, which a list
must not pay for. Fetch `GET /assessments/{assessment_id}` for a single run's rows.

**Filtering**

- `config_id` — return only the runs pinned to that config. Omit it for every run.
- `version` — the config version, applied **only** when `config_id` is given. Defaults
  to `1`, so asking for a config without naming a version returns its v1 runs.
- `method` — narrow to `BATCH`, `RUN` or `RESPONSE`. Omit for all three.

`limit` defaults to 50 (max 100) and `offset` pages through.
