Export results for all child runs under a RUN assessment.

For `json`, returns a flat list in the API response. For `csv`/`xlsx`,
returns one file for a single run or a ZIP archive when multiple runs exist.

Returns `422` for a BATCH assessment: its rows are served by
`GET /assessments/{assessment_id}`.
