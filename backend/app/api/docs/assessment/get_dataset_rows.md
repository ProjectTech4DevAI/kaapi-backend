Return every row of an assessment dataset as column-keyed JSON.

Fetches and parses the underlying CSV/XLSX file in full, returning the column
`headers`, all `rows` (each a `{column: value}` dict), and `total_rows`. Intended
for the frontend to assemble a directly-usable API-batch input file client-side.

Unlike the `limit_rows` preview on `GET /datasets/{dataset_id}`, this returns the
entire dataset and always fetches the file, so responses scale with dataset size.
