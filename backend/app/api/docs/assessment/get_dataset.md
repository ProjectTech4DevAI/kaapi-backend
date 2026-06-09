Get a single assessment dataset by ID.

Optionally include a signed URL to download the original uploaded file.

Pass `limit_rows=N` (1-100) to additionally include a lightweight preview
of the dataset's column headers and the first N data rows. When omitted,
the underlying file is not fetched and the response stays small.
