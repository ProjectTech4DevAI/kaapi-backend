Fetch a BATCH assessment's status and every row it has produced so far.

Safe to poll while the run is in flight. `items` always holds exactly `total_items`
entries, in submission order, including placeholders for rows the provider has not
returned yet (`output.assessment` is `null`) and for rows a pre-filter gated out.

Each row carries:

- `row_index` — position in the original submission; the stable correlator.
- `input` — the submitted row, echoed back only when `include_input=true`. That flag
  re-reads the stored submission from object storage on every call, so leave it off in
  a tight poll loop and set it once the run is terminal. `null` if the stored submission
  could not be read.
- `output` — identical in shape to the webhook payload's item, so one parser serves both.
- `error` — the provider's error for that row, when it failed.

Stop polling once `status` is `COMPLETED`, `COMPLETED_WITH_ERRORS` or `FAILED`. On a
failed run, `error` carries the reason.

Returns 404 when the assessment does not exist in this project, and 422 when it is not
a BATCH assessment.
