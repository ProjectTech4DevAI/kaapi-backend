Check whether a submission duplicates an entry in a corpus. The request is accepted immediately and the result is delivered to your webhook.

Two checks run in parallel, each with its own saved config:

* **strict**: one verdict, `is_duplicate`, with the matching corpus entry as `citation` (null when it is not a duplicate).
* **lenient**: a list of similar corpus entries, each with a `score` from 1 (loosely related) to 5 (almost certainly a duplicate) and its `citation`. Empty when nothing is similar.

`config.strict` and `config.lenient` each pin a saved config version (`id` + `version`) tagged `ASSESSMENT`. Each must use the `google-gcp` provider and set `model`, `knowledge_base_ids` (the corpus to search) and `max_num_results` (how many corpus entries to compare against) in its assessment params. Its `json_output_schema` must declare the fields the check returns (`is_duplicate` and `citation` for strict; `matches` whose items have `score` and `citation` for lenient); it may declare more. Otherwise the request is rejected with `422`.

The response is `202` with the job `id` and status `PENDING`. When both checks finish, the result is POSTed to `callback_url` (HTTPS only), signed with the project webhook secret when one is configured. `request_metadata` is returned unchanged in the callback.

If one check fails, the other is still delivered: the failed one is null and its reason is in `errors`, and the status is `COMPLETED_WITH_ERRORS`.
