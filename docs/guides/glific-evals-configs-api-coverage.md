# Glific Evals & Configs UI → Kaapi API Coverage

This maps the **Evals** and **Configs** flows in the Glific `AI Assistants`
prototype (`glific-evals-v13.html`) to the corresponding Kaapi backend APIs, and
marks whether each API already exists.

**How the prototype's concepts map to Kaapi:**

| Prototype concept | Kaapi concept |
| --- | --- |
| An "Assistant" | A **Config** (`/api/v1/configs`) |
| A saved "Version" (prompt + model + settings) | A **Config version** (`/configs/{id}/versions`) |
| A "Golden Q&A set" | An **evaluation dataset** (`/evaluations/datasets`) |
| A "Run" / evaluation | An **evaluation run** (`/api/v2/evaluations`) |

---

## Configs flow

| API | Available |
| --- | --- |
| Get Configs (list assistants) | **Yes** |
| Get Config (open an assistant) | **Yes** |
| Create Config (create assistant) | **Yes** |
| Update Config (rename / edit) | **Yes** |
| Delete Config (delete assistant) | **Yes** |
| Create Config Version (Save Version) | **Yes** |
| Get Config Versions (version dropdown) | **Yes** |
| Get Config Version (load a version) | **Yes** |
| Duplicate Config (Duplicate assistant) | **No** |
| Publish / Set-Live Config Version (Publish & go live) | **No** |

### Endpoints & notes

| API | Available | Kaapi endpoint / notes |
| --- | --- | --- |
| Get Configs | ✅ Yes | `GET /api/v1/configs` |
| Get Config | ✅ Yes | `GET /api/v1/configs/{config_id}` |
| Create Config | ✅ Yes | `POST /api/v1/configs` — also creates version 1 in the same call |
| Update Config | ✅ Yes | `PATCH /api/v1/configs/{config_id}` |
| Delete Config | ✅ Yes | `DELETE /api/v1/configs/{config_id}` |
| Create Config Version | ✅ Yes | `POST /api/v1/configs/{config_id}/versions` — matches "Save Version" (each save = a new version) |
| Get Config Versions | ✅ Yes | `GET /api/v1/configs/{config_id}/versions` |
| Get Config Version | ✅ Yes | `GET /api/v1/configs/{config_id}/versions/{version_number}` |
| Duplicate Config | ❌ No | No clone/copy endpoint. The UI "Duplicate" would need a new API (or client-side create-config from a fetched blob). |
| Publish / Set-Live Config Version | ❌ No | **No concept of a live/published/active version in Kaapi.** The `ConfigVersion` model has no `is_live`/`published`/`active` field, and there is no promote/go-live endpoint. The prototype's "Publish & go live", the LIVE badge, and "was live" states have no backing API. |

---

## Evals flow

| API | Available |
| --- | --- |
| Upload Dataset (add Golden Q&A set) | **Yes** |
| List Datasets (Manage sets) | **Yes** |
| Get Dataset (view a set) | **Yes** (partial) |
| Delete Dataset (delete a set) | **Yes** |
| Export Dataset (export set CSV) | **No** |
| Run Eval (Run evaluation) | **Yes** |
| Get Eval (in-progress / completed status) | **Yes** |
| Get Eval Results (metrics + per-question) | **Yes** |
| List Evals (History) | **Yes** |
| Export Eval Results (Export CSV) | **No** |
| Improve Prompt (What to change next) | **Yes** |
| Run-time / online evaluation (live conversation scoring) | **No** |

### Endpoints & notes

| API | Available | Kaapi endpoint / notes |
| --- | --- | --- |
| Upload Dataset | ✅ Yes | `POST /api/v2/evaluations/datasets` (v1 also exists). CSV columns `question`, `answer`, optional `category` — matches the prototype's CSV. |
| List Datasets | ✅ Yes | `GET /api/v1/evaluations/datasets` (v1 only; no v2 variant) |
| Get Dataset | ⚠️ Partial | `GET /api/v1/evaluations/datasets/{dataset_id}` returns the dataset record, but there is **no per-item/questions listing route**. The prototype's "View set" question table would read rows from the stored CSV (`signed_url`), not a questions API. |
| Delete Dataset | ✅ Yes | `DELETE /api/v1/evaluations/datasets/{dataset_id}` |
| Export Dataset | ❌ No | The prototype exports the set as CSV client-side; there's no API for it (the source CSV is already retrievable via the dataset's `signed_url`). |
| Run Eval | ✅ Yes | `POST /api/v2/evaluations` — body `dataset_id`, `experiment_name`, `config_id`, `config_version`. **Mismatch:** the prototype's per-run duplication (1× / 5×) does not map to the run endpoint — in Kaapi `duplication_factor` is set at **dataset upload** time (`1–5`), not per run. |
| Get Eval (status) | ✅ Yes | `GET /api/v1/evaluations/{evaluation_id}` — `status` goes `processing → completed`/`failed`, backing the "in progress" / "completed" job banner. |
| Get Eval Results | ✅ Yes | Same `GET /api/v1/evaluations/{evaluation_id}` — run-level `score` + per-row judge scores/reasoning in the `score_trace_url` trace. Covers the overall gauge + question-level table. |
| List Evals (History) | ✅ Yes | `GET /api/v1/evaluations` (`limit`/`offset`). The prototype's version/set filters and sorting would be applied client-side. |
| Export Eval Results (CSV) | ❌ No | No CSV/file export route. `GET /api/v1/evaluations/{id}` has an `export_format` param but it only accepts `row`/`grouped` and just restructures the JSON — it does not produce a CSV. The prototype's "⤓ Export CSV" has no API. |
| Improve Prompt | ✅ Yes | `POST /api/v2/evaluations/{evaluation_id}/improve-prompt` (v1 also exists). Backs the "What to change next" / suggested-prompt-change panel. Async — delivers to an HTTPS `callback_url`. |
| Run-time / online evaluation | ❌ No | **No API.** The entire "Run-time Evaluations" tab (continuous scoring of real conversations, rolling trend, flagged log) has no backing endpoint — Kaapi only does the on-demand Golden Q&A run. |

---

## Adjacent UI surfaces (outside Evals & Configs)

Called out for completeness — these prototype tabs aren't part of the Evals/Configs
flow but affect a full build:

| API | Available | Notes |
| --- | --- | --- |
| Try It Out (run one prompt against a saved config) | ❌ No | No single-prompt sandbox route takes a `config_id`. The only config-by-reference execution is the full STS chain `POST /api/v1/llm/chain/sts`, not a text playground. `POST /responses` exists but doesn't reference a config. |
| Knowledge Base (list / add / remove files, vector store) | — | KB management lives outside the config/evaluations routes; not evaluated here. The config blob only references `knowledge_base_ids`. |

---

## Summary of gaps

Everything the Evals & Configs flow needs **exists today except**:

1. **Publish / go-live for a config version** — no live/published/active concept in Kaapi at all (the biggest gap; the prototype's whole version-lifecycle UI depends on it).
2. **Duplicate config** — no clone endpoint.
3. **Run-time / online evaluation** — no live-traffic scoring; only on-demand runs.
4. **CSV export** of eval results and of a dataset — no export API.
5. **Per-run duplication factor** — Kaapi sets it at dataset-upload time, not per run.
6. **Dataset questions listing** — `GET dataset` returns the record, not an items API (rows come from the stored CSV).
