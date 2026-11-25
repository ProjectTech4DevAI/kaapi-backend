# Duplicate Detection

Checks one new submission against earlier entries and tells you whether it is a duplicate. One request runs two checks, **strict** and **lenient**, and the combined result is delivered to your webhook.

---

## 1. Strict and lenient

The two checks answer different questions, and each searches its own vector store.
both strict and lenient has its own seprate vectostore

### Strict: "Is this an exact duplicate?"

- Searches the **strict vector store** (for example, entries that were already shortlisted).
- Answers **yes or no**, and on yes returns the one entry it duplicates.
- "Exact" means the same problem solved the same way: the same devices, method and materials, with nothing extra or missing on either side. Different wording or language does not matter.
- Built to be careful: wrongly calling an original idea a duplicate is worse than missing one, so when unsure it says no.

Example:

| New submission | Earlier entry | Strict answer |
| :---- | :---- | :---- |
| Walking stick with an ultrasonic sensor that beeps near obstacles | Stick for the blind with an ultrasonic sensor and a buzzer | **Duplicate** (same devices, same job) |
| Walking stick with an ultrasonic sensor that **vibrates** near obstacles | Stick for the blind with an ultrasonic sensor and a buzzer | **Duplicate** (vibration motor and buzzer do the same job: warn the user on the spot) |
| Walking stick with an ultrasonic sensor and a buzzer that **also sends an SMS alert to a family member** | Stick for the blind with an ultrasonic sensor and a buzzer | **Not a duplicate** (the new idea adds a feature the earlier one does not have: the SMS alert) |

### Lenient: "What is similar?"

- Searches the **lenient vector store** (for example, all earlier entries).
- Returns a **ranked list** of similar entries, each with a score:

| Score | Meaning |
| :---- | :---- |
| 5 | Same idea, only the wording differs |
| 4 | Same idea with one small difference |
| 3 | Same problem, shares one important part |
| 2 | Same problem, solved differently |
| 1 | Only the topic is shared |

- Built to catch as much as possible, so a reviewer can look at the borderline ones.

---

## 2. Configs

Each check runs on its own saved config. Create **two configs** through the config API, both with tag `ASSESSMENT`: one for strict and one for lenient.

| Config field | What it does |
| :---- | :---- |
| `assessment.provider` | Must be `google-gcp` |
| `params.model` | The Gemini model for this check |
| `params.instructions` | The check's prompt (sent to Gemini as the system instruction) |
| `params.submission` | Template for the user input; `{query}` (or any `{column}` from `input_schema`) is replaced with the submission text |
| `params.knowledge_base_ids` | The vector store (RAG corpus) this check searches |
| `params.max_num_results` | How many similar entries the search returns |
| `params.json_output_schema` | The JSON shape Gemini must answer in |
| `params.temperature`, `params.thinking_level` | Passed to Gemini as set |

The output schema must declare the fields the code reads:

- **Strict**: `is_duplicate` and `citation`
- **Lenient**: `matches[].score` and `matches[].citation`

Extra fields are allowed and ignored by the code, for example a `difference` field that makes the model name the deciding difference before it answers.

The project also needs a `google-gcp` credential. Its service account is used for both the search and the Gemini call.

A config that cannot run a check (wrong provider, missing params, or missing output fields) is rejected with `422` when the request is submitted, not later.

---

## 3. API call

### Submit

```text
POST /api/v1/assessments/duplicates
```

```json
{
  "query": "Title: Smart stick\nProblem: Blind people bump into obstacles.\nSolution: A walking stick with an ultrasonic sensor that beeps near obstacles.",
  "config": {
    "strict":  { "id": "<strict config id>",  "version": 1 },
    "lenient": { "id": "<lenient config id>", "version": 1 }
  },
  "callback_url": "https://your-host.example/webhooks/duplicates",
  "request_metadata": { "submission_id": "S-1042" }
}
```

| Field | Required | Notes |
| :---- | :---- | :---- |
| `query` | Yes | The submission text; must not be empty |
| `config.strict`, `config.lenient` | Yes | `{id, version}` of each config |
| `callback_url` | Yes | HTTPS only, and not a private or local address |
| `request_metadata` | No | Any JSON object, returned unchanged in the webhook |

Returns `202` with the job id and status `PENDING`. The result arrives later on the webhook.

### Look up one job

```text
GET /api/v1/assessments/duplicates/{id}
```

Returns the job's status and any results so far, or `404` if the job is not in your project. Use it for a single lookup (for example after a missed webhook), not for polling.

### Webhook

```json
{
  "success": true,
  "data": {
    "id": "<job id>",
    "status": "COMPLETED",
    "strict": { "is_duplicate": true, "citation": "<full text of the matching entry>" },
    "lenient": [
      { "score": 5, "citation": "<full text of an entry>" },
      { "score": 3, "citation": "<full text of an entry>" }
    ],
    "errors": { "strict": null, "lenient": null },
    "request_metadata": { "submission_id": "S-1042" }
  },
  "error": null,
  "metadata": null
}
```

If the project has a `webhook_secret` credential, the request is signed with `X-Webhook-Signature` and `X-Webhook-Timestamp` headers.

---

## 4. Core engine: how one check works

Both checks follow the same five steps (`detector.py`), and they run **at the same time**, so a job takes as long as the slower check.

Example submission: *"A walking stick with an ultrasonic sensor that beeps near obstacles."*

### Step 1: Search

Vertex AI RAG Engine searches the check's vector store and returns the `max_num_results` most similar entries.

```text
1. Stick for the blind with an ultrasonic sensor and a buzzer
2. Smart cap that warns blind people with a voice message
3. Glasses with an IR sensor for obstacle detection
```

If an entry is long, the search can return it in pieces; the pieces of one entry are joined back together.

### Step 2: Tag

Each result gets a short label, `C1`, `C2`, `C3`..., and is wrapped in a `<candidate>` tag:

```text
<candidates>
<candidate id="C1">Stick for the blind with an ultrasonic sensor and a buzzer</candidate>
<candidate id="C2">Smart cap that warns blind people with a voice message</candidate>
<candidate id="C3">Glasses with an IR sensor for obstacle detection</candidate>
</candidates>
```

### Step 3: Ask

One Gemini call per check:

- **System instruction**: the config's prompt (strict or lenient).
- **Input**: the submission, then the tagged candidates.

Gemini answers in the config's JSON shape and refers to entries **by label only**:

```json
// strict
{ "is_duplicate": true, "citation": "C1" }

// lenient
{ "all_scores": "C1:5 C2:2 C3:3", "matches": [ { "score": 5, "citation": "C1" }, { "score": 3, "citation": "C3" } ] }
```

### Step 4: Map

The code checks the answer and swaps each label for that entry's full text:

```text
"C1"  →  "Stick for the blind with an ultrasonic sensor and a buzzer"
```

Why labels: the model never writes the citation text itself, so a citation is always a real entry from the vector store, never something the model made up. An answer that cites a label it was not given, or has the wrong shape, is treated as an error for that check.

### Step 5: Result

```json
// strict
{ "is_duplicate": true, "citation": "Stick for the blind with an ultrasonic sensor and a buzzer" }

// lenient (highest score first)
[
  { "score": 5, "citation": "Stick for the blind with an ultrasonic sensor and a buzzer" },
  { "score": 3, "citation": "Glasses with an IR sensor for obstacle detection" }
]
```

Edge cases:

- The search finds nothing: strict says not a duplicate, lenient returns an empty list.
- Lenient cites the same entry twice: the higher score is kept.

---

## 5. What happens around the engine

```text
POST /assessments/duplicates
  1. check callback_url, load both configs, reject unusable ones (422)
  2. save the job: 1 assessment row + 2 run rows (strict, lenient), status PENDING
  3. hand it to Celery                                  → 202 to the caller

Celery task
  4. claim the job: PENDING → PROCESSING (only one worker can)
  5. read configs and credentials, then release the DB connection
  6. run strict and lenient in parallel (section 4)
  7. save both results and the final status
  8. send the webhook
```

database entry, in the existing assessment tables:

- `assessment`: one row per job; `method = DUPLICATE_DETECTION`, `input` holds the request without `config`.
- `assessment_run`: two rows; `execution.level` is `strict` or `lenient`, `execution.result` holds the result, `error_message` holds the error.

### Job status

| Strict | Lenient | Status | Webhook `success` |
| :---- | :---- | :---- | :---- |
| ok | ok | `COMPLETED` | `true` |
| one fails | the other ok | `COMPLETED_WITH_ERRORS` | `true` |
| fails | fails | `FAILED` | `false` |

One check failing never drops the other check's result.

### Failure handling

| What happens | Result |
| :---- | :---- |
| Config missing or unusable at submit | `404` / `422`, nothing saved |
| Task cannot be queued | Job `FAILED`, `503` to the caller |
| Search or Gemini fails for one check | That check gets an error; the other still returns |
| Unexpected error in the task | Both checks get an error, job `FAILED`, webhook sent |
| Worker process killed mid-run | The failure handler `fail_duplicate_detection` marks the job `FAILED` and sends the webhook |
| Same task delivered twice | Only one worker can claim the job, so it runs once |

Known gap: if the whole worker container stops, no failure handler runs and the job stays `PROCESSING`.

### Error messages

| Starts with | Meaning |
| :---- | :---- |
| `[VERTEX] Corpus search failed` | The search failed (corpus name, region or service account access) |
| `[GOOGLE_GCP] ...` | The Gemini call failed (quota, timeout, config) |
| `[KAAPI] The model's answer for the ... check did not match the expected shape` | Invalid answer, or a citation label that was not given |
| `[KAAPI] Duplicate detection could not run` | Unexpected error inside the task |
| `[KAAPI] Duplicate detection was interrupted before it finished` | The worker stopped before the job finished |

### Observability (Sentry)

- Each step is its own bar in the waterfall trace: `duplicate_detection.strict.search`, `duplicate_detection.lenient.search`, `chat <model>` (one per check, with model and token counts) and `duplicate_detection.send_callback`.
- Finished steps stay visible even if the worker dies mid-run.
- DB and HTTP auto-instrumentation is turned off for this feature to keep the trace readable; submission text and retrieved entries are not put on spans.

### Settings .env.example

| Setting | Default | Meaning |
| :---- | :---- | :---- |
| `DEDUP_SEARCH_TIMEOUT` | `5` | Seconds for one search |
| `DEDUP_MODEL_TIMEOUT` | `60` | Seconds for one Gemini call |
| `DEDUP_MODEL_ATTEMPTS` | `3` | Attempts per Gemini call |

### Files

| File | Role |
| :---- | :---- |
| `app/api/routes/assessment/duplicate_detection.py` | `POST` and `GET` routes |
| `app/services/assessment/duplicate_detection/service.py` | Submit, status, run, save, webhook, failure handling |
| `app/services/assessment/duplicate_detection/detector.py` | The core engine (section 4) |
| `app/crud/assessment/duplicate_detection.py` | Create, read, claim and save a job |
| `app/models/assessment/duplicate_detection.py` | Request, result and config models |
| `app/celery/tasks/job_execution.py` | `run_duplicate_detection`, `fail_duplicate_detection` |
| `app/celery/utils.py` | `start_duplicate_detection` (queues the task with its failure handler) |
| `app/tests/assessment/test_duplicate_detection_api.py` | Tests |
