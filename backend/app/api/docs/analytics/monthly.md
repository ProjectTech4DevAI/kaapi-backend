Read live monthly analytics for the current organization.

The response is shaped as a list of data points, one per
`(month, modality, provider)` combination — aggregated across every project
in the caller's organization. Each point contains a single numeric `value` —
what that value represents depends on the `metric` query parameter. This
lets the frontend pivot the response directly into chart series without
further post-processing.

Data is computed on-demand from `llm_call`, `llm_chain`, and
`evaluation_run`, so every request reflects the current database state with
no caching layer in between. A row inserted seconds ago will already be
visible in the response.

---

## Authentication & default scope

Any authenticated user with an organization context can call this endpoint.
Scope is decided per-request from the caller's auth context:

| Caller's context | Default scope |
| ---------------- | ------------- |
| Currently selected project | Analytics for **just that project**. |
| Org-level (no project selected) | Analytics across **all projects in the caller's org**. |

The implicit org-id filter is always applied first, so data from other
organizations is never returned. To override the default and look at a
specific project (e.g. an org admin comparing two projects), pass the
`project_id` query parameter — it must reference a project inside the
caller's organization. A `project_id` from a different org returns an
empty result, not a leak.

---

## Query parameters

| Parameter   | Type     | Required | Default | Description |
| ----------- | -------- | -------- | ------- | ----------- |
| `metric`    | enum     | **yes**  | —       | Which metric the `value` field carries on each point. One of: `requests`, `cost`, `eval_runs`, `eval_cost`. |
| `from_month`| date     | no       | 24 months before `to_month` (or before today if `to_month` is also omitted) | Inclusive lower bound. Must be a first-of-month date, e.g. `2026-01-01`. Pass an explicit value to query further back. The default exists to cap worst-case scan size as `llm_call` grows. |
| `to_month`  | date     | no       | — (no upper bound) | Inclusive upper bound. Must be a first-of-month date, e.g. `2026-05-01`. |
| `modality`  | enum     | no       | — (all) | Filter to a single modality bucket. One of: `T-FS-T`, `S-FS-S`, `STT`, `TTS`, `OTHER`. |
| `provider`  | string   | no       | — (all) | Filter to a single provider, e.g. `openai`, `google`, `sarvamai`, `elevenlabs`. |
| `project_id`| integer  | no       | Caller's current project, if any; else all projects in the org. | Override the default scope. Must reference a project inside the caller's organization. Cross-organization access is rejected (the org filter is always applied first). |

### `metric` values

| Value        | What `value` contains on each point |
| ------------ | ----------------------------------- |
| `requests`   | `total_llm_call_requests + total_llm_chain_requests` — the total number of inference requests in the bucket (LLM calls plus chain orchestrations). |
| `cost`       | Sum of LLM call cost in USD for the bucket. Chains are NOT added on top — a chain's cost equals the sum of its child calls, which are already counted. |
| `eval_runs`  | Count of evaluation runs in the bucket. |
| `eval_cost`  | Sum of evaluation run cost in USD for the bucket. |

### `modality` values and how they're derived

| Modality | LLM call (`input_type` → `output_type`) | Evaluation run `type` |
| -------- | --------------------------------------- | --------------------- |
| `T-FS-T` | `text` → `text`                         | `text`                |
| `S-FS-S` | `audio` → `audio`                       | —                     |
| `STT`    | `audio` → `text`                        | `stt`                 |
| `TTS`    | `text` → `audio`                        | `tts`                 |
| `OTHER`  | anything else (image, pdf, multimodal)  | `assessment`, any other type |

LLM chains are attributed to the modality of their **first child call**.

---

## Response shape

```json
{
  "success": true,
  "data": [
    {
      "month": "2026-03-01",
      "modality": "T-FS-T",
      "provider": "openai",
      "value": "12450",
      "input_tokens": 1250000,
      "output_tokens": 820000,
      "total_tokens": 2070000
    },
    {
      "month": "2026-04-01",
      "modality": "T-FS-T",
      "provider": "openai",
      "value": "18230",
      "input_tokens": 1840000,
      "output_tokens": 1210000,
      "total_tokens": 3050000
    },
    {
      "month": "2026-04-01",
      "modality": "STT",
      "provider": "sarvamai",
      "value": "1402",
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    }
  ],
  "error": null,
  "metadata": null
}
```

Rows are sorted by `month`, then `modality`, then `provider`. Cost values
are decimal strings with up to 6 decimal places (e.g. `"12.450000"`).

Token fields (`input_tokens`, `output_tokens`, `total_tokens`) are sourced
from `llm_call.usage` and are independent of the chosen `metric` — they
are populated on every point regardless of whether you asked for
`requests`, `cost`, or eval metrics. This lets the frontend render token
usage in a tooltip or secondary axis without a second API call.

Tokens contributed only by `llm_call` rows. Chains and evaluation runs
add nothing to token totals — chain tokens are the sum of their child
calls (would double-count), and eval tokens live in a separate domain.

If no data matches the filters, `data` is an empty array — this is not an
error.

---

## Example requests

### 1. Total monthly cost across all modalities and providers

```
GET /api/analytics/monthly?metric=cost&from_month=2026-01-01&to_month=2026-05-01
```

### 2. Just the OpenAI text-to-text request volume

```
GET /api/analytics/monthly?metric=requests&modality=T-FS-T&provider=openai
```

### 3. STT evaluation run costs this year

```
GET /api/analytics/monthly?metric=eval_cost&modality=STT&from_month=2026-01-01
```

---

## Notes on accuracy

- **Live reads**: every request runs a fresh `GROUP BY` against the source
  tables, so the response always reflects the current database. There is
  no daily aggregation cron and no staleness window.
- **Default time window** is the last 24 months. When `from_month` is
  omitted, the query is bounded to that range so an unfiltered call can't
  trigger a full-table scan as the source tables grow. Pass an explicit
  `from_month` to query further back.
- **Missing pricing** for a provider/model yields a cost of `0` for those
  rows rather than failing the whole query. Make sure your
  `ModelConfig.pricing` is populated for every provider/model you use if
  you want accurate cost numbers.
- **Cost is not double-counted across chains**: a chain row contributes
  only to the `requests` metric (via the chain count), never to `cost` —
  its dollars come from the underlying `llm_call` rows.
- **Cost computed on summed tokens per (provider, model) group**, which is
  equivalent to per-row pricing because `estimate_model_cost` is linear in
  token counts.
