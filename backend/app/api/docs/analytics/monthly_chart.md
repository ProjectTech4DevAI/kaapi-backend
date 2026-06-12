Chart-shaped live monthly analytics for the current organization.

Use this endpoint when you want to render the data directly as a line, bar,
or stacked-area chart. Numbers are computed on-demand from `llm_call`,
`llm_chain`, and `evaluation_run` — no caching layer, so the chart always
reflects the current database state. The response shape is compatible with
most chart libraries (Recharts, Chart.js, ApexCharts, Highcharts, ECharts):

- `labels[]` — the x-axis values (one entry per month).
- `series[]` — one entry per chart line/bar, each with a human-readable
  `name` and a `data[]` array. `series[i].data[j]` corresponds to
  `labels[j]`. Missing months are filled with `0` so every series has the
  same length as `labels`.

For a flat row-per-bucket shape (suitable when you want to do your own
pivoting), use `GET /api/analytics/monthly` instead.

---

## Authentication & default scope

Any authenticated user with an organization context can call this endpoint.
By default it returns data scoped to the caller's **currently selected
project**; if the caller has no project selected, it falls back to all
projects in the caller's organization. Pass `project_id` to override the
default — it must reference a project inside the caller's organization, so
cross-organization access is never possible.

---

## Query parameters

| Parameter   | Type    | Required | Default                | Description |
| ----------- | ------- | -------- | ---------------------- | ----------- |
| `metric`    | enum    | **yes**  | —                      | Which metric to plot. One of: `requests`, `cost`, `eval_runs`, `eval_cost`. |
| `group_by`  | enum    | no       | `modality_provider`    | How to split the data into series. See the table below. |
| `from_month`| date    | no       | 24 months before `to_month` (or before today if `to_month` is also omitted) | Inclusive lower bound (first-of-month), e.g. `2026-01-01`. Pass an explicit value to query further back. The default exists to cap worst-case scan size as the source tables grow. |
| `to_month`  | date    | no       | — (no upper bound)     | Inclusive upper bound (first-of-month), e.g. `2026-05-01`. |
| `modality`  | enum    | no       | — (all)                | Pre-filter to a single modality bucket. |
| `provider`  | string  | no       | — (all)                | Pre-filter to a single provider. |
| `project_id`| integer | no       | Caller's current project, if any; else all projects in the org. | Override the default scope. Must reference a project inside the caller's organization. |

### `group_by` values

| Value                 | Series produced                                                              |
| --------------------- | ---------------------------------------------------------------------------- |
| `modality_provider`   | One series per `(modality, provider)` combination. Series name: `"T-FS-T · openai"`. |
| `modality`            | One series per modality, summed across providers. Series name: `"T-FS-T"`.   |
| `provider`            | One series per provider, summed across modalities. Series name: `"openai"`.  |
| `total`               | A single series containing the per-month grand total. Series name: `"total"`. |

---

## Response shape

```json
{
  "success": true,
  "data": {
    "metric": "cost",
    "group_by": "modality_provider",
    "labels": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
    "series": [
      {
        "name": "T-FS-T · openai",
        "data": ["10.500000", "15.400000", "18.700000", "22.100000"],
        "total_input_tokens": 4250000,
        "total_output_tokens": 2810000,
        "total_tokens": 7060000
      },
      {
        "name": "T-FS-T · google",
        "data": ["5.100000", "6.300000", "8.200000", "12.400000"],
        "total_input_tokens": 1820000,
        "total_output_tokens": 1240000,
        "total_tokens": 3060000
      },
      {
        "name": "STT · sarvamai",
        "data": ["0", "0.800000", "1.200000", "1.900000"],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0
      }
    ]
  },
  "error": null,
  "metadata": null
}
```

- `labels` are sorted chronologically (oldest → newest).
- `series` are sorted alphabetically by `name`.
- All `series[].data` arrays have the same length as `labels`. Months with
  no data for a given series are filled with `0`, so the chart library
  doesn't have to align points itself.
- Cost values are decimal strings with up to 6 decimal places.
- `total_input_tokens`, `total_output_tokens`, and `total_tokens` on each
  series are series-wide sums across every label, sourced from
  `llm_call.usage`. They are independent of the chosen `metric` — populated
  whether you're charting requests, cost, or eval numbers. Chains and
  evaluation runs contribute zero to token totals.
- An empty result returns `labels: []` and `series: []`.

---

## Example requests

### 1. Monthly cost grouped by provider (one line per provider)

```
GET /api/analytics/monthly/chart?metric=cost&group_by=provider
```

### 2. Total request volume across all dimensions (single line)

```
GET /api/analytics/monthly/chart?metric=requests&group_by=total
```

### 3. STT-only eval cost trend for the year

```
GET /api/analytics/monthly/chart?metric=eval_cost&modality=STT&from_month=2026-01-01
```

### 4. Cost split by modality for a specific project

```
GET /api/analytics/monthly/chart?metric=cost&group_by=modality&project_id=42
```

---

## Frontend integration tips

**Recharts**: pass `labels` as the X-axis source and render one `<Line>`,
`<Bar>`, or `<Area>` per item in `series`, using `series[i].name` as the
key and the values from `series[i].data`.

**Chart.js / ApexCharts**: the shape is almost their native config — the
`labels` array maps to their `labels`/`categories`, and each series object
maps to `datasets[]` / `series[]`.

For a **stacked area chart** of cost by provider over time, use
`metric=cost` and `group_by=provider` — the response is already
chart-ready.
