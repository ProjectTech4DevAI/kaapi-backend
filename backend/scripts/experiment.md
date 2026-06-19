# Celery Worker Starvation Experiment

## Goal

Characterize CPU/memory behavior and starvation risk for the staging Celery
worker pool (prefork, concurrency=4) under realistic per-bucket load. Decide
afterwards whether to scale ECS containers, split the pool by bucket, or do
nothing.

## Hypothesis

With a single shared 4-slot prefork pool, a heavy bucket (e.g. STT or SpeechToSpeech)
saturating all 4 slots will block latency-sensitive LLM requests. Priority on
a shared queue does not fix slot-pinning; only dedicated worker pools per
bucket (or more slots via scaling) will.

## Buckets (Celery task groupings)

A "bucket" = a group of Celery tasks that share resource character. The
worker pool is shared across all of them; starvation analysis is per bucket.

| Bucket           | Underlying Celery tasks                                                                 | Driven via API by                                          |
|------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------|
| **llm**          | `run_llm_job`, `run_llm_chain_job`, `run_response_job`                                   | `POST /llm/call`, `POST /llm/chain`, `POST /responses`     |
| **docstore**     | `run_doctransform_job`, `run_collection_setup_job`, `run_collection_batch_job`, `run_delete_collection_job` | `POST /documents`, `POST /collections`, `DELETE /collections/{id}` |
| **speech_eval**  | `run_stt_batch_submission`, `run_stt_metric_computation`, `run_tts_batch_submission`, `run_tts_result_processing` | `POST /evaluations/stt/runs`, `POST /evaluations/tts/runs` |
| **evaluation**   | `run_evaluation_fast`                                                                    | `POST /evaluations`                                        |

### Sub-variants exercised within the `llm` bucket

All four hit the same Celery tasks (`run_llm_job` or `run_llm_chain_job`)
but differ in payload shape and latency profile:

| Variant | Endpoint           | Body shape                                  |
|---------|--------------------|---------------------------------------------|
| text    | `POST /llm/call`   | Text in, stored config (`config.id`)         |
| stt     | `POST /llm/call`   | Audio in (base64 opus), `completion.type=stt` |
| tts     | `POST /llm/call`   | Text in, `completion.type=tts`, ogg out      |
| sts     | `POST /llm/chain`  | Audio in + `knowledge_base_ids` (RAG chain)  |

### Scope of this experiment

- `scripts/load_sim.py` currently **only drives the llm bucket** (across all
  four sub-variants). This characterizes intra-llm-bucket behavior on the
  shared pool — the heaviest path in current traffic.
- `docstore`, `speech_eval`, `evaluation` buckets are **deferred**: they need
  separate drivers (multipart upload for docstore, seeded eval rows for
  the others). Add when intra-llm characterization shows headroom and we
  want true inter-bucket starvation data.

## Metrics per bucket

Compute every metric below per bucket, per scenario. The minimum useful set
is starred (★) — start there; collect the rest only if a question can't be
answered from the starred five.

### Throughput
- `jobs_enqueued` — from `load_sim` logs.
- `jobs_succeeded` / `jobs_failed` — from worker logs (`succeeded in` vs `Task ... raised`).
- ★ `sustained_rate` (jobs/s) = `jobs_succeeded / window_s`, measured over the
  steady-state window (after warmup, before cooldown).

### Latency
- `queue_wait_ms` — enqueue → prerun. Requires the Celery signal hook;
  from logs alone, approximate as `client_http_latency_ms − run_ms` only when
  the API enqueues synchronously.
- ★ `run_ms` p50 / p95 / p99 — direct from `Task ... succeeded in Xs` lines.
- `client_http_latency_ms` p50 / p95 — from `load_sim` log (`latency_ms=...`).

### Resource cost
- ★ `peak_rss_per_child_mb` — high-water mark from `top` timeline during the run.
- ★ `rss_delta_per_child_mb` = `peak − idle_floor` (~228 MB baseline). Honest per-task cost.
- ★ `mean_cpu_pct_per_child` — average `%CPU` while child is in `R` state.
  Distinguishes CPU-bound (~100%) from I/O-bound (much lower, e.g. STT waiting
  on provider API).
- `container_total_rss_peak_mb` — `top` header `MiB Mem used`. Drives ECS sizing.

### Slot occupancy (starvation predictor)
- `slot_demand` = `sustained_rate × mean_run_s`. If `sum(slot_demand)` across
  buckets > 4, starvation is structural — no amount of priority tuning fixes it.
- `pct_time_pool_saturated` = fraction of `top` timeline rows where all 4
  children are in `R`. >50% during a steady-state window = pool is the bottleneck.

### Starvation (only meaningful in mixed scenarios)
- ★ `latency_inflation` = `mixed.run_ms_p95 / solo_baseline.run_ms_p95` per bucket.
  >1.5× = this bucket is being starved by something else in the mix.
- `queue_depth_at_steady_state` — from `celery inspect reserved` or RabbitMQ
  `messages_ready`. Should be flat; linear growth means past saturation.

### Reliability
- `error_rate` = `jobs_failed / jobs_enqueued`.
- `http_4xx_rate` / `http_5xx_rate` — from `load_sim`. 5xx spikes during stress
  point at the API or DB, not the worker.

### Output: per-scenario summary row

For each scenario, produce one row joining the bucket and its metrics. Example:

| scenario_id   | bucket | sustained_rate | run_ms_p95 | rss_delta_mb | cpu_pct | latency_inflation | error_rate |
|---------------|--------|----------------|------------|--------------|---------|-------------------|------------|
| baseline_llm  | llm    |                |            |              |         | 1.00 (ref)        |            |
| mixed_balanced| llm    |                |            |              |         |                   |            |

## Tooling

- **Load driver**: `scripts/load_sim.py`. CLI: `--load bucket:rate[,bucket:rate] --duration <s>`.
  - Threaded enqueuer per bucket; sends real prod-shaped JSON bodies.
  - Jitter modes: `poisson` (default, exponential inter-arrival), `uniform`, `none`.
  - Each request stamps `run_id` into `request_metadata.test_id` for log slicing.
- **Worker observation**: `docker exec -it <celery-container> bash`, then `top`.
  - Sort by RSS: `Shift+M`. Show cmdline: `c`. Filter: `top -c -p $(pgrep -d',' -f celery)`.
  - Capture timeline: `while true; do date; top -b -n 1 -c -p $(pgrep -d',' -f celery); sleep 5; done > top_<scenario>.log`.
- **Task durations**: from Celery's built-in `Task <name>[<id>] succeeded in Xs` log lines.
- **Logs sink**: staging CloudWatch (already configured). Slice by `run_id`.

## Handy commands

### Shell into the celery container
```bash
docker compose ps                                  # find container name
docker compose exec celery-worker bash             # adjust service name
# or
docker exec -it <container_id> bash
```

### `top` for prefork children
```bash
# interactive: sort by RSS, show cmdline, filter to celery
top -c -p $(pgrep -d',' -f celery)
# inside top: press M (sort by RSS), c (toggle cmdline), 1 (per-CPU view)

# one-shot snapshot
top -b -n 1 -c -p $(pgrep -d',' -f celery)

# timeline capture during a scenario
while true; do
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  top -b -n 1 -c -p $(pgrep -d',' -f celery)
  sleep 5
done > top_<scenario_id>.log
```

### Find prefork children directly
```bash
pgrep -af celery                                   # all celery processes
ps -eo pid,ppid,rss,cmd | grep -i celery | grep -v grep
# parent has lowest PID among celery procs; children share its PPID
```

### docker compose logs
```bash
docker compose logs --tail=200 celery-worker
docker compose logs -f celery-worker | tee worker_<scenario_id>.log
docker compose logs celery-worker --since 10m | grep "succeeded in"
docker compose logs celery-worker | grep "<run_id>"   # slice by --run-id stamp
```

### Celery introspection (from inside the container or any env with the app)
```bash
celery -A app.celery.celery_app inspect active            # currently running tasks
celery -A app.celery.celery_app inspect reserved          # picked up, not started
celery -A app.celery.celery_app inspect stats             # pool size, mem
celery -A app.celery.celery_app purge                     # drain queue between scenarios
```

### Broker queue depth (RabbitMQ)
```bash
# via management plugin (if enabled) — replace <vhost>
curl -u guest:guest http://<broker-host>:15672/api/queues/<vhost> \
  | jq '.[] | {name, messages, messages_ready, messages_unacknowledged}'

# via rabbitmqctl inside broker container
docker compose exec rabbitmq rabbitmqctl list_queues name messages consumers
```

### Extract task durations from worker logs
```bash
# all tasks
grep -E "succeeded in [0-9.]+s" worker_<scenario_id>.log

# specific task name
grep "run_llm_job.*succeeded in" worker_<scenario_id>.log \
  | awk -F'succeeded in ' '{print $2}' | awk '{print $1}' | sed 's/s$//'

# mean + count
grep "run_llm_job.*succeeded in" worker_<scenario_id>.log \
  | awk -F'succeeded in ' '{print $2}' | awk '{sum+=$1; n++} END {print sum/n, n}'
```

### Slice CloudWatch by run_id
```
fields @timestamp, @message
| filter @message like /sim-<run_id>/
| sort @timestamp asc
| limit 1000
```

### Run-the-sim quick reference
```bash
export STAGING_API_URL=https://api-staging.kaapi.ai/api/v1
export STAGING_API_KEY=...
python scripts/load_sim.py --selftest                                  # sanity
python scripts/load_sim.py --load llm:1.0 --duration 300               # solo
python scripts/load_sim.py --load llm:1.0,stt:0.3,sts:0.2 \
       --duration 600 --run-id mixed_balanced                          # mixed
python scripts/load_sim.py --load sts:0.4 --duration 300 \
       --jitter none --run-id stress_sts_fixed                         # fixed interval
```

## Worker topology assumptions

- Single prefork worker process tree: 1 parent + 4 children.
- Idle baseline per child ≈ 226–228 MB RSS, parent ≈ 260 MB (one-time measurement, re-take if image changes).
- Container memory limit: TBD — confirm against ECS task definition before stress runs.

## Phase 1 — Rate-finding pre-run

For each bucket independently:

1. Drain queue (`celery -A app.celery.celery_app purge`); confirm `inspect active` empty.
2. Enqueue **22 jobs** back-to-back at `--load <bucket>:5 --duration 5`. Rate doesn't matter; the goal is to measure per-task duration without queue effects.
3. Wait until `inspect active` is empty.
4. From worker logs, grep the 22 `succeeded in Xs` lines. Drop the first 2 (cold-start). Compute mean and p95.
5. Saturation rate per bucket = `4 / mean_duration_sec`.
6. Record peak RSS observed in `top` log during the batch.

Output table (fill in after the runs). All rows belong to the **llm bucket**;
columns track behavior per sub-variant.

| llm sub-variant | mean_dur_s | p95_dur_s | sat_rate (j/s) | peak_rss_per_child_mb |
|-----------------|------------|-----------|----------------|------------------------|
| text            |            |           |                |                        |
| stt             |            |           |                |                        |
| tts             |            |           |                |                        |
| sts             |            |           |                |                        |

Derive scenario rates from this table:
- `baseline_rate = 0.3 × sat_rate` (comfortable steady state)
- `stress_rate   = 1.5 × sat_rate` (deliberate oversubscription)

## Phase 2 — Scenarios

Each scenario: purge → 30s warmup → run → 60s cooldown / wait for drain.
Stamp `--run-id <scenario_id>` so CloudWatch slicing is trivial.

All scenarios below exercise the **llm bucket** in different sub-variant
mixes. Inter-bucket scenarios (llm vs docstore, etc.) are out of scope until
the other bucket drivers exist.

### 2.1 Baseline (solo, per llm sub-variant)

```
python scripts/load_sim.py --load llm:<baseline_rate> --duration 300 --run-id baseline_llm
python scripts/load_sim.py --load stt:<baseline_rate> --duration 300 --run-id baseline_stt
python scripts/load_sim.py --load tts:<baseline_rate> --duration 300 --run-id baseline_tts
python scripts/load_sim.py --load sts:<baseline_rate> --duration 300 --run-id baseline_sts
```

Captures: idle floor vs. running RSS, %CPU per child, no queue growth expected.

### 2.2 Stress (solo, per llm sub-variant)

```
python scripts/load_sim.py --load llm:<stress_rate> --duration 300 --run-id stress_llm
# ... and so on for stt / tts / sts
```

Expected: queue depth grows, RSS peaks, possibly OOM-adjacent behavior for sts.

### 2.3 Mixed balanced (all llm sub-variants)

All four llm sub-variants at `baseline_rate` concurrently. If nothing starves here,
the steady-state pool is fine and the rest of Phase 2 can be skipped.

```
python scripts/load_sim.py \
    --load llm:<b_llm>,stt:<b_stt>,tts:<b_tts>,sts:<b_sts> \
    --duration 600 --run-id mixed_balanced
```

### 2.4 Adversarial (heavy llm sub-variant vs. latency-sensitive text)

Slot-demand sum (rate × mean_dur) is the key prediction. Aim for sum > 4 so
LLM has nowhere to land.

```
# sts pinning LLM
python scripts/load_sim.py --load sts:<stress_sts>,llm:<baseline_llm> \
    --duration 600 --run-id adv_sts_vs_llm

# stt pinning LLM
python scripts/load_sim.py --load stt:<stress_stt>,llm:<baseline_llm> \
    --duration 600 --run-id adv_stt_vs_llm
```

Expected: LLM `succeeded in` log lines drop or stop; queue wait climbs.

## Phase 3 — Decision rubric

After Phase 2 data is in hand:

- **Mixed balanced is fine + adversarial only starves at unrealistic rates**
  → do nothing. Set a queue-depth alert. Stop.
- **Mixed balanced already saturates 4 slots at realistic prod rates**
  → scale ECS containers (2×–3×) and re-run mixed balanced. Pool split likely unnecessary.
- **Even at 2× ECS containers, one bucket still pins all slots**
  → split the pool: separate queues per bucket with dedicated worker concurrency
    (e.g. 2 slots LLM, 1 slot STS, 1 slot STT/TTS). Priority on a shared
    queue will NOT fix prefork slot pinning.

## Caveats

- **Prefork RSS is process-wide, not per-task** — but since each prefork child
  runs one task at a time, child RSS during a task is effectively per-task RSS.
  Aggregate RSS across children includes copy-on-write shared pages; trust
  `top`'s header `MiB Mem used`, not the sum of child RSS, for container sizing.
- **Single-host artifact**: staging may not match prod ECS host CPU/memory.
  Saturation rates derived here are a lower bound for prod sizing, not exact.
- **Audio sample**: one short opus clip reused for every stt/sts request.
  Real prod audio length varies; per-task duration spread will be wider in prod.
- **`worker_max_tasks_per_child`**: if set, expect periodic RSS resets as
  children recycle. Note the value before interpreting RSS trends.

## Out of scope

- Splitting the pool / introducing priorities (Phase 3 follow-up, only if rubric demands it).
- Per-task structured instrumentation (Celery signals). Adds queue-wait
  visibility; revisit only if log-based timing is insufficient.
- Burst / time-of-day arrival modeling. Poisson is the default; capture a
  real arrival timeline from CloudWatch and replay only if Poisson is a poor fit.
