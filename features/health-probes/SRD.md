# Health Probes SRD

## Introduction & Purpose

This SRD redefines Kaapi's provider health probes so a probe run is a real, checkable test of the `POST /llm/call` path, not a call that only confirms a Celery task got enqueued.

The current probe (`app/services/health_probes.py`) calls `provider.execute()` directly, bundles all ~13 provider/modality combinations into a single Celery task, and wraps the Sentry check-in around that same enqueue-only task. This bypasses `LLMCallConfig` resolution and the provider mappers entirely, so a malformed per-provider config (ElevenLabs' TTS param is `model_id`, not `model`; Sarvam's TTS/STT calls require a `language` param absent from the probe payload) never reaches validation and never flags. There is also no independent per-probe record: pass/fail exists only inside a single combined Sentry check-in for the whole batch.

The feature produces, per cron tick:
- Exactly one probe fired through the real `/llm/call` job pipeline (config blob -> mapper -> provider), using the same `job` table record every production LLM call gets.
- A Sentry check-in that reflects the previous tick's actual job outcome (`SUCCESS`/`FAILED`), not just that a task was enqueued.

**Phase 1:** round-robin one-probe-per-tick firing through `start_job`, previous-probe status check via the existing `job` table, Redis-held rotation state.
**Phase 2+:** none identified; the hardcoded probe registry is maintained by hand as providers/models change.

Intent: the probe is a real end-to-end regression guard against config/mapper/backward-compat breaks, with provider uptime as a secondary signal, not the reverse.

## Goals

- A probe traverses the full `/llm/call` path (`LLMCallConfig` resolution, mapper, provider call) via the same job pipeline production traffic uses, not a direct provider client call.
- Malformed or incompatible per-provider params (e.g. a wrong param name, a missing required param) surface as a validation or job failure, because the probe now goes through the same config validation every real call goes through.
- The registry's ElevenLabs TTS and Sarvam TTS entries carry the provider-required Kaapi params (`voice` and `language` respectively) so those two probes pass, not merely fail loudly.
- Each cron tick tests one probe from the registry, round-robin, instead of all combinations at once.
- Sentry reflects the real outcome of a specific probe's job, not just that a Celery task was enqueued.
- A missing rotation-state key in Redis degrades to a logged, Sentry-visible entry and the flow continues; it never fails the cron tick.

## Assumptions & Constraints

- **Out of scope:** a probe results table or dashboard (the `job` table is the record of pass/fail); config storage for probes (each probe's config is hardcoded in its `LLMCallRequest` payload); alerting rules beyond the existing Sentry cron-monitor check-in pattern.
- **Cadence:** cron fires every `HEALTH_PROBE_INTERVAL_MINUTES` (~3); one probe per tick. With N probes in the registry, a given probe's effective retest cadence is `N * HEALTH_PROBE_INTERVAL_MINUTES`.
- **Rotation state, no new table:** the round-robin index and the last-fired job ID live in Redis (`REDIS_URL`, already Kaapi's Celery result backend), not Postgres. Keys: `health_probe:index` (advanced via atomic `INCR`, `mod` over registry length, so two overlapping ticks can never claim the same slot) and `health_probe:last_job_id`.
- **Registry:** the probe list stays a hardcoded Python list, same shape as the current `_PROBES` (provider, model, modality), plus an explicit per-entry `params` dict for any Kaapi param a provider's mapper requires beyond `model`. Concretely: the Sarvam TTS entry carries `language` (its mapper hard-requires it for `target_language_code`, `app/services/llm/mappers.py` `map_kaapi_to_sarvam_params`), and the ElevenLabs TTS entry carries `voice` (its mapper hard-requires it to resolve `voice_id`, `map_kaapi_to_elevenlabs_params`). Edited by hand as providers/models change (e.g. drop a model once Kaapi stops using it).
- **Reuse:** no new tables. The probe's job rides the existing `job` table (`JobType.LLM_API`) via `start_job`, the same entry point `POST /llm/call` uses. `HEALTH_PROBE_ORG_ID` / `HEALTH_PROBE_PROJECT_ID` (existing settings) scope the probe's tenant.
- **Cost:** each probe tick is a real, billed provider call against the configured health-probe org/project; this pollutes that org's usage aggregation by one call per tick. Accepted cost, the point of routing through the real pipeline is the end-to-end regression guard, not a free check.
- **No callback:** the probe's `LLMCallRequest` carries no `callback_url`. Result reading is by polling `Job.status` on the next tick, not by callback delivery, so there is nothing for a callback to add, and no SSRF-guard/domain concern to solve (see Design Decisions).

## Detailed Design (Execution Flow)

Each cron tick does two things in one request: resolve the previous tick's probe result, then fire the next one.

### Cron tick: check previous, fire next

---

**>> PLACE IMAGE HERE: `assets/flow-a.png`, cron tick, check previous probe result then fire the next.**
System-level sequence: the external scheduler, the Kaapi backend, Redis, Postgres, Sentry, the LLM call pipeline, and the provider.

---

Resolution order: read `health_probe:last_job_id` from Redis first. If present, look up that `Job` row; `SUCCESS` means the LLM was actually invoked and returned, `FAILED` (including a provider-side error surfaced through the pipeline) or still `PENDING` past the tick means a real failure, and either drives the Sentry check-in status. If the key is missing (first run, or a Redis eviction), log it and continue without failing the tick, no check-in is possible for a probe that was never recorded.

Rotation: claim `health_probe:index` via an atomic `INCR` (default to `0` on a missing key or a Redis error, logged, non-fatal), select `registry[index mod len(registry)]`, call `start_job` with that probe's hardcoded `LLMCallRequest`, then write the new job ID to `health_probe:last_job_id`. The probe travels through the identical pipeline a real `/llm/call` request does; a provider-side error or a config validation error both land the job in `FAILED`, which is what the next tick reports.

## Functional Requirements (Testing)

| ID | What (user-facing behavior) | Acceptance criteria | Status |
|----|-----------------------------|---------------------|--------|
| FR-1 | One probe fired per cron tick | A single `GET /cron/health-probes` invocation enqueues exactly one `LLMCallRequest` job, not all registry entries | Not Started |
| FR-2 | Probe uses the real LLM call pipeline | The enqueued job resolves through `LLMCallConfig` and the provider mapper (the same path `POST /llm/call` uses), not a direct provider client call | Not Started |
| FR-3 | Round-robin selection | Across consecutive ticks, the probe index advances by one (mod registry length) each time; after `len(registry)` ticks every registry entry has been fired exactly once | Not Started |
| FR-4 | Missing rotation key does not fail the tick | With `health_probe:index` absent from Redis, the tick still fires a probe (defaulting to index 0) and logs the miss | Not Started |
| FR-5 | Missing last-job key does not fail the tick | With `health_probe:last_job_id` absent from Redis, the tick skips the previous-result check (logs it), and still fires the next probe | Not Started |
| FR-6 | Real job failure reported | When the previous probe's `Job.status` is `FAILED` (including a provider-side error), the tick's Sentry check-in reports an error state, not OK | Not Started |
| FR-7 | Real job success reported | When the previous probe's `Job.status` is `SUCCESS`, the tick's Sentry check-in reports OK | Not Started |
| FR-8 | Malformed per-provider config now surfaces | A registry entry whose hardcoded config is invalid for that provider (e.g. a required or misnamed param) fails at config validation/mapper resolution and lands the job in `FAILED`, it is not silently swallowed | Not Started |
| FR-9 | Concurrent ticks never collide | Two overlapping calls to the index claim never return the same index; across N concurrent claims, every registry slot is claimed the same number of times | Not Started |
| FR-10 | Sarvam TTS probe carries the required `language` param | The Sarvam TTS registry entry's mapped native params include `target_language_code`; its job reaches the provider call instead of failing at mapper resolution with "Missing required 'language' parameter for TTS" | Not Started |
| FR-11 | ElevenLabs TTS probe carries the required `voice` param | The ElevenLabs TTS registry entry's mapped native params include `voice_id`; its job reaches the provider call instead of failing at mapper resolution with "Missing required 'voice' parameter for TTS" | Not Started |

## Endpoints

### `GET /cron/health-probes` (existing, behavior replaced)

Still hidden from Swagger, still superuser-only. No longer enqueues a bespoke all-probes Celery task; instead runs the check-previous-then-fire-next tick described above and returns immediately once the next probe is enqueued.

**Response:**

```json
{
  "enqueued": true,
  "job_id": "9c2e4d6f-1a2b-3c4d-5e6f-7a8b9c0d1e2f",
  "probe_index": 4,
  "previous_job_status": "SUCCESS"
}
```

`previous_job_status` is `null` when `health_probe:last_job_id` was missing (first tick, or the key expired).

## Database Schema

No new tables. The probe's per-tick result rides the existing `job` table (`models/job.py`), the same table every `/llm/call` request writes to; `JobType.LLM_API`, scoped to `HEALTH_PROBE_PROJECT_ID`.

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| HEALTH_PROBE_INTERVAL_MINUTES | int | 3 | Cron tick interval, one probe per tick |
| HEALTH_PROBE_ORG_ID | int \| None | existing | Org the probe's job is created under |
| HEALTH_PROBE_PROJECT_ID | int \| None | existing | Project the probe's job is created under |

## Design Decisions / Known Limitations

- **Probe fires through `start_job`, not `provider.execute()`.** This is the core fix: the old design's biggest gap was bypassing config resolution and mappers, which is exactly where the ElevenLabs/Sarvam param mismatches lived undetected. Routing through the same entry point as production traffic means a config-shape regression fails the probe the same way it would fail a real caller.
- **No probe-specific Sentry span inside a bespoke Celery task.** The previous design's Sentry instrumentation sat on the enqueue route, so it never saw the actual provider call. This redesign removes the need for that span entirely: the probe now runs inside the standard LLM job pipeline (already traced), and the check-in is driven by reading the `Job.status` a tick later, which is a stronger signal than a span around a single synchronous call.
- **Redis over a new table for rotation state.** Redis is already wired up as the Celery result backend with unused capacity; a probe-index table would duplicate what two keys already do, and rotation state has no need for query, audit, or FK/tenant isolation.
- **Atomic `INCR`, not GET-then-SET, for the rotation index.** A read-then-write pair lets two overlapping ticks read the same index, fire the same probe, and silently skip another registry entry for a cycle. `INCR` is one atomic Redis command, so concurrent callers always land on distinct, consecutive slots.
- **No callback_url, no callback endpoint.** An earlier pass added a dummy `POST /internal/health-probe-callback` on the assumption the pipeline needs a callback target and Kaapi's SSRF guard would reject `localhost`. Neither holds: `callback_url` on `LLMCallRequest` is optional, `Job.status` is set to `SUCCESS`/`FAILED` in `execute_job` unconditionally, and callback delivery is a best-effort side channel that never gates it. Since the probe already reads its result by polling `Job.status` on the next tick, the callback bought nothing, so `callback_url` is simply omitted.
- **Calling `start_job` directly, not `POST /llm/call` over HTTP.** `start_job` is exactly what the route calls after its two thin wrapper concerns (the auth dependency and callback SSRF validation, both irrelevant to a cron-triggered internal probe with no callback). Going over HTTP instead would add API-key provisioning, a self-referential network hop, and JSON (de)serialization, all to rebuild the same `LLMCallRequest` this call already builds in Python, for no additional coverage.
- **Known limitation:** if a probe's config becomes stale (a model retired, a param shape changed upstream) the registry must be edited and redeployed by hand, there is no self-healing or drift detection on the registry itself.
- **ElevenLabs and Sarvam TTS payloads are fixed, not just detected.** The prior probe passed only `model` for every provider; ElevenLabs TTS silently needs `voice` (to resolve `voice_id`) and Sarvam TTS silently needs `language` (to resolve `target_language_code`), both hard requirements in their respective mappers. Phase 1 adds these fields to the registry entries directly, so both probes are expected to pass on first tick, not merely fail loudly. The redesigned pipeline (routing through the real mapper) is what surfaces this class of gap at all; the registry fix is what closes it for these two entries specifically.
