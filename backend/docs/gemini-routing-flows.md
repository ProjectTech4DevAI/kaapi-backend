# Gemini Provider Routing — Flow Report

Registry: `app/services/llm/providers/registry.py`
Env: `GEMINI_DEFAULT_INFERENCE_ROUTE` (`"aistudio"` | `"vertex"`, default `"aistudio"`)
Platform fallbacks: `GEMINI_API_KEY` (aistudio), `GCP_VERTEX_API_KEY` + `GCP_PROJECT_ID` + `GCP_VERTEX_LOCATION` + `GCP_SA_KEY` (vertex)

## Provider names

| Name | Meaning |
|---|---|
| `google` / `google-native` | Platform-routed. Backend chosen by env. |
| `google-vertex` / `google-vertex-native` | Explicit Vertex AI. Uses caller's creds. |
| `google-aistudio` / `google-aistudio-native` | Explicit AI Studio. Uses caller's creds. |

## Happy paths

### H1 — Explicit `google-vertex` with BYOK creds
Caller: `provider_type="google-vertex"`, project has `credential` row `provider='google-vertex'`.
Flow: `get_provider_class` → `GoogleVertexAIProvider`. Credential lookup key = `google-vertex`. Row found → creds injected into `create_client` → BYOK client returned.

### H2 — Explicit `google-aistudio` with BYOK creds
Caller: `provider_type="google-aistudio"`, project has `credential` row `provider='google-aistudio'`.
Flow: `get_provider_class` → `GoogleAIProvider`. Lookup key = `google-aistudio`. Row found → BYOK api_key used.

### H3 — Platform-routed `google`, env=`vertex`, BYOK vertex creds
Caller: `provider_type="google"`, `GEMINI_DEFAULT_INFERENCE_ROUTE="vertex"`, row `provider='google-vertex'` exists.
Flow: `_GOOGLE_ROUTED` hit → env → vertex. Lookup key rewritten to `google-vertex`. Row found → BYOK vertex client.

### H4 — Platform-routed `google`, env=`aistudio`, BYOK aistudio creds
Symmetric to H3 with `GoogleAIProvider` + `google-aistudio` lookup key.

### H5 — Platform-routed `google`, env=`vertex`, no BYOK → platform fallback
Caller: `provider_type="google"`, env=vertex, no `google-vertex` row.
Flow: creds = None. `is_platform_routed=True` so the missing-creds raise is skipped. `credentials={}` passed to `GoogleVertexAIProvider.create_client`, which per-field falls back to `GCP_VERTEX_*` settings. Client built from platform defaults.

### H6 — Platform-routed `google`, env=`aistudio`, no BYOK → platform fallback
Symmetric to H5. `GoogleAIProvider.create_client` uses `settings.GEMINI_API_KEY`.

### H7 — Any non-Google provider (e.g. `openai-native`)
Unchanged: lookup key = `openai`, missing creds raise, present creds → client.

## Edge cases

### E1 — Explicit `google-vertex` without creds
`get_provider_credential` returns None. `is_platform_routed=False` → raise
`ValueError("Credentials for provider 'google-vertex' not configured for this project.")`.
Rationale: explicit provider names bypass platform fallback by design.

### E2 — Explicit `google-aistudio` without creds
Symmetric to E1. Raises for `'google-aistudio'`.

### E3 — Platform-routed `google` with no BYOK and no platform key
env=aistudio, `google-aistudio` row missing, `GEMINI_API_KEY=""`.
Flow: creds=None, `credentials={}`, `create_client` → `api_key` empty → raises `ValueError("API Key for Google Gemini Not Set")`.
Same shape on vertex if all `GCP_*` platform vars are empty and the row is missing (per existing vertex behavior).

### E4 — env unset (empty string)
`settings.GEMINI_DEFAULT_INFERENCE_ROUTE=""`. In `get_provider_class`: `route == "aistudio"` is False → falls to vertex branch. Effectively `""` behaves like `"vertex"`.
Not enforced as an error. If you want to hard-fail on bad values, add the `Literal` back.

### E5 — env set to garbage (e.g. `"gemini-pro"`)
Treated as "not aistudio" → routes to vertex. Silent misconfiguration.
Same mitigation as E4.

### E6 — `-native` variants
`google-native` behaves identically to `google` (env-routed).
`google-vertex-native`, `google-aistudio-native` behave identically to their non-native peers.
`-native` is stripped for the credential lookup key.

### E7 — Legacy `google` credential rows (pre-migration)
If DB still has rows keyed `provider='google'` from the old semantics, this code will not find them when the caller asks for `google-vertex` or when `google`+env=vertex remaps to `google-vertex`. Requires the pending data migration to rename `google` → `google-vertex`.

### E8 — Non-ValueError exception in `create_client`
Caught by the outer `except Exception` in `get_llm_provider`. Logged with `exc_info`, re-raised as `RuntimeError(f"Could not connect to {provider_type} services.")`. `ValueError` from missing keys is preserved as-is.

### E9 — Invalid provider name
`get_provider_class` → not in `_registry`, not in `_GOOGLE_ROUTED` → raises `ValueError("Provider '<x>' is not supported...")` with the full supported list.

## Decision table

| provider_type | env | BYOK row present | Result |
|---|---|---|---|
| `google-vertex` | any | yes | vertex, BYOK |
| `google-vertex` | any | no | raise |
| `google-aistudio` | any | yes | aistudio, BYOK |
| `google-aistudio` | any | no | raise |
| `google` | `vertex` | `google-vertex` yes | vertex, BYOK |
| `google` | `vertex` | no | vertex, platform fallback |
| `google` | `aistudio` | `google-aistudio` yes | aistudio, BYOK |
| `google` | `aistudio` | no | aistudio, platform fallback |
| `google` | `""` / other | — | vertex (silent) |

## Code verification

Manual trace of `registry.py` against each documented flow:

| Flow | Routing check | Credential key check | `create_client` behavior | Match |
|---|---|---|---|---|
| H1 — explicit `google-vertex` + BYOK | `_registry["google-vertex"]` → `GoogleVertexAIProvider` | `"google-vertex"` lookup | `credentials.get("api_key")` is truthy → BYOK used | OK |
| H2 — explicit `google-aistudio` + BYOK | `_registry["google-aistudio"]` → `GoogleAIProvider` | `"google-aistudio"` lookup | `credentials.get("api_key")` is truthy → BYOK used | OK |
| H3 — platform `google`, env=vertex, BYOK | `_GOOGLE_ROUTED` hit → `GoogleVertexAIProvider` (env != aistudio) | `"google-vertex"` lookup | `credentials.get("api_key")` truthy → BYOK used | OK |
| H4 — platform `google`, env=aistudio, BYOK | `_GOOGLE_ROUTED` hit → `GoogleAIProvider` (env == aistudio) | `"google-aistudio"` lookup | `credentials.get("api_key")` truthy → BYOK used | OK |
| H5 — platform `google`, env=vertex, no BYOK | `_GOOGLE_ROUTED` hit → `GoogleVertexAIProvider` | no `google-vertex` row → `credentials={}` | `credentials.get("api_key")` falsy → falls back to `settings.GCP_VERTEX_API_KEY` | OK |
| H6 — platform `google`, env=aistudio, no BYOK | `_GOOGLE_ROUTED` hit → `GoogleAIProvider` | no `google-aistudio` row → `credentials={}` | `credentials.get("api_key")` falsy → falls back to `settings.GEMINI_API_KEY` | OK |
| H7 — non-Google (e.g. `openai-native`) | `_registry["openai-native"]` → `OpenAIProvider` | `"openai"` lookup (strip `-native`) | unchanged existing flow | OK |
| E1 — explicit `google-vertex` no creds | `GoogleVertexAIProvider` | `get_provider_credential` returns None | `is_platform_routed=False` → `ValueError` raised | OK |
| E2 — explicit `google-aistudio` no creds | `GoogleAIProvider` | `get_provider_credential` returns None | `is_platform_routed=False` → `ValueError` raised | OK |
| E3 — platform `google`, no BYOK, no platform key | `GoogleAIProvider` (env=aistudio) | no `google-aistudio` row → `credentials={}` | `api_key` empty → `ValueError("API Key for Google Gemini Not Set")` | OK |
| E4 — env unset (`""`) | `route == "aistudio"` is False → `GoogleVertexAIProvider` | — | — | OK |
| E5 — env garbage (e.g. `"gemini-pro"`) | same as E4 | — | — | OK |
| E6 — `-native` variants | `google-native` in `_GOOGLE_ROUTED`; others in `_registry`; `-native` stripped from lookup key | `google-vertex` / `google-aistudio` lookup key | identical to non-native peers | OK |
| E7 — legacy `google` rows | lookup key is `google-vertex`/`google-aistudio`, not `google` | — | — | OK (requires migration) |
| E8 — non-`ValueError` in `create_client` | — | — | caught → logged `exc_info=True` → re-raised `RuntimeError`; `ValueError` preserved | OK |
| E9 — invalid provider name | not in `_GOOGLE_ROUTED`, not in `_registry` → `ValueError` with supported list | — | — | OK |

## Pending / not covered here

- DB data migration renaming `credential.provider='google'` rows (to `'google-vertex'` per the new semantics). Tracked separately.
- `.env.example` update for `GEMINI_API_KEY` and `GEMINI_DEFAULT_INFERENCE_ROUTE`.
