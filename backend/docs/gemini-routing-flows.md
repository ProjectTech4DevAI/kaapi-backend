# Gemini Provider Routing — Flow Report

Registry: `app/services/llm/providers/registry.py`
Env: `GEMINI_DEFAULT_INFERENCE_ROUTE` (`"google-aistudio"` | `"google-gcp"`) — boot-validated by a `Literal` in `app/core/config.py`, so a garbage value fails startup and can never reach the registry.
Platform fallbacks: `GOOGLE_AISTUDIO_API_KEY` (aistudio), `GOOGLE_GCP_API_KEY` + `GOOGLE_GCP_PROJECT_ID` + `GOOGLE_GCP_PROJECT_LOCATION` + `GCP_SA_KEY` (gcp)

## Provider names

| Name | Meaning |
|---|---|
| `google` / `google-native` | Platform-routed. Backend chosen by env. |
| `google-gcp` / `google-gcp-native` | Explicit Vertex AI on GCP. Uses caller's creds. |
| `google-aistudio` / `google-aistudio-native` | Explicit AI Studio. Uses caller's creds. |

## Happy paths

### H1 — Explicit `google-gcp` with BYOK creds
Caller: `provider_type="google-gcp"`, project has `credential` row `provider='google-gcp'`.
Flow: `get_provider_class` → `GoogleGCPProvider`. Credential lookup key = `google-gcp`. Row found → creds injected into `create_client` → BYOK client returned.

### H2 — Explicit `google-aistudio` with BYOK creds
Caller: `provider_type="google-aistudio"`. Lookup chain = `google-aistudio` → `google`; first row found wins → `GoogleAIProvider` with BYOK api_key.

### H3 — Platform-routed `google`, env=`google-gcp`, BYOK gcp creds
Caller: `provider_type="google"`, `GEMINI_DEFAULT_INFERENCE_ROUTE="google-gcp"`, row `provider='google-gcp'` exists.
Flow: `_GOOGLE_ROUTED` hit → env → gcp. Lookup key rewritten to `google-gcp`. Row found → BYOK GCP client.

### H4 — Platform-routed `google`, env=`google-aistudio`, BYOK `google` creds
`_GOOGLE_ROUTED` hit → `GoogleAIProvider`. Lookup key is the tenant's `google` row (not `google-aistudio`). Row found → BYOK api_key.

### H5 — Platform-routed `google`, env=`google-gcp`, no BYOK → platform fallback
Flow: creds = None. `is_platform_routed=True` so the missing-creds raise is skipped. `credentials={}` passed to `GoogleGCPProvider.create_client`, which per-field falls back to `GOOGLE_GCP_*` / `GCP_SA_KEY` settings.

### H6 — Any non-Google provider (e.g. `openai-native`)
Unchanged: lookup key = `openai`, missing creds raise, present creds → client.

## Edge cases

### E1 — Explicit `google-gcp` without creds
`get_provider_credential` returns None, `is_platform_routed=False` → raise
`ValueError("Credentials for provider 'google-gcp' not configured for this project.")`.
Rationale: explicit provider names bypass platform fallback by design.

### E2 — Explicit `google-aistudio` without creds
Neither `google-aistudio` nor `google` row exists → raise for `'google-aistudio'` (first key in the chain).

### E3 — Platform-routed `google`, env=`google-aistudio`, no `google` row → platform fallback
Symmetric with the gcp route: `is_platform_routed=True` skips the missing-creds raise; `credentials={}` reaches `GoogleAIProvider.create_client`, which falls back to `settings.GOOGLE_AISTUDIO_API_KEY` (raises `ValueError("API Key for Google Gemini Not Set")` only if that is also empty).

### E4 — Unknown/absent env route value
GCP is the default flow: only the exact string `"google-aistudio"` selects `GoogleAIProvider`; anything else resolves to `GoogleGCPProvider`. In practice unreachable — the config `Literal["google-aistudio", "google-gcp"]` rejects other values at boot.

### E5 — `-native` variants
`google-native` behaves identically to `google` (env-routed). `google-gcp-native` and `google-aistudio-native` behave identically to their non-native peers; `-native` is stripped for the credential lookup key.

### E6 — Non-ValueError exception in `create_client`
Caught by the outer `except Exception` in `get_llm_provider`. Logged with `exc_info`, re-raised as `RuntimeError(f"Could not connect to {provider_type} services.")`. `ValueError` is preserved as-is.

### E7 — Invalid provider name
`get_provider_class` → not in `_registry`, not in `_GOOGLE_ROUTED` → raises `ValueError("Provider '<x>' is not supported...")` with the full supported list.

## Decision table

| provider_type | env | BYOK row | Result |
|---|---|---|---|
| `google-gcp` | any | `google-gcp` yes | gcp, BYOK |
| `google-gcp` | any | no | raise |
| `google-aistudio` | any | `google-aistudio` or `google` yes | aistudio, BYOK |
| `google-aistudio` | any | no | raise |
| `google` | `google-gcp` | `google-gcp` yes | gcp, BYOK |
| `google` | `google-gcp` | no | gcp, platform fallback |
| `google` | `google-aistudio` | `google` yes | aistudio, BYOK |
| `google` | `google-aistudio` | no | aistudio, platform fallback (`GOOGLE_AISTUDIO_API_KEY`; raise if unset) |
