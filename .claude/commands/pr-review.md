---
description: Review the current branch's diff against main using kaapi-backend conventions and a personal review checklist.
---

You are reviewing the current branch's local diff against `main` in this FastAPI + PostgreSQL + SQLModel + Alembic + Celery service.

## Gather the diff

Refresh the remote baseline, then pin the base so every run reviews the same file set against the latest `main` — an ambiguous or stale base is why the same branch yields 13 files one run and 39 the next.

1. **Pull this branch** — `git pull origin $(git rev-parse --abbrev-ref HEAD)` — fast-forward local to the latest pushed commits so the review runs on current work, not a stale HEAD. If the pull fails (no upstream, diverged, conflicts), stop and report — don't review stale code silently.
2. **Fetch the latest remote `main`** — `git fetch origin main`. The baseline is `origin/main` (true GitHub main), not the stale local `main` ref. No checkout, no rebase — the tree stays on your branch.
3. `BASE=$(git merge-base origin/main HEAD)` — pin the merge-base SHA once; use it for every command below so the scope can't drift between runs.
4. `git diff $BASE...HEAD --stat` for the file/line counts, `git diff $BASE...HEAD` for the review, `git log $BASE..HEAD --oneline` for the commits.
5. **Separate this branch's own work from merged-in commits.** A branch that merged another open PR (e.g. #956) carries that PR's files into the diff and inflates scope. Scan `git log $BASE..HEAD --oneline` for merge commits / unrelated PR numbers; if found, review only this branch's commits and note at the top: "Excluded N files from merged-in #<pr> — review under its own PR." Never silently review two features as one.
6. State the resolved base SHA + file count at the top of the review so a second run is comparable to the first.
7. `Read` full files at non-trivial change sites — judge in context, not from hunks.
8. `Grep` for duplication, reused literals, unused symbols.

## What to check

### Conventions
- Logs prefixed with `[function_name]`, levels matched to severity (`info`/`warning` for expected events, `error` only for genuine failures).
- Route descriptions via `description=load_description("<domain>/<action>.md")`, never inline strings. `response_model` set; no untyped `dict` responses.
- DB columns get `sa_column_kwargs={"comment": "..."}` when purpose isn't obvious (status fields, JSON, foreign keys).
- Type hints on every parameter and return. `-> Any` is not an annotation — narrow it or drop it.
- `uv` is the runner, not `pip`.

### Layering & duplication
- `HTTPException` belongs in routes, not `crud/`. CRUD returns data / `None` / raises domain errors. Third-party network calls also don't belong in `crud/` — that's DB-only.
- Routes thin, business logic in `services/`, DB access in `crud/`.
- Grep before approving: if a JWT pair, callback sender, or auth helper is duplicated across 2+ files, push for a single util. Before suggesting "extract a helper", confirm one doesn't already exist — the touched domain's `docs/wiki/modules/*.md` page lists its existing services/crud/utils, use it as the grep starting point.
- New table in the diff → check `docs/wiki/domain-map.md`: does an existing entity/flow already cover it? Ask for the justification if the page suggests reuse.
- Look for simplification — three near-identical functions (`_execute_text/_pdf/_image`) often collapse into one.

### Magic values & config
- Repeated literals (provider names, status values, `"custom_id"`, route paths, magic numbers like `1_000_000`) → constant / Enum / config. Name the other location where it's reused.
- Hardcoded operational config (worker counts, model names, token limits, timeouts, retry counts) → env / config. Defaults lean toward smallest/cheapest, not most expensive.
- Dict crossing function boundaries where a Pydantic model belongs.

### Naming
- `list_*` for plural fetch, `get_*` for singletons. Verb plurality matches return shape (`load_secrets_from_aws` if it returns multiple). Suffix `Enum` on enum classes. snake_case funcs/vars, PascalCase classes, UPPER_SNAKE constants.
- No leftover names from copy-paste of a sibling file.
- Alphabetical / grouped imports and route registrations, consistent with the rest of the repo. PEP 8 import order (stdlib first).

### Error handling
- `try` wraps *only* the line(s) that throw. Bloated try blocks are bugs waiting to happen.
- Nested `try/except`: trace the path. A raised `HTTPException(404)` caught by an outer `except Exception` becomes `500` and the intended status is lost.
- Concrete exception types, not `except Exception:` / `except:`.
- Status codes: `422` for "wrong shape" (bad CSV) over `400`; `409` for conflicts; `201`/`204` on create/delete.
- Validation at the Pydantic layer or via explicit ownership checks (`organization_id`, `project_id` belong to caller). `assert` is not validation in production code.
- Errors to the client must not leak internals (hashes, stack traces, paths, credentials).

### Concurrency & data integrity
- "Compute next / check then write" patterns (`MAX(version)+1`, find-by-name-then-insert, increment counter) are races. Push for unique constraints, transactions, or DB-side sequences.
- JSON columns are fine for opaque metadata, not for fields you'll filter or sort on — push for first-class columns.
- Cross-codebase consistency: timestamp names (`created_at` vs `inserted_at`), HTTP code choices, route shape (`/list` suffix is redundant).

### API & response design
- Can the caller use this field? Is `data.id` the id of *what*? Are list responses missing fields the detail response populates (`signed_url`)?
- Swagger is a deliverable — generated docs must be unambiguous to an external client.

### FastAPI
- Router prefixes/tags/versioning consistent with the rest of `app/api/`.
- `Depends` correctly wired for auth, db, current user/org/project.
- Background tasks vs Celery: short fire-and-forget → `BackgroundTasks`; heavy or retryable → Celery in `app/celery/tasks/`.

### Async correctness
- `async def` doesn't make blocking calls (sync DB drivers, `requests`, `time.sleep`, sync file I/O).
- `await` only on coroutines. CPU-bound work → threadpool / Celery / sync route.

### Security
- No secrets / `.env` changes committed.
- Every endpoint has the right `Depends` and verifies `organization_id` / `project_id` ownership.
- API keys / hashes never returned raw — mask after a known prefix.
- **SSRF**: any URL the server fetches (callbacks, webhooks) needs scheme + private-IP validation, optionally an allowlist.
- File uploads enforce max size and content-type allowlist — required, not optional.
- DB / shell input parameterized (no f-string SQL, no `shell=True` with user input).

### Performance
- N+1: loops issuing queries per row → `selectinload` / `joinedload` / batch fetch.
- New filter / FK columns → `index=True`. Pagination on list endpoints.

### Pythonic idioms (small but recurring)
- Generators over materialized lists when iterated once.
- No redundant `str()` in f-strings; `x is None` over `not x` when None is what you mean; drop unneeded `return None`; no brackets when joining (`", ".join(p.value for p in Provider)`).
- Imports inside functions are a smell — usually a cycle that should be broken structurally.
- `setattr` on Pydantic / SQLModel objects → use `model_copy(update={...})` or `dataclasses.replace`.

### Edge cases
- For each new path, ask: input is `None`? list is empty? upstream call fails partway? what does the downgrade migration leave behind?

### Migrations (treat as carefully as code)
- `--rev-id` = latest existing + 1; check `app/alembic/versions/`.
- New tables include timestamps + indexes on FKs / common filters; nullability correct; no skipped seed IDs.
- `downgrade()` implemented and reversible — empty downgrade is a blocker.
- Backfills live in `upgrade()` SQL, not a separate manual script.

### Wiki freshness
- Diff adds/removes a route, table, model, service, crud file, or Celery task but the matching `docs/wiki/modules/*.md` page (or `docs/wiki/domain-map.md` for entity/edge changes) is not updated in the same diff → flag it. Stale wiki misleads every future session.

### Cleanup
- Unused imports / functions / params / dead paths.
- Empty `__init__.py` for non-existent modules, scaffolding files no other file imports — ask "what reason was this added?"
- Commented-out blocks and `print(...)` debug removed.

### Tests
- New behavior → test. Bug fix → regression test. Non-trivial code with zero tests → say so.
- Tests assert behavior, not implementation. Flag tautological / framework-only tests.
- Use the `app/tests/` factory pattern — no hardcoded `organization_id=1`.
- Random values seeded. Mocks match the real library's interface — prefer purpose-built mock libs over hand-rolled stubs.

## How to write the comments

- Cite `path:line`. Show the suggested change inline when short.
- **Name the failure mode**, not just the smell. Weak: "this try/except is too broad." Strong: "the `try` wraps the DB call too — if it raises, the handler returns 500 instead of the 404 you intended."
- **Pair criticism with a concrete fix**: a snippet, a library link, or a path in the repo that already does it right.
- **Question form** for judgment calls ("Why hardcode four workers?"). **Direct form** for unambiguous bugs.
- Hedge ("maybe", "I think") on judgment, not on correctness.
- Defer non-blocking work explicitly: "Not for this PR — worth a follow-up." Don't let style nits gate a merge.
- Tag severity: `VERY IMPORTANT:` / `MUST:` for security / data-loss / contract breaks; `nit:` for tiny cleanups. Approval can carry caveats ("approved, please resolve X before merging") — never just rubber-stamp.

## Output format

```
**Branch**: <head> → main @ <short base SHA> · <file count> files (<additions>+ / <deletions>-)

## Summary
<1–3 sentences: what the PR does + verdict (approve / approve with nits / request changes). Caveats on approval are fine.>

## Blocking issues
- <correctness, security, or convention violations. Each: path:line, what's wrong, why it breaks, suggested fix. Prefix VERY IMPORTANT: / MUST: when warranted.>

## Suggestions
- <non-blocking improvements>

## Nits
- <style, naming, tiny cleanups — prefix `nit:`>

## What's done well
- <genuine positives only — a correct race fix, a well-placed constraint, a tricky edge handled. Omit the whole section if there's nothing real to say.>
```

**One bullet per item, one section per item; drop any empty section; don't pad.** Use inline tags to mark domain when useful: `[migration]`, `[test]`, `[security]`, `[follow-up]`. Severity drives the section; the tag adds the domain colour. Use `[follow-up]` for cross-cutting work the author should track separately rather than fix in this PR; place it in `Suggestions`.

Do not modify files during the review — read-only.
