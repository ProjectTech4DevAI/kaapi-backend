# Test suite performance: why it was slow, what fixed it

The full suite (~3,080 tests) took **~20 minutes**. After the fixes
below it runs in **~51 s** parallel (~74 s with coverage via
`scripts/test.sh`). Profiled 2026-08 with `pytest --durations` and a
per-test CSV from `--junitxml`.

## Slowness causes, biggest first

**Fix:** point `.env.test` at the local Docker Postgres
(`127.0.0.1`, dedicated test database). Same file, 242 s → 1.2 s.
Suite: 20 min → 3.5 min.

**Rule:** the test database must be local. Latency multiplies by
query count times test count; nothing else in the suite can compete
with WAN round-trips.

### 2. bcrypt at production cost in tests (~58% of remaining test time)

~300 API-route tests sat at a uniform ~1.6 s because every
authenticated request verifies a bcrypt hash at 12 rounds (~175 ms
of pure CPU per hash/verify). Details, safety argument, and rules:
[test-bcrypt-rounds.md](test-bcrypt-rounds.md).

**Fix:** `bcrypt__rounds=4` when `ENVIRONMENT == "testing"`
(~0.7 ms). Slowest tests dropped from ~1.5 s to ~20 ms.

### 3. Sequential execution

`scripts/test.sh` ran one test at a time in a single process.

**Fix:** `pytest-xdist` (`-n auto`), one worker per core. Suite:
3.5 min → ~54 s. This required making the session-scoped
`seed_baseline` fixture single-shooter: `seed_database` starts by
deleting everything, so concurrent workers would wipe each other's
seed — `conftest.py` now takes a `FileLock` and only the first
worker seeds.

**Trade-off:** parallel output is an anonymous dot stream — pytest
can't group dots by file when workers interleave files. Failures
still print full paths; use plain `pytest path/to/test.py -v` when
watching a single module.

### 4. Coverage wrapping the whole run (~20–40% overhead)

`coverage run -m pytest` traces every line and can't merge across
xdist workers.

**Fix:** `pytest-cov` (`--cov=app`), which starts coverage inside
each worker and merges automatically. Coverage stays in
`scripts/test.sh` / CI; skip `--cov` for local iteration.

## What was *not* the problem

**Test count.** 3,080 tests was the original suspect ("delete some
tests"), but the math never blamed volume: 20 min ÷ 3,000 ≈ 400 ms per
test, and no honest unit test costs 400 ms of CPU. That per-test cost
was I/O and crypto overhead. Post-fix the median test is 5 ms, and all
3,080 run in under a minute — culling tests would have bought minutes
while permanently costing coverage.

## Re-profiling recipe

```bash
# slowest N to the terminal
uv run pytest app/tests -n auto --durations=25 -q

# per-test CSV (sequential = contention-free numbers)
uv run pytest app/tests -q --junitxml=report.xml
# then parse testcase[@time] from report.xml, sort desc
```

Watch for the same signatures: a *uniform* duration across many tests
means a shared fixture or dependency (auth, network, sleep), not the
tests themselves; a *long tail everywhere* means per-test
infrastructure cost (DB latency, app startup).
