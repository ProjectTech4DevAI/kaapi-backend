# Why tests run bcrypt at 4 rounds

`app/core/security.py` configures the shared `CryptContext` with
`bcrypt__rounds=4` when `ENVIRONMENT == "testing"`, and the production
default of 12 everywhere else.

## The problem

bcrypt is *deliberately* slow — its cost parameter exists to make
brute-forcing stolen hashes expensive. At the default 12 rounds, one
hash or verify takes ~175 ms of pure CPU.

That security property is pure overhead in tests. The suite hits bcrypt
constantly:

- every API-key-authenticated request verifies the key hash
  (`APIKeyManager.verify`)
- every login (JWT token fixtures) verifies the user's password
  (`crud/user.py`)
- seeding hashes user passwords and API keys

Profiling the suite (2026-08) showed a uniform ~1.6 s tail across ~300
API-route tests — almost entirely bcrypt. The slowest 10% of tests
accounted for 58% of total test time.

## The fix

4 is bcrypt's minimum cost factor: ~0.7 ms per operation, a ~250×
speedup. Example: `test_users.py` went from ~1.5 s per test to 20 ms.

This is safe because:

- **Nothing real is protected.** Test hashes guard seeded throwaway
  credentials in a local database that is wiped every run.
- **The code path is identical.** Tests still exercise real bcrypt
  hashing and verification — only the work factor changes, and bcrypt
  stores the cost inside each hash, so verification is
  self-describing and format-compatible with production hashes.
- **Production is untouched.** Any environment other than `testing`
  gets 12 rounds explicitly.

## Rules

- Never raise test rounds back "for realism" — it buys nothing and
  costs minutes per suite run.
- Never point `ENVIRONMENT=testing` at a database holding real
  credentials (low-cost hashes would then be weakly protected).
- All hashing must go through the shared `pwd_context` in
  `app/core/security.py` (`APIKeyManager.pwd_context` aliases it).
  Creating a separate `CryptContext` silently reverts to slow
  defaults — that's how the test seed data ended up with its own
  12-round context before this change.
