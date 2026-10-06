# Sentry Release Tracking: git sha se release, deploy se link

**Audience:** Prashant / jo bhi kaapi-backend ka deploy pipeline sambhalta hai.
**Time:** ~30 min. Code aur workflow changes PR `feat/sentry-release-tracking` mein hain; baaki Sentry/GitHub settings haath se.

## Abhi kya galat hai

- Sentry mein release `0.5.0` April 2026 se static hai. Har deploy same release pe jaata hai, isliye:
  - "Regressed in release", "Resolved in next release" kaam nahi karte.
  - Suspect commits nahi dikhte (`commitCount: 0`, `deployCount: 0`).
  - Kaunsa deploy kab gaya, Sentry se pata nahi chalta.
- `resolve_sentry_release()` sirf `SENTRY_RELEASE` env ya `kaapi-backend@API_VERSION` deta hai. `API_VERSION` haath se badalta hai, deploy pe nahi.
- Achhi khabar: Dockerfile mein `ARG GIT_SHA` / `ENV GIT_SHA` already hai, aur `create-release.yml` + `deploy-staging-ecs.yml` dono `--build-arg GIT_SHA=${{ github.sha }}` pass karte hain. Container ke andar sha available hai, bas koi padhta nahi.
- Environment theek hai: prod traffic `production` naam se aa raha hai. Sentry mein ek purana `prod` environment bhi hai, usko hide karna hai.

## Target

Release string: `kaapi-backend@<tag>+<sha12>`
Example: `kaapi-backend@v0.5.1+3c5333f8ca82`

- `<tag>` = git tag jo prod deploy trigger karta hai (`v0.5.1`). Staging pe `main`.
- `+<sha12>` = commit sha ke pehle 12 chars. Sentry `+` ke baad wala build metadata maanta hai.
- Same string API aur Celery dono se jaaye, warna Sentry do release dikhayega.

## Step 1: Code, release string sha se banao

**File:** `backend/app/core/config.py`

`SENTRY_RELEASE` ke paas do settings add karo:

```python
GIT_SHA: str = "unknown"
RELEASE_TAG: str | None = None
```

Dockerfile ka `ENV GIT_SHA` pydantic-settings automatically uthayega. `RELEASE_TAG` workflow se aayega (Step 2).

**File:** `backend/app/core/telemetry/sentry/init.py`

`resolve_sentry_release()` badlo:

```python
GIT_SHA_SHORT_LEN = 12


def resolve_sentry_release() -> str:
    """Release id shared by API and worker; SENTRY_RELEASE overrides, else tag+sha from the build."""
    if settings.SENTRY_RELEASE:
        return settings.SENTRY_RELEASE
    version = settings.RELEASE_TAG or settings.API_VERSION
    sha = settings.GIT_SHA[:GIT_SHA_SHORT_LEN]
    if sha and sha != "unknown":
        return f"{settings.BACKEND_SERVICE_NAME}@{version}+{sha}"
    return f"{settings.BACKEND_SERVICE_NAME}@{version}"
```

Local dev pe `GIT_SHA` unknown hoga, to purana format hi aayega. Koi `.env` change nahi.

**Tests:** `backend/app/tests/core/telemetry/sentry/test_sentry_init.py` mein `TestResolveSentryRelease` mein do case add karo: sha set hai to `+sha` aata hai, `unknown` hai to nahi aata.

Commit: `feat(telemetry): build Sentry release from git sha and release tag`

## Step 2: Dockerfile, tag bhi container tak pahunchao

**File:** `backend/Dockerfile`, `ARG GIT_SHA` ke neeche:

```dockerfile
ARG RELEASE_TAG=""
ENV RELEASE_TAG=$RELEASE_TAG
```

**File:** `.github/workflows/create-release.yml`, "Build and Push Docker Image" step mein:

```yaml
docker build \
  --build-arg GIT_SHA=${{ github.sha }} \
  --build-arg RELEASE_TAG=${{ github.ref_name }} \
  ...
```

**File:** `.github/workflows/deploy-staging-ecs.yml`, same step:

```yaml
docker build --build-arg GIT_SHA=${{ github.sha }} --build-arg RELEASE_TAG=main ...
```

Staging release `kaapi-backend@main+<sha>` banega. Har main push pe naya release, sahi hai.

## Step 3: Sentry ko batao ki release bana aur deploy hua

Ye step suspect commits aur deploy markers deta hai. Bina iske Sentry ko release string dikhegi par commits nahi.

### 3a. Sentry token banao

Sentry → Settings → Developer Settings → **Organization Tokens** (user token nahi, org token) → Create. Scopes: `project:releases`, `org:read`. Naam: `github-actions-releases`.

GitHub repo → Settings → Secrets and variables → Actions:
- Secret `SENTRY_AUTH_TOKEN` = upar wala token
- Variable `SENTRY_ORG` = `project-tech4dev`
- Variable `SENTRY_PROJECT` = `kaapi-production`
- Variable `SENTRY_STAGING_PROJECT` = `kaapi-staging`

Chat mein jo user token diya tha wo yahan mat use karo, aur usko rotate kar do.

### 3b. Repo link verify

Sentry → Settings → Integrations → GitHub → `ProjectTech4DevAI/kaapi-backend` already linked hai (active). Kuch nahi karna.

### 3c. Workflow step (PR mein already hai, yahan sirf samajhne ke liye)

Dono workflows mein checkout ke baad ek step release id banata hai, taaki code aur Sentry mein string exactly same ho (sha ke 12 chars):

```yaml
      - name: Resolve release id
        id: release
        run: echo "id=kaapi-backend@${GITHUB_REF_NAME}+${GITHUB_SHA::12}" >> "$GITHUB_OUTPUT"
```

Staging mein `${GITHUB_REF_NAME}` ki jagah `main` hard-coded hai, kyunki wo workflow `workflow_dispatch` se chalta hai.

Deploy steps ke **baad** (deploy fail hua to Sentry mein deploy marker nahi banna chahiye):

```yaml
      - name: Register release and deploy in Sentry
        uses: getsentry/action-release@v3
        env:
          SENTRY_AUTH_TOKEN: ${{ secrets.SENTRY_AUTH_TOKEN }}
          SENTRY_ORG: ${{ vars.SENTRY_ORG }}
          SENTRY_PROJECT: ${{ vars.SENTRY_PROJECT }}        # staging: vars.SENTRY_STAGING_PROJECT
        with:
          release: ${{ steps.release.outputs.id }}
          environment: production                           # staging: staging
          set_commits: auto
          ignore_missing: true
```

Checkout step pe `fetch-depth: 0` hai, warna `set_commits: auto` ko pichhle release tak ke commits nahi milte.

## Step 4: Sentry UI cleanup

1. Sentry → kaapi-production → Settings → Environments → `prod` → **Hide**. Sirf `production` rahe.
2. Settings → Inbound Filters → **Filter out known web crawlers** → ON. (Alag kaam, par yahin ho jaata hai.)

## Step 5: Verify, pehle deploy ke baad

1. Sentry → Releases → naya entry `kaapi-backend@v0.5.1+...` dikhe, `production` environment ke saath, commit count > 0.
2. Kisi bhi naye issue pe "First seen in release" wahi string ho.
3. API aur Celery dono se same release aa rahi hai:
   Discover → spans → `field: release, count()` → sirf ek release row active deploy ke liye.
4. Purani `0.5.0` release pe naye events aana band.

Agar step 3 mein API aur worker alag release dikhayein, to Celery task-definition mein `GIT_SHA`/`RELEASE_TAG` env override to nahi? Dono same image use karte hain, to nahi hona chahiye, par ECS task def mein explicit env check kar lo.

## Baad mein (optional)

- `SENTRY_RELEASE` env ab kisi ko set nahi karna chahiye. Agar ECS task def ya Secrets Manager mein set hai, hata do, warna wo override kar dega.
- Release health (crash-free sessions) backend ke liye nahi chahiye, `auto_session_tracking` off hi rehne do.
