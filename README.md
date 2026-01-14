# Kaapi

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![](https://github.com/ProjectTech4DevAI/kaapi-backend/workflows/Continuous%20Integration/badge.svg)
[![Code coverage badge](https://img.shields.io/codecov/c/github/ProjectTech4DevAI/kaapi-backend/staging.svg)](https://codecov.io/gh/ProjectTech4DevAI/kaapi-backend/branch/staging)
![GitHub issues](https://img.shields.io/github/issues-raw/ProjectTech4DevAI/kaapi-backend)
[![codebeat badge](https://codebeat.co/badges/dd951390-5f51-4c98-bddc-0b618bdb43fd)](https://codebeat.co/projects/github-com-ProjectTech4DevAI/kaapi-backend-staging)
[![Commits](https://img.shields.io/github/commit-activity/m/ProjectTech4DevAI/kaapi-backend)](https://img.shields.io/github/commit-activity/m/ProjectTech4DevAI/kaapi-backend)

## Pre-requisites

- [docker](https://docs.docker.com/get-started/get-docker/) Docker
- [uv](https://docs.astral.sh/uv/) for Python package and environment management.
- **Poppler** – Install Poppler, required for PDF processing.

## Project Setup

You can **just fork or clone** this repository and use it as is.

✨ It just works. ✨

### Configure

Create env file using example file

```bash
cp .env.example .env
```

You can then update configs in the `.env` files to customize your configurations.

⚠️ Some services depend on these environment variables being set correctly. Missing or invalid values may cause startup issues.

### Generate Secret Keys


You have to change them with a secret key, to generate secret keys you can run the following command:

```bash

python -c "import secrets; print(secrets.token_urlsafe(32))"

````

Copy the content and use that as password / secret key. And run that again to generate another secure key.

## Bootstrap & development mode

You have two options to start this dockerized setup, depending on whether you want to reset the database:
### Option A: Run migrations & seed data (will reset DB)

Use the prestart profile to automatically run database migrations and seed data.
This profile also resets the database, so use it only when you want a fresh start.
```bash
docker compose --profile prestart up
```

### Option B: Start normally without resetting DB

If you don't want to reset the database, start the project directly:
```bash
docker compose watch
```
This will start all services in watch mode for development — ideal for local iterations.

### Rebuilding Images

While the backend service supports live code reloading via `docker compose watch`, **Celery does not support auto-reload**. When you make changes to Celery tasks, workers, or related code, you need to rebuild the Docker image:

```bash
docker compose up --build
```

This is also necessary when:
- Dependencies change in `pyproject.toml` or `uv.lock`
- You modify Dockerfile configurations
- Changes aren't being reflected in the running containers

## Backend Development

Backend docs: [backend/README.md](./backend/README.md).

## Deployment

Deployment docs: [deployment.md](./deployment.md).

## Development

General development docs: [development.md](./development.md).

This includes using Docker Compose, custom local domains, `.env` configurations, etc.

## Release Notes

Check the file [release-notes.md](./release-notes.md).

## Credits

This project was created using [full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template). A big thank you to the team for creating and maintaining the template!!!
