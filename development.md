# Kaapi - Development

## Docker Compose

* Start the local stack with Docker Compose:

```bash
docker compose watch
```

* Now you can open your browser and interact with these URLs:

Backend, JSON-based web API based on OpenAPI: <http://localhost:8000>

Automatic interactive documentation with Swagger UI (from the OpenAPI backend): <http://localhost:8000/docs>

Alternative interactive documentation (ReDoc): <http://localhost:8000/redoc>

Adminer, database web administration: <http://localhost:8080>

RabbitMQ Management UI: <http://localhost:15672>

Celery Flower (task monitoring): <http://localhost:5555>

**Note**: The first time you start your stack, it might take a minute for it to be ready. While the backend waits for the database to be ready and configures everything. You can check the logs to monitor it.

To check the logs, run (in another terminal):

```bash
docker compose logs
```

To check the logs of a specific service, add the name of the service, e.g.:

```bash
docker compose logs backend
```

## Local Development

The Docker Compose files are configured so that each of the services is available in a different port in `localhost`.

You can stop the `backend` Docker Compose service and run the local development server instead:

```bash
docker compose stop backend
```

And then start the local development server for the backend:

```bash
cd backend
fastapi run --reload app/main.py
```

This way the backend runs on the same port (<http://localhost:8000>) whether it's in Docker or running locally.

### Running Completely Without Docker

If you want to run everything locally without Docker, you'll need to set up RabbitMQ, Redis, Celery, and optionally Celery Flower manually.

#### 1. Install and Start RabbitMQ & Redis

**macOS (using Homebrew):**

```bash
brew install rabbitmq redis

brew services start rabbitmq
brew services start redis
```

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install rabbitmq-server redis-server

sudo systemctl enable --now rabbitmq-server
sudo systemctl enable --now redis-server
```

**Verify services are running:**

- RabbitMQ Management UI: <http://localhost:15672> (default credentials: `guest`/`guest`)
- Redis: `redis-cli ping` (should return `PONG`)

#### 2. Start the Backend Server

In your first terminal:

```bash
cd backend
fastapi run --reload app/main.py
```

The backend will be available at <http://localhost:8000>

#### 3. Start Celery Worker

In a second terminal, start the Celery worker:

```bash
cd backend
uv run celery -A app.celery.celery_app worker --loglevel=info
```

Leave this process running. This handles background tasks like document processing and LLM job execution.

#### 4. (Optional) Start Celery Flower for Task Monitoring

Flower provides a web UI to monitor Celery tasks and workers.

**Start Flower in a fourth terminal:**

```bash
cd backend
uv run celery -A app.celery.celery_app flower --port=5555
```

Flower will be available at: <http://localhost:5555>

> **Note:** If you start Flower before running any Celery workers, you may see warnings like:
> ```text
> WARNING - flower.inspector - Inspect method ... failed
> ```
> This just means there are no active workers yet. Once you start a Celery worker (step 3),
> Flower will be able to inspect it and the warnings will stop.

---

## Docker Compose files and env vars

The `docker-compose.yml` file contains all the configurations for the stack, including services like PostgreSQL, Redis, RabbitMQ, backend, Celery workers, and Adminer.

The Docker Compose file uses the `.env` file containing configurations to be injected as environment variables in the containers.

After changing environment variables, make sure you restart the stack:

```bash
docker compose watch
```

## The .env file

The `.env` file is the one that contains all your configurations, generated keys and passwords, etc.

Depending on your workflow, you could want to exclude it from Git, for example if your project is public. In that case, you would have to make sure to set up a way for your CI tools to obtain it while building or deploying your project.

One way to do it could be to add each environment variable to your CI/CD system, and updating the `docker-compose.yml` file to read that specific env var instead of reading the `.env` file.

## Pre-commits and code linting

we are using a tool called [pre-commit](https://pre-commit.com/) for code linting and formatting.

When you install it, it runs right before making a commit in git. This way it ensures that the code is consistent and formatted even before it is committed.

You can find a file `.pre-commit-config.yaml` with configurations at the root of the project.

#### Install pre-commit to run automatically

`pre-commit` is already part of the dependencies of the project, but you could also install it globally if you prefer to, following [the official pre-commit docs](https://pre-commit.com/).

After having the `pre-commit` tool installed and available, you need to "install" it in the local repository, so that it runs automatically before each commit.

Using `uv`, you could do it with:

```bash
❯ uv run pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

Now whenever you try to commit, e.g. with:

```bash
git commit
```

...pre-commit will run and check and format the code you are about to commit, and will ask you to add that code (stage it) with git again before committing.

Then you can `git add` the modified/fixed files again and now you can commit.

#### Running pre-commit hooks manually

you can also run `pre-commit` manually on all the files, you can do it using `uv` with:

```bash
❯ uv run pre-commit run --all-files
check for added large files..............................................Passed
check toml...............................................................Passed
check yaml...............................................................Passed
ruff.....................................................................Passed
ruff-format..............................................................Passed
eslint...................................................................Passed
prettier.................................................................Passed
```

## Development URLs

All services are available on localhost with different ports:

**Backend**: <http://localhost:8000>

**Swagger UI** (Interactive API Docs): <http://localhost:8000/docs>

**ReDoc** (Alternative API Docs): <http://localhost:8000/redoc>

**Adminer** (Database Management): <http://localhost:8080>

**RabbitMQ Management**: <http://localhost:15672> (username: guest, password: guest)

**Celery Flower** (Task Monitoring): <http://localhost:5555>
