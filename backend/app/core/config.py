import multiprocessing
import os
import secrets
import warnings
from typing import Any, Literal, Self

from pydantic import (
    EmailStr,
    HttpUrl,
    PostgresDsn,
    computed_field,
    model_validator,
)
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(origins: Any) -> list[str] | str:
    # If it's a plain comma-separated string, split it into a list
    if isinstance(origins, str) and not origins.startswith("["):
        return [origin.strip() for origin in origins.split(",")]
    # If it's already a list or JSON-style string, just return it
    elif isinstance(origins, (list, str)):
        return origins
    raise ValueError(f"Invalid CORS origins format: {origins!r}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # env_file will be set dynamically in get_settings()
        env_ignore_empty=True,
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"
    # v2 hosts
    API_V2_STR: str = "/api/v2"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 1 days = 1 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 1
    # 60 minutes * 24 hours * 7 days = 7 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    ENVIRONMENT: Literal[
        "development", "testing", "staging", "production"
    ] = "development"

    PROJECT_NAME: str
    API_VERSION: str = "0.5.0"
    SENTRY_DSN: HttpUrl | None = None
    POSTGRES_SERVER: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""
    KAAPI_GUARDRAILS_AUTH: str = ""
    KAAPI_GUARDRAILS_URL: str = ""

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""

    # Frontend URL for magic links
    FRONTEND_HOST: str = "http://localhost:3000"

    # Invitation token expiry (default 24 hours)
    INVITE_TOKEN_EXPIRE_HOURS: int = 24

    # Magic link login token expiry (default 15 minutes)
    MAGIC_LINK_TOKEN_EXPIRE_MINUTES: int = 15

    # SMTP / Email
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    EMAILS_FROM_EMAIL: str = ""
    EMAILS_FROM_NAME: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def emails_enabled(self) -> bool:
        return bool(self.SMTP_HOST and self.EMAILS_FROM_EMAIL)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> PostgresDsn:
        return MultiHostUrl.build(
            scheme="postgresql+psycopg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_SERVER,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48
    EMAIL_TEST_USER: EmailStr

    FIRST_SUPERUSER: EmailStr
    FIRST_SUPERUSER_PASSWORD: str

    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_DEFAULT_REGION: str = ""
    AWS_S3_BUCKET_PREFIX: str = ""
    # KMS key (ID, ARN, or alias) for credential encryption
    AWS_KMS_KEY_ID: str = ""

    # GCP Vertex AI platform defaults. Used when a project does not register
    # its own ``google`` credential row (BYOK is all-or-nothing — see the
    # Provider.GOOGLE comment in app/core/providers.py).
    GCP_VERTEX_API_KEY: str = ""
    GCP_VERTEX_LOCATION: str = ""
    GCP_PROJECT_ID: str = ""
    # Filesystem path to the platform-default GCP service-account JSON.
    # Used by the registry fallback when a project has no ``google`` row.
    GCP_SA_KEY: str = ""
    GCS_AUDIO_BUCKET: str = ""

    # RabbitMQ configuration for Celery broker
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    RABBITMQ_VHOST: str = "/"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def RABBITMQ_URL(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/{self.RABBITMQ_VHOST}"

    # Redis configuration for Celery result backend
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def AWS_S3_BUCKET(self) -> str:
        return f"{self.AWS_S3_BUCKET_PREFIX}-{self.ENVIRONMENT}"

    LOG_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    OTEL_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "kaapi-backend"
    BACKEND_SERVICE_NAME: str = "kaapi-backend"
    CRON_SERVICE_NAME: str = "kaapi-cron"

    # Threshold Request Rate per minute
    THRESHOLD_LLM_CALL_RATE: int = 15
    THRESHOLD_COLLECTIONS_RATE: int = 3
    THRESHOLD_EVALUATIONS_RATE: int = 3

    # Celery Configuration
    CELERY_WORKER_CONCURRENCY: int | None = None
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = 150
    CELERY_WORKER_MAX_MEMORY_PER_CHILD: int = 300000
    CELERY_TASK_SOFT_TIME_LIMIT: int = 300
    CELERY_TASK_TIME_LIMIT: int = 600
    CELERY_TASK_MAX_RETRIES: int = 3
    CELERY_TASK_DEFAULT_RETRY_DELAY: int = 60
    CELERY_RESULT_EXPIRES: int = 3600
    CELERY_BROKER_POOL_LIMIT: int = 10
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1
    CELERY_ENABLE_UTC: bool = True
    CELERY_TIMEZONE: str = "UTC"

    # callback timeouts and limits
    CALLBACK_CONNECT_TIMEOUT: int = 3
    CALLBACK_READ_TIMEOUT: int = 10

    # Evaluation cron invocation interval (minutes). In staging/production the
    # endpoint is triggered by AWS EventBridge on this cadence; locally it can
    # be driven by scripts/python/invoke-cron.py. The Sentry cron monitor reads
    # this same value so its expected schedule stays aligned with the trigger.
    CRON_INTERVAL_MINUTES: int = 5

    PENDING_JOB_MONITOR_INTERVAL_MINUTES: int = 5
    PENDING_RECENT_GRACE_MINUTES: int = 3
    LLM_PENDING_THRESHOLD_MINUTES: int = 30
    COLLECTION_PENDING_THRESHOLD_MINUTES: int = 30
    DOC_TRANSFORMATION_PENDING_THRESHOLD_MINUTES: int = 30
    # A fast run stuck in `processing` past this with no chunk progress is stalled;
    # the cron healer re-enqueues its missing chunk tasks.
    EVAL_FAST_STALL_THRESHOLD_MINUTES: int = 15
    PENDING_JOB_QUERY_TIMEOUT_MS: int = 1000

    # AI-assisted prompt improvement settings.
    # See docs/srd-ai-prompt-improvement.md for the full design rationale.
    # Platform-owned Anthropic key shared by every org/project for this feature,
    # so prompt improvement works without per-project credentials.
    ANTHROPIC_API_KEY: str = ""
    PROMPT_IMPROVEMENT_MODEL: str = "claude-opus-4-8"

    # Fast evaluation (run_mode="fast") configuration.
    # See "Fast Evaluation SRD.md" for the full design rationale.
    EVAL_FAST_MAX_UNIQUE_ROWS: int = 100
    EVAL_FAST_FAILURE_THRESHOLD: float = 0.5
    # Capped at 4 by default: higher values (8-10) across multiple Celery
    # workers can cause memory pressure on smaller EC2 instances.
    EVAL_FAST_API_CONCURRENCY: int = 4
    # Items per responses chunk task; smaller = more parallel workers and each
    # task well under CELERY_TASK_SOFT_TIME_LIMIT.
    EVAL_FAST_CHUNK_SIZE: int = 50

    EVAL_JUDGE_MODEL: str = "gpt-5-mini"

    # Reasoning effort for the judge model; "minimal" keeps per-row judging fast.
    # One of: none | minimal | low | medium | high | xhigh.
    EVAL_JUDGE_REASONING_EFFORT: str = "minimal"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def COMPUTED_CELERY_WORKER_CONCURRENCY(self) -> int:
        """Auto-calculate worker concurrency if not set explicitly."""
        if self.CELERY_WORKER_CONCURRENCY is not None:
            return self.CELERY_WORKER_CONCURRENCY
        # Use CPU cores * 2 as default
        return multiprocessing.cpu_count() * 2

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", '
                "for security, please change it, at least for deployments."
            )
            if self.ENVIRONMENT in ["development", "testing"]:
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_default_secret("SECRET_KEY", self.SECRET_KEY)
        self._check_default_secret("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD)
        self._check_default_secret(
            "FIRST_SUPERUSER_PASSWORD", self.FIRST_SUPERUSER_PASSWORD
        )

        return self


def get_settings() -> Settings:
    """Get settings with appropriate env file based on ENVIRONMENT."""
    environment = os.getenv("ENVIRONMENT", "development")

    # Determine env file
    env_files = {"testing": "../.env.test", "development": "../.env"}
    env_file = env_files.get(environment, "../.env")

    # Create Settings instance with the appropriate env file
    return Settings(_env_file=env_file)


# Export settings instance
settings = get_settings()
