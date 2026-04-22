import boto3
import json
import logging
import secrets
import warnings
import os
import multiprocessing
from typing import Any, Literal

from pydantic import (
    EmailStr,
    HttpUrl,
    PostgresDsn,
    computed_field,
    model_validator,
)
from pydantic_core import MultiHostUrl
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)


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
    POSTGRES_SERVER: str = ""
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = ""
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

    # AWS Secrets Manager
    USE_AWS_SECRETS: bool = False
    AWS_SECRETS_REGION: str = ""
    AWS_POSTGRES_SECRET_NAME: str = ""

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
    def _apply_aws_secrets(self) -> Self:
        """Overlay service credentials from AWS Secrets Manager when enabled.

        Each AWS_*_SECRET_NAME points to a secret whose SecretString is a
        JSON object (e.g. the format produced by RDS/ElastiCache rotation).
        Each secret is processed independently with its own field map so
        services that share key names (username/password/host/port) do not
        overwrite each other. Keys absent from a secret leave the existing
        .env-derived value untouched.
        """
        if not self.USE_AWS_SECRETS:
            return self

        secret_configs: list[tuple[str, dict[str, str]]] = [
            (
                self.AWS_POSTGRES_SECRET_NAME,
                {
                    "username": "POSTGRES_USER",
                    "password": "POSTGRES_PASSWORD",
                    "host": "POSTGRES_SERVER",
                    "port": "POSTGRES_PORT",
                },
            ),
        ]

        active = [(name, field_map) for name, field_map in secret_configs if name]
        if not active:
            return self

        region = self.AWS_SECRETS_REGION or self.AWS_DEFAULT_REGION
        client_kwargs: dict[str, Any] = {"region_name": region} if region else {}
        client = boto3.client("secretsmanager", **client_kwargs)

        for secret_name, field_map in active:
            self._load_secret(client, secret_name, field_map)

        return self

    def _load_secret(
        self, client: Any, secret_name: str, field_map: dict[str, str]
    ) -> None:
        data = json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])
        if not isinstance(data, dict):
            raise ValueError(
                f"Secret '{secret_name}' must be a JSON object "
                f"(got {type(data).__name__})"
            )
        applied: list[str] = []
        for key, field in field_map.items():
            if key in data:
                setattr(self, field, data[key])
                applied.append(field)
        logger.info(
            f"[_load_secret] Loaded AWS secret | "
            f"{{'name': '{secret_name}', 'fields': {sorted(applied)}}}"
        )

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

    return Settings(_env_file=env_file)


# Export settings instance
settings = get_settings()
