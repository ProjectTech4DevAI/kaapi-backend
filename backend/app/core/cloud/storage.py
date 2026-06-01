import os
import json
import mimetypes
from sqlmodel import Session
from uuid import UUID, uuid4
import logging
import functools as ft
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import ParseResult, urlparse, urlunparse

from abc import ABC, abstractmethod
import boto3
from fastapi import UploadFile
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from google.cloud import storage as gcs
from google.oauth2 import service_account

from app.core.config import settings


def _mask(value: str | None) -> str:
    # Lazy to break a top-level cycle: app.utils transitively imports
    # app.services.llm.providers, which imports this module.
    from app.utils import mask_string

    return mask_string(value)


logger = logging.getLogger(__name__)


class CloudStorageError(Exception):
    pass


class AmazonCloudStorageClient:
    @ft.cached_property
    def client(self):
        kwargs = {}
        cred_params = (
            ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
            ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
            ("region_name", "AWS_DEFAULT_REGION"),
        )

        for i, j in cred_params:
            kwargs[i] = os.environ.get(j, getattr(settings, j))

        client = boto3.client("s3", **kwargs)
        return client

    def create(self):
        try:
            self.client.head_bucket(Bucket=settings.AWS_S3_BUCKET)
        except ValueError as err:
            logger.error(
                f"[AmazonCloudStorageClient.create] Invalid bucket configuration | "
                f"{{'bucket': '{_mask(settings.AWS_S3_BUCKET)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(err) from err
        except ClientError as err:
            response = int(err.response["Error"]["Code"])
            if response != 404:
                logger.error(
                    f"[AmazonCloudStorageClient.create] Unexpected AWS error | "
                    f"{{'bucket': '{_mask(settings.AWS_S3_BUCKET)}', 'error': '{str(err)}', 'code': {response}}}",
                    exc_info=True,
                )
                raise CloudStorageError(err) from err
            logger.warning(
                f"[AmazonCloudStorageClient.create] Bucket not found, creating | "
                f"{{'bucket': '{_mask(settings.AWS_S3_BUCKET)}'}}"
            )
            try:
                self.client.create_bucket(
                    Bucket=settings.AWS_S3_BUCKET,
                    CreateBucketConfiguration={
                        "LocationConstraint": settings.AWS_DEFAULT_REGION,
                    },
                )
                logger.info(
                    f"[AmazonCloudStorageClient.create] Bucket created successfully | "
                    f"{{'bucket': '{_mask(settings.AWS_S3_BUCKET)}'}}"
                )
            except ClientError as create_err:
                logger.error(
                    f"[AmazonCloudStorageClient.create] Failed to create bucket | "
                    f"{{'bucket': '{_mask(settings.AWS_S3_BUCKET)}', 'error': '{str(create_err)}'}}",
                    exc_info=True,
                )
                raise CloudStorageError(create_err) from create_err


@dataclass(frozen=True)
class SimpleStorageName:
    Key: str
    Bucket: str = settings.AWS_S3_BUCKET

    def __str__(self):
        return urlunparse(self.to_url())

    def to_url(self):
        kwargs = {
            "scheme": "s3",
            "netloc": self.Bucket,
            "path": self.Key,
        }
        for k in ParseResult._fields:
            kwargs.setdefault(k)
        return ParseResult(**kwargs)

    @classmethod
    def from_url(cls, url: str):
        url = urlparse(url)
        path = Path(url.path)
        if path.is_absolute():
            path = path.relative_to(path.root)
        return cls(Bucket=url.netloc, Key=str(path))


class CloudStorage(ABC):
    def __init__(self, project_id: int, storage_path: UUID):
        self.project_id = project_id
        self.storage_path = str(storage_path)

    @abstractmethod
    def put(self, source: UploadFile, filepath: Path) -> SimpleStorageName:
        """Upload a file to storage"""
        pass

    @abstractmethod
    def stream(self, url: str) -> StreamingBody:
        """Stream a file from storage"""
        pass

    @abstractmethod
    def get(self, url: str) -> bytes:
        """Get file contents as bytes (for files that fit in memory)"""
        pass

    @abstractmethod
    def get_file_size_kb(self, url: str) -> float:
        """Return the file size in KB"""
        pass

    @abstractmethod
    def get_signed_url(self, url: str, expires_in: int = 3600) -> str:
        """Generate a signed URL with an optional expiry"""
        pass

    @abstractmethod
    def delete(self, url: str) -> None:
        """Delete a file from storage"""
        pass


class AmazonCloudStorage(CloudStorage):
    def __init__(self, project_id: int, storage_path: UUID):
        super().__init__(project_id, storage_path)
        self.aws = AmazonCloudStorageClient()

    def put(self, source: UploadFile, file_path: Path) -> SimpleStorageName:
        if file_path.is_absolute():
            raise ValueError("file_path must be relative to the project's storage root")
        key = Path(self.storage_path) / file_path
        destination = SimpleStorageName(key.as_posix())
        kwargs = asdict(destination)

        try:
            self.aws.client.upload_fileobj(
                source.file,
                ExtraArgs={
                    "ContentType": source.content_type,
                },
                **kwargs,
            )
            logger.info(
                f"[AmazonCloudStorage.put] File uploaded successfully | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(destination.Bucket)}', 'key': '{_mask(destination.Key)}'}}"
            )
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.put] AWS upload error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(destination.Bucket)}', 'key': '{_mask(destination.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}"') from err

        return destination

    def stream(self, url: str) -> StreamingBody:
        name = SimpleStorageName.from_url(url)
        kwargs = asdict(name)
        try:
            body = self.aws.client.get_object(**kwargs).get("Body")
            logger.info(
                f"[AmazonCloudStorage.stream] File streamed successfully | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}'}}"
            )
            return body
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.stream] AWS stream error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}" ({url})') from err

    def get(self, url: str) -> bytes:
        name = SimpleStorageName.from_url(url)
        kwargs = asdict(name)
        try:
            body = self.aws.client.get_object(**kwargs).get("Body")
            content = body.read()
            logger.info(
                f"[AmazonCloudStorage.get] File retrieved successfully | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'size_bytes': {len(content)}}}"
            )
            return content
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.get] AWS get error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}" ({url})') from err

    def get_file_size_kb(self, url: str) -> float:
        name = SimpleStorageName.from_url(url)
        kwargs = asdict(name)
        try:
            response = self.aws.client.head_object(**kwargs)
            size_bytes = response["ContentLength"]
            size_kb = round(size_bytes / 1024, 2)
            logger.info(
                f"[AmazonCloudStorage.get_file_size_kb] File size retrieved successfully | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'size_kb': {size_kb}}}"
            )
            return size_kb
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.get_file_size_kb] AWS head object error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}" ({url})') from err

    # Maximum allowed expiry for signed URLs (24 hours)
    MAX_SIGNED_URL_EXPIRY = 86400

    def get_signed_url(self, url: str, expires_in: int = 3600) -> str:
        """
        Generate a signed S3 URL for the given file.
        :param url: S3 url (e.g., s3://bucket/key)
        :param expires_in: Expiry time in seconds (default: 1 hour, max: 24 hours)
        :return: Signed URL as string
        """
        # Cap expiry at maximum allowed value to prevent excessively long-lived URLs
        expires_in = min(expires_in, self.MAX_SIGNED_URL_EXPIRY)

        name = SimpleStorageName.from_url(url)
        try:
            signed_url = self.aws.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": name.Bucket, "Key": name.Key},
                ExpiresIn=expires_in,
            )
            logger.info(
                f"[AmazonCloudStorage.get_signed_url] Signed URL generated | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}'}}"
            )
            return signed_url
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.get_signed_url] AWS presign error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}" ({url})') from err

    def delete(self, url: str) -> None:
        name = SimpleStorageName.from_url(url)
        kwargs = asdict(name)
        try:
            self.aws.client.delete_object(**kwargs)
            logger.info(
                f"[AmazonCloudStorage.delete] File deleted successfully | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}'}}"
            )
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.delete] AWS delete error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}" ({url})') from err


def get_cloud_storage(session: Session, project_id: int) -> CloudStorage:
    """
    Method to create and configure a cloud storage instance.
    """
    # Lazy import to avoid a top-level cycle: storage.py is imported from
    # app.services.llm.providers.gai_vertex, which itself is wired into the
    # provider registry that app.crud transitively pulls in.
    from app.crud import get_project_by_id

    project = get_project_by_id(session=session, project_id=project_id)
    if not project:
        raise ValueError(f"Invalid project_id: {project_id}")

    storage_path = project.storage_path

    try:
        return AmazonCloudStorage(project_id=project_id, storage_path=storage_path)
    except Exception as err:
        logger.error(
            f"[get_cloud_storage] Failed to initialize storage for project_id={project_id}: {err}",
            exc_info=True,
        )
        raise


# ──────────────────────────────────────────────────────────────────────────────
# GCP service-account fetch (AWS Secrets Manager) + GCS upload util.
# BYOK-ready: every util takes explicit secret_name / bucket / project_id so
# per-project credentials can be passed in. Settings provide the platform
# defaults for the shared SA path.
# ──────────────────────────────────────────────────────────────────────────────

GCS_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


class SecretsManagerError(Exception):
    pass


def upsert_byok_secret_for_provider(
    provider: str,
    credentials: dict,
    *,
    org_id: int,
    project_id: int,
) -> dict:
    """Persist provider-specific BYOK secrets to AWS Secrets Manager and
    rewrite the credentials dict so only references (not raw secrets) are
    stored in the DB.

    Currently only ``google-vertex`` needs this: when ``sa_key`` is present,
    the SA JSON is uploaded to SM under a deterministic per-project name,
    and the dict is rewritten to carry ``gcp_sa_secret_name`` /
    ``gcp_sa_secret_region`` instead.

    Returns the (possibly rewritten) credentials dict. No-op for providers
    without BYOK secrets or when the optional ``sa_key`` field is absent.
    """
    if provider == "google-vertex":
        sa_key = credentials.get("sa_key")
        # The validator only checks key presence, not shape/truthiness — so
        # null, empty dict, or a JSON string would slip through and leave a
        # partial-BYOK row (user api_key + platform SA), which is exactly
        # the broken hybrid BYOK enforcement is meant to prevent.
        if not isinstance(sa_key, dict) or not sa_key:
            raise ValueError(
                "google-vertex 'sa_key' must be a non-empty service-account JSON object"
            )
        secret_name = (
            f"kaapi/{settings.ENVIRONMENT}/orgs/{org_id}"
            f"/projects/{project_id}/google-vertex/sa"
        )
        put_gcp_service_account(sa_key, secret_name=secret_name)
        rewritten = {k: v for k, v in credentials.items() if k != "sa_key"}
        rewritten["gcp_sa_secret_name"] = secret_name
        rewritten["gcp_sa_secret_region"] = settings.GCP_SA_SECRET_REGION
        return rewritten
    return credentials


def put_gcp_service_account(
    sa_info: dict,
    *,
    secret_name: str,
    region_name: str | None = None,
) -> None:
    """Create or update a GCP service-account JSON key in AWS Secrets Manager.

    Idempotent: tries CreateSecret first, falls back to PutSecretValue when
    the secret already exists. Validates SA shape upfront so we never store
    junk. Invalidates the ``get_gcp_service_account`` LRU cache on success
    so the next read picks up the rotated key.
    """
    if sa_info.get("type") != "service_account":
        raise SecretsManagerError(
            f"Refusing to write secret '{secret_name}': not a GCP service-account key "
            f"(got type={sa_info.get('type')!r})"
        )

    region = region_name or settings.GCP_SA_SECRET_REGION
    payload = json.dumps(sa_info)

    sm_client = boto3.session.Session().client(
        service_name="secretsmanager", region_name=region
    )

    try:
        try:
            sm_client.create_secret(Name=secret_name, SecretString=payload)
            action = "created"
        except sm_client.exceptions.ResourceExistsException:
            sm_client.put_secret_value(SecretId=secret_name, SecretString=payload)
            action = "updated"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(
            f"[put_gcp_service_account] Secret write failed | "
            f"secret={_mask(secret_name)}, region={region}, code={code}"
        )
        raise SecretsManagerError(
            f"Failed to write secret '{secret_name}' (code={code}): {e}"
        ) from e

    get_gcp_service_account.cache_clear()
    logger.info(
        f"[put_gcp_service_account] Secret {action} | "
        f"secret={_mask(secret_name)}, region={region}, "
        f"project_id={sa_info.get('project_id')}, "
        f"client_email={_mask(sa_info.get('client_email', ''))}"
    )


@ft.lru_cache(maxsize=32)
def get_gcp_service_account(
    secret_name: str | None = None,
    region_name: str | None = None,
) -> dict:
    """Fetch a GCP service-account JSON key from AWS Secrets Manager.

    Cached per (secret_name, region) — restart the process or call
    ``get_gcp_service_account.cache_clear()`` to pick up a rotated key.

    BYOK: pass a project-owned ``secret_name``. Defaults to the platform-shared
    secret configured in settings.
    """
    secret = secret_name or settings.GCP_SA_SECRET_NAME
    region = region_name or settings.GCP_SA_SECRET_REGION

    sm_client = boto3.session.Session().client(
        service_name="secretsmanager", region_name=region
    )

    try:
        response = sm_client.get_secret_value(SecretId=secret)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(
            f"[get_gcp_service_account] Secret fetch failed | "
            f"secret={_mask(secret)}, region={region}, code={code}"
        )
        raise SecretsManagerError(
            f"Failed to fetch secret '{secret}' (code={code}): {e}"
        ) from e

    if "SecretString" not in response:
        raise SecretsManagerError(
            f"Secret '{secret}' has no SecretString (binary secret unsupported)"
        )

    try:
        sa_info = json.loads(response["SecretString"])
    except json.JSONDecodeError as e:
        raise SecretsManagerError(f"Secret '{secret}' is not valid JSON: {e}") from e

    if sa_info.get("type") != "service_account":
        raise SecretsManagerError(
            f"Secret '{secret}' is not a GCP service-account key "
            f"(got type={sa_info.get('type')!r})"
        )

    logger.info(
        f"[get_gcp_service_account] Loaded SA key | "
        f"secret={_mask(secret)}, project_id={sa_info.get('project_id')}, "
        f"client_email={_mask(sa_info.get('client_email', ''))}"
    )
    return sa_info


_MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/webm": ".webm",
    "audio/aac": ".aac",
    "audio/aiff": ".aiff",
}


def upload_audio_to_gcs(
    *,
    bucket_name: str,
    sa_info: dict,
    audio_bytes: bytes | None = None,
    local_path: str | None = None,
    content_type: str | None = None,
    project_id: str | None = None,
    key_prefix: str = "audio",
) -> str:
    """Upload audio to GCS and return its ``gs://bucket/key`` URI.

    Pass exactly one of ``audio_bytes`` or ``local_path``.

    BYOK: caller supplies ``sa_info`` and ``bucket_name``. The returned URI
    plugs directly into Vertex ``fileData.fileUri``.
    """
    if (audio_bytes is None) == (local_path is None):
        raise ValueError("Pass exactly one of audio_bytes or local_path")

    if local_path is not None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Audio file not found: {local_path}")
        size = os.path.getsize(local_path)
        ext = Path(local_path).suffix or ""
        mime = content_type or mimetypes.guess_type(local_path)[0] or "audio/wav"
    else:
        size = len(audio_bytes)
        mime = content_type or "audio/wav"
        ext = _MIME_TO_EXT.get(mime, "")

    key = f"{key_prefix}/{uuid4().hex}{ext}"

    try:
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=list(GCS_SCOPES)
        )
        client = gcs.Client(
            project=project_id or sa_info.get("project_id"), credentials=creds
        )
        blob = client.bucket(bucket_name).blob(key)
        if local_path is not None:
            blob.upload_from_filename(local_path, content_type=mime)
        else:
            blob.upload_from_string(audio_bytes, content_type=mime)
    except Exception as e:
        logger.error(
            f"[upload_audio_to_gcs] Upload failed | "
            f"bucket={bucket_name}, key={key}, error={e}",
            exc_info=True,
        )
        raise CloudStorageError(f"GCS upload failed: {e}") from e

    uri = f"gs://{bucket_name}/{key}"
    logger.info(
        f"[upload_audio_to_gcs] Uploaded | "
        f"uri={uri}, mime={mime}, size_kb={size / 1024:.1f}"
    )
    return uri
