import os
import mimetypes
import filetype
from sqlmodel import Session
from uuid import UUID, uuid4
import logging
import functools as ft
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import ParseResult, urlparse, urlunparse

from abc import ABC, abstractmethod
from typing import Any, NamedTuple
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


class ObjectNotFoundError(CloudStorageError):
    pass


MISSING_OBJECT_CODES = ("404", "NoSuchKey")


def _to_storage_error(err: ClientError, url: str) -> CloudStorageError:
    """Map a botocore ClientError onto the storage exception hierarchy."""
    message = f'AWS Error: "{err}" ({url})'
    code = err.response.get("Error", {}).get("Code")
    if code in MISSING_OBJECT_CODES:
        return ObjectNotFoundError(message)
    return CloudStorageError(message)


class SignedUpload(NamedTuple):
    url: str
    # Effective expiry after capping, which may be shorter than what the caller asked for.
    expires_in: int


class AmazonCloudStorageClient:
    @ft.cached_property
    def client(self):
        kwargs = {
            "region_name": os.environ.get(
                "AWS_DEFAULT_REGION", settings.AWS_DEFAULT_REGION
            )
        }
        if settings.ENVIRONMENT == "development":
            kwargs["aws_access_key_id"] = os.environ.get(
                "AWS_ACCESS_KEY_ID", settings.AWS_ACCESS_KEY_ID
            )
            kwargs["aws_secret_access_key"] = os.environ.get(
                "AWS_SECRET_ACCESS_KEY", settings.AWS_SECRET_ACCESS_KEY
            )

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


# Leads the key rather than following storage_path: S3 lifecycle filters are literal
# prefixes with no wildcard, so a per-project segment in front makes abandoned uploads
# unreapable by a single rule.
# WARNING: a live S3 lifecycle rule deletes everything under this prefix after 1 day
# (staging + production buckets). Never write anything here that must outlive a day.
STAGING_PREFIX = "pending"


class CloudStorage(ABC):
    def __init__(self, project_id: int, storage_path: UUID):
        self.project_id = project_id
        self.storage_path = str(storage_path)

    def url_for(self, file_path: Path) -> SimpleStorageName:
        """Resolve a project-relative path into a fully qualified storage name.

        The single place storage_path is joined — callers never build keys themselves.
        """
        if file_path.is_absolute():
            raise ValueError("file_path must be relative to the project's storage root")
        key = Path(self.storage_path) / file_path
        return SimpleStorageName(key.as_posix())

    def staging_url_for(self, file_path: Path) -> SimpleStorageName:
        """Resolve a pre-registration staging key, under the bucket-wide staging prefix."""
        name = self.url_for(file_path)
        return SimpleStorageName(f"{STAGING_PREFIX}/{name.Key}", name.Bucket)

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
    def get_signed_upload_url(
        self, file_path: Path, expires_in: int = 3600
    ) -> SignedUpload:
        """Generate a signed URL the client can upload (PUT) to directly.

        Always resolves through the staging prefix: nothing is ever presigned to a
        final key, or an abandoned upload would be indistinguishable from a document.
        """
        pass

    @abstractmethod
    def copy(self, source_url: str, destination: Path) -> SimpleStorageName:
        """Server-side copy of an existing object to a project-relative path"""
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
        destination = self.url_for(file_path)
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
            raise _to_storage_error(err, url) from err

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
            raise _to_storage_error(err, url) from err

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
            raise _to_storage_error(err, url) from err

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

    def get_signed_upload_url(
        self, file_path: Path, expires_in: int = 3600
    ) -> SignedUpload:
        """
        Generate a signed S3 URL the client can PUT raw bytes to, under the staging prefix.
        No content type is signed, so the client sends no headers beyond the body.
        """
        expires_in = min(expires_in, self.MAX_SIGNED_URL_EXPIRY)

        name = self.staging_url_for(file_path)
        try:
            signed_url = self.aws.client.generate_presigned_url(
                "put_object",
                Params=asdict(name),
                ExpiresIn=expires_in,
            )
            return SignedUpload(url=signed_url, expires_in=expires_in)
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.get_signed_upload_url] AWS presign error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(name.Bucket)}', 'key': '{_mask(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise _to_storage_error(err, str(name)) from err

    def copy(self, source_url: str, destination: Path) -> SimpleStorageName:
        source = SimpleStorageName.from_url(source_url)
        target = self.url_for(destination)
        try:
            self.aws.client.copy_object(
                Bucket=target.Bucket,
                Key=target.Key,
                CopySource={"Bucket": source.Bucket, "Key": source.Key},
            )
            return target
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.copy] AWS copy error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{_mask(target.Bucket)}', 'source_key': '{_mask(source.Key)}', 'key': '{_mask(target.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise _to_storage_error(err, source_url) from err

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
            raise _to_storage_error(err, url) from err


def get_cloud_storage(session: Session, project_id: int) -> CloudStorage:
    """
    Method to create and configure a cloud storage instance.
    """
    # Lazy import to avoid a top-level cycle: storage.py is imported from
    # app.services.llm.providers.google_gcp, which itself is wired into the
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


GCS_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def build_gcp_sa_credentials(sa_key: dict[str, Any]) -> service_account.Credentials:
    """Build signing-capable SA credentials from a service-account key dict."""
    return service_account.Credentials.from_service_account_info(
        sa_key, scopes=list(GCS_SCOPES)
    )


MAX_AUDIO_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

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
        if not audio_bytes:
            raise ValueError("audio_bytes is empty")
        size = len(audio_bytes)
        mime = content_type or "audio/wav"
        ext = _MIME_TO_EXT.get(mime, "")

    if mime not in _MIME_TO_EXT:
        raise ValueError(
            f"Unsupported content_type '{mime}'. Allowed: "
            f"{', '.join(sorted(_MIME_TO_EXT))}"
        )

    # Sniff the actual bytes — content_type is caller-supplied and spoofable.
    sniff_source = audio_bytes if audio_bytes is not None else local_path
    detected = filetype.guess(sniff_source)
    if detected is None or not detected.mime.startswith("audio/"):
        raise ValueError(
            f"Uploaded content is not a recognised audio file "
            f"(detected={detected.mime if detected else 'unknown'})"
        )

    if size > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError(
            f"Audio exceeds {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB limit "
            f"(got {size / (1024 * 1024):.1f} MB)"
        )

    key = f"{key_prefix}/{uuid4().hex}{ext}"

    try:
        creds = build_gcp_sa_credentials(sa_info)
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
