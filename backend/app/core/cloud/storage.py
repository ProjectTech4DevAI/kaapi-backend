import os
from sqlmodel import Session
from uuid import UUID
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

from app.crud import get_project_by_id
from app.core.config import settings
from app.utils import mask_string

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
                f"{{'bucket': '{mask_string(settings.AWS_S3_BUCKET)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(err) from err
        except ClientError as err:
            response = int(err.response["Error"]["Code"])
            if response != 404:
                logger.error(
                    f"[AmazonCloudStorageClient.create] Unexpected AWS error | "
                    f"{{'bucket': '{mask_string(settings.AWS_S3_BUCKET)}', 'error': '{str(err)}', 'code': {response}}}",
                    exc_info=True,
                )
                raise CloudStorageError(err) from err
            logger.warning(
                f"[AmazonCloudStorageClient.create] Bucket not found, creating | "
                f"{{'bucket': '{mask_string(settings.AWS_S3_BUCKET)}'}}"
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
                    f"{{'bucket': '{mask_string(settings.AWS_S3_BUCKET)}'}}"
                )
            except ClientError as create_err:
                logger.error(
                    f"[AmazonCloudStorageClient.create] Failed to create bucket | "
                    f"{{'bucket': '{mask_string(settings.AWS_S3_BUCKET)}', 'error': '{str(create_err)}'}}",
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
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(destination.Bucket)}', 'key': '{mask_string(destination.Key)}'}}"
            )
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.put] AWS upload error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(destination.Bucket)}', 'key': '{mask_string(destination.Key)}', 'error': '{str(err)}'}}",
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
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}'}}"
            )
            return body
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.stream] AWS stream error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}', 'error': '{str(err)}'}}",
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
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}', 'size_kb': {size_kb}}}"
            )
            return size_kb
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.get_file_size_kb] AWS head object error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}" ({url})') from err

    def get_signed_url(self, url: str, expires_in: int = 3600) -> str:
        """
        Generate a signed S3 URL for the given file.
        :param url: S3 url (e.g., s3://bucket/key)
        :param expires_in: Expiry time in seconds (default: 1 hour)
        :return: Signed URL as string
        """
        name = SimpleStorageName.from_url(url)
        try:
            signed_url = self.aws.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": name.Bucket, "Key": name.Key},
                ExpiresIn=expires_in,
            )
            logger.info(
                f"[AmazonCloudStorage.get_signed_url] Signed URL generated | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}'}}"
            )
            return signed_url
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.get_signed_url] AWS presign error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}', 'error': '{str(err)}'}}",
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
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}'}}"
            )
        except ClientError as err:
            logger.error(
                f"[AmazonCloudStorage.delete] AWS delete error | "
                f"{{'project_id': '{self.project_id}', 'bucket': '{mask_string(name.Bucket)}', 'key': '{mask_string(name.Key)}', 'error': '{str(err)}'}}",
                exc_info=True,
            )
            raise CloudStorageError(f'AWS Error: "{err}" ({url})') from err


def get_cloud_storage(session: Session, project_id: int) -> CloudStorage:
    """
    Method to create and configure a cloud storage instance.
    """
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
