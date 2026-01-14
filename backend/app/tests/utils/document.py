import itertools as it
import functools as ft
from typing import Any, Generator
from uuid import UUID
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from urllib.parse import ParseResult, urlunparse

import pytest
from httpx import Response
from sqlmodel import Session, delete
from fastapi.testclient import TestClient

from app.core.config import settings
from app.crud.project import get_project_by_id
from app.models import Document, DocumentPublic, Project
from app.utils import APIResponse
from app.tests.utils.auth import TestAuthContext

from .utils import SequentialUuidGenerator


def httpx_to_standard(response: Response):
    return APIResponse(**response.json())


class DocumentMaker:
    def __init__(self, project_id: int, session: Session):
        self.project_id = project_id
        self.session = session
        self.project: Project = get_project_by_id(
            session=self.session, project_id=self.project_id
        )
        self.index = SequentialUuidGenerator()

    def __iter__(self):
        return self

    def __next__(self):
        doc_id = next(self.index)
        key = f"{self.project.storage_path}/{doc_id}.txt"
        object_store_url = f"s3://{settings.AWS_S3_BUCKET}/{key}"

        return Document(
            id=doc_id,
            project_id=self.project.id,
            fname=f"{doc_id}.xyz",
            object_store_url=object_store_url,
            is_deleted=False,
        )


class DocumentStore:
    def __init__(self, db: Session, project_id: int) -> None:
        self.db = db
        self.documents = DocumentMaker(project_id=project_id, session=db)
        self.clear(self.db)

    @staticmethod
    def clear(db: Session) -> None:
        db.exec(delete(Document))
        db.commit()

    @property
    def project(self) -> Project:
        return self.documents.project

    def put(self) -> Document:
        doc = next(self.documents)
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def extend(self, n: int) -> Generator[Document, None, None]:
        for _ in range(n):
            yield self.put()

    def fill(self, n: int) -> list[Document]:
        return list(self.extend(n))


class Route:
    _empty = ParseResult(*it.repeat("", len(ParseResult._fields)))
    _root = Path(settings.API_V1_STR, "documents")

    def __init__(self, endpoint: str | Path, **qs_args: Any) -> None:
        self.endpoint = endpoint
        self.qs_args = qs_args

    def __str__(self) -> str:
        return urlunparse(self.to_url())

    def to_url(self) -> ParseResult:
        path = self._root.joinpath(self.endpoint)
        kwargs = {
            "path": str(path),
        }
        if self.qs_args:
            query = "&".join(it.starmap("{}={}".format, self.qs_args.items()))
            kwargs["query"] = query

        return self._empty._replace(**kwargs)

    def append(self, doc: Document, suffix: str | None = None) -> "Route":
        segments = [self.endpoint, str(doc.id)]
        if suffix:
            segments.append(suffix)
        endpoint = Path(*segments)
        return type(self)(endpoint, **self.qs_args)


@dataclass
class WebCrawler:
    client: TestClient
    user_api_key: TestAuthContext

    def get(self, route: Route) -> Response:
        return self.client.get(
            str(route),
            headers={"X-API-KEY": self.user_api_key.key},
        )

    def delete(self, route: Route) -> Response:
        return self.client.delete(
            str(route),
            headers={"X-API-KEY": self.user_api_key.key},
        )


class DocumentComparator:
    """Compare a Document model against the DocumentPublic API response."""

    @ft.singledispatchmethod
    @staticmethod
    def to_string(value: Any) -> Any:
        return value

    @to_string.register
    @staticmethod
    def _(value: UUID) -> str:
        return str(value)

    @to_string.register
    @staticmethod
    def _(value: datetime) -> str:
        return value.isoformat()

    def __init__(self, document: Document) -> None:
        self.document = document

    def __eq__(self, other: dict) -> bool:
        this = dict(self.to_public_dict())
        return this == other

    def to_public_dict(self) -> dict:
        """Convert Document to dict matching DocumentPublic schema."""
        field_names = DocumentPublic.model_fields.keys()

        result = {}
        for field in field_names:
            value = getattr(self.document, field, None)
            result[field] = self.to_string(value)

        return result


@pytest.fixture
def crawler(client: TestClient, user_api_key: TestAuthContext) -> WebCrawler:
    return WebCrawler(client, user_api_key=user_api_key)
