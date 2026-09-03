import os

import pytest
from starlette.testclient import TestClient

from app.core.config import settings
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.document import WebCrawler


@pytest.fixture
def crawler(client: TestClient, user_api_key: TestAuthContext) -> WebCrawler:
    """Provides a WebCrawler instance for document API testing."""
    return WebCrawler(client, user_api_key=user_api_key)


@pytest.fixture(scope="class")
def aws_credentials() -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = settings.AWS_DEFAULT_REGION
