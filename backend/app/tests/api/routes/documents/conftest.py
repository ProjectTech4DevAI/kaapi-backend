import pytest
from starlette.testclient import TestClient

from app.tests.utils.auth import TestAuthContext
from app.tests.utils.document import WebCrawler


@pytest.fixture
def crawler(client: TestClient, user_api_key: TestAuthContext) -> WebCrawler:
    """Provides a WebCrawler instance for document API testing."""
    return WebCrawler(client, user_api_key=user_api_key)
