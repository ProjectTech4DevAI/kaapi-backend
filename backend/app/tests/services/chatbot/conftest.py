import pytest
from sqlmodel import Session

from app.models import Config
from app.tests.utils.chatbot import create_test_anthropic_config


@pytest.fixture
def config(db: Session) -> Config:
    return create_test_anthropic_config(db)
