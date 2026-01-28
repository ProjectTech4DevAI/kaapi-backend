"""
Local conftest for LangfuseTracer unit tests.

These tests don't require database access, so we override the session-scoped
seed_baseline fixture to skip database seeding.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def seed_baseline():
    """Override the global seed_baseline fixture to skip database seeding."""
    yield
