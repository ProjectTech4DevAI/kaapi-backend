"""Tests for the Sentry config settings declared in core/config.py.

Defaults are read off the model's declared fields (env-independent), so no full
Settings instance (with its many required env vars) is needed. Profiling values
are inline constants in core/telemetry.py, not settings, and are covered there.
"""

import pytest

from app.core.config import Settings


class TestSentryConfigDefaults:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("SENTRY_TRACES_SAMPLE_RATE", 1.0),
            ("SENTRY_RELEASE", None),
            ("SENTRY_SEND_DEFAULT_PII", False),
            ("SENTRY_ERROR_SAMPLE_RATE", 1.0),
        ],
    )
    def test_declared_default(self, field, expected):
        assert Settings.model_fields[field].default == expected
