import pytest

from app.core.config import Settings


class TestSentryConfigDefaults:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("SENTRY_TRACES_SAMPLE_RATE", 1.0),
            ("SENTRY_SEND_DEFAULT_PII", False),
            ("SENTRY_ERROR_SAMPLE_RATE", 1.0),
        ],
    )
    def test_declared_default(
        self, field: str, expected: float | str | bool | None
    ) -> None:
        assert Settings.model_fields[field].default == expected
