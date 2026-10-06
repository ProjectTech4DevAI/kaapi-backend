import pytest
from pydantic import TypeAdapter, ValidationError

from app.core.config import Settings


class TestSentryConfigDefaults:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("SENTRY_TRACES_SAMPLE_RATE", 1.0),
            ("SENTRY_RELEASE", None),
            ("SENTRY_SEND_DEFAULT_PII", False),
            ("SENTRY_ERROR_SAMPLE_RATE", 1.0),
            ("SENTRY_PROFILE_SESSION_SAMPLE_RATE", 1.0),
            ("SENTRY_PROFILE_LIFECYCLE", "trace"),
        ],
    )
    def test_declared_default(
        self, field: str, expected: float | str | bool | None
    ) -> None:
        assert Settings.model_fields[field].default == expected


class TestSentryProfilingValidation:
    @staticmethod
    def _adapter(field: str) -> TypeAdapter[object]:
        return TypeAdapter(Settings.model_fields[field].rebuild_annotation())

    @pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
    def test_session_sample_rate_accepts_unit_interval(self, value: float) -> None:
        assert (
            self._adapter("SENTRY_PROFILE_SESSION_SAMPLE_RATE").validate_python(value)
            == value
        )

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_session_sample_rate_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(ValidationError):
            self._adapter("SENTRY_PROFILE_SESSION_SAMPLE_RATE").validate_python(value)

    @pytest.mark.parametrize("value", ["manual", "trace"])
    def test_lifecycle_accepts_supported_modes(self, value: str) -> None:
        assert self._adapter("SENTRY_PROFILE_LIFECYCLE").validate_python(value) == value

    def test_lifecycle_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter("SENTRY_PROFILE_LIFECYCLE").validate_python("always")
