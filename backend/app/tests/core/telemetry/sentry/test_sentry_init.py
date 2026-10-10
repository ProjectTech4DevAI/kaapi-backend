from unittest.mock import MagicMock, patch

from sentry_sdk.integrations import Integration

from app.core.telemetry.sentry import init as sentry_init


class _FakeIntegration(Integration):
    identifier = "fake"

    @staticmethod
    def setup_once() -> None:
        return None


class TestInitSentry:
    def test_noop_without_dsn(self) -> None:
        fake = MagicMock()
        with (
            patch.object(sentry_init.settings, "SENTRY_DSN", None),
            patch.object(sentry_init, "sentry_sdk", fake),
        ):
            sentry_init.init_sentry(integrations=[], disabled_integrations=[])

        fake.init.assert_not_called()

    def test_merges_caller_integrations_with_shared_config(self) -> None:
        fake = MagicMock()
        extra = _FakeIntegration()
        disabled = _FakeIntegration()
        with (
            patch.object(
                sentry_init.settings, "SENTRY_DSN", "https://k@o.ingest.sentry.io/1"
            ),
            patch.object(sentry_init.settings, "SENTRY_SEND_DEFAULT_PII", False),
            patch.object(sentry_init, "sentry_sdk", fake),
            patch.object(sentry_init, "genai_privacy_integrations", return_value=[]),
        ):
            sentry_init.init_sentry(
                integrations=[extra], disabled_integrations=[disabled]
            )

        fake.init.assert_called_once()
        kwargs = fake.init.call_args.kwargs
        assert kwargs["instrumenter"] == "otel"
        assert kwargs["send_default_pii"] is False
        assert kwargs["include_local_variables"] is False
        assert kwargs["max_request_body_size"] == "never"
        assert kwargs["before_send"] is sentry_init.before_send_error_filter
        assert (
            kwargs["before_send_transaction"]
            is sentry_init.before_send_transaction_filter
        )
        assert kwargs["before_send_log"] is sentry_init.before_send_log_filter
        assert extra in kwargs["integrations"]
        assert kwargs["disabled_integrations"] == [disabled]
