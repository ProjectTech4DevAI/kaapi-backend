from unittest.mock import MagicMock, patch

from app.core.telemetry import metrics


def _active_sentry() -> MagicMock:
    fake = MagicMock()
    fake.get_client.return_value.is_active.return_value = True
    return fake


class TestRecordStalePendingJobs:
    def test_table_level_metrics(self) -> None:
        fake = _active_sentry()
        with patch.object(metrics, "sentry_sdk", fake):
            metrics.record_stale_pending_jobs(
                table="batch_job",
                status="pending",
                stale_count=3,
                oldest_age_seconds=90,
            )
        names = {c.kwargs["name"]: c.kwargs for c in fake.metrics.gauge.call_args_list}
        assert names["jobs.pending.stale.count"]["value"] == 3
        assert names["jobs.pending.oldest_age_seconds"]["unit"] == "second"
        assert (
            names["jobs.pending.stale.count"]["attributes"]["job.table"] == "batch_job"
        )

    def test_dimensional_metrics_without_age(self) -> None:
        fake = _active_sentry()
        with patch.object(metrics, "sentry_sdk", fake):
            metrics.record_stale_pending_jobs(
                table="job",
                status="pending",
                stale_count=1,
                oldest_age_seconds=None,
                job_type="llm",
                action_type="call",
                dimensional=True,
            )
        names = {c.kwargs["name"]: c.kwargs for c in fake.metrics.gauge.call_args_list}
        assert list(names) == ["jobs.pending.stale.by_dimension.count"]
        attrs = names["jobs.pending.stale.by_dimension.count"]["attributes"]
        assert attrs["job.type"] == "llm"
        assert attrs["job.action_type"] == "call"
