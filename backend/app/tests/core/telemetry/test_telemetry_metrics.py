from unittest.mock import MagicMock, mock_open, patch

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


class TestProcessRss:
    def test_reads_statm_pages(self) -> None:
        with (
            patch("builtins.open", mock_open(read_data="100 50 10\n")),
            patch.object(metrics.resource, "getpagesize", return_value=4096),
        ):
            assert metrics._current_rss_bytes() == 50 * 4096

    def test_falls_back_to_rusage_without_statm(self) -> None:
        rusage = MagicMock(ru_maxrss=7)
        with (
            patch("builtins.open", side_effect=OSError),
            patch.object(metrics.resource, "getrusage", return_value=rusage),
        ):
            assert metrics._current_rss_bytes() == 7 * metrics._RSS_BYTES_PER_UNIT

    def test_record_gauges_and_tags_span(self) -> None:
        fake = _active_sentry()
        span = MagicMock()
        span.is_recording.return_value = True
        with (
            patch.object(metrics, "_current_rss_bytes", return_value=123),
            patch.object(metrics, "sentry_sdk", fake),
            patch.object(metrics.trace, "get_current_span", return_value=span),
        ):
            assert (
                metrics.record_process_rss(role="task", task_name="run_llm_job") == 123
            )

        kwargs = fake.metrics.gauge.call_args.kwargs
        assert kwargs["name"] == metrics.PROCESS_RSS_METRIC
        assert kwargs["attributes"] == {
            "process.role": "task",
            "celery.task_name": "run_llm_job",
        }
        span.set_attribute.assert_called_once_with(metrics.PROCESS_RSS_ATTRIBUTE, 123)
