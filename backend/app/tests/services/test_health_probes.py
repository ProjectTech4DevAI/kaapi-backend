import threading
from types import TracebackType
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session

from app.core.config import settings
from app.crud.jobs import JobCrud
from app.models.job import JobStatus, JobType, JobUpdate
from app.models.llm.request import KaapiCompletionConfig
from app.services import health_probes
from app.services.health_probes import _PROBES, _build_probe_request
from app.services.llm.mappers import transform_kaapi_config_to_native
from app.tests.utils.test_data import create_test_project


class _NonClosingSession:
    """Stands in for `Session(engine)` inside health_probes, backed by the
    conftest transactional `db` session instead of closing a real one."""

    def __init__(self, session: Session):
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


class _FakeRedis:
    """In-memory stand-in for `_redis_client`. `incr` is lock-protected so it
    actually replicates Redis's atomicity for the concurrency test below."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = str(value)

    def incr(self, key: str) -> int:
        with self._lock:
            count = int(self.store.get(key, "0")) + 1
            self.store[key] = str(count)
            return count


@pytest.fixture
def probe_env(db: Session, monkeypatch: pytest.MonkeyPatch):
    project = create_test_project(db)
    monkeypatch.setattr(settings, "HEALTH_PROBE_ORG_ID", project.organization_id)
    monkeypatch.setattr(settings, "HEALTH_PROBE_PROJECT_ID", project.id)
    monkeypatch.setattr(
        health_probes, "Session", lambda _engine: _NonClosingSession(db)
    )
    fake_redis = _FakeRedis()
    monkeypatch.setattr(health_probes, "_redis_client", fake_redis)
    return project, fake_redis


def _create_job(db: Session, project_id: int, status: JobStatus) -> UUID:
    job = JobCrud(session=db).create(job_type=JobType.LLM_API, project_id=project_id)
    JobCrud(session=db).update(job_id=job.id, job_update=JobUpdate(status=status))
    return job.id


class TestFiresExactlyOneProbe:
    def test_tick_enqueues_exactly_one_job(
        self, probe_env: tuple[Any, _FakeRedis]
    ) -> None:
        _, _fake_redis = probe_env
        expected_job_id = uuid4()

        with patch(
            "app.services.health_probes.start_job", return_value=expected_job_id
        ) as start_job_mock:
            result = health_probes.run_health_probe_tick()

        start_job_mock.assert_called_once()
        assert result["enqueued"] is True
        assert result["job_id"] == expected_job_id


class TestRoundRobinRotation:
    def test_index_advances_and_wraps_across_full_registry(
        self, probe_env: tuple[Any, _FakeRedis]
    ) -> None:
        seen_indexes = []
        with patch(
            "app.services.health_probes.start_job", side_effect=lambda **_k: uuid4()
        ):
            for _ in range(len(_PROBES)):
                result = health_probes.run_health_probe_tick()
                seen_indexes.append(result["probe_index"])

            assert seen_indexes == list(range(len(_PROBES)))

            wrapped = health_probes.run_health_probe_tick()
            assert wrapped["probe_index"] == 0

    def test_concurrent_claims_never_collide_or_skip_a_slot(
        self, probe_env: tuple[Any, _FakeRedis]
    ) -> None:
        # N concurrent claims must land on N distinct, evenly spread slots.
        from concurrent.futures import ThreadPoolExecutor

        rounds = 5
        total_claims = rounds * len(_PROBES)
        with ThreadPoolExecutor(max_workers=16) as pool:
            claimed = list(
                pool.map(
                    lambda _: health_probes._claim_next_probe_index(),
                    range(total_claims),
                )
            )

        counts = {i: claimed.count(i) for i in range(len(_PROBES))}
        assert all(count == rounds for count in counts.values()), counts


class TestMissingRedisKeysDoNotFailTick:
    def test_missing_index_key_defaults_to_zero(
        self, db: Session, probe_env: tuple[Any, _FakeRedis]
    ) -> None:
        project, fake_redis = probe_env
        # last_job_id present (not a first-ever run), index key absent.
        existing_job_id = _create_job(db, project.id, JobStatus.SUCCESS)
        fake_redis.store[health_probes._LAST_JOB_ID_KEY] = str(existing_job_id)

        with patch(
            "app.services.health_probes.start_job", return_value=uuid4()
        ) as start_job_mock:
            result = health_probes.run_health_probe_tick()

        start_job_mock.assert_called_once()
        assert result["enqueued"] is True
        assert result["probe_index"] == 0

    def test_missing_last_job_id_key_skips_checkin_but_still_fires(
        self, probe_env: tuple[Any, _FakeRedis]
    ) -> None:
        _, fake_redis = probe_env
        fake_redis.store[health_probes._INDEX_KEY] = "2"

        with (
            patch(
                "app.services.health_probes.start_job", return_value=uuid4()
            ) as start_job_mock,
            patch("app.services.health_probes.capture_checkin") as checkin_mock,
        ):
            result = health_probes.run_health_probe_tick()

        start_job_mock.assert_called_once()
        checkin_mock.assert_not_called()
        assert result["enqueued"] is True
        assert result["probe_index"] == 2
        assert result["previous_job_status"] is None


class TestPreviousProbeCheckin:
    def test_previous_job_failed_reports_error_checkin(
        self, db: Session, probe_env: tuple[Any, _FakeRedis]
    ) -> None:
        project, fake_redis = probe_env
        failed_job_id = _create_job(db, project.id, JobStatus.FAILED)
        fake_redis.store[health_probes._LAST_JOB_ID_KEY] = str(failed_job_id)

        with (
            patch("app.services.health_probes.start_job", return_value=uuid4()),
            patch("app.services.health_probes.capture_checkin") as checkin_mock,
        ):
            result = health_probes.run_health_probe_tick()

        checkin_mock.assert_called_once()
        assert (
            checkin_mock.call_args.kwargs["status"] == health_probes.MonitorStatus.ERROR
        )
        assert result["previous_job_status"] == JobStatus.FAILED.value

    def test_previous_job_success_reports_ok_checkin(
        self, db: Session, probe_env: tuple[Any, _FakeRedis]
    ) -> None:
        project, fake_redis = probe_env
        success_job_id = _create_job(db, project.id, JobStatus.SUCCESS)
        fake_redis.store[health_probes._LAST_JOB_ID_KEY] = str(success_job_id)

        with (
            patch("app.services.health_probes.start_job", return_value=uuid4()),
            patch("app.services.health_probes.capture_checkin") as checkin_mock,
        ):
            result = health_probes.run_health_probe_tick()

        checkin_mock.assert_called_once()
        assert checkin_mock.call_args.kwargs["status"] == health_probes.MonitorStatus.OK
        assert result["previous_job_status"] == JobStatus.SUCCESS.value


class TestProbeRegistryPayloadsAreValid:
    # FR-8/FR-10/FR-11: every registry entry must resolve through the real
    # Kaapi -> native mapper with zero warnings (catches e.g. Sarvam TTS
    # missing `language`, ElevenLabs TTS missing `voice`).

    @pytest.mark.parametrize(
        "probe",
        _PROBES,
        ids=[f"{p.provider}-{p.model}-{p.modality}" for p in _PROBES],
    )
    def test_probe_config_resolves_with_no_mapper_warnings(
        self, db: Session, probe: health_probes.Probe
    ) -> None:
        request = _build_probe_request(probe, index=0)

        kaapi_config = request.config.blob.completion  # type: ignore[union-attr]
        assert isinstance(kaapi_config, KaapiCompletionConfig)

        _native_config, warnings = transform_kaapi_config_to_native(
            session=db, kaapi_config=kaapi_config
        )

        assert warnings == []
