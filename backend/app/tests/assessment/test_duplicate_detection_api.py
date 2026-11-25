"""Tests for the duplicate-detection endpoint, its job runner and its detector."""

from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.crud.assessment import api
from app.models.assessment import (
    Assessment,
    AssessmentMethod,
    AssessmentRun,
    AssessmentStatus,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.config.config import ConfigTag
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import create_test_config
from app.tests.utils.utils import random_lower_string

URL = f"{settings.API_V1_STR}/assessments/duplicates"
SERVICE = "app.services.assessment.duplicate_detection.service"
CALLBACK = "https://hook.example.com/dedup"

STRICT_SCHEMA = {
    "type": "object",
    "properties": {
        "difference": {"type": "string"},
        "is_duplicate": {"type": "boolean"},
        "citation": {"type": "string", "nullable": True},
    },
    "required": ["difference", "is_duplicate", "citation"],
}
LENIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "citation": {"type": "string"},
                },
                "required": ["score", "citation"],
            },
        }
    },
    "required": ["matches"],
}


def _blob_dict(*, schema: dict, provider: str = "google-gcp", **params) -> dict:
    base = {
        "model": "gemini-3.8-flash",
        "instructions": "Compare the submission with the candidates.",
        "submission": "<new_submission>\n{query}\n</new_submission>",
        "knowledge_base_ids": ["projects/p/locations/l/ragCorpora/1"],
        "max_num_results": 5,
        "json_output_schema": schema,
    }
    base.update(params)
    return {
        "input_schema": {"query": {"type": "text"}},
        "assessment": {"provider": provider, "type": "text", "params": base},
    }


def _config(db: Session, project_id: int, blob: dict) -> dict:
    config = create_test_config(
        db,
        project_id=project_id,
        name=f"dedup-{random_lower_string()}",
        config_blob=AssessmentConfigBlob.model_validate(blob),
        tag=ConfigTag.ASSESSMENT,
    )
    return {"id": str(config.id), "version": 1}


def _body(strict: dict, lenient: dict, **overrides) -> dict:
    body = {
        "query": "Title: Rain alarm\nSolution: a rain sensor rings a buzzer",
        "config": {"strict": strict, "lenient": lenient},
        "callback_url": CALLBACK,
        "request_metadata": {"submission_id": "s-1"},
    }
    body.update(overrides)
    return body


@pytest.fixture
def configs(db: Session, user_api_key: TestAuthContext) -> tuple[dict, dict]:
    return (
        _config(db, user_api_key.project_id, _blob_dict(schema=STRICT_SCHEMA)),
        _config(db, user_api_key.project_id, _blob_dict(schema=LENIENT_SCHEMA)),
    )


@pytest.fixture
def dispatch():
    # The callback guard resolves DNS; the broker is external. Neither is under test here.
    with (
        patch(f"{SERVICE}.validate_callback_url"),
        patch(f"{SERVICE}.start_duplicate_detection", return_value="task-1") as start,
    ):
        yield start


class TestSubmitRoute:
    def test_valid_request_is_accepted_and_stored_as_one_job_with_two_runs(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        configs: tuple[dict, dict],
        dispatch,
    ) -> None:
        strict, lenient = configs
        resp = client.post(
            URL, headers=user_api_key_header, json=_body(strict, lenient)
        )

        assert resp.status_code == 202
        data = resp.json()["data"]
        assert data["status"] == "PENDING"
        assert data["strict"] is None and data["lenient"] is None
        assert data["config"] == {"strict": strict, "lenient": lenient}
        assert data["request_metadata"] == {"submission_id": "s-1"}

        job = db.get(Assessment, data["id"])
        assert job.method == AssessmentMethod.DUPLICATE_DETECTION
        assert job.input["callback_url"] == CALLBACK
        runs = api.list_executions(session=db, assessment_id=job.id)
        assert sorted(run.execution["level"] for run in runs) == ["lenient", "strict"]
        assert dispatch.call_args.args[0] == data["id"]

    def test_unknown_config_is_404(
        self, client: TestClient, user_api_key_header, configs, dispatch
    ) -> None:
        strict, lenient = configs
        missing = {"id": str(uuid4()), "version": 1}

        resp = client.post(
            URL, headers=user_api_key_header, json=_body(missing, lenient)
        )

        assert resp.status_code == 404
        dispatch.assert_not_called()

    def test_another_projects_config_is_404(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header,
        superuser_api_key: TestAuthContext,
        configs,
        dispatch,
    ) -> None:
        _, lenient = configs
        foreign = _config(
            db, superuser_api_key.project_id, _blob_dict(schema=STRICT_SCHEMA)
        )

        resp = client.post(
            URL, headers=user_api_key_header, json=_body(foreign, lenient)
        )

        assert resp.status_code == 404

    @pytest.mark.parametrize(
        ("blob", "message"),
        [
            (
                _blob_dict(schema=STRICT_SCHEMA, knowledge_base_ids=[]),
                "knowledge_base_ids",
            ),
            (
                _blob_dict(schema=STRICT_SCHEMA, max_num_results=None),
                "max_num_results",
            ),
            (
                _blob_dict(schema=STRICT_SCHEMA, provider="openai", model="gpt-4o"),
                "google-gcp",
            ),
            (
                _blob_dict(
                    schema={
                        "type": "object",
                        "properties": {"is_duplicate": {"type": "boolean"}},
                    }
                ),
                "must declare: citation",
            ),
        ],
        ids=[
            "empty-corpus-list",
            "no-candidate-count",
            "wrong-provider",
            "no-citation",
        ],
    )
    def test_strict_config_that_cannot_run_the_check_is_422(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header,
        configs,
        dispatch,
        blob: dict,
        message: str,
    ) -> None:
        _, lenient = configs
        strict = _config(db, user_api_key.project_id, blob)

        resp = client.post(
            URL, headers=user_api_key_header, json=_body(strict, lenient)
        )

        assert resp.status_code == 422
        assert resp.json()["error"].startswith("The strict config cannot run")
        assert message in resp.json()["error"]
        dispatch.assert_not_called()

    def test_lenient_config_without_a_match_score_is_422(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header,
        configs,
        dispatch,
    ) -> None:
        strict, _ = configs
        no_score = {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"citation": {"type": "string"}},
                    },
                }
            },
        }
        lenient = _config(db, user_api_key.project_id, _blob_dict(schema=no_score))

        resp = client.post(
            URL, headers=user_api_key_header, json=_body(strict, lenient)
        )

        assert resp.status_code == 422
        assert "matches[].score" in resp.json()["error"]

    def test_strict_config_used_as_lenient_is_422(
        self, client: TestClient, user_api_key_header, configs, dispatch
    ) -> None:
        strict, _ = configs

        resp = client.post(URL, headers=user_api_key_header, json=_body(strict, strict))

        assert resp.status_code == 422
        assert resp.json()["error"].startswith("The lenient config cannot run")

    def test_extra_output_fields_are_allowed(
        self, client: TestClient, user_api_key_header, configs, dispatch
    ) -> None:
        # STRICT_SCHEMA carries `difference`, a field the check never reads.
        strict, lenient = configs

        resp = client.post(
            URL, headers=user_api_key_header, json=_body(strict, lenient)
        )

        assert resp.status_code == 202

    @pytest.mark.parametrize(
        "callback", ["http://hook.example.com/dedup", "https://127.0.0.1/dedup"]
    )
    def test_unsafe_callback_url_is_422(
        self, client: TestClient, user_api_key_header, configs, callback: str
    ) -> None:
        strict, lenient = configs

        with patch(f"{SERVICE}.start_duplicate_detection") as start:
            resp = client.post(
                URL,
                headers=user_api_key_header,
                json=_body(strict, lenient, callback_url=callback),
            )

        assert resp.status_code == 422
        start.assert_not_called()

    def test_empty_query_is_422(
        self, client: TestClient, user_api_key_header, configs, dispatch
    ) -> None:
        strict, lenient = configs

        resp = client.post(
            URL, headers=user_api_key_header, json=_body(strict, lenient, query="")
        )

        assert resp.status_code == 422

    def test_missing_api_key_is_401(
        self, client: TestClient, configs, dispatch
    ) -> None:
        strict, lenient = configs

        resp = client.post(URL, json=_body(strict, lenient))

        assert resp.status_code == 401

    def test_dispatch_failure_is_503_and_fails_the_job(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header,
        configs,
        dispatch,
    ) -> None:
        strict, lenient = configs
        dispatch.side_effect = RuntimeError("broker down")

        resp = client.post(
            URL, headers=user_api_key_header, json=_body(strict, lenient)
        )

        assert resp.status_code == 503
        job = db.exec(
            select(Assessment)
            .where(Assessment.project_id == user_api_key.project_id)
            .where(Assessment.method == AssessmentMethod.DUPLICATE_DETECTION)
            .order_by(Assessment.inserted_at.desc())
        ).first()
        assert job.status == AssessmentStatus.FAILED


class TestStatusRoute:
    def _submit(self, client, headers, configs) -> str:
        strict, lenient = configs
        with (
            patch(f"{SERVICE}.validate_callback_url"),
            patch(f"{SERVICE}.start_duplicate_detection"),
        ):
            return client.post(
                URL, headers=headers, json=_body(strict, lenient)
            ).json()["data"]["id"]

    def test_returns_the_job(
        self, client: TestClient, user_api_key_header, configs
    ) -> None:
        job_id = self._submit(client, user_api_key_header, configs)

        resp = client.get(f"{URL}/{job_id}", headers=user_api_key_header)

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["id"] == job_id
        assert data["status"] == "PENDING"
        assert data["errors"] == {"strict": None, "lenient": None}

    def test_another_projects_job_is_404(
        self,
        client: TestClient,
        user_api_key_header,
        superuser_api_key: TestAuthContext,
        configs,
    ) -> None:
        job_id = self._submit(client, user_api_key_header, configs)

        resp = client.get(
            f"{URL}/{job_id}", headers={"X-API-KEY": superuser_api_key.key}
        )

        assert resp.status_code == 404

    def test_an_assessment_of_another_method_is_404(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header,
    ) -> None:
        other = api.create_assessment(
            session=db,
            method=AssessmentMethod.RESPONSE,
            input={"data": {"a": "one"}},
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )

        resp = client.get(f"{URL}/{other.id}", headers=user_api_key_header)

        assert resp.status_code == 404

    def test_unknown_id_is_404(self, client: TestClient, user_api_key_header) -> None:
        resp = client.get(f"{URL}/{uuid4()}", headers=user_api_key_header)

        assert resp.status_code == 404


class _SessionCtx:
    """Hands the service the test's transactional session instead of a new connection."""

    def __init__(self, session: Session):
        self._session = session

    def __enter__(self) -> Session:
        return self._session

    def __exit__(self, *_exc) -> bool:
        return False


def _seed_job(db: Session, auth: TestAuthContext, status: AssessmentStatus) -> UUID:
    from app.crud.assessment import duplicate_detection as crud
    from app.models.assessment.duplicate_detection import DuplicateDetectionCreate

    strict = _config(db, auth.project_id, _blob_dict(schema=STRICT_SCHEMA))
    lenient = _config(db, auth.project_id, _blob_dict(schema=LENIENT_SCHEMA))
    job, _ = crud.create_assessment(
        session=db,
        request=DuplicateDetectionCreate.model_validate(_body(strict, lenient)),
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )
    if status != AssessmentStatus.PENDING:
        api.update_status(session=db, obj=job, status=status)
    return job.id


def _result(*, strict=None, lenient=None, strict_error=None, lenient_error=None):
    from app.models.assessment.duplicate_detection import (
        DetectionErrors,
        DuplicateDetectionResult,
    )

    return DuplicateDetectionResult(
        strict=strict,
        lenient=lenient,
        errors=DetectionErrors(strict=strict_error, lenient=lenient_error),
    )


class _FakeDetector:
    """Stands in for the Vertex calls; returns or raises what the test sets."""

    outcome: object = None

    def __init__(self, _client) -> None:
        pass

    @staticmethod
    def create_client(_credentials) -> None:
        return None

    def detect(self, **_):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        pass


@pytest.fixture
def worker(db: Session):
    """Runs the worker functions against the test session; collects the webhooks they send."""
    from app.services.assessment.duplicate_detection import service

    sent: list[dict] = []
    _FakeDetector.outcome = None

    def _capture(url, payload, webhook_secret=None, **_log_kwargs):
        sent.append({"url": url, "payload": payload, "secret": webhook_secret})
        return True

    with (
        patch.object(service, "Session", lambda _engine: _SessionCtx(db)),
        patch.object(service, "DuplicateDetector", _FakeDetector),
        patch.object(service, "send_callback", side_effect=_capture),
    ):
        yield SimpleNamespace(service=service, sent=sent)


def _runs(db: Session, job_id) -> dict[str, AssessmentRun]:
    return {
        run.execution["level"]: run
        for run in api.list_executions(session=db, assessment_id=job_id)
    }


class TestRunJob:
    def test_both_levels_succeed(self, db: Session, user_api_key, worker) -> None:
        from app.models.assessment.duplicate_detection import (
            LenientMatch,
            StrictVerdict,
        )

        job_id = _seed_job(db, user_api_key, AssessmentStatus.PENDING)
        _FakeDetector.outcome = _result(
            strict=StrictVerdict(
                is_duplicate=True, citation="CID: 7\nTitle: Rain alarm"
            ),
            lenient=[LenientMatch(score=4, citation="CID: 9\nTitle: Drying rack")],
        )

        worker.service.run_duplicate_detection(assessment_id=job_id)

        assert db.get(Assessment, job_id).status == AssessmentStatus.COMPLETED
        runs = _runs(db, job_id)
        assert runs["strict"].execution["result"] == {
            "is_duplicate": True,
            "citation": "CID: 7\nTitle: Rain alarm",
        }
        assert runs["lenient"].execution["result"] == [
            {"score": 4, "citation": "CID: 9\nTitle: Drying rack"}
        ]
        assert [run.status for run in runs.values()] == [AssessmentStatus.COMPLETED] * 2

        [webhook] = worker.sent
        assert webhook["url"] == CALLBACK
        payload = webhook["payload"]
        assert payload["success"] is True
        assert payload["data"]["id"] == str(job_id)
        assert payload["data"]["status"] == AssessmentStatus.COMPLETED
        assert payload["data"]["strict"]["is_duplicate"] is True
        assert payload["data"]["lenient"][0]["score"] == 4
        assert payload["data"]["request_metadata"] == {"submission_id": "s-1"}

    def test_one_level_failing_still_delivers_the_other(
        self, db: Session, user_api_key, worker
    ) -> None:
        from app.models.assessment.duplicate_detection import StrictVerdict

        job_id = _seed_job(db, user_api_key, AssessmentStatus.PENDING)
        _FakeDetector.outcome = _result(
            strict=StrictVerdict(is_duplicate=False),
            lenient_error="[VERTEX] Corpus search failed (code: 404).",
        )

        worker.service.run_duplicate_detection(assessment_id=job_id)

        assert (
            db.get(Assessment, job_id).status == AssessmentStatus.COMPLETED_WITH_ERRORS
        )
        runs = _runs(db, job_id)
        assert runs["strict"].status == AssessmentStatus.COMPLETED
        assert runs["lenient"].status == AssessmentStatus.FAILED
        assert runs["lenient"].error_message.startswith("[VERTEX]")
        payload = worker.sent[0]["payload"]
        assert payload["success"] is True
        assert payload["data"]["strict"] == {"is_duplicate": False, "citation": None}
        assert payload["data"]["errors"]["lenient"].startswith("[VERTEX]")

    def test_both_levels_failing_fails_the_job(
        self, db: Session, user_api_key, worker
    ) -> None:
        job_id = _seed_job(db, user_api_key, AssessmentStatus.PENDING)
        _FakeDetector.outcome = _result(strict_error="boom", lenient_error="boom")

        worker.service.run_duplicate_detection(assessment_id=job_id)

        assert db.get(Assessment, job_id).status == AssessmentStatus.FAILED
        assert worker.sent[0]["payload"]["success"] is False

    def test_unexpected_error_fails_both_levels_and_is_delivered(
        self, db: Session, user_api_key, worker
    ) -> None:
        job_id = _seed_job(db, user_api_key, AssessmentStatus.PENDING)
        _FakeDetector.outcome = KeyError("knowledge_base_ids")

        worker.service.run_duplicate_detection(assessment_id=job_id)

        assert db.get(Assessment, job_id).status == AssessmentStatus.FAILED
        errors = worker.sent[0]["payload"]["data"]["errors"]
        assert errors["strict"] == errors["lenient"]
        assert "could not run (code: KeyError)" in errors["strict"]

    @pytest.mark.parametrize(
        "status",
        [
            AssessmentStatus.PROCESSING,
            AssessmentStatus.COMPLETED,
            AssessmentStatus.FAILED,
        ],
    )
    def test_a_job_that_is_not_pending_is_left_alone(
        self, db: Session, user_api_key, worker, status: AssessmentStatus
    ) -> None:
        # A second delivery of the same task must not run or report the job again.
        job_id = _seed_job(db, user_api_key, status)
        _FakeDetector.outcome = _result(strict_error="must not run", lenient_error="x")

        worker.service.run_duplicate_detection(assessment_id=job_id)

        assert db.get(Assessment, job_id).status == status
        assert worker.sent == []


class TestFailInterrupted:
    @pytest.mark.parametrize(
        "status", [AssessmentStatus.PENDING, AssessmentStatus.PROCESSING]
    )
    def test_an_unfinished_job_is_failed_and_reported(
        self, db: Session, user_api_key, worker, status: AssessmentStatus
    ) -> None:
        job_id = _seed_job(db, user_api_key, status)

        worker.service.fail_interrupted(assessment_id=job_id)

        assert db.get(Assessment, job_id).status == AssessmentStatus.FAILED
        runs = _runs(db, job_id)
        assert all("interrupted" in run.error_message for run in runs.values())
        [webhook] = worker.sent
        assert webhook["payload"]["success"] is False
        assert webhook["payload"]["data"]["status"] == AssessmentStatus.FAILED

    @pytest.mark.parametrize(
        "status",
        [
            AssessmentStatus.COMPLETED,
            AssessmentStatus.COMPLETED_WITH_ERRORS,
            AssessmentStatus.FAILED,
        ],
    )
    def test_a_finished_job_keeps_its_result(
        self, db: Session, user_api_key, worker, status: AssessmentStatus
    ) -> None:
        job_id = _seed_job(db, user_api_key, status)

        worker.service.fail_interrupted(assessment_id=job_id)

        assert db.get(Assessment, job_id).status == status
        assert worker.sent == []


class TestDispatch:
    def test_the_job_is_enqueued_with_a_failure_handler(self) -> None:
        from app.celery.utils import start_duplicate_detection

        with patch(
            "app.celery.tasks.job_execution.run_duplicate_detection.apply_async"
        ) as apply_async:
            apply_async.return_value = SimpleNamespace(id="task-1")
            task_id = start_duplicate_detection("job-1", "trace-1")

        assert task_id == "task-1"
        kwargs = apply_async.call_args.kwargs
        assert kwargs["kwargs"] == {"assessment_id": "job-1", "trace_id": "trace-1"}
        handler = kwargs["link_error"]
        assert handler.task == "app.celery.tasks.job_execution.fail_duplicate_detection"
        assert handler.kwargs["assessment_id"] == "job-1"


DETECTOR = "app.services.assessment.duplicate_detection.detector"


def _search_response(*contexts: tuple[str, str]) -> SimpleNamespace:
    payload = {
        "contexts": {
            "contexts": [{"sourceUri": uri, "text": text} for uri, text in contexts]
        }
    }
    return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)


@pytest.fixture
def detector():
    """A detector whose corpus search and model call are stubbed per test."""
    from app.services.assessment.duplicate_detection.detector import DuplicateDetector

    with (
        patch(f"{DETECTOR}.build_gcp_sa_credentials"),
        patch(f"{DETECTOR}.AuthorizedSession") as http,
    ):
        built = DuplicateDetector(SimpleNamespace(sa_info={}, genai=None))
        built.http = http.return_value
        yield built


def _answer(detector, text: str | None, error: str | None = None) -> None:
    response = (
        None
        if text is None
        else SimpleNamespace(
            usage=None,
            response=SimpleNamespace(
                model=None,
                output=SimpleNamespace(content=SimpleNamespace(value=text)),
            ),
        )
    )
    detector.execute = lambda **_: (response, error)


def _level_blob(schema: dict) -> AssessmentConfigBlob:
    return AssessmentConfigBlob.model_validate(_blob_dict(schema=schema))


class TestSearch:
    def test_chunks_of_one_entry_are_merged_in_rank_order(self, detector) -> None:
        detector.http.post.return_value = _search_response(
            ("gs://b/a.txt", "CID: 1\nTitle: A"),
            ("gs://b/b.txt", "CID: 2\nTitle: B"),
            ("gs://b/a.txt", "Solution: rest of A"),
        )

        found, error = detector.search(
            ["projects/p/locations/asia-south1/ragCorpora/1"], "q", 5
        )

        assert error is None
        assert found == ["CID: 1\nTitle: A\nSolution: rest of A", "CID: 2\nTitle: B"]
        assert (
            "asia-south1-aiplatform.googleapis.com"
            in detector.http.post.call_args.args[0]
        )

    def test_a_failed_search_is_reported_not_raised(self, detector) -> None:
        import requests

        detector.http.post.side_effect = requests.ConnectionError("down")

        found, error = detector.search(["projects/p/locations/l/ragCorpora/1"], "q", 5)

        assert found == []
        assert error.startswith("[VERTEX] Corpus search failed")


class TestCheckStrict:
    def _with_candidates(self, detector, *texts: str) -> None:
        detector.http.post.return_value = _search_response(
            *[(f"gs://b/{i}", text) for i, text in enumerate(texts)]
        )

    def test_the_cited_tag_is_replaced_by_the_corpus_entry(self, detector) -> None:
        self._with_candidates(
            detector, "CID: 1\nTitle: Fan", "CID: 2\nTitle: Rain alarm"
        )
        _answer(
            detector, '{"difference": "none", "is_duplicate": true, "citation": "C2"}'
        )

        verdict, error = detector.check_strict(_level_blob(STRICT_SCHEMA), "rain alarm")

        assert error is None
        assert verdict.model_dump() == {
            "is_duplicate": True,
            "citation": "CID: 2\nTitle: Rain alarm",
        }

    def test_not_a_duplicate_has_no_citation(self, detector) -> None:
        self._with_candidates(detector, "CID: 1\nTitle: Fan")
        _answer(
            detector, '{"difference": "x", "is_duplicate": false, "citation": "C1"}'
        )

        verdict, error = detector.check_strict(_level_blob(STRICT_SCHEMA), "q")

        assert error is None
        assert verdict.model_dump() == {"is_duplicate": False, "citation": None}

    def test_a_tag_that_was_never_offered_is_an_error(self, detector) -> None:
        self._with_candidates(detector, "CID: 1\nTitle: Fan")
        _answer(detector, '{"difference": "x", "is_duplicate": true, "citation": "C9"}')

        verdict, error = detector.check_strict(_level_blob(STRICT_SCHEMA), "q")

        assert verdict is None
        assert "strict check did not match the expected shape" in error

    def test_no_candidates_is_not_a_duplicate_without_asking_the_model(
        self, detector
    ) -> None:
        self._with_candidates(detector)
        detector.execute = lambda **_: pytest.fail("the model must not be called")

        verdict, error = detector.check_strict(_level_blob(STRICT_SCHEMA), "q")

        assert (verdict.is_duplicate, error) == (False, None)

    def test_a_search_failure_is_the_level_error(self, detector) -> None:
        import requests

        detector.http.post.side_effect = requests.Timeout("slow")
        detector.execute = lambda **_: pytest.fail("the model must not be called")

        verdict, error = detector.check_strict(_level_blob(STRICT_SCHEMA), "q")

        assert verdict is None
        assert error.startswith("[VERTEX]")

    def test_a_model_failure_is_the_level_error(self, detector) -> None:
        self._with_candidates(detector, "CID: 1\nTitle: Fan")
        _answer(detector, None, error="[GOOGLE_GCP] Resource not found (code: 404)")

        verdict, error = detector.check_strict(_level_blob(STRICT_SCHEMA), "q")

        assert verdict is None
        assert error == "[GOOGLE_GCP] Resource not found (code: 404)"


class TestCheckLenient:
    def test_matches_are_ranked_and_a_repeated_entry_keeps_its_best_score(
        self, detector
    ) -> None:
        detector.http.post.return_value = _search_response(
            ("gs://b/1", "CID: 1\nTitle: A"),
            ("gs://b/2", "CID: 2\nTitle: B"),
            ("gs://b/3", "CID: 3\nTitle: C"),
        )
        _answer(
            detector,
            '{"matches": [{"score": 4, "citation": "C1"}, {"score": 5, "citation": "C3"},'
            ' {"score": 2, "citation": "C1"}]}',
        )

        matches, error = detector.check_lenient(_level_blob(LENIENT_SCHEMA), "q")

        assert error is None
        assert [m.model_dump() for m in matches] == [
            {"score": 5, "citation": "CID: 3\nTitle: C"},
            {"score": 4, "citation": "CID: 1\nTitle: A"},
        ]

    def test_a_score_outside_one_to_five_is_an_error(self, detector) -> None:
        detector.http.post.return_value = _search_response(("gs://b/1", "CID: 1"))
        _answer(detector, '{"matches": [{"score": 9, "citation": "C1"}]}')

        matches, error = detector.check_lenient(_level_blob(LENIENT_SCHEMA), "q")

        assert matches is None
        assert "lenient check did not match the expected shape" in error


class TestDetect:
    def test_runs_both_levels_and_keeps_their_errors_apart(self, detector) -> None:
        from app.models.assessment.duplicate_detection import StrictVerdict

        detector.check_strict = lambda blob, query: (
            StrictVerdict(is_duplicate=False),
            None,
        )
        detector.check_lenient = lambda blob, query: (None, "lenient broke")

        result = detector.detect(
            query="q",
            strict=_level_blob(STRICT_SCHEMA),
            lenient=_level_blob(LENIENT_SCHEMA),
        )

        assert result.strict.is_duplicate is False
        assert result.lenient is None
        assert result.errors.model_dump() == {
            "strict": None,
            "lenient": "lenient broke",
        }
