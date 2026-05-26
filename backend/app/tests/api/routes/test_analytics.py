from datetime import date, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Job
from app.models.evaluation import EvaluationDataset, EvaluationRun
from app.models.llm.request import LlmCall, LlmChain
from app.models.model_config import ModelConfig
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.llm import create_llm_job
from app.tests.utils.test_data import create_test_evaluation_dataset

MONTHLY_URL = f"{settings.API_V1_STR}/analytics/monthly"
CHART_URL = f"{settings.API_V1_STR}/analytics/monthly/chart"


@pytest.fixture
def llm_job(db: Session) -> Job:
    return create_llm_job(db)


@pytest.fixture
def eval_dataset(db: Session, user_api_key: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


@pytest.fixture
def model_pricing(db: Session) -> ModelConfig:
    """Seed pricing so estimate_model_cost returns a value during tests."""
    model = ModelConfig(
        provider="openai",
        model_name=f"gpt-4o-analytics-test-{uuid4().hex[:8]}",
        completion_type="text",
        pricing={
            "response": {"input_token_cost": 1.0, "output_token_cost": 2.0},
        },
        is_active=True,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


# ----- Helpers ----------------------------------------------------------------


def _make_llm_call(
    db: Session,
    *,
    job_id,
    project_id: int,
    organization_id: int,
    provider: str = "openai",
    model: str = "gpt-4o",
    input_type: str = "text",
    output_type: str | None = "text",
    input_tokens: int = 100,
    output_tokens: int = 50,
    inserted_at: datetime | None = None,
) -> LlmCall:
    call = LlmCall(
        job_id=job_id,
        project_id=project_id,
        organization_id=organization_id,
        input="hi",
        input_type=input_type,
        output_type=output_type,
        provider=provider,
        model=model,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )
    if inserted_at is not None:
        call.inserted_at = inserted_at
    db.add(call)
    db.commit()
    db.refresh(call)
    return call


def _make_llm_chain(
    db: Session,
    *,
    job_id,
    project_id: int,
    organization_id: int,
    first_call_id: str | None = None,
) -> LlmChain:
    chain = LlmChain(
        job_id=job_id,
        project_id=project_id,
        organization_id=organization_id,
        total_blocks=1,
        input="chain",
        block_sequences=[first_call_id] if first_call_id else [],
    )
    db.add(chain)
    db.commit()
    db.refresh(chain)
    return chain


def _make_eval_run(
    db: Session,
    *,
    dataset_id: int,
    project_id: int,
    organization_id: int,
    type_: str = "text",
    cost_usd: float = 1.50,
) -> EvaluationRun:
    run = EvaluationRun(
        run_name=f"run_{uuid4().hex[:8]}",
        dataset_name="ds",
        dataset_id=dataset_id,
        type=type_,
        status="completed",
        organization_id=organization_id,
        project_id=project_id,
        cost={"total_cost_usd": cost_usd},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _ok(response):
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    return body["data"]


# ----- /analytics/monthly ----
class TestMonthlyAnalytics:
    def test_requires_authentication(self, client: TestClient):
        response = client.get(MONTHLY_URL, params={"metric": "requests"})
        assert response.status_code in (401, 403)

    def test_metric_requests_counts_llm_calls(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="anlz-provider-A",
            input_tokens=100,
            output_tokens=50,
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "requests", "provider": "anlz-provider-A"},
                headers=user_api_key_header,
            )
        )
        assert len(data) == 1
        point = data[0]
        assert point["modality"] == "T-FS-T"
        assert point["provider"] == "anlz-provider-A"
        assert int(point["value"]) == 1
        assert point["input_tokens"] == 100
        assert point["output_tokens"] == 50
        assert point["total_tokens"] == 150

    def test_metric_cost_uses_estimate_model_cost(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
        model_pricing: ModelConfig,
    ):
        # 1M input @ $1 + 500k output @ $2 = $2.00
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="openai",
            model=model_pricing.model_name,
            input_tokens=1_000_000,
            output_tokens=500_000,
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "cost", "provider": "openai"},
                headers=user_api_key_header,
            )
        )
        total = sum(float(p["value"]) for p in data)
        assert total == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "input_type,output_type,expected_modality",
        [
            ("text", "text", "T-FS-T"),
            ("audio", "audio", "S-FS-S"),
            ("audio", "text", "STT"),
            ("text", "audio", "TTS"),
            ("image", "text", "OTHER"),
        ],
    )
    def test_modality_derivation_from_io_types(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
        input_type: str,
        output_type: str,
        expected_modality: str,
    ):
        provider = f"mod-{expected_modality.lower()}"
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider=provider,
            input_type=input_type,
            output_type=output_type,
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "requests", "provider": provider},
                headers=user_api_key_header,
            )
        )
        assert len(data) == 1
        assert data[0]["modality"] == expected_modality

    def test_filter_modality(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="filter-mod",
            input_type="text",
            output_type="text",
        )
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="filter-mod",
            input_type="audio",
            output_type="text",
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={
                    "metric": "requests",
                    "modality": "STT",
                    "provider": "filter-mod",
                },
                headers=user_api_key_header,
            )
        )
        assert len(data) == 1
        assert data[0]["modality"] == "STT"

    def test_filter_provider(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="prov-only-this",
        )
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="prov-other",
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "requests", "provider": "prov-only-this"},
                headers=user_api_key_header,
            )
        )
        assert len(data) == 1
        assert data[0]["provider"] == "prov-only-this"

    def test_filter_excludes_data_outside_window(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="window-test",
        )

        # to_month far in the past — current row should be out of range.
        data = _ok(
            client.get(
                MONTHLY_URL,
                params={
                    "metric": "requests",
                    "provider": "window-test",
                    "from_month": "2020-01-01",
                    "to_month": "2020-01-01",
                },
                headers=user_api_key_header,
            )
        )
        assert data == []

    def test_metric_eval_runs(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        eval_dataset: EvaluationDataset,
    ):
        _make_eval_run(
            db,
            dataset_id=eval_dataset.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            type_="text",
            cost_usd=3.0,
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "eval_runs"},
                headers=user_api_key_header,
            )
        )
        # Find at least one bucket with eval_runs >= 1 in text modality.
        assert any(p["modality"] == "T-FS-T" and int(p["value"]) >= 1 for p in data)

    def test_metric_eval_cost_stt(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        eval_dataset: EvaluationDataset,
    ):
        _make_eval_run(
            db,
            dataset_id=eval_dataset.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            type_="stt",
            cost_usd=3.5,
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "eval_cost"},
                headers=user_api_key_header,
            )
        )
        assert any(p["modality"] == "STT" and float(p["value"]) >= 3.5 for p in data)

    def test_llm_chain_attributed_to_first_block(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        call = _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="chain-provider",
            input_type="audio",
            output_type="text",
        )
        _make_llm_chain(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            first_call_id=str(call.id),
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "requests", "provider": "chain-provider"},
                headers=user_api_key_header,
            )
        )
        # Both the call and the chain should be attributed to STT/chain-provider.
        assert len(data) == 1
        assert data[0]["modality"] == "STT"
        assert int(data[0]["value"]) == 2

    def test_llm_chain_without_first_block_falls_back_to_other(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_chain(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            first_call_id=None,
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "requests", "provider": "unknown"},
                headers=user_api_key_header,
            )
        )
        assert any(
            p["modality"] == "OTHER" and p["provider"] == "unknown" for p in data
        )

    def test_explicit_nonexistent_project_id_returns_empty(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="proj-scoping",
        )

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={
                    "metric": "requests",
                    "project_id": 999_999_999,
                    "provider": "proj-scoping",
                },
                headers=user_api_key_header,
            )
        )
        assert data == []

    def test_soft_deleted_llm_calls_excluded(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        call = _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="soft-deleted",
        )
        call.deleted_at = datetime.now()
        db.add(call)
        db.commit()

        data = _ok(
            client.get(
                MONTHLY_URL,
                params={"metric": "requests", "provider": "soft-deleted"},
                headers=user_api_key_header,
            )
        )
        assert data == []


# ----- /analytics/monthly/chart ----
class TestMonthlyChartAnalytics:
    def test_requires_authentication(self, client: TestClient):
        response = client.get(CHART_URL, params={"metric": "requests"})
        assert response.status_code in (401, 403)

    def test_chart_default_group_by_modality_provider(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="chart-default",
        )

        data = _ok(
            client.get(
                CHART_URL,
                params={"metric": "requests", "provider": "chart-default"},
                headers=user_api_key_header,
            )
        )
        assert data["metric"] == "requests"
        assert data["group_by"] == "modality_provider"
        assert isinstance(data["labels"], list)
        assert len(data["labels"]) >= 1
        series_names = {s["name"] for s in data["series"]}
        assert "T-FS-T · chart-default" in series_names
        # Each series.data is aligned with labels.
        for s in data["series"]:
            assert len(s["data"]) == len(data["labels"])

    @pytest.mark.parametrize(
        "group_by,expected_name",
        [
            ("modality", "T-FS-T"),
            ("provider", "chart-gb"),
            ("total", "total"),
        ],
    )
    def test_chart_group_by_variants(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
        group_by: str,
        expected_name: str,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="chart-gb",
        )

        data = _ok(
            client.get(
                CHART_URL,
                params={
                    "metric": "requests",
                    "group_by": group_by,
                    "provider": "chart-gb",
                },
                headers=user_api_key_header,
            )
        )
        assert data["group_by"] == group_by
        names = {s["name"] for s in data["series"]}
        assert expected_name in names

    def test_chart_token_totals_aggregate_across_months(
        self,
        client: TestClient,
        db: Session,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        llm_job: Job,
    ):
        _make_llm_call(
            db,
            job_id=llm_job.id,
            project_id=user_api_key.project_id,
            organization_id=user_api_key.organization_id,
            provider="chart-tokens",
            input_tokens=200,
            output_tokens=100,
        )

        data = _ok(
            client.get(
                CHART_URL,
                params={
                    "metric": "requests",
                    "group_by": "total",
                    "provider": "chart-tokens",
                },
                headers=user_api_key_header,
            )
        )
        total = next(s for s in data["series"] if s["name"] == "total")
        assert total["total_input_tokens"] == 200
        assert total["total_output_tokens"] == 100
        assert total["total_tokens"] == 300

    def test_chart_with_no_data_returns_empty_labels_and_series(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
    ):
        # Far-future window guarantees no data.
        future = date(date.today().year + 5, 1, 1)
        data = _ok(
            client.get(
                CHART_URL,
                params={
                    "metric": "requests",
                    "from_month": future.isoformat(),
                    "to_month": future.isoformat(),
                },
                headers=user_api_key_header,
            )
        )
        assert data["labels"] == []
        assert data["series"] == []
