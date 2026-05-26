import logging
from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.model_config import KNOWN_PROVIDERS
from app.models import (
    AnalyticsChartGroupBy,
    AnalyticsChartResponse,
    AnalyticsChartSeries,
    AnalyticsMetric,
    AnalyticsMonthlyMetricPoint,
    Modality,
)
from app.services.analytics import Bucket, aggregate_monthly_metrics
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# Default lookback when the caller omits `from_month`. Caps the worst-case
# scan size so an unfiltered request can't trigger a full-table scan on
# llm_call / llm_chain / evaluation_run as the source tables grow.
DEFAULT_LOOKBACK_MONTHS = 24


def _default_from_month(anchor: date) -> date:
    """First-of-month DEFAULT_LOOKBACK_MONTHS calendar months before anchor."""
    year = anchor.year
    month = anchor.month - DEFAULT_LOOKBACK_MONTHS
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1)


def _snap_to_first_of_month(d: date | None) -> date | None:
    """Coerce a date to the first of its month.

    The analytics window is bucketed monthly, so a caller passing
    `2026-03-15` would otherwise filter `inserted_at >= 2026-03-15` and
    return a partial March bucket that looks indistinguishable from a
    real month. Snap to `2026-03-01` so the response always represents
    whole months.
    """
    if d is None:
        return None
    return date(d.year, d.month, 1)


def _validate_provider_filter(provider: str | None) -> None:
    """Reject provider filters that aren't one of the canonical enum values.

    A typo like `opena1` would otherwise silently return an empty result
    set, which is indistinguishable from "no activity for openai". Surface
    it as a 400 instead.
    """
    if provider is not None and provider not in KNOWN_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown provider '{provider}'. "
                f"Expected one of: {sorted(KNOWN_PROVIDERS)}."
            ),
        )


def _bucket_value(bucket: Bucket, metric: AnalyticsMetric) -> Decimal:
    if metric is AnalyticsMetric.REQUESTS:
        return Decimal(
            int(bucket["llm_call_requests"]) + int(bucket["llm_chain_requests"])
        )
    if metric is AnalyticsMetric.COST:
        return Decimal(bucket["cost_usd"])
    if metric is AnalyticsMetric.EVAL_RUNS:
        return Decimal(int(bucket["eval_runs"]))
    return Decimal(bucket["eval_cost_usd"])  # EVAL_COST


def _series_name(
    modality: Modality, provider: str, group_by: AnalyticsChartGroupBy
) -> str:
    if group_by is AnalyticsChartGroupBy.MODALITY_PROVIDER:
        return f"{modality.value} · {provider}"
    if group_by is AnalyticsChartGroupBy.MODALITY:
        return modality.value
    if group_by is AnalyticsChartGroupBy.PROVIDER:
        return provider
    return "total"  # AnalyticsChartGroupBy.TOTAL


@router.get(
    "/monthly",
    description=load_description("analytics/monthly.md"),
    response_model=APIResponse[list[AnalyticsMonthlyMetricPoint]],
    dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))],
)
def get_monthly_analytics(
    session: SessionDep,
    current_user: AuthContextDep,
    metric: AnalyticsMetric = Query(
        ...,
        description="Which metric to return (requests | cost | eval_runs | eval_cost)",
    ),
    from_month: date
    | None = Query(
        None,
        description=(
            "Inclusive lower bound (first-of-month). When omitted, defaults to "
            f"{DEFAULT_LOOKBACK_MONTHS} months before `to_month` (or before today "
            "if `to_month` is also omitted). Pass an explicit value to query "
            "further back."
        ),
    ),
    to_month: date
    | None = Query(
        None,
        description="Inclusive upper bound (first-of-month). Defaults to no upper bound.",
    ),
    modality: Modality
    | None = Query(None, description="Filter to a single modality bucket."),
    provider: str
    | None = Query(
        None,
        description=(
            "Filter to a single provider. Must be one of the canonical "
            "model_config values: 'openai', 'google', 'sarvamai', 'elevenlabs'. "
            "Anything else returns 400."
        ),
    ),
    project_id: int
    | None = Query(
        None,
        description=(
            "Optional: scope to a single project within the organization. "
            "Defaults to the caller's current project if one is selected; "
            "otherwise aggregates across every project in the caller's org."
        ),
    ),
):
    """Live monthly analytics for the caller's current project (or whole org
    if no project is selected), shaped per-point.

    Each point is `{month, modality, provider, value, ...tokens}`. Data is
    computed on-demand from llm_call/llm_chain/evaluation_run — no
    aggregation table or background job, so reads always reflect the
    current database state.
    """
    effective_project_id = (
        project_id
        if project_id is not None
        else (current_user.project.id if current_user.project else None)
    )
    _validate_provider_filter(provider)
    from_month = _snap_to_first_of_month(from_month)
    to_month = _snap_to_first_of_month(to_month)
    effective_from_month = from_month or _default_from_month(to_month or date.today())
    buckets = aggregate_monthly_metrics(
        session=session,
        organization_id=current_user.organization_.id,
        from_month=effective_from_month,
        to_month=to_month,
        modality_filter=modality,
        provider_filter=provider,
        project_id=effective_project_id,
    )

    points: list[AnalyticsMonthlyMetricPoint] = []
    for key in sorted(buckets.keys(), key=lambda k: (k[0], k[1].value, k[2])):
        month, mod, prov = key
        bucket = buckets[key]
        input_tokens = int(bucket["input_tokens"])
        output_tokens = int(bucket["output_tokens"])
        points.append(
            AnalyticsMonthlyMetricPoint(
                month=month,
                modality=mod,
                provider=prov,
                value=_bucket_value(bucket, metric),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        )

    return APIResponse.success_response(points)


@router.get(
    "/monthly/chart",
    description=load_description("analytics/monthly_chart.md"),
    response_model=APIResponse[AnalyticsChartResponse],
    dependencies=[Depends(require_permission(Permission.REQUIRE_ORGANIZATION))],
)
def get_monthly_analytics_chart(
    session: SessionDep,
    current_user: AuthContextDep,
    metric: AnalyticsMetric = Query(
        ...,
        description="Which metric to plot (requests | cost | eval_runs | eval_cost).",
    ),
    from_month: date
    | None = Query(
        None,
        description=(
            "Inclusive lower bound (first-of-month). When omitted, defaults to "
            f"{DEFAULT_LOOKBACK_MONTHS} months before `to_month` (or before today "
            "if `to_month` is also omitted). Pass an explicit value to query "
            "further back."
        ),
    ),
    to_month: date
    | None = Query(
        None,
        description="Inclusive upper bound (first-of-month). Defaults to no upper bound.",
    ),
    modality: Modality
    | None = Query(None, description="Filter to a single modality bucket."),
    provider: str
    | None = Query(
        None,
        description=(
            "Filter to a single provider. Must be one of the canonical "
            "model_config values: 'openai', 'google', 'sarvamai', 'elevenlabs'. "
            "Anything else returns 400."
        ),
    ),
    project_id: int
    | None = Query(
        None,
        description=(
            "Optional: scope to a single project within the organization. "
            "Defaults to the caller's current project if one is selected; "
            "otherwise aggregates across every project in the caller's org."
        ),
    ),
    group_by: AnalyticsChartGroupBy = Query(
        AnalyticsChartGroupBy.MODALITY_PROVIDER,
        description=(
            "How to split data into chart series. "
            "`modality_provider` = one series per (modality, provider) combo. "
            "`modality` = one per modality, summing across providers. "
            "`provider` = one per provider, summing across modalities. "
            "`total` = a single series with the grand total per month."
        ),
    ),
):
    """Live analytics shaped for direct rendering as a chart.

    Scoped to the caller's current project by default, or to the whole
    organization if the caller has no project selected.
    """
    effective_project_id = (
        project_id
        if project_id is not None
        else (current_user.project.id if current_user.project else None)
    )
    _validate_provider_filter(provider)
    from_month = _snap_to_first_of_month(from_month)
    to_month = _snap_to_first_of_month(to_month)
    effective_from_month = from_month or _default_from_month(to_month or date.today())
    buckets = aggregate_monthly_metrics(
        session=session,
        organization_id=current_user.organization_.id,
        from_month=effective_from_month,
        to_month=to_month,
        modality_filter=modality,
        provider_filter=provider,
        project_id=effective_project_id,
    )

    labels: list[date] = sorted({month for (month, _, _) in buckets.keys()})
    label_index = {m: i for i, m in enumerate(labels)}

    series_acc: dict[str, list[Decimal]] = {}
    series_tokens: dict[str, dict[str, int]] = {}
    for (month, mod, prov), bucket in buckets.items():
        name = _series_name(mod, prov, group_by)
        if name not in series_acc:
            series_acc[name] = [Decimal("0")] * len(labels)
            series_tokens[name] = {"input_tokens": 0, "output_tokens": 0}
        series_acc[name][label_index[month]] += _bucket_value(bucket, metric)
        series_tokens[name]["input_tokens"] += int(bucket["input_tokens"])
        series_tokens[name]["output_tokens"] += int(bucket["output_tokens"])

    series = [
        AnalyticsChartSeries(
            name=name,
            data=values,
            total_input_tokens=series_tokens[name]["input_tokens"],
            total_output_tokens=series_tokens[name]["output_tokens"],
            total_tokens=(
                series_tokens[name]["input_tokens"]
                + series_tokens[name]["output_tokens"]
            ),
        )
        for name, values in sorted(series_acc.items())
    ]

    return APIResponse.success_response(
        AnalyticsChartResponse(
            metric=metric,
            group_by=group_by,
            labels=labels,
            series=series,
        )
    )
