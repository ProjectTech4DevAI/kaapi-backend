import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import get_args

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from app.api.deps import AuthContextDep, SessionDep
from app.api.permissions import Permission, require_permission
from app.crud.model_config import Provider, estimate_model_cost
from app.models import (
    AnalyticsChartGroupBy,
    AnalyticsChartResponse,
    AnalyticsChartSeries,
    AnalyticsMetric,
    AnalyticsMonthlyMetricPoint,
    Modality,
)
from app.models.config.version import ConfigVersion
from app.models.evaluation import EvaluationRun
from app.models.llm.request import LlmCall, LlmChain
from app.models.model_config import ModelConfig
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# (input_type, output_type) -> modality bucket for llm_call rows.
_LLM_MODALITY: dict[tuple[str | None, str | None], Modality] = {
    ("text", "text"): Modality.T_FS_T,
    ("audio", "audio"): Modality.S_FS_S,
    ("audio", "text"): Modality.STT,
    ("text", "audio"): Modality.TTS,
}

# evaluation_run.type (lowercased) -> modality bucket.
_EVAL_TYPE_TO_MODALITY: dict[str, Modality] = {
    "text": Modality.T_FS_T,
    "stt": Modality.STT,
    "tts": Modality.TTS,
}

# Values accepted by the `global.provider_enum` column on model_config.
_KNOWN_PROVIDERS: frozenset[str] = frozenset(get_args(Provider))


def _derive_llm_modality(input_type: str | None, output_type: str | None) -> Modality:
    return _LLM_MODALITY.get((input_type, output_type), Modality.OTHER)


def _first_of_next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


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


def _llm_modality_case() -> sa.sql.ColumnElement[str]:
    """SQL CASE mapping llm_call.input_type/output_type to a modality string."""
    return sa.case(
        (
            sa.and_(LlmCall.input_type == "text", LlmCall.output_type == "text"),
            Modality.T_FS_T.value,
        ),
        (
            sa.and_(LlmCall.input_type == "audio", LlmCall.output_type == "audio"),
            Modality.S_FS_S.value,
        ),
        (
            sa.and_(LlmCall.input_type == "audio", LlmCall.output_type == "text"),
            Modality.STT.value,
        ),
        (
            sa.and_(LlmCall.input_type == "text", LlmCall.output_type == "audio"),
            Modality.TTS.value,
        ),
        else_=Modality.OTHER.value,
    )


def _empty_bucket() -> dict[str, int | Decimal]:
    return {
        "llm_call_requests": 0,
        "llm_chain_requests": 0,
        "cost_usd": Decimal("0"),
        "input_tokens": 0,
        "output_tokens": 0,
        "eval_runs": 0,
        "eval_cost_usd": Decimal("0"),
    }


def _aggregate_live(
    session: Session,
    *,
    organization_id: int,
    from_month: date | None,
    to_month: date | None,
    modality_filter: Modality | None,
    provider_filter: str | None,
    project_id: int | None,
) -> dict[tuple[date, Modality, str], dict[str, int | Decimal]]:
    """Live aggregation against llm_call, llm_chain, evaluation_run.

    Each source is GROUP BY'd in Postgres; per-group cost for llm_call is
    computed in Python via estimate_model_cost using the summed tokens.
    estimate_model_cost is linear in tokens, so summing first and pricing
    once per (provider, model) is equivalent to per-row pricing.

    Returns: {(month, modality, provider) -> totals dict}.
    """
    end_date = _first_of_next_month(to_month) if to_month else None
    buckets: dict[tuple[date, Modality, str], dict[str, int | Decimal]] = defaultdict(
        _empty_bucket
    )

    # ---- For the llm_call ----
    month_col = (
        sa.func.date_trunc("month", LlmCall.inserted_at).cast(sa.Date).label("month")
    )
    modality_col = _llm_modality_case().label("modality")
    provider_col = sa.func.coalesce(LlmCall.provider, "unknown").label("provider")
    input_tokens_col = sa.func.coalesce(
        sa.func.sum(sa.cast(LlmCall.usage["input_tokens"].astext, sa.Integer)),
        0,
    ).label("input_tokens")
    output_tokens_col = sa.func.coalesce(
        sa.func.sum(sa.cast(LlmCall.usage["output_tokens"].astext, sa.Integer)),
        0,
    ).label("output_tokens")
    count_col = sa.func.count().label("request_count")

    llm_stmt = (
        select(
            month_col,
            modality_col,
            provider_col,
            LlmCall.model,
            count_col,
            input_tokens_col,
            output_tokens_col,
        )
        .where(
            LlmCall.deleted_at.is_(None),
            LlmCall.organization_id == organization_id,
        )
        .group_by(month_col, modality_col, provider_col, LlmCall.model)
    )
    if from_month is not None:
        llm_stmt = llm_stmt.where(LlmCall.inserted_at >= from_month)
    if end_date is not None:
        llm_stmt = llm_stmt.where(LlmCall.inserted_at < end_date)
    if project_id is not None:
        llm_stmt = llm_stmt.where(LlmCall.project_id == project_id)
    if provider_filter is not None:
        llm_stmt = llm_stmt.where(LlmCall.provider == provider_filter)

    for row in session.exec(llm_stmt).all():
        modality_enum = Modality(row.modality)
        if modality_filter is not None and modality_enum is not modality_filter:
            continue
        key = (row.month, modality_enum, row.provider)
        bucket = buckets[key]
        bucket["llm_call_requests"] += row.request_count

        input_tokens = int(row.input_tokens or 0)
        output_tokens = int(row.output_tokens or 0)
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        if (input_tokens or output_tokens) and row.provider in _KNOWN_PROVIDERS:
            estimate = estimate_model_cost(
                session=session,
                provider=row.provider,  # type: ignore[arg-type]
                model_name=row.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if estimate is not None:
                bucket["cost_usd"] += Decimal(str(estimate.get("total_cost", 0)))

    # ---- llm_chain ----
    # A chain is attributed to the modality+provider of its first child call.
    # Fetch chains with the first-block UUID, then do one batched lookup
    # against llm_call to resolve modality+provider.
    chain_first_block = LlmChain.block_sequences[0].astext.label("first_call_id")
    chain_month_col = (
        sa.func.date_trunc("month", LlmChain.inserted_at).cast(sa.Date).label("month")
    )

    chain_stmt = select(chain_month_col, chain_first_block).where(
        LlmChain.organization_id == organization_id,
    )
    if from_month is not None:
        chain_stmt = chain_stmt.where(LlmChain.inserted_at >= from_month)
    if end_date is not None:
        chain_stmt = chain_stmt.where(LlmChain.inserted_at < end_date)
    if project_id is not None:
        chain_stmt = chain_stmt.where(LlmChain.project_id == project_id)

    chain_rows = session.exec(chain_stmt).all()
    first_call_ids = {row.first_call_id for row in chain_rows if row.first_call_id}

    first_call_map: dict[str, sa.Row] = {}
    if first_call_ids:
        lookup_stmt = select(
            LlmCall.id, LlmCall.input_type, LlmCall.output_type, LlmCall.provider
        ).where(LlmCall.id.in_(first_call_ids))
        for call_row in session.exec(lookup_stmt).all():
            first_call_map[str(call_row.id)] = call_row

    for row in chain_rows:
        first = first_call_map.get(row.first_call_id) if row.first_call_id else None
        if first is not None:
            chain_modality = _derive_llm_modality(first.input_type, first.output_type)
            chain_provider = first.provider or "unknown"
        else:
            chain_modality = Modality.OTHER
            chain_provider = "unknown"

        if modality_filter is not None and chain_modality is not modality_filter:
            continue
        if provider_filter is not None and chain_provider != provider_filter:
            continue
        buckets[(row.month, chain_modality, chain_provider)]["llm_chain_requests"] += 1

    # ---- evaluation_run ----
    mc_lookup = (
        select(ModelConfig.model_name, ModelConfig.provider)
        .distinct(ModelConfig.model_name)
        .order_by(ModelConfig.model_name, ModelConfig.provider)
        .subquery()
    )

    cv_provider_normalized = sa.func.split_part(
        sa.cast(ConfigVersion.config_blob["completion"]["provider"].astext, sa.String),
        "-native",
        1,
    )

    eval_provider_expr = sa.func.coalesce(
        sa.cast(mc_lookup.c.provider, sa.String),
        sa.func.nullif(cv_provider_normalized, ""),
        "unknown",
    )

    eval_month_col = (
        sa.func.date_trunc("month", EvaluationRun.inserted_at)
        .cast(sa.Date)
        .label("month")
    )
    eval_type_lower = sa.func.lower(sa.func.coalesce(EvaluationRun.type, "")).label(
        "type_lower"
    )
    eval_provider_col = eval_provider_expr.label("provider")
    eval_count_col = sa.func.count().label("eval_count")
    eval_cost_col = sa.func.coalesce(
        sa.func.sum(sa.cast(EvaluationRun.cost["total_cost_usd"].astext, sa.Numeric)),
        0,
    ).label("eval_cost_usd")

    eval_stmt = (
        select(
            eval_month_col,
            eval_type_lower,
            eval_provider_col,
            eval_count_col,
            eval_cost_col,
        )
        .select_from(EvaluationRun)
        .outerjoin(
            mc_lookup,
            mc_lookup.c.model_name == EvaluationRun.providers[0].astext,
        )
        .outerjoin(
            ConfigVersion,
            sa.and_(
                ConfigVersion.config_id == EvaluationRun.config_id,
                ConfigVersion.version == EvaluationRun.config_version,
            ),
        )
        .where(EvaluationRun.organization_id == organization_id)
        .group_by(eval_month_col, eval_type_lower, eval_provider_col)
    )
    if from_month is not None:
        eval_stmt = eval_stmt.where(EvaluationRun.inserted_at >= from_month)
    if end_date is not None:
        eval_stmt = eval_stmt.where(EvaluationRun.inserted_at < end_date)
    if project_id is not None:
        eval_stmt = eval_stmt.where(EvaluationRun.project_id == project_id)
    if provider_filter is not None:
        eval_stmt = eval_stmt.where(eval_provider_expr == provider_filter)

    for row in session.exec(eval_stmt).all():
        eval_modality = _EVAL_TYPE_TO_MODALITY.get(row.type_lower, Modality.OTHER)
        if modality_filter is not None and eval_modality is not modality_filter:
            continue
        key = (row.month, eval_modality, row.provider)
        bucket = buckets[key]
        bucket["eval_runs"] += row.eval_count
        bucket["eval_cost_usd"] += Decimal(str(row.eval_cost_usd or 0))

    return buckets


def _bucket_value(bucket: dict[str, int | Decimal], metric: AnalyticsMetric) -> Decimal:
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
        None, description="Filter to a single provider (e.g. 'openai', 'google')."
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
    effective_from_month = from_month or _default_from_month(to_month or date.today())
    buckets = _aggregate_live(
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
    provider: str | None = Query(None, description="Filter to a single provider."),
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
    effective_from_month = from_month or _default_from_month(to_month or date.today())
    buckets = _aggregate_live(
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
