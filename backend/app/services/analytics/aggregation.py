"""Live monthly aggregation across llm_call, llm_chain, and evaluation_run.

The route handlers in `app/api/routes/analytics.py` call
`aggregate_monthly_metrics`, which fans out to three private helpers — one
per source table. Each helper does its own GROUP BY in Postgres and merges
results into the shared `buckets` dict in place. Splitting the sources lets
future endpoints (e.g. daily rollups, scheduled jobs) reuse individual
aggregations without dragging in the others.
"""

import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import aliased
from sqlmodel import Session, select

from app.crud.model_config import (
    KNOWN_PROVIDERS,
    NATIVE_PROVIDER_SUFFIX,
    estimate_model_cost,
)
from app.models.analytics import Modality
from app.models.config.version import ConfigVersion
from app.models.evaluation import EvaluationRun
from app.models.llm.request import LlmCall, LlmChain
from app.models.model_config import ModelConfig

logger = logging.getLogger(__name__)


BucketKey = tuple[date, Modality, str]
Bucket = dict[str, int | Decimal]


# evaluation_run.type (lowercased) -> modality bucket.
_EVAL_TYPE_TO_MODALITY: dict[str, Modality] = {
    "text": Modality.T_FS_T,
    "stt": Modality.STT,
    "tts": Modality.TTS,
}


def _first_of_next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _empty_bucket() -> Bucket:
    return {
        "llm_call_requests": 0,
        "llm_chain_requests": 0,
        "cost_usd": Decimal("0"),
        "input_tokens": 0,
        "output_tokens": 0,
        "eval_runs": 0,
        "eval_cost_usd": Decimal("0"),
    }


def _llm_modality_case(call=LlmCall) -> sa.sql.ColumnElement[str]:
    """SQL CASE mapping llm_call.input_type/output_type to a modality string.

    Accepts the `LlmCall` class or an `aliased(LlmCall)` so callers that
    join llm_call into another query (e.g. chain → first-block call) can
    reuse the same classification logic.
    """
    return sa.case(
        (
            sa.and_(call.input_type == "text", call.output_type == "text"),
            Modality.T_FS_T.value,
        ),
        (
            sa.and_(call.input_type == "audio", call.output_type == "audio"),
            Modality.S_FS_S.value,
        ),
        (
            sa.and_(call.input_type == "audio", call.output_type == "text"),
            Modality.STT.value,
        ),
        (
            sa.and_(call.input_type == "text", call.output_type == "audio"),
            Modality.TTS.value,
        ),
        else_=Modality.OTHER.value,
    )


def _aggregate_llm_calls(
    session: Session,
    buckets: dict[BucketKey, Bucket],
    *,
    organization_id: int,
    from_month: date | None,
    end_date: date | None,
    modality_filter: Modality | None,
    provider_filter: str | None,
    project_id: int | None,
) -> None:
    """GROUP BY llm_call → merge counts, tokens, and per-group cost into buckets.

    Cost is computed per (provider, model) group via `estimate_model_cost`
    using the summed tokens. The pricing function is linear in tokens, so
    summing first and pricing once is equivalent to per-row pricing. Skipped
    for providers outside `KNOWN_PROVIDERS` (the `model_config.provider`
    enum) since the lookup would raise InvalidTextRepresentation.
    """
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

    stmt = (
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
        stmt = stmt.where(LlmCall.inserted_at >= from_month)
    if end_date is not None:
        stmt = stmt.where(LlmCall.inserted_at < end_date)
    if project_id is not None:
        stmt = stmt.where(LlmCall.project_id == project_id)
    if provider_filter is not None:
        stmt = stmt.where(LlmCall.provider == provider_filter)

    for row in session.exec(stmt).all():
        modality_enum = Modality(row.modality)
        if modality_filter is not None and modality_enum is not modality_filter:
            continue
        bucket = buckets[(row.month, modality_enum, row.provider)]
        bucket["llm_call_requests"] += row.request_count

        input_tokens = int(row.input_tokens or 0)
        output_tokens = int(row.output_tokens or 0)
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        if (input_tokens or output_tokens) and row.provider in KNOWN_PROVIDERS:
            estimate = estimate_model_cost(
                session=session,
                provider=row.provider,  # type: ignore[arg-type]
                model_name=row.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if estimate is not None:
                bucket["cost_usd"] += Decimal(str(estimate.get("total_cost", 0)))


def _aggregate_chains(
    session: Session,
    buckets: dict[BucketKey, Bucket],
    *,
    organization_id: int,
    from_month: date | None,
    end_date: date | None,
    modality_filter: Modality | None,
    provider_filter: str | None,
    project_id: int | None,
) -> None:
    """GROUP BY llm_chain → merge counts into buckets.

    A chain is attributed to the modality+provider of its first child call.
    Single LEFT JOIN (llm_chain → llm_call on the first block UUID) +
    GROUP BY in Postgres — no per-row Python materialization. Chains
    with no resolvable first call land in (OTHER, "unknown") because the
    joined columns are NULL.
    """
    first_call = aliased(LlmCall)
    first_block_uuid = sa.cast(
        LlmChain.block_sequences[0].astext, PG_UUID(as_uuid=True)
    )
    month_col = (
        sa.func.date_trunc("month", LlmChain.inserted_at).cast(sa.Date).label("month")
    )
    modality_col = _llm_modality_case(first_call).label("modality")
    provider_col = sa.func.coalesce(first_call.provider, "unknown").label("provider")
    count_col = sa.func.count().label("chain_count")

    stmt = (
        select(month_col, modality_col, provider_col, count_col)
        .select_from(LlmChain)
        .outerjoin(first_call, first_call.id == first_block_uuid)
        .where(LlmChain.organization_id == organization_id)
        .group_by(month_col, modality_col, provider_col)
    )
    if from_month is not None:
        stmt = stmt.where(LlmChain.inserted_at >= from_month)
    if end_date is not None:
        stmt = stmt.where(LlmChain.inserted_at < end_date)
    if project_id is not None:
        stmt = stmt.where(LlmChain.project_id == project_id)
    if provider_filter is not None:
        stmt = stmt.where(
            sa.func.coalesce(first_call.provider, "unknown") == provider_filter
        )

    for row in session.exec(stmt).all():
        chain_modality = Modality(row.modality)
        if modality_filter is not None and chain_modality is not modality_filter:
            continue
        buckets[(row.month, chain_modality, row.provider)][
            "llm_chain_requests"
        ] += row.chain_count


def _aggregate_evals(
    session: Session,
    buckets: dict[BucketKey, Bucket],
    *,
    organization_id: int,
    from_month: date | None,
    end_date: date | None,
    modality_filter: Modality | None,
    provider_filter: str | None,
    project_id: int | None,
) -> None:
    """GROUP BY evaluation_run → merge counts and cost into buckets.

    `EvaluationRun.providers` is misnamed: per its column comment it actually
    stores *model names* (e.g. ['gemini-2.5-pro']), not providers. To recover
    the real provider we look up the model in `model_config`.

    CAVEAT: `DISTINCT ON (model_name) ORDER BY (model_name, provider)` picks
    the alphabetically-first provider when the same model_name exists under
    multiple providers (the unique key is (provider, model_name), so this is
    legal). Attribution is therefore best-effort and may be wrong for
    models shared across providers. The `ConfigVersion.config_blob` branch
    below is the authoritative source when present.
    """
    mc_lookup = (
        select(ModelConfig.model_name, ModelConfig.provider)
        .distinct(ModelConfig.model_name)
        .order_by(ModelConfig.model_name, ModelConfig.provider)
        .subquery()
    )

    cv_provider_normalized = sa.func.split_part(
        sa.cast(ConfigVersion.config_blob["completion"]["provider"].astext, sa.String),
        NATIVE_PROVIDER_SUFFIX,
        1,
    )

    provider_expr = sa.func.coalesce(
        sa.cast(mc_lookup.c.provider, sa.String),
        sa.func.nullif(cv_provider_normalized, ""),
        "unknown",
    )

    month_col = (
        sa.func.date_trunc("month", EvaluationRun.inserted_at)
        .cast(sa.Date)
        .label("month")
    )
    type_lower = sa.func.lower(sa.func.coalesce(EvaluationRun.type, "")).label(
        "type_lower"
    )
    provider_col = provider_expr.label("provider")
    count_col = sa.func.count().label("eval_count")
    cost_col = sa.func.coalesce(
        sa.func.sum(sa.cast(EvaluationRun.cost["total_cost_usd"].astext, sa.Numeric)),
        0,
    ).label("eval_cost_usd")

    stmt = (
        select(month_col, type_lower, provider_col, count_col, cost_col)
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
        .group_by(month_col, type_lower, provider_col)
    )
    if from_month is not None:
        stmt = stmt.where(EvaluationRun.inserted_at >= from_month)
    if end_date is not None:
        stmt = stmt.where(EvaluationRun.inserted_at < end_date)
    if project_id is not None:
        stmt = stmt.where(EvaluationRun.project_id == project_id)
    if provider_filter is not None:
        stmt = stmt.where(provider_expr == provider_filter)

    for row in session.exec(stmt).all():
        eval_modality = _EVAL_TYPE_TO_MODALITY.get(row.type_lower, Modality.OTHER)
        if modality_filter is not None and eval_modality is not modality_filter:
            continue
        bucket = buckets[(row.month, eval_modality, row.provider)]
        bucket["eval_runs"] += row.eval_count
        bucket["eval_cost_usd"] += Decimal(str(row.eval_cost_usd or 0))


def aggregate_monthly_metrics(
    session: Session,
    *,
    organization_id: int,
    from_month: date | None,
    to_month: date | None,
    modality_filter: Modality | None,
    provider_filter: str | None,
    project_id: int | None,
) -> dict[BucketKey, Bucket]:
    """Live aggregation across llm_call, llm_chain, and evaluation_run.

    Returns a dict keyed by (month, modality, provider) with per-bucket
    totals (requests, tokens, cost, eval runs, eval cost). The caller
    decides which fields to surface for a given metric.
    """
    end_date = _first_of_next_month(to_month) if to_month else None
    buckets: dict[BucketKey, Bucket] = defaultdict(_empty_bucket)
    common = dict(
        organization_id=organization_id,
        from_month=from_month,
        end_date=end_date,
        modality_filter=modality_filter,
        provider_filter=provider_filter,
        project_id=project_id,
    )
    _aggregate_llm_calls(session, buckets, **common)
    _aggregate_chains(session, buckets, **common)
    _aggregate_evals(session, buckets, **common)
    return buckets
