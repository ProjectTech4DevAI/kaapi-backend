from datetime import date
from decimal import Decimal
from enum import Enum

from sqlmodel import SQLModel


class Modality(str, Enum):
    """High-level modality bucket for analytics grouping.

    Derived from llm_call.input_type + output_type, or from evaluation_run.type.
    """

    T_FS_T = "T-FS-T"  # text -> text
    S_FS_S = "S-FS-S"  # audio -> audio
    STT = "STT"  # audio -> text
    TTS = "TTS"  # text -> audio
    OTHER = "OTHER"  # anything else (image, pdf, multimodal, assessment, ...)


class AnalyticsMetric(str, Enum):
    """Metric selector for the analytics endpoints."""

    REQUESTS = "requests"
    COST = "cost"
    EVAL_RUNS = "eval_runs"
    EVAL_COST = "eval_cost"


class AnalyticsMonthlyMetricPoint(SQLModel):
    """One data point in a metric-shaped response.

    `value` carries the metric the caller asked for. Token fields are
    sourced from `llm_call.usage` and are independent of the chosen metric,
    so the frontend can render token usage alongside any metric without an
    extra API call. Chains and evaluation runs do not contribute tokens —
    chain tokens are the sum of their child calls (would double-count), and
    eval tokens live in a separate domain.
    """

    month: date
    modality: Modality
    provider: str
    value: Decimal
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AnalyticsChartGroupBy(str, Enum):
    """Dimension to split the chart series by."""

    MODALITY_PROVIDER = "modality_provider"
    MODALITY = "modality"
    PROVIDER = "provider"
    TOTAL = "total"


class AnalyticsChartSeries(SQLModel):
    """A single line / bar on the chart.

    `data[i]` aligns with `labels[i]` from the parent response. The
    `total_*_tokens` fields are series-wide sums (across every month in
    `labels`) sourced from `llm_call.usage`, so the chart can render a
    secondary axis or tooltip totals without an extra API call.
    """

    name: str
    data: list[Decimal]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0


class AnalyticsChartResponse(SQLModel):
    """Chart-shaped analytics response.

    Compatible with most chart libraries (Recharts, Chart.js, ApexCharts,
    Highcharts, ECharts): each series has the same length as `labels`, and
    `data[i]` corresponds to `labels[i]`.
    """

    metric: AnalyticsMetric
    group_by: AnalyticsChartGroupBy
    labels: list[date]
    series: list[AnalyticsChartSeries]
