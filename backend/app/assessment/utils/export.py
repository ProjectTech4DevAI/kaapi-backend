"""Export utilities for assessment results (CSV, XLSX, JSON)."""

import csv
import io
import json
import logging
import re
from typing import Any, Literal

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.assessment.batch import _load_dataset_rows
from app.assessment.models import AssessmentExportRow
from app.assessment.processing import parse_assessment_output
from app.assessment.utils.parsing import parse_stored_results, usage_totals
from app.core.cloud import get_cloud_storage
from app.core.storage_utils import generate_timestamped_filename
from app.crud.job import get_batch_job
from app.assessment.models import Assessment
from app.models.batch_job import BatchJob
from app.models.evaluation import EvaluationDataset, EvaluationRun

logger = logging.getLogger(__name__)


def _safe_filename_part(value: str) -> str:
    """Build a filesystem-safe filename component."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "assessment_results"


def _expand_input_columns(
    row_payload: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand ``input_data`` dict into separate input columns.

    Uses the original column names from the dataset (no prefix).

    Returns:
        (expanded_rows with input_data replaced by individual columns,
         ordered list of input column names)
    """
    input_keys: list[str] = []
    seen_keys: dict[str, None] = {}

    for row in row_payload:
        input_data = row.get("input_data")
        if isinstance(input_data, dict):
            for k in input_data:
                if k not in seen_keys:
                    seen_keys[k] = None
                    input_keys.append(k)

    if not input_keys:
        for row in row_payload:
            row.pop("input_data", None)
        return row_payload, []

    expanded: list[dict[str, Any]] = []
    for row in row_payload:
        input_data = row.pop("input_data", None) or {}
        new_row = {}
        for k in input_keys:
            new_row[k] = input_data.get(k)
        new_row.update(row)
        expanded.append(new_row)

    return expanded, list(input_keys)


def _drop_empty_columns(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove columns where every row has a null or empty-string value."""
    non_empty_fields: list[str] = []
    for field in fieldnames:
        if any(
            row.get(field) is not None and str(row.get(field, "")).strip() != ""
            for row in rows
        ):
            non_empty_fields.append(field)

    if len(non_empty_fields) == len(fieldnames):
        return rows, fieldnames

    pruned = [{k: row.get(k) for k in non_empty_fields} for row in rows]
    return pruned, non_empty_fields


def _expand_output_columns(
    row_payload: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand the ``output`` field into separate columns when it contains valid JSON.

    Returns:
        (expanded_rows, ordered_fieldnames)
    """
    # First expand input columns
    row_payload, input_col_names = _expand_input_columns(row_payload)

    base_fields = [
        f for f in AssessmentExportRow.model_fields.keys()
        if f not in ("output", "input_data")
    ]

    parsed_outputs: list[dict[str, Any] | None] = []
    output_keys: list[str] = []
    seen_keys: dict[str, None] = {}  # ordered set
    has_unparsed_output = False

    for row in row_payload:
        raw = row.get("output")
        if raw is None:
            parsed_outputs.append(None)
            continue

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
        elif isinstance(raw, dict):
            parsed = raw
        else:
            parsed = None

        if not isinstance(parsed, dict):
            has_unparsed_output = True
            parsed_outputs.append(None)
            continue

        parsed_outputs.append(parsed)
        for k in parsed:
            if k not in seen_keys:
                seen_keys[k] = None
                output_keys.append(k)

    if not output_keys:
        # Keep original layout with output as a single column
        fieldnames = input_col_names + list(AssessmentExportRow.model_fields.keys())
        fieldnames = [f for f in fieldnames if f != "input_data"]
        return row_payload, fieldnames

    # Build expanded rows
    expanded: list[dict[str, Any]] = []
    for row, parsed in zip(row_payload, parsed_outputs, strict=True):
        new_row = {k: v for k, v in row.items() if k != "output"}
        if parsed:
            for k in output_keys:
                new_row[k] = parsed.get(k)
        else:
            for k in output_keys:
                new_row[k] = None
            if row.get("output") is not None:
                new_row["output_raw"] = row.get("output")
        expanded.append(new_row)

    # Build fieldnames: input columns + base fields + output columns
    output_idx = base_fields.index("result_status") + 1  # after result_status
    fieldnames = (
        input_col_names
        + base_fields[:output_idx]
        + output_keys
        + base_fields[output_idx:]
    )
    if has_unparsed_output:
        fieldnames.insert(
            len(input_col_names) + output_idx + len(output_keys), "output_raw"
        )

    return expanded, fieldnames


def serialize_export_rows(
    export_rows: list[AssessmentExportRow],
    export_format: Literal["json", "csv", "xlsx"],
) -> tuple[bytes, str]:
    """Serialize export rows into the requested file format."""
    row_payload = [row.model_dump(mode="json") for row in export_rows]

    if export_format == "json":
        expanded, _ = _expand_output_columns(row_payload)
        return (
            json.dumps(expanded, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
        )

    # For CSV/XLSX, expand output keys into separate columns
    expanded, fieldnames = _expand_output_columns(row_payload)

    if export_format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded)
        return output.getvalue().encode("utf-8"), "text/csv"

    try:
        import pandas as pd
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="XLSX export requires pandas/openpyxl support in the backend runtime",
        ) from exc

    # XLSX shows input columns + output columns only (no metadata fields).
    metadata_fields = {
        f for f in AssessmentExportRow.model_fields.keys()
        if f not in ("output", "input_data")
    }
    excel_fields = [f for f in fieldnames if f not in metadata_fields]
    if not excel_fields:
        excel_fields = ["output"]

    # Drop columns where every row is null/empty
    expanded, excel_fields = _drop_empty_columns(expanded, excel_fields)

    buf = io.BytesIO()
    data_frame = pd.DataFrame(expanded, columns=excel_fields)
    with pd.ExcelWriter(buf) as writer:
        data_frame.to_excel(writer, index=False, sheet_name="results")
    return (
        buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def build_json_export_rows(
    export_rows: list[AssessmentExportRow],
) -> list[dict[str, Any]]:
    """Return JSON rows with structured output expanded into top-level keys."""
    row_payload = [row.model_dump(mode="json") for row in export_rows]
    expanded, _ = _expand_output_columns(row_payload)
    return expanded


def build_export_response(
    export_rows: list[AssessmentExportRow],
    export_format: Literal["json", "csv", "xlsx"],
    base_name: str,
) -> StreamingResponse:
    """Return a file download response for assessment exports."""
    payload, media_type = serialize_export_rows(export_rows, export_format)
    filename = generate_timestamped_filename(
        _safe_filename_part(base_name),
        extension=export_format,
    )
    return StreamingResponse(
        io.BytesIO(payload),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _load_parsed_results_for_run(
    session: Any,
    run: EvaluationRun,
    batch_job: BatchJob,
) -> list[dict[str, Any]] | None:
    """Fetch and parse the stored batch results for a run.

    Tries object store first; falls back to downloading directly from the
    batch provider (e.g. OpenAI file API) when the S3 copy is unavailable.
    """
    # 1. Try object store (S3)
    if run.object_store_url:
        try:
            storage = get_cloud_storage(session, project_id=run.project_id)
            body = storage.stream(run.object_store_url)
            raw_results = parse_stored_results(body.read().decode("utf-8"))
            if raw_results:
                return parse_assessment_output(raw_results, batch_job.provider)
            logger.warning(
                "[_load_parsed_results_for_run] S3 file was empty for run id=%s",
                run.id,
            )
        except Exception as exc:
            logger.warning(
                "[_load_parsed_results_for_run] S3 download failed for run id=%s: %s",
                run.id,
                exc,
            )

    # 2. Fallback: download directly from batch provider
    if batch_job.provider_output_file_id:
        try:
            from app.assessment.processing import _get_batch_provider
            from app.core.batch import download_batch_results

            provider = _get_batch_provider(
                session=session,
                provider_name=batch_job.provider,
                organization_id=run.organization_id,
                project_id=run.project_id,
            )
            raw_results = download_batch_results(provider=provider, batch_job=batch_job)
            return parse_assessment_output(raw_results, batch_job.provider)
        except Exception as exc:
            logger.error(
                "[_load_parsed_results_for_run] Provider download also failed for run id=%s: %s",
                run.id,
                exc,
                exc_info=True,
            )

    logger.warning(
        "[_load_parsed_results_for_run] No results available for run id=%s "
        "(object_store_url=%s, provider_output_file_id=%s)",
        run.id,
        run.object_store_url,
        batch_job.provider_output_file_id,
    )
    return None


def _load_dataset_rows_for_run(
    session: Any,
    run: EvaluationRun,
) -> list[dict[str, str]]:
    """Load original dataset rows for input-output correlation.

    Returns an empty list if the dataset is not available.
    """
    try:
        dataset = session.get(EvaluationDataset, run.dataset_id)
        if not dataset or not dataset.object_store_url:
            logger.warning(
                "[_load_dataset_rows_for_run] Dataset not available for run id=%s",
                run.id,
            )
            return []
        return _load_dataset_rows(session, dataset)
    except Exception as exc:
        logger.warning(
            "[_load_dataset_rows_for_run] Failed to load dataset for run id=%s: %s",
            run.id,
            exc,
        )
        return []


def load_export_rows_for_run(
    session: Any,
    run: EvaluationRun,
    assessment: Assessment | None = None,
) -> list[AssessmentExportRow]:
    """Load flattened export rows for a single child assessment run."""
    if not run.batch_job_id:
        logger.warning(
            "[load_export_rows_for_run] No batch_job_id for run id=%s", run.id
        )
        return []

    batch_job = get_batch_job(session=session, batch_job_id=run.batch_job_id)
    if not batch_job:
        logger.warning(
            "[load_export_rows_for_run] Missing batch job for run id=%s",
            run.id,
        )
        return []

    parsed_results = _load_parsed_results_for_run(
        session=session,
        run=run,
        batch_job=batch_job,
    )
    if parsed_results is None:
        return []

    # Load original dataset rows for input correlation
    dataset_rows = _load_dataset_rows_for_run(session, run)

    experiment_name = assessment.experiment_name if assessment else run.run_name
    dataset_id = assessment.dataset_id if assessment else run.dataset_id
    dataset_name = assessment.dataset_name if assessment else run.dataset_name

    export_rows: list[AssessmentExportRow] = []
    for item in parsed_results:
        input_tokens, output_tokens, total_tokens = usage_totals(item.get("usage"))

        # Correlate with original input row via row_id (format: "row_{idx}")
        input_data: dict[str, str] | None = None
        row_id_str = str(item.get("row_id", ""))
        if dataset_rows and row_id_str.startswith("row_"):
            try:
                row_idx = int(row_id_str.split("_", 1)[1])
                if 0 <= row_idx < len(dataset_rows):
                    input_data = dataset_rows[row_idx]
            except (ValueError, IndexError):
                pass

        export_rows.append(
            AssessmentExportRow(
                assessment_id=run.assessment_id or 0,
                experiment_name=experiment_name,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                run_id=run.id,
                run_name=run.run_name,
                run_status=run.status,
                config_id=run.config_id,
                config_version=run.config_version,
                row_id=row_id_str,
                result_status="failed" if item.get("error") else "passed",
                input_data=input_data,
                output=item.get("output"),
                error=item.get("error"),
                response_id=item.get("response_id"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                updated_at=run.updated_at,
            )
        )

    return export_rows


def sort_export_rows(
    export_rows: list[AssessmentExportRow],
) -> list[AssessmentExportRow]:
    """Sort exported rows for stable downloads across runs/configs."""
    export_rows.sort(
        key=lambda row: (
            row.config_version or 0,
            row.row_id,
            row.run_id,
        )
    )
    return export_rows
