"""Export utilities for assessment results (CSV, XLSX, JSON)."""

import csv
import io
import json
import logging
import re
import zipfile
from typing import Any, Literal

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.core.cloud import get_cloud_storage
from app.core.storage_utils import generate_timestamped_filename
from app.crud.assessment.processing import parse_assessment_output
from app.crud.job import get_batch_job
from app.models.assessment import Assessment, AssessmentExportRow, AssessmentRun
from app.models.batch_job import BatchJob
from app.models.evaluation import EvaluationDataset
from app.services.assessment.utils.parsing import parse_stored_results, usage_totals
from app.utils import APIResponse

_L1_JSON_COLUMNS = ["topic_relevance", "duplicate_detection"]

logger = logging.getLogger(__name__)


def _load_dataset_rows(
    session: Session,
    dataset: EvaluationDataset,
) -> list[dict[str, str]]:
    from app.crud.assessment.batch import _load_dataset_rows as load_dataset_rows

    return load_dataset_rows(session, dataset)


def _load_l1_results(
    session: Session,
    run: AssessmentRun,
    assessment: Assessment,
) -> dict[str, dict[str, Any]]:
    """Load L1 results from object store, keyed by row_id. Returns {} if unavailable."""
    if not run.l1_object_store_url:
        return {}
    try:
        storage = get_cloud_storage(session, project_id=assessment.project_id)
        body = storage.stream(run.l1_object_store_url)
        raw = body.read().decode("utf-8")
        results: list[dict[str, Any]] = json.loads(raw)
        return {str(item["row_id"]): item for item in results if "row_id" in item}
    except Exception as exc:
        logger.warning(
            "[_load_l1_results] Failed to load L1 results for run id=%s: %s",
            run.id,
            exc,
        )
        return {}


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
            for input_key in input_data:
                if input_key not in seen_keys:
                    seen_keys[input_key] = None
                    input_keys.append(input_key)

    if not input_keys:
        for row in row_payload:
            row.pop("input_data", None)
        return row_payload, []

    reserved_fields = set(AssessmentExportRow.model_fields.keys()) - {"input_data"}
    key_map: dict[str, str] = {}
    for input_key in input_keys:
        col = f"input_{input_key}" if input_key in reserved_fields else input_key
        key_map[input_key] = col

    collisions = {key: value for key, value in key_map.items() if key != value}
    if collisions:
        logger.warning(
            "[_expand_input_columns] Input dataset columns conflict with reserved "
            "export fields and were namespaced: %s",
            collisions,
        )

    expanded: list[dict[str, Any]] = []
    for row in row_payload:
        input_data = row.pop("input_data", None) or {}
        new_row = {}
        for input_key in input_keys:
            new_row[key_map[input_key]] = input_data.get(input_key)
        new_row.update(row)
        expanded.append(new_row)

    return expanded, [key_map[input_key] for input_key in input_keys]


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

    pruned = [{field: row.get(field) for field in non_empty_fields} for row in rows]
    return pruned, non_empty_fields


def _parse_json_col(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _expand_output_columns(
    row_payload: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[str]]:
    """Expand ``output``, ``topic_relevance``, and ``duplicate_detection`` JSON columns
    into separate flat columns when they contain valid JSON objects.

    Returns:
        (expanded_rows, ordered_fieldnames)
    """
    row_payload, input_col_names = _expand_input_columns(row_payload)

    json_expand_cols = {"output", "input_data"} | set(_L1_JSON_COLUMNS)
    base_fields = [
        field
        for field in AssessmentExportRow.model_fields.keys()
        if field not in json_expand_cols
    ]

    # L1 columns are prefixed with their parent name to avoid key collisions
    parsed_cols: dict[str, list[dict[str, Any] | None]] = {
        col: [] for col in ["output"] + _L1_JSON_COLUMNS
    }
    col_keys: dict[str, list[str]] = {col: [] for col in ["output"] + _L1_JSON_COLUMNS}
    col_seen: dict[str, dict[str, None]] = {
        col: {} for col in ["output"] + _L1_JSON_COLUMNS
    }
    has_unparsed_output = False

    for row in row_payload:
        for col in ["output"] + _L1_JSON_COLUMNS:
            parsed = _parse_json_col(row.get(col))
            if parsed is None and col == "output" and row.get(col) is not None:
                has_unparsed_output = True
            parsed_cols[col].append(parsed)
            if parsed:
                for k in parsed:
                    prefixed = f"{col}_{k}" if col in _L1_JSON_COLUMNS else k
                    if prefixed not in col_seen[col]:
                        col_seen[col][prefixed] = None
                        col_keys[col].append(prefixed)

    def _get_prefixed(parsed: dict[str, Any] | None, col: str) -> dict[str, Any]:
        if not parsed:
            return {}
        if col in _L1_JSON_COLUMNS:
            return {f"{col}_{k}": v for k, v in parsed.items()}
        return parsed

    # Build expanded rows
    expanded: list[dict[str, Any]] = []
    for i, row in enumerate(row_payload):
        new_row = {k: v for k, v in row.items() if k not in json_expand_cols}
        for col in ["output"] + _L1_JSON_COLUMNS:
            parsed = parsed_cols[col][i]
            keys = col_keys[col]
            prefixed_vals = _get_prefixed(parsed, col)
            if prefixed_vals:
                for k in keys:
                    new_row[k] = prefixed_vals.get(k)
            else:
                for k in keys:
                    new_row[k] = None
                if col == "output" and row.get("output") is not None:
                    new_row["output_raw"] = row.get("output")
        expanded.append(new_row)

    l1_keys = col_keys["topic_relevance"] + col_keys["duplicate_detection"]
    output_keys = col_keys["output"]

    all_output_keys = l1_keys + output_keys
    if not all_output_keys:
        fieldnames = input_col_names + list(AssessmentExportRow.model_fields.keys())
        fieldnames = [f for f in fieldnames if f != "input_data"]
        return row_payload, fieldnames, input_col_names, [], []

    fieldnames = input_col_names + l1_keys + output_keys + base_fields
    if has_unparsed_output:
        fieldnames.insert(
            len(input_col_names) + len(l1_keys) + len(output_keys), "output_raw"
        )

    return expanded, fieldnames, input_col_names, l1_keys, output_keys


def serialize_export_rows(
    export_rows: list[AssessmentExportRow],
    export_format: Literal["json", "csv", "xlsx"],
    post_processing_config: dict[str, Any] | None = None,
) -> tuple[bytes, str]:
    """Serialize export rows into the requested file format."""
    from app.services.assessment.utils.post_processing import apply_post_processing

    row_payload = [row.model_dump(mode="json") for row in export_rows]

    if export_format == "json":
        expanded, *_ = _expand_output_columns(row_payload)
        expanded = apply_post_processing(expanded, post_processing_config)
        return (
            json.dumps(expanded, ensure_ascii=False, indent=2).encode("utf-8"),
            "application/json",
        )

    (
        expanded,
        fieldnames,
        input_col_names,
        l1_keys,
        output_keys,
    ) = _expand_output_columns(row_payload)
    expanded = apply_post_processing(expanded, post_processing_config)

    # Add any new computed columns to fieldnames so they appear in output
    existing = set(fieldnames)
    computed_names = [
        c["name"]
        for c in (post_processing_config or {}).get("computed_columns") or []
        if c.get("name") and c["name"] not in existing
    ]
    if computed_names:
        fieldnames = fieldnames + computed_names

    if export_format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
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

    # Explicit ordering: inputs → L1 → L2 → computed columns
    excel_fields = input_col_names + l1_keys + output_keys + computed_names
    if not excel_fields:
        excel_fields = output_keys or ["output"]

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
    expanded, fieldnames, *_ = _expand_output_columns(row_payload)
    return [{k: row.get(k) for k in fieldnames if k in row} for row in expanded]


def build_export_response(
    export_rows: list[AssessmentExportRow],
    export_format: Literal["json", "csv", "xlsx"],
    base_name: str,
    post_processing_config: dict[str, Any] | None = None,
) -> StreamingResponse:
    """Return a file download response for assessment exports."""
    payload, media_type = serialize_export_rows(
        export_rows, export_format, post_processing_config
    )
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
    session: Session,
    run: AssessmentRun,
    batch_job: BatchJob,
) -> list[dict[str, Any]] | None:
    """Fetch and parse the stored batch results for a run.

    Tries object store first; falls back to downloading directly from the
    batch provider (e.g. OpenAI file API) when the S3 copy is unavailable.
    """
    parent = session.get(Assessment, run.assessment_id)
    if not parent:
        logger.warning(
            "[_load_parsed_results_for_run] Parent assessment not found for run id=%s",
            run.id,
        )
        return None

    # 1. Try object store (S3)
    if run.object_store_url:
        try:
            storage = get_cloud_storage(session, project_id=parent.project_id)
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
            from app.core.batch import download_batch_results
            from app.crud.assessment.processing import _get_batch_provider

            provider = _get_batch_provider(
                session=session,
                provider_name=batch_job.provider,
                organization_id=parent.organization_id,
                project_id=parent.project_id,
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
    session: Session,
    run: AssessmentRun,
    assessment: Assessment,
) -> list[dict[str, str]]:
    """Load original dataset rows for input-output correlation.

    Returns an empty list if the dataset is not available.
    """
    try:
        dataset = session.get(EvaluationDataset, assessment.dataset_id)
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


def _extract_l1_json_columns(
    l1_item: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return topic_relevance and duplicate_detection as JSON strings for export expansion."""
    if not l1_item:
        return {"topic_relevance": None, "duplicate_detection": None}

    tr = l1_item.get("topic_relevance")
    dup = l1_item.get("duplicate_detection")

    tr_flat: dict[str, Any] | None = None
    if tr:
        tr_flat = {}
        for col, val in (tr.get("column_relevance") or {}).items():
            tr_flat[col] = val
        tr_flat["decision"] = tr.get("decision")
        tr_flat["reasoning"] = tr.get("reasoning")

    dup_flat: dict[str, Any] | None = None
    if dup:
        dup_flat = {k: v for k, v in dup.items() if k != "row_id"}

    return {
        "topic_relevance": json.dumps(tr_flat, ensure_ascii=False) if tr_flat else None,
        "duplicate_detection": json.dumps(dup_flat, ensure_ascii=False)
        if dup_flat
        else None,
    }


def load_export_rows_for_run(
    session: Session,
    run: AssessmentRun,
    assessment: Assessment | None = None,
) -> list[AssessmentExportRow]:
    """Load flattened export rows for a single child assessment run.

    When L1 results exist, ALL dataset rows are included in output.
    L1-rejected rows have L1 columns filled and L2 columns empty.
    L1-passed rows have all columns filled.
    Without L1, behaviour is unchanged (only L2 result rows returned).
    """
    if assessment is None:
        assessment = session.get(Assessment, run.assessment_id)
    if assessment is None:
        logger.warning(
            "[load_export_rows_for_run] Parent assessment missing for run id=%s",
            run.id,
        )
        return []

    dataset = session.get(EvaluationDataset, assessment.dataset_id)
    dataset_name = dataset.name if dataset else None
    dataset_rows = _load_dataset_rows_for_run(session, run, assessment)

    # Load L1 results (empty dict if no L1 was run)
    l1_by_row_id = _load_l1_results(session, run, assessment)

    # Load L2 results (may be None if batch not complete)
    l2_by_row_id: dict[str, dict[str, Any]] = {}
    if run.batch_job_id:
        batch_job = get_batch_job(session=session, batch_job_id=run.batch_job_id)
        if batch_job:
            parsed_results = _load_parsed_results_for_run(
                session=session, run=run, batch_job=batch_job
            )
            if parsed_results:
                l2_by_row_id = {
                    str(item["row_id"]): item
                    for item in parsed_results
                    if "row_id" in item
                }

    has_l1 = bool(l1_by_row_id)

    if has_l1 and dataset_rows:
        # All rows in output — build from full dataset
        export_rows: list[AssessmentExportRow] = []
        for row_idx, input_data in enumerate(dataset_rows):
            row_id_str = f"row_{row_idx}"
            l1_item = l1_by_row_id.get(row_id_str)
            l1_cols = _extract_l1_json_columns(l1_item)
            l2_item = l2_by_row_id.get(row_id_str)

            input_tokens, output_tokens, total_tokens = usage_totals(
                l2_item.get("usage") if l2_item else None
            )
            l1_passed = (l1_item or {}).get("l1_passed", True)
            result_status = (
                "l1_rejected"
                if not l1_passed
                else ("failed" if l2_item and l2_item.get("error") else "passed")
            )

            export_rows.append(
                AssessmentExportRow(
                    assessment_id=run.assessment_id,
                    experiment_name=assessment.experiment_name,
                    dataset_id=assessment.dataset_id,
                    dataset_name=dataset_name,
                    run_id=run.id,
                    run_name=assessment.experiment_name,
                    run_status=run.status,
                    config_id=run.config_id,
                    config_version=run.config_version,
                    row_id=row_id_str,
                    result_status=result_status,
                    input_data=input_data,
                    topic_relevance=l1_cols.get("topic_relevance"),
                    duplicate_detection=l1_cols.get("duplicate_detection"),
                    output=l2_item.get("output") if l2_item else None,
                    error=l2_item.get("error") if l2_item else None,
                    response_id=l2_item.get("response_id") if l2_item else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    updated_at=run.updated_at,
                )
            )
        return export_rows

    # No L1 — original behaviour: only L2 result rows
    if not run.batch_job_id:
        logger.warning(
            "[load_export_rows_for_run] No batch_job_id for run id=%s", run.id
        )
        return []

    batch_job = get_batch_job(session=session, batch_job_id=run.batch_job_id)
    if not batch_job:
        logger.warning(
            "[load_export_rows_for_run] Missing batch job for run id=%s", run.id
        )
        return []

    parsed_results = _load_parsed_results_for_run(
        session=session, run=run, batch_job=batch_job
    )
    if not parsed_results:
        logger.warning(
            "[load_export_rows_for_run] Parsed results empty for run id=%s", run.id
        )
        return []

    export_rows = []
    for item in parsed_results:
        input_tokens, output_tokens, total_tokens = usage_totals(item.get("usage"))
        input_data = None
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
                assessment_id=run.assessment_id,
                experiment_name=assessment.experiment_name,
                dataset_id=assessment.dataset_id,
                dataset_name=dataset_name,
                run_id=run.id,
                run_name=assessment.experiment_name,
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

    def _row_index(row_id: str) -> int:
        if not row_id.startswith("row_"):
            return 0
        try:
            return int(row_id.split("_", 1)[1])
        except (ValueError, IndexError):
            return 0

    export_rows.sort(
        key=lambda row: (
            row.config_version or 0,
            _row_index(row.row_id),
            row.run_id,
        )
    )
    return export_rows


def build_assessment_results_response(
    session: Session,
    assessment: Assessment,
    runs: list[AssessmentRun],
    export_format: Literal["json", "csv", "xlsx"],
) -> APIResponse[list[dict[str, Any]]] | StreamingResponse:
    """Bundle child-run results for a parent assessment into a download response.

    JSON returns a flat list. CSV/XLSX with one run returns a single file;
    multiple runs are zipped one-file-per-run.
    """
    runs_with_rows: list[tuple[AssessmentRun, list[AssessmentExportRow]]] = []
    all_rows: list[AssessmentExportRow] = []
    for run in runs:
        rows = load_export_rows_for_run(session=session, run=run, assessment=assessment)
        if rows:
            runs_with_rows.append((run, sort_export_rows(rows)))
            all_rows.extend(rows)

    all_rows = sort_export_rows(all_rows)

    if export_format == "json":
        return APIResponse.success_response(data=build_json_export_rows(all_rows))

    if len(runs_with_rows) <= 1:
        return build_export_response(
            export_rows=all_rows,
            export_format=export_format,
            base_name=(
                f"{assessment.experiment_name}_assessment_{assessment.id}_results"
            ),
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for run, rows in runs_with_rows:
            config_label = (
                f"config_v{run.config_version}"
                if run.config_version
                else f"run_{run.id}"
            )
            config_id_short = str(run.config_id)[:8] if run.config_id else ""
            file_base = _safe_filename_part(f"{config_label}_{config_id_short}")
            file_bytes, _ = serialize_export_rows(rows, export_format)
            zf.writestr(f"{file_base}.{export_format}", file_bytes)

    zip_buffer.seek(0)
    zip_filename = generate_timestamped_filename(
        _safe_filename_part(f"{assessment.experiment_name}_assessment_{assessment.id}"),
        extension="zip",
    )
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )
