"""Assessment batch JSONL construction and submission.

Builds provider-specific JSONL files from dataset rows + config,
then submits them via the core batch infrastructure.
"""

import csv
import io
import logging
from typing import Any

import openpyxl
from openpyxl.utils.exceptions import InvalidFileException
from sqlmodel import Session

from app.core.batch import BATCH_KEY, start_batch_job
from app.core.batch.openai import OpenAIBatchProvider
from app.core.cloud import get_cloud_storage
from app.models.assessment import (
    Assessment,
    AssessmentAttachment,
    AssessmentRun,
)
from app.models.batch_job import BatchJob, BatchJobType
from app.models.evaluation import EvaluationDataset
from app.models.llm.request import ConfigBlob
from app.services.assessment.mappers import (
    map_kaapi_to_google_params,
    map_kaapi_to_openai_params,
    normalize_llm_text,
)
from app.services.assessment.utils.attachments import (
    resolve_attachment_values,
    resolve_image_mime_and_payload,
    split_attachment_urls,
    split_data_url,
    to_direct_attachment_url,
)
from app.services.llm.providers.registry import LLMProvider

logger = logging.getLogger(__name__)


def _load_dataset_rows(
    session: Session,
    dataset: EvaluationDataset,
) -> list[dict[str, str]]:
    """Load dataset rows from object store.

    Returns a list of dicts (one per row) with column-name keys.
    """
    if not dataset.object_store_url:
        raise ValueError(f"Dataset {dataset.id} has no object_store_url")

    storage = get_cloud_storage(session=session, project_id=dataset.project_id)

    # Download the file content via stream()
    body = storage.stream(dataset.object_store_url)
    file_content = body.read()
    if not file_content:
        raise ValueError(f"Failed to download dataset from {dataset.object_store_url}")

    metadata = dataset.dataset_metadata or {}
    file_ext = metadata.get("file_extension", ".csv")

    if file_ext == ".xls":
        raise ValueError(
            "Legacy Excel format (.xls) is not supported. Please upload .xlsx or .csv."
        )
    if file_ext == ".xlsx":
        return _parse_excel_rows(file_content)
    return _parse_csv_rows(file_content)


def _parse_csv_rows(content: bytes) -> list[dict[str, str]]:
    """Parse CSV content into list of row dicts."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        text = content.decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader if any(v and v.strip() for v in row.values())]


def _parse_excel_rows(content: bytes) -> list[dict[str, str]]:
    """Parse Excel content into list of row dicts."""
    wb = None
    try:
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            return []

        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            return []

        columns = [
            str(col_header) if col_header is not None else f"col_{idx}"
            for idx, col_header in enumerate(header)
        ]
        result = []
        for row in rows_iter:
            if row and any(cell is not None for cell in row):
                row_dict = {
                    columns[idx]: str(cell) if cell is not None else ""
                    for idx, cell in enumerate(row)
                    if idx < len(columns)
                }
                result.append(row_dict)

        return result
    except InvalidFileException as e:
        logger.warning("[_parse_excel_rows] Invalid XLSX file content: %s", e)
        raise
    except Exception as e:
        logger.warning(
            "[_parse_excel_rows] Failed to parse XLSX rows | %s", e, exc_info=True
        )
        raise ValueError("Failed to parse XLSX dataset rows") from e
    finally:
        if wb is not None:
            wb.close()


def _build_text_prompt(
    row: dict[str, str],
    text_columns: list[str],
    prompt_template: str | None,
) -> str:
    """Build the text prompt for a single row.

    If prompt_template is provided, placeholders like {column_name} are replaced.
    Otherwise, all text column values are concatenated with newlines.
    """
    if prompt_template:
        prompt = normalize_llm_text(prompt_template)
        for col in text_columns:
            placeholder = "{" + col + "}"
            prompt = prompt.replace(placeholder, normalize_llm_text(row.get(col, "")))
        return prompt

    # No template: concatenate text columns
    parts = [
        normalize_llm_text(row.get(col, ""))
        for col in text_columns
        if row.get(col, "").strip()
    ]
    return "\n".join(parts)


def build_openai_jsonl(
    rows: list[dict[str, str]],
    text_columns: list[str],
    attachments: list[AssessmentAttachment],
    prompt_template: str | None,
    openai_params: dict,
) -> list[dict[str, Any]]:
    """Build OpenAI batch JSONL data from dataset rows.

    Each line follows the OpenAI batch format:
    {
        "custom_id": "row_0",
        "method": "POST",
        "url": "/v1/responses",
        "body": { model, instructions, temperature, input: [{role, content: [...]}] }
    }
    """
    jsonl_data = []

    for idx, row in enumerate(rows):
        # Build input array
        input_parts: list[dict[str, Any]] = []

        # Text prompt
        text_prompt = _build_text_prompt(row, text_columns, prompt_template)
        if text_prompt.strip():
            input_parts.append({"type": "input_text", "text": text_prompt})

        # Attachments
        for att in attachments:
            cell_value = row.get(att.column, "")
            input_parts.extend(resolve_attachment_values(cell_value, att))

        if not input_parts:
            logger.warning("[build_openai_jsonl] Skipping empty row | idx=%s", idx)
            continue

        # Build body from mapped params
        body = dict(openai_params)
        body["input"] = [
            {
                "role": "user",
                "content": input_parts,
            }
        ]

        jsonl_data.append(
            {
                BATCH_KEY: f"row_{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )

    return jsonl_data


def build_google_jsonl(
    rows: list[dict[str, str]],
    text_columns: list[str],
    attachments: list[AssessmentAttachment],
    prompt_template: str | None,
    google_params: dict,
) -> list[dict[str, Any]]:
    """Build Google (Gemini) batch JSONL data from dataset rows.

    Each line follows the Gemini batch format:
    {
        "key": "row_0",
        "request": { "contents": [{ "parts": [...], "role": "user" }] }
    }
    """
    jsonl_data = []

    for idx, row in enumerate(rows):
        parts: list[dict[str, Any]] = []

        # Text prompt
        text_prompt = _build_text_prompt(row, text_columns, prompt_template)
        if text_prompt.strip():
            parts.append({"text": text_prompt})

        # Attachments (Gemini uses file_data for inline content)
        for att in attachments:
            cell_value = row.get(att.column, "").strip()
            if not cell_value:
                continue

            cell_values = (
                split_attachment_urls(cell_value)
                if att.format == "url"
                else [cell_value]
            )

            for item_value in cell_values:
                normalized_value = (
                    to_direct_attachment_url(item_value, att.type)
                    if att.format == "url"
                    else item_value
                )
                if att.type == "image":
                    mime_type, payload = resolve_image_mime_and_payload(
                        normalized_value,
                        att.format,
                    )
                    if att.format == "url":
                        parts.append(
                            {
                                "fileData": {
                                    "mimeType": mime_type,
                                    "fileUri": normalized_value,
                                }
                            }
                        )
                    else:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": payload,
                                }
                            }
                        )
                elif att.type == "pdf":
                    if att.format == "url":
                        parts.append(
                            {
                                "fileData": {
                                    "mimeType": "application/pdf",
                                    "fileUri": normalized_value,
                                }
                            }
                        )
                    else:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": "application/pdf",
                                    "data": split_data_url(normalized_value)[1],
                                }
                            }
                        )

        if not parts:
            logger.warning("[build_google_jsonl] Skipping empty row | idx=%s", idx)
            continue

        system_instruction = google_params.get("instructions")
        request: dict[str, Any] = {
            "contents": [{"parts": parts, "role": "user"}],
        }
        if system_instruction:
            request["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        generation_config: dict[str, Any] = {}
        temperature = google_params.get("temperature")
        if temperature is not None:
            generation_config["temperature"] = temperature
        top_p = google_params.get("top_p")
        if top_p is not None:
            generation_config["topP"] = top_p
        max_output_tokens = google_params.get("max_output_tokens")
        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = max_output_tokens
        thinking_config = google_params.get("thinking_config")
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config
        output_schema = google_params.get("output_schema")
        if output_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = output_schema
        if generation_config:
            request["generationConfig"] = generation_config

        jsonl_data.append(
            {
                "key": f"row_{idx}",
                "request": request,
            }
        )

    return jsonl_data


def submit_assessment_batch(
    session: Session,
    run: AssessmentRun,
    assessment: Assessment,
    dataset: EvaluationDataset,
    config_blob: ConfigBlob,
    assessment_input: dict[str, Any],
    organization_id: int,
    project_id: int,
) -> BatchJob:
    """Build JSONL and submit a batch for one assessment run.

    Args:
        session: Database session
        run: The AssessmentRun to process
        dataset: The dataset to read rows from
        config_blob: Resolved configuration blob
        assessment_input: Assessment input config (prompt_template, text_columns, etc.)
        organization_id: Organization ID
        project_id: Project ID

    Returns:
        Created BatchJob record
    """
    text_columns = assessment_input.get("text_columns", [])
    prompt_template = assessment_input.get("prompt_template")
    system_instruction = assessment_input.get("system_instruction")
    attachments_raw = assessment_input.get("attachments", [])
    output_schema = assessment_input.get("output_schema")
    attachments = [AssessmentAttachment(**a) for a in attachments_raw]

    # Load dataset rows
    rows = _load_dataset_rows(session, dataset)
    if not rows:
        raise ValueError(f"Dataset {dataset.id} has no rows")

    logger.info(
        "[submit_assessment_batch] Building JSONL | run_id=%s | rows=%s | provider=%s",
        run.id,
        len(rows),
        config_blob.completion.provider,
    )

    # Determine provider and build params
    completion = config_blob.completion
    provider_name = completion.provider or "openai"

    params = dict(completion.params)
    params.pop("instructions", None)
    params.pop("system_instruction", None)
    if isinstance(system_instruction, str) and system_instruction.strip():
        params["instructions"] = system_instruction
    if output_schema:
        params["output_schema"] = output_schema

    # Determine the base provider (openai or google)
    base_provider = provider_name.replace("-native", "")

    if base_provider == LLMProvider.OPENAI:
        mapped_params, warnings = map_kaapi_to_openai_params(
            session=session,
            kaapi_params=params,
        )
        if warnings:
            logger.info("[submit_assessment_batch] Mapper warnings: %s", warnings)

        jsonl_data = build_openai_jsonl(
            rows=rows,
            text_columns=text_columns,
            attachments=attachments,
            prompt_template=prompt_template,
            openai_params=mapped_params,
        )

        # Get OpenAI client and submit
        from app.utils import get_openai_client

        openai_client = get_openai_client(
            session=session,
            org_id=organization_id,
            project_id=project_id,
        )
        provider = OpenAIBatchProvider(client=openai_client)

        batch_config = {
            "endpoint": "/v1/responses",
            "description": f"Assessment: {assessment.experiment_name}",
            "completion_window": "24h",
        }

        batch_job = start_batch_job(
            session=session,
            provider=provider,
            provider_name="openai",
            job_type=BatchJobType.ASSESSMENT,
            organization_id=organization_id,
            project_id=project_id,
            jsonl_data=jsonl_data,
            config=batch_config,
        )

    elif base_provider == LLMProvider.GOOGLE:
        mapped_params, warnings = map_kaapi_to_google_params(params)
        if warnings:
            logger.info("[submit_assessment_batch] Mapper warnings: %s", warnings)

        jsonl_data = build_google_jsonl(
            rows=rows,
            text_columns=text_columns,
            attachments=attachments,
            prompt_template=prompt_template,
            google_params=mapped_params,
        )

        # Get Gemini client and submit
        from app.core.batch import GeminiBatchProvider
        from app.core.batch.client import GeminiClient

        gemini_client = GeminiClient.from_credentials(
            session=session,
            org_id=organization_id,
            project_id=project_id,
        )
        provider = GeminiBatchProvider(
            client=gemini_client.client,
            model=f"models/{mapped_params.get('model', 'gemini-2.5-pro')}",
        )

        batch_config = {
            "display_name": f"assessment-{assessment.experiment_name}",
            "model": f"models/{mapped_params.get('model', 'gemini-2.5-pro')}",
        }

        batch_job = start_batch_job(
            session=session,
            provider=provider,
            provider_name="google",
            job_type=BatchJobType.ASSESSMENT,
            organization_id=organization_id,
            project_id=project_id,
            jsonl_data=jsonl_data,
            config=batch_config,
        )

    else:
        raise ValueError(
            f"Unsupported provider for assessment batches: {provider_name}"
        )

    logger.info(
        "[submit_assessment_batch] Submitted batch | run_id=%s | batch_job_id=%s | provider=%s | items=%s",
        run.id,
        batch_job.id,
        base_provider,
        len(jsonl_data),
    )

    return batch_job
