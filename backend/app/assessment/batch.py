"""Assessment batch JSONL construction and submission.

Builds provider-specific JSONL files from dataset rows + config,
then submits them via the core batch infrastructure.
"""

import csv
import io
import logging
import re
import base64
import binascii
from urllib.parse import urlparse
from typing import Any
from uuid import UUID

from sqlmodel import Session

from app.assessment.mappers import (
    map_kaapi_to_google_params,
    map_kaapi_to_openai_params,
    normalize_llm_text,
)
from app.assessment.models import (
    AssessmentAttachment,
    AssessmentRun,
    AssessmentTextLLMParams,
)
from app.core.batch import BATCH_KEY, start_batch_job
from app.core.batch.openai import OpenAIBatchProvider
from app.core.cloud import get_cloud_storage
from app.core.storage_utils import load_json_from_object_store
from app.crud.config.version import ConfigVersionCrud
from app.models.batch_job import BatchJob, BatchJobType
from app.models.evaluation import EvaluationDataset
from app.models.llm.request import ConfigBlob, KaapiCompletionConfig, LLMCallConfig
from app.services.llm.jobs import resolve_config_blob

logger = logging.getLogger(__name__)

# Provider name → native provider suffix mapping
_NATIVE_PROVIDERS = {
    "openai": "openai",
    "openai-native": "openai",
    "google": "google",
    "google-native": "google",
}

_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def _resolve_config(
    session: Session,
    config_id: UUID,
    config_version: int,
    project_id: int,
) -> tuple[ConfigBlob | None, str | None]:
    """Resolve a stored config into a ConfigBlob."""
    config_crud = ConfigVersionCrud(
        session=session,
        config_id=config_id,
        project_id=project_id,
    )
    return resolve_config_blob(
        config_crud=config_crud,
        config=LLMCallConfig(id=config_id, version=config_version),
    )


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

    if file_ext in (".xlsx", ".xls"):
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
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        wb.close()
        return []

    rows_iter = ws.iter_rows(values_only=True)
    header = next(rows_iter, None)
    if header is None:
        wb.close()
        return []

    columns = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(header)]
    result = []
    for row in rows_iter:
        if row and any(cell is not None for cell in row):
            row_dict = {
                columns[i]: str(cell) if cell is not None else ""
                for i, cell in enumerate(row)
                if i < len(columns)
            }
            result.append(row_dict)

    wb.close()
    return result


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


def _split_attachment_urls(value: str) -> list[str]:
    """Split comma/newline separated attachment URLs from a single dataset cell."""
    return [part.strip() for part in re.split(r"[\n,]+", value) if part.strip()]


def _to_direct_attachment_url(url: str, attachment_type: str) -> str:
    """Normalize share-page attachment URLs into provider-fetchable direct URLs.

    This currently handles common Google Drive share URL shapes. The file must
    still be publicly accessible to the model provider.
    """
    url = url.strip()
    file_id = None

    match = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url)
    if match:
        file_id = match.group(1)

    if not file_id:
        match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if match and (
            "drive.google.com" in url or "drive.usercontent.google.com" in url
        ):
            file_id = match.group(1)

    if not file_id:
        return url

    if attachment_type == "image":
        return f"https://lh3.googleusercontent.com/d/{file_id}"

    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _split_data_url(value: str) -> tuple[str | None, str]:
    """Return (mime_type, base64_payload) for a data URL; otherwise (None, value)."""
    match = re.match(
        r"^data:([^;]+);base64,(.+)$",
        value.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, value.strip()
    return match.group(1).strip().lower(), match.group(2).strip()


def _guess_image_mime_from_url(url: str) -> str | None:
    path = urlparse(url).path or ""
    for ext, mime in _IMAGE_MIME_BY_EXT.items():
        if path.lower().endswith(ext):
            return mime
    return None


def _decode_base64_prefix(payload: str, max_chars: int = 256) -> bytes | None:
    compact = re.sub(r"\s+", "", payload)
    if not compact:
        return None
    sample = compact[:max_chars]
    padding = "=" * (-len(sample) % 4)
    try:
        return base64.b64decode(sample + padding, validate=False)
    except (binascii.Error, ValueError):
        return None


def _guess_image_mime_from_base64(payload: str) -> str | None:
    blob = _decode_base64_prefix(payload)
    if not blob:
        return None
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if blob.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if blob.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if blob.startswith(b"BM"):
        return "image/bmp"
    if len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    if blob.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return None


def _resolve_image_mime_and_payload(
    value: str,
    format_type: str,
) -> tuple[str, str]:
    """Resolve image mime type and raw base64 payload (for base64 format)."""
    if format_type == "url":
        return _guess_image_mime_from_url(value) or "image/png", value

    data_url_mime, payload = _split_data_url(value)
    if data_url_mime and data_url_mime.startswith("image/"):
        return data_url_mime, payload

    return _guess_image_mime_from_base64(payload) or "image/png", payload


def _resolve_attachment_values(
    value: str,
    att: AssessmentAttachment,
) -> list[dict[str, Any]]:
    """Convert one dataset cell into one or more OpenAI-style input objects."""
    value = value.strip()
    if not value:
        return []

    if att.format == "url":
        values = _split_attachment_urls(value)
    else:
        values = [value]

    resolved: list[dict[str, Any]] = []
    for item_value in values:
        normalized_value = (
            _to_direct_attachment_url(item_value, att.type)
            if att.format == "url"
            else item_value
        )

        if att.type == "image":
            if att.format == "url":
                resolved.append({"type": "input_image", "image_url": normalized_value})
            else:
                mime_type, payload = _resolve_image_mime_and_payload(
                    normalized_value,
                    "base64",
                )
                resolved.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{payload}",
                    }
                )
        elif att.type == "pdf":
            if att.format == "url":
                resolved.append(
                    {
                        "type": "input_file",
                        "file_url": normalized_value,
                    }
                )
            else:
                _, payload = _split_data_url(normalized_value)
                resolved.append(
                    {
                        "type": "input_file",
                        "file_data": f"data:application/pdf;base64,{payload}",
                        "filename": "document.pdf",
                    }
                )

    return resolved


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
            input_parts.extend(_resolve_attachment_values(cell_value, att))

        if not input_parts:
            logger.warning(f"[build_openai_jsonl] Skipping empty row | idx={idx}")
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
                _split_attachment_urls(cell_value)
                if att.format == "url"
                else [cell_value]
            )

            for item_value in cell_values:
                normalized_value = (
                    _to_direct_attachment_url(item_value, att.type)
                    if att.format == "url"
                    else item_value
                )
                if att.type == "image":
                    mime_type, payload = _resolve_image_mime_and_payload(
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
                                    "data": _split_data_url(normalized_value)[1],
                                }
                            }
                        )

        if not parts:
            logger.warning(f"[build_google_jsonl] Skipping empty row | idx={idx}")
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
                "metadata": {"key": f"row_{idx}"},
                "request": request,
            }
        )

    return jsonl_data


def submit_assessment_batch(
    session: Session,
    run: AssessmentRun,
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
    attachments_raw = assessment_input.get("attachments", [])
    output_schema = assessment_input.get("output_schema")
    attachments = [AssessmentAttachment(**a) for a in attachments_raw]

    # Load dataset rows
    rows = _load_dataset_rows(session, dataset)
    if not rows:
        raise ValueError(f"Dataset {dataset.id} has no rows")

    logger.info(
        f"[submit_assessment_batch] Building JSONL | "
        f"run_id={run.id} | rows={len(rows)} | "
        f"provider={config_blob.completion.provider}"
    )

    # Determine provider and build params
    completion = config_blob.completion
    provider_name = completion.provider or "openai"

    params = dict(completion.params)
    if output_schema:
        params["output_schema"] = output_schema

    # Determine the base provider (openai or google)
    base_provider = _NATIVE_PROVIDERS.get(provider_name, "openai")

    if base_provider == "openai":
        mapped_params, warnings = map_kaapi_to_openai_params(params)
        if warnings:
            logger.info(f"[submit_assessment_batch] Mapper warnings: {warnings}")

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
            "description": f"Assessment: {run.run_name}",
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

    elif base_provider == "google":
        mapped_params, warnings = map_kaapi_to_google_params(params)
        if warnings:
            logger.info(f"[submit_assessment_batch] Mapper warnings: {warnings}")

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
            "display_name": f"assessment-{run.run_name}",
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
        f"[submit_assessment_batch] Submitted batch | "
        f"run_id={run.id} | batch_job_id={batch_job.id} | "
        f"provider={base_provider} | items={len(jsonl_data)}"
    )

    return batch_job
