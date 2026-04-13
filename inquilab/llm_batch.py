"""Batch API clients for OpenAI and Gemini. Submit bulk requests, poll, return results."""

from __future__ import annotations

import base64
import json
import time
import tempfile
import logging
import httpx
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =====================================================================
#  Shared helpers
# =====================================================================


def _download_and_compress(url: str, max_size: int = 768, quality: int = 70) -> tuple[str, str] | None:
    """Download image, compress to JPEG, return (base64_data, mime_type) or None on failure."""
    from PIL import Image

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        logger.warning(f"[_download_and_compress] {resp.status_code} - {url}")
        return None
    except Exception as e:
        logger.warning(f"[_download_and_compress] {type(e).__name__} - {url}")
        return None

    try:
        img = Image.open(BytesIO(resp.content))
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
    except Exception as e:
        logger.warning(f"[_download_and_compress] Failed to process image {url}: {e}")
        return None


def _download_images_parallel(
    urls: list[str], max_workers: int = 20,
) -> dict[str, tuple[str, str] | None]:
    """Download and compress many images in parallel. Returns {url: (base64, mime) or None}."""
    from concurrent.futures import ThreadPoolExecutor

    unique_urls = list(set(urls))
    if not unique_urls:
        return {}

    print(f"  Downloading & compressing {len(unique_urls)} unique images ({max_workers} threads)...", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_download_and_compress, unique_urls))

    cache = dict(zip(unique_urls, results))
    valid = sum(1 for v in cache.values() if v is not None)
    skipped = len(unique_urls) - valid
    print(f"  Download done: {valid} valid, {skipped} skipped", flush=True)

    return cache


def _validate_image_url(url: str, retries: int = 2) -> bool:
    """Check if image URL is reachable — HEAD first, fall back to 1-byte range GET. Retries on timeout."""
    for attempt in range(retries + 1):
        try:
            resp = httpx.head(url, follow_redirects=True, timeout=15)
            if resp.status_code == 200:
                return True
            if resp.status_code in (403, 405):
                resp = httpx.get(url, follow_redirects=True, timeout=15, headers={"Range": "bytes=0-0"})
                return resp.status_code in (200, 206)
            return False
        except httpx.TimeoutException:
            if attempt < retries:
                continue
            return False
        except Exception:
            return False
    return False


def _validate_image_urls_parallel(
    urls: list[str], max_workers: int = 20,
) -> dict[str, bool]:
    """Validate many image URLs in parallel. Returns {url: is_reachable}."""
    from concurrent.futures import ThreadPoolExecutor

    unique_urls = list(set(urls))
    if not unique_urls:
        return {}

    print(f"  Validating {len(unique_urls)} unique image URLs ({max_workers} threads)...", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_validate_image_url, unique_urls))

    valid = sum(results)
    skipped = len(unique_urls) - valid
    print(f"  Validation done: {valid} valid, {skipped} skipped", flush=True)

    return dict(zip(unique_urls, results))


def _strip_additional_properties(schema: dict) -> dict:
    schema = dict(schema)
    schema.pop("additionalProperties", None)
    if "properties" in schema:
        schema["properties"] = {
            k: _strip_additional_properties(v) if isinstance(v, dict) else v
            for k, v in schema["properties"].items()
        }
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _strip_additional_properties(schema["items"])
    return schema


# =====================================================================
#  OpenAI Batch API
# =====================================================================

def create_openai_batch_requests(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Build OpenAI batch request objects.

    Each row dict must have:
      - custom_id, model, system_prompt, user_text
      - image_urls: list[str] | None
      - output_schema: dict | None
      - temperature: float

    Returns:
      (requests, skipped_images) where skipped_images maps custom_id -> list of unreachable URLs
    """
    # Download and compress all images in parallel — embed as base64 data URIs
    all_urls = []
    for row in rows:
        if row.get("image_urls"):
            all_urls.extend(row["image_urls"])
    image_cache = _download_images_parallel(all_urls) if all_urls else {}

    requests = []
    skipped_images: dict[str, list[str]] = {}

    for row in rows:
        content: list[dict] = [{"type": "text", "text": row["user_text"]}]
        custom_id = row["custom_id"]

        if row.get("image_urls"):
            for url in row["image_urls"]:
                cached = image_cache.get(url)
                if cached:
                    b64_data, mime_type = cached
                    data_uri = f"data:{mime_type};base64,{b64_data}"
                    content.append({"type": "image_url", "image_url": {"url": data_uri}})
                else:
                    skipped_images.setdefault(custom_id, []).append(url)
                    logger.warning(f"[create_openai_batch_requests] Skipping unreachable image: {url}")

        body: dict[str, Any] = {
            "model": row["model"],
            "messages": [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": content},
            ],
            "temperature": row.get("temperature", 0.4),
        }

        if row.get("output_schema"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "strict": True,
                    "schema": row["output_schema"],
                },
            }

        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    return requests, skipped_images


MAX_BATCH_FILE_SIZE = 150 * 1024 * 1024  # 150MB safe limit (OpenAI max is 200MB)


def _chunk_requests_by_size(
    requests: list[dict[str, Any]],
    max_bytes: int = MAX_BATCH_FILE_SIZE,
) -> list[list[dict[str, Any]]]:
    """Split requests into chunks that each fit under max_bytes when serialized to JSONL."""
    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_size = 0

    for req in requests:
        line_size = len(json.dumps(req).encode("utf-8")) + 1  # +1 for newline
        if current_chunk and current_size + line_size > max_bytes:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(req)
        current_size += line_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _submit_single_openai_batch(
    client: Any,
    chunk: list[dict[str, Any]],
    chunk_idx: int,
    total_chunks: int,
) -> str:
    """Upload one chunk and create a batch job. Returns batch ID."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for req in chunk:
        tmp.write(json.dumps(req) + "\n")
    tmp.close()
    jsonl_path = tmp.name

    label = f"[Chunk {chunk_idx + 1}/{total_chunks}]"
    print(f"  {label} Uploading batch file ({len(chunk)} requests)...")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  {label} File uploaded: {file_obj.id}")

    Path(jsonl_path).unlink(missing_ok=True)

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  {label} Batch created: {batch.id}")
    return batch.id


def _poll_openai_batches(
    client: Any,
    batch_ids: list[str],
    poll_interval: int = 30,
) -> list[Any]:
    """Poll multiple batch jobs until all finish. Returns list of completed batch objects."""
    pending = set(batch_ids)
    finished: dict[str, Any] = {}

    while pending:
        for batch_id in list(pending):
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else "?"
            failed = batch.request_counts.failed if batch.request_counts else 0

            print(f"  [{batch_id[:20]}] {status} | {completed}/{total} done | {failed} failed", flush=True)

            if status in ("completed", "failed", "expired", "cancelled"):
                finished[batch_id] = batch
                pending.discard(batch_id)

        if pending:
            time.sleep(poll_interval)

    return [finished[bid] for bid in batch_ids]


def _collect_openai_batch_results(
    client: Any,
    batch: Any,
) -> dict[str, dict[str, Any] | None]:
    """Download and parse results from a completed batch."""
    results: dict[str, dict[str, Any] | None] = {}

    if batch.status != "completed":
        print(f"  ERROR: Batch {batch.id} ended with status '{batch.status}'")
        if batch.errors and batch.errors.data:
            for err in batch.errors.data:
                print(f"    - {err.code}: {err.message}")
        return results

    output_file_id = batch.output_file_id
    if not output_file_id:
        print(f"  ERROR: No output file for batch {batch.id}")
        return results

    print(f"  Downloading results from {output_file_id}...")
    content = client.files.content(output_file_id)

    for line in content.text.strip().split("\n"):
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        response = entry.get("response", {})
        body = response.get("body", {})

        if response.get("status_code") == 200:
            choices = body.get("choices", [])
            if choices:
                raw = choices[0].get("message", {}).get("content", "")
                try:
                    results[custom_id] = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"  WARNING: Failed to parse JSON for {custom_id}")
                    results[custom_id] = None
            else:
                results[custom_id] = None
        else:
            error = body.get("error", {})
            print(f"  WARNING: Request {custom_id} failed: {error.get('message', 'unknown error')}")
            results[custom_id] = None

    return results


def submit_openai_batch(
    api_key: str,
    requests: list[dict[str, Any]],
    poll_interval: int = 30,
) -> dict[str, dict[str, Any] | None]:
    """Submit batch to OpenAI (auto-chunks if too large), poll, return {custom_id: parsed_json}."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    chunks = _chunk_requests_by_size(requests)
    print(f"  Split into {len(chunks)} chunk(s)")

    # Submit all chunks
    batch_ids = []
    for i, chunk in enumerate(chunks):
        batch_id = _submit_single_openai_batch(client, chunk, i, len(chunks))
        batch_ids.append(batch_id)

    # Poll all batches
    print(f"\n  Polling {len(batch_ids)} batch(es)...")
    finished_batches = _poll_openai_batches(client, batch_ids, poll_interval)

    # Collect and merge results
    all_results: dict[str, dict[str, Any] | None] = {}
    for batch in finished_batches:
        chunk_results = _collect_openai_batch_results(client, batch)
        all_results.update(chunk_results)

    return all_results


# =====================================================================
#  Gemini Flex API (synchronous, 50% cheaper, concurrent)
# =====================================================================

def submit_gemini_flex(
    api_key: str,
    model: str,
    rows: list[dict[str, Any]],
    max_workers: int = 10,
    max_retries: int = 3,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, list[str]]]:
    """Process requests via Gemini Flex tier concurrently.

    Args:
        api_key: Gemini API key
        model: model name (e.g. "gemini-2.5-flash")
        rows: list of dicts with custom_id, system_prompt, user_text, image_urls, output_schema, temperature
        max_workers: concurrent threads (default 10)
        max_retries: retries on 503/429 with exponential backoff (default 3)

    Returns:
        (results, skipped_images) where results maps custom_id -> parsed JSON
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Pre-validate all image URLs in parallel
    all_urls = []
    for row in rows:
        if row.get("image_urls"):
            all_urls.extend(row["image_urls"])
    url_status = _validate_image_urls_parallel(all_urls) if all_urls else {}

    skipped_images: dict[str, list[str]] = {}

    def _guess_mime(url: str) -> str:
        ext = url.rsplit(".", 1)[-1].lower() if "." in url else "jpeg"
        return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    def _call_flex(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        custom_id = row["custom_id"]

        parts: list[dict] = [{"text": row["user_text"]}]
        if row.get("image_urls"):
            for url in row["image_urls"]:
                if url_status.get(url, False):
                    parts.append({"file_data": {"file_uri": url, "mime_type": _guess_mime(url)}})
                else:
                    skipped_images.setdefault(custom_id, []).append(url)

        body: dict[str, Any] = {
            "contents": [{"parts": parts, "role": "user"}],
            "system_instruction": {"parts": [{"text": row["system_prompt"]}]},
            "generationConfig": {
                "temperature": row.get("temperature", 0.4),
            },
            "service_tier": "flex",
        }

        if row.get("output_schema"):
            clean_schema = _strip_additional_properties(row["output_schema"])
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = clean_schema

        for attempt in range(max_retries + 1):
            try:
                resp = httpx.post(api_url, json=body, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return custom_id, json.loads(raw_text)
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", 0)
                is_retryable = (
                    status in (429, 503)
                    or isinstance(e, (httpx.TimeoutException, httpx.ConnectError))
                    or "nodename" in str(e).lower()
                )
                if is_retryable and attempt < max_retries:
                    delay = 5 * (2 ** attempt)
                    logger.warning(f"[gemini_flex] {custom_id} retry {attempt + 1}/{max_retries} in {delay}s: {status or type(e).__name__}")
                    time.sleep(delay)
                else:
                    logger.warning(f"[gemini_flex] {custom_id} failed: {status or type(e).__name__}")
                    return custom_id, None
                return custom_id, None

        return custom_id, None

    total = len(rows)
    print(f"  Gemini Flex: processing {total} requests ({max_workers} concurrent)...", flush=True)

    results: dict[str, dict[str, Any] | None] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call_flex, row): row["custom_id"] for row in rows}

        for future in as_completed(futures):
            custom_id, result = future.result()
            results[custom_id] = result
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} done", flush=True)

    succeeded = sum(1 for v in results.values() if v is not None)
    failed = total - succeeded
    print(f"  Flex done: {succeeded} succeeded, {failed} failed", flush=True)

    return results, skipped_images
