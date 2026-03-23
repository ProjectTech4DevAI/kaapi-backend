# Inquilab — Backend phase

---

## What backend receives

```json
{
  "experiment_name": "hindi_qa_batch",
  "dataset_id": "uuid",
  "prompt_template": "# Problem: {problem}\n# Solution: {solution}\n\nAnalyze the attached image(s).",
  "text_columns": ["problem", "solution"],
  "attachments": [
    { "column": "image_url", "type": "image" },
    { "column": "doc_url",   "type": "pdf"   }
  ],
  "configs": [
    { "config_id": "uuid-1", "config_version": 1 },
    { "config_id": "uuid-2", "config_version": 1 }
  ]
}
```

---

## Step 1: Validate and create records

- Validate dataset, configs, columns exist
- Create one **experiment** record (status: `pending`)
- Create N **runner** records — one per config (status: `pending`)
- Dispatch one Celery task
- Return `202`

---

## Step 2: Construct inputs — row by row

Each dataset row becomes one JSONL line. The construction has two stages:

### Stage A: Row → `list[QueryInput]`

For each row, build a list of `QueryInput` objects using existing models.

**Text prompt:**

Interpolate template with row values → create `TextInput`:

```
TextInput(
    type="text",
    content=TextContent(format="text", value="# Problem: पुराने समय में...\n# Solution: पुराने समय में लोग...\n\nAnalyze the attached image(s).")
)
```

**Attachments — per cell detection:**

For each attachment column, read the cell value and detect:

- **Empty** → drop, nothing added to the list
- **`str` starting with `http`** → single URL

```
ImageInput(
    type="image",
    content=[ImageContent(format="url", value="https://img.com/cave.jpg")]
)
```

- **`str` not starting with `http`** → single base64

```
ImageInput(
    type="image",
    content=[ImageContent(format="base64", value="iVBORw0KGgo...")]
)
```

- **`list[str]`** (cell starts with `[`) → parse JSON array, detect each item independently

```
PDFInput(
    type="pdf",
    content=[
        PDFContent(format="url", value="https://docs.com/a.pdf"),
        PDFContent(format="base64", value="JVBERi0xLjQ...")
    ]
)
```

Mixed formats within same list is fine — each item detected on its own.

**Result per row — `list[QueryInput]`:**

Row with all columns populated:

```
[TextInput, ImageInput, PDFInput]
```

Row where image cell is empty:

```
[TextInput, PDFInput]
```

Row where both attachment cells are empty:

```
[TextInput]
```

### Stage B: `list[QueryInput]` → `resolve_input()` → `MultiModalInput`

The existing `resolve_input()` function already handles `list[QueryInput]`. It iterates the list and collects all parts:

```
resolve_input([TextInput, ImageInput, PDFInput])
```

Internally:

- `TextInput` → `TextContent` added to parts
- `ImageInput` → `resolve_image_content()` → `list[ImageContent]` added to parts (mime_type defaulted to `image/png` if missing)
- `PDFInput` → `resolve_pdf_content()` → `list[PDFContent]` added to parts (mime_type defaulted to `application/pdf` if missing)

Output:

```
MultiModalInput(
    parts=[
        TextContent(format="text", value="# Problem: पुराने समय में..."),
        ImageContent(format="url", value="https://img.com/cave.jpg", mime_type="image/png"),
        PDFContent(format="url", value="https://docs.com/a.pdf", mime_type="application/pdf"),
        PDFContent(format="base64", value="JVBERi0xLjQ...", mime_type="application/pdf")
    ]
)
```

If the `list[QueryInput]` has only a `TextInput` (no attachments), `resolve_input()` still returns `MultiModalInput` with just a `TextContent` part. Or alternatively, if there's only one item and it's `TextInput`, it returns the plain string directly — both work with existing provider adapters.

---

## Step 3: `MultiModalInput` → Provider-specific JSONL line

Each config has a `ConfigBlob` → `CompletionConfig` → tells us the provider and params.

The provider adapter takes the `MultiModalInput` and transforms it into the provider's expected format.

**OpenAI (Responses API):**

```json
{
  "custom_id": "row_0",
  "method": "POST",
  "url": "/v1/responses",
  "body": {
    "model": "gpt-4o",
    "instructions": "...",
    "temperature": 0.1,
    "input": [
      { "type": "input_text", "text": "# Problem: पुराने समय में..." },
      { "type": "input_image", "image_url": "https://img.com/cave.jpg" },
      { "type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0xLjQ...", "filename": "doc.pdf" }
    ]
  }
}
```

**OpenAI (Chat Completions API):**

```json
{
  "custom_id": "row_0",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-4o",
    "temperature": 0.1,
    "messages": [
      { "role": "system", "content": "..." },
      { "role": "user", "content": [
        { "type": "text", "text": "# Problem: पुराने समय में..." },
        { "type": "image_url", "image_url": { "url": "https://img.com/cave.jpg" } }
      ]}
    ]
  }
}
```

**Anthropic (Message Batches):**

```json
{
  "custom_id": "row_0",
  "params": {
    "model": "claude-sonnet-4-20250514",
    "system": "...",
    "temperature": 0.1,
    "messages": [
      { "role": "user", "content": [
        { "type": "text", "text": "# Problem: पुराने समय में..." },
        { "type": "image", "source": { "type": "url", "url": "https://img.com/cave.jpg" } },
        { "type": "document", "source": { "type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQ..." } }
      ]}
    ]
  }
}
```

**Same `MultiModalInput`, different wire format.** The provider adapter reads `CompletionConfig.provider` and transforms accordingly. The `MultiModalInput.parts` list maps 1:1 to each provider's content blocks.

---

## Step 4: Build JSONL per config

The resolved `MultiModalInput` per row is constructed **once**.

For each config:

1. Load `ConfigBlob` from `config_id` + `config_version`
2. Get provider + params from `CompletionConfig`
3. For each row, wrap the `MultiModalInput` in that provider's batch format
4. Write all rows as JSONL

Same inputs, different wrappers. N configs = N JSONL files.

---

## Step 5: Submit batches

For each config/runner:

1. Get batch provider (e.g., `OpenAIBatchProvider`)
2. Upload JSONL
3. Submit → get `provider_batch_id`
4. Link to runner, update status to `processing`

All submitted within one Celery task.

---

## Step 6: Poll for completion

Poll each runner's batch status every ~30-60 seconds.

- **In progress** → update `completed_items` if available
- **Completed** → fetch results → Step 7
- **Failed** → mark runner failed, store error

---

## Step 7: Process results

Match results back to rows using `custom_id`.

Per row: store model output, input reference, latency, usage, error (if any).

Update runner status to `completed`.

---

## Step 8: Finalize

All runners done → experiment `completed` (or `failed` if all failed).

Frontend polls status, fetches results when ready.

---

## Summary

```
Step 1   Validate + create records
Step 2   Row → list[QueryInput] → resolve_input() → MultiModalInput
Step 3   MultiModalInput → provider-specific JSONL line
Step 4   Build JSONL per config (same inputs, different wrapper)
Step 5   Submit batches
Step 6   Poll
Step 7   Process results
Step 8   Finalize
```

The pipeline: **Row → QueryInput models → resolve_input() → MultiModalInput → provider adapter → JSONL line**. All existing models used as-is.