Upload a document to Kaapi.

- If only a file is provided, the document will be uploaded and stored, and its ID will be returned.
- If a target format is specified, a transformation job will also be created to transform document into target format in the background. The response will include both the uploaded document details and information about the transformation job.
- If a callback URL is provided, you will receive a notification at that URL once the document transformation job is completed.

### File Size Restrictions

- **Maximum file size**: 512MB (configurable via `MAX_DOCUMENT_UPLOAD_SIZE_MB` environment variable)
- Files exceeding the size limit will be rejected with a 413 (Payload Too Large) error
- Empty files will be rejected with a 422 (Unprocessable Entity) error

### Supported Transformations

The following (source_format → target_format) transformations are supported:

- pdf → markdown
  - zerox

### Transformers

Available transformer names and their implementations, default transformer is zerox:

- `zerox`
