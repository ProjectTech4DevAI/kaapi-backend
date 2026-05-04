This operation marks the document as deleted in the database while retaining its metadata. However, the actual file is
permanently deleted from cloud storage (e.g., S3) and cannot be recovered. Only the database record remains for reference
purposes.

When `tag` is omitted, this endpoint resolves only general documents:
documents tagged `default`. Pass an explicit
tag such as `ASSESSMENT` for tagged document surfaces.

If the document is part of an active collection, those collections
will be deleted using the collections delete interface. Noteably, this
means all OpenAI Vector Store's and Assistant's to which this document
belongs will be deleted.
