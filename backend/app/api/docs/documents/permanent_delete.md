Permanently delete a document from cloud storage.

This operation marks the document as deleted in the database while retaining its metadata. However, the actual file is
permanently deleted from cloud storage (e.g., S3) and cannot be recovered. Only the database record remains for reference
purposes.

If the document belongs to any active collections, those collections will also be deleted. This includes all associated knowledge bases — for example, any OpenAI vector stores that were created through this platform with this document.
