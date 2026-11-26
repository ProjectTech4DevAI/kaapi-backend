Permanently delete a document.

The file is removed from cloud storage (e.g., S3) and the database record is deleted. This cannot be undone. Records
that depend on the document, such as its transformed documents, transformation jobs, fine-tuning jobs and model
evaluations, are deleted along with it.

Collections and knowledge bases that contain this document are not touched. Only active documents can be permanently
deleted; a document that was already soft deleted returns 404.
