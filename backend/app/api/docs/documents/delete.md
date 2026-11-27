Soft delete a document.

This marks the document as deleted and stops returning it from document endpoints and from collection document
listings. The file stays in cloud storage and the database record is kept.

Collections and knowledge bases that contain this document are not touched, so its content stays searchable in them
until the collection is deleted.
