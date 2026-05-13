Perform a delete of the document.

This marks the document as deleted and hides it from all API responses. It does not delete the document
from cloud storage or its information from the database.

If the document belongs to any active collections, those collections will also be deleted. This includes all associated knowledge bases — for example, any OpenAI vector stores that were created through this platform with this document.
