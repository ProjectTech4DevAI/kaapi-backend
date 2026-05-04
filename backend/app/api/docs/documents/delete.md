Perform a delete of the document. This makes the
document invisible. It does not delete the document from cloud storage
or its information from the database.

If the document is part of an active collection, those collections
will be deleted using the collections delete interface. Noteably, this
means all OpenAI Vector Store's and Assistant's to which this document
belongs will be deleted.
