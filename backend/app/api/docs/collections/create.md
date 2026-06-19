Setup and configure the Vector store that is a requirement for File search
pipeline:

* Create a vector store from the document IDs you received after uploading your
  documents through the Documents module.
* Given Documents are automatically batched during vector store creation to handle large uploads efficiently. A new batch starts when the total size reaches 30 MB or the file count reaches 200, whichever comes first.

If any step in the LLM service interaction fails, all previously created resources are cleaned up automatically. Failures can be caused by service downtime, invalid parameter values, or unsupported document types — the latter is especially common with PDFs that cannot be parsed.

The Vector store will be created asynchronously.
The immediate response from this endpoint is
going to contain the collection "job ID" and status. Once the collection has
been created, information about the collection will be returned to the user via
the callback URL. If a callback URL is not provided, clients can check the
`collection job info` endpoint with the `job_id`, to retrieve
information about the created collection.
