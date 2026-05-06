Setup and configure the Vector store that is pertinent to the File search
pipeline:

* Create a vector store from the document IDs you received after uploading your
  documents through the Documents module.
* Documents are automatically batched when creating the vector store to optimize
  the upload process for large document sets. A new batch is created when either
  the cumulative size reaches 30 MB of documents given to upload to a vector store
  or the document count reaches 200 files in a batch, whichever limit is hit first.
* [Deprecated] Attach the Vector Store to an OpenAI
  [Assistant]. Use
  parameters in the request body relevant to an Assistant to flesh out
  its configuration. Note that an assistant will only be created when you pass both
  "model" and "instruction" in the request body otherwise only a vector store will be
  created from the documents given.

If any step in the LLM service interaction fails, all previously created resources are cleaned up automatically. For example, if the vector store creation fails, any files already uploaded to OpenAI are removed. Failures can be caused by service downtime, invalid parameter values, or unsupported document types — the latter is especially common with PDFs that cannot be parsed.

The Vector store/assistant will be created asynchronously.
The immediate response from this endpoint is `collection_job` object which is
going to contain the collection "job ID" and status. Once the collection has
been created, information about the collection will be returned to the user via
the callback URL. If a callback URL is not provided, clients can check the
`collection job info` endpoint with the `job_id`, to retrieve
information about the created collection.
