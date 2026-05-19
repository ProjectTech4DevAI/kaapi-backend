Retrieve detailed information about a specific collection by its collection id.

**Including Documents:**

If the `include_docs` flag in the parameter is true then you will get a list of document IDs associated with a given collection as well. Note that, documents returned are not only stored by Kaapi, but also by Vector store provider.

Additionally, if you set the `include_url` parameter to true, a signed URL will be included in the response, which is a clickable link to access the retrieved document. If you don't set it to true, the URL will not be included in the response.
