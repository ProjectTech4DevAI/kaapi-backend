List documents uploaded to Kaapi.

When `tag` is omitted, this endpoint returns only general documents:
documents tagged `default`. Pass an explicit
tag such as `ASSESSMENT` to list documents for that tagged surface.

If you set the ``include_url`` parameter to true, a signed URL will be included in the response, which is a clickable link to access the retrieved documents. If you don't set it to true, the URL will not be included in the response.
