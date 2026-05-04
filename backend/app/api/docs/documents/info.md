Retrieve all information about a given document.

When `tag` is omitted, this endpoint resolves only general documents:
documents tagged `default`. Pass an explicit
tag such as `ASSESSMENT` for tagged document surfaces.

If you set the ``include_url`` parameter to true, a signed URL will be included in the response, which is a clickable link to access the retrieved document. If you don't set it to true, the URL will not be included in the response.
