Remove a collection from the platform.

No action is taken on the documents themselves: the contents of the
documents that were a part of the collection remain unchanged, those
documents can still be accessed via the documents endpoints. The endpoint returns the job ID and status of the collection delete operation. When you take the id returned and use the `collection job info` endpoint,
if the job is successful, you will get the status as successful.
Additionally, if a `callback_url` was provided in the request body,
you will receive a message indicating whether the deletion was successful or if it failed.
