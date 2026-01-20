Remove a collection from the platform. This is a two step process:

1. Delete all resources that were allocated: file(s), the Vector
   Store, and the Assistant.
2. Delete the collection entry from the kaapi database.

No action is taken on the documents themselves: the contents of the
documents that were a part of the collection remain unchanged, those
documents can still be accessed via the documents endpoints. The response from this
endpoint will be a `collection_job` object which will contain the collection `job_id` and
status. When you take the id returned and use the `collection job info` endpoint,
if the job is successful, you will get the status as successful.
Additionally, if a `callback_url` was provided in the request body,
you will receive a message indicating whether the deletion was successful or if it failed.
