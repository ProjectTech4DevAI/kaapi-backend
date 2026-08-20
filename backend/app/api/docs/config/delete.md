Delete a configuration and all its versions.

This operation performs a delete, marking the configuration and all
associated versions as deleted in the database while retaining records
for audit purposes.

The lookup is by ID alone (within your project) and is **not** tag-scoped —
there is no `tag` parameter. A `default` or `ASSESSMENT` config is deleted by
its ID either way, along with all of its versions. Only a config that does not
exist in your project returns 404.
