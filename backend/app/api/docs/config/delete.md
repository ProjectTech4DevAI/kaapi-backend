Delete a configuration and all its versions.

This operation performs a delete, marking the configuration and all
associated versions as deleted in the database while retaining records
for audit purposes.

The lookup is scoped by `tag`. It defaults to `default`; pass
`ASSESSMENT` to delete an assessment config. A config that exists under a
different tag is reported as not found.
