Retrieve a specific configuration by its ID.

Returns the configuration metadata including name, description, and
timestamps. This endpoint provides configuration-level details but does
not include version information.

The lookup is by ID alone (within your project) and is **not** tag-scoped —
there is no `tag` parameter. A config is returned regardless of whether it is a
`default` or `ASSESSMENT` config; its own tag is included in the response. Only a
config that does not exist in your project returns 404.
