Retrieve a specific configuration by its ID.

Returns the configuration metadata including name, description, and
timestamps. This endpoint provides configuration-level details but does
not include version information.

The lookup is scoped by `tag`. It defaults to `default`; pass
`ASSESSMENT` to fetch an assessment config. A config that exists under a
different tag is reported as not found.
