query: Search string used for partial matching across configurations
skip: Number of records to skip for pagination (default: 0)
limit: Maximum number of records to return (default: 100, max: 100)

Retrieve all configurations for the current project.

When `tag` is omitted, this endpoint returns only general configurations:
configs tagged `default`. Pass an explicit
tag such as `ASSESSMENT` to list configs for that tagged surface.

Returns a paginated list of configurations ordered by most recently updated
first. Each configuration includes metadata (name, description, timestamps)
but excludes version details for performance.
