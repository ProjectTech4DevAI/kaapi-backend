Create a new version for an existing configuration.

To create a new version, provide the `config_id` in the URL path and the new
configuration parameters in the request body. The system will automatically
create a new version under the same configuration with an incremented version number.
Version numbers are automatically incremented sequentially (1, 2, 3, etc.)
and cannot be manually set or skipped.

When `tag` is omitted, this endpoint only resolves general configurations:
configs tagged `default`. Pass an explicit
tag such as `ASSESSMENT` for tagged config surfaces.

## Important
- This endpoint accepts partial updates using dict[str, Any] for config_blob.
- Only the fields that need to be updated should be provided.
- The `type` field is inherited from the existing configuration and cannot be changed. Provider and model can change between versions.
