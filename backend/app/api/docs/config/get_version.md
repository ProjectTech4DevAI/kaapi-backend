Retrieve a specific version of a configuration.

When `tag` is omitted, this endpoint only resolves versions for general
configurations: configs tagged `default`. Pass
an explicit tag such as `ASSESSMENT` for tagged config surfaces.

Returns the complete version details including the full configuration
blob (config_blob) with all LLM parameters.
