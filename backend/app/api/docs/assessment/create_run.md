Start an assessment across one or more stored config versions.

Creates an assessment and one child assessment run per config, then submits each
run to batch processing.

Optional `system_instruction` is forwarded into each generated provider request
as the system/developer instruction for that assessment run.
