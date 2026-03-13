Get details of a specific dataset by ID.

Returns comprehensive dataset information including metadata (ID, name, item counts, duplication factor), Langfuse integration details (dataset ID), and the object store URL for the CSV file.

Use `include_signed_url=true` to include a time-limited signed URL for downloading the dataset CSV directly from object storage. The signed URL expires after 1 hour.
