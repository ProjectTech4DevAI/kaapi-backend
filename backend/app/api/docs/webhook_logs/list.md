List webhook callback deliveries for the current project, newest first.

Each row is one `send_callback` delivery: the URL, outcome (`DELIVERED`, `FAILED`, or `REJECTED` when the URL failed SSRF validation and no request was sent), attempt count, last HTTP status, the exact payload sent, and the receiver's truncated response. Filter by `resource_type`, `resource_id`, or `status`; the response `metadata` carries `skip`, `limit`, and `total`.
