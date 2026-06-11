List active models grouped by provider.

Returns a dict keyed by provider; each value is the list of that provider's active models.

Pagination (`skip` / `limit`) is applied **before** grouping — adjust `limit` if expecting many models per provider.
