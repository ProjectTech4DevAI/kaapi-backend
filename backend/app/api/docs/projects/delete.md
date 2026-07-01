Delete a project. **Requires superuser access.**

Supports two delete modes, selected by an optional request body:

```json
{ "hard_delete": false }
```

- **`hard_delete: false` (default)** — *Soft delete.* The project is marked **inactive**. No data is removed, so the project can be reactivated later. It simply stops appearing in listings and can no longer be used. Omitting the body entirely also performs a soft delete.
- **`hard_delete: true`** — *Permanent delete.* The project and everything owned by it — collections, documents, credentials, assistants, fine-tunings, conversations, and user-project mappings — are permanently removed. **This cannot be undone.**

In both modes, user **accounts are never deleted** (a user may belong to other projects). Any user left without an active project afterwards is marked **inactive** and can no longer log in until they are added to an active project again.
