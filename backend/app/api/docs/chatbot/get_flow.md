Get a version of the chatbot's React Flow graph: `{id, chatbot_id, version, is_live, nodes, edges, inserted_at, updated_at}`.

With `?version=N` that version is returned. Without it, the draft is returned if one exists (the highest version when it is not live), otherwise the live version. `nodes` and `edges` come back exactly as saved, with React Flow keys (`sourceHandle`, `targetHandle`, and UI-only keys such as `measured`). Agent nodes of a published version carry the pinned `data.config_version`.

Returns 404 if the chatbot doesn't exist in this project, has no flow yet, or has no such version.
