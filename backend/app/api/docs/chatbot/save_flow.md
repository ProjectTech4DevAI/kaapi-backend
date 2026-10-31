Save the chatbot's flow: `{nodes, edges}` in React Flow shape. Returns the saved version.

The **first** save of a chatbot creates version 1 and makes it live immediately, so the chatbot can run without a separate publish. Because it goes live, it gets the same strict validation as publish (see the publish endpoint) and its missing `config_version`s are pinned to the latest; an invalid first flow returns 422 and nothing is saved.

Every later save is a draft that needs a publish to go live: if the highest version is not live it is overwritten, otherwise a new version `max + 1` is created with `is_live: false`. Live versions are never modified, so editing never affects running conversations.

Draft saves check only the shape, so half-built flows save: each node needs `id` (1–255 chars), `type` (`input`, `output`, `coordinator`, `onboarding`, `rag`), `position {x, y}` and an object `data`; agent nodes (`coordinator`, `onboarding`, `rag`) may carry `data.config_id` (UUID), `data.config_version` (≥ 1) and `data.prompt`. Each edge needs `id`, `source`, `target`; `sourceHandle`, `targetHandle` and `data` (e.g. `intent`, `description`) are optional. Extra keys are stored and returned as sent. Returns 404 for an unknown chatbot, 409 if the flow was changed concurrently (reload and retry), and 422 for an invalid first save.
