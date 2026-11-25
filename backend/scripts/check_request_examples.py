"""Fail when a request payload changed but its examples did not.

Compares two exported OpenAPI specs (base branch vs PR head). For every
operation present in both, the JSON request-body schema is resolved
($refs inlined, descriptions dropped) and compared. If the payload changed
while the operation's request examples stayed identical, the examples in
`app/api/docs/<module>/examples.py` were most likely forgotten.

Usage:
    python scripts/check_request_examples.py base-openapi.yaml head-openapi.yaml
"""

import sys
from pathlib import Path
from typing import Any

import yaml

JSON_MEDIA_TYPE = "application/json"
METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
# Doc-only keys: changing them does not change the payload.
IGNORED_KEYS = {"description", "title", "example", "examples", "deprecated"}


def load(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def resolve(node: Any, components: dict[str, Any], seen: tuple[str, ...] = ()) -> Any:
    """Inline $refs and drop doc-only keys; a recursive ref stays a ref."""
    if isinstance(node, list):
        return [resolve(item, components, seen) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in seen or name not in components:
            return {"$ref": name}
        return resolve(components[name], components, (*seen, name))
    resolved = {
        key: resolve(value, components, seen)
        for key, value in node.items()
        if key not in IGNORED_KEYS and key != "properties"
    }
    if isinstance(node.get("properties"), dict):
        resolved["properties"] = {
            name: resolve(sub, components, seen)
            for name, sub in node["properties"].items()
        }
    return resolved


def request_bodies(spec: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """`"METHOD /path" -> (resolved body schema, its examples)`."""
    components = spec.get("components", {}).get("schemas", {})
    bodies = {}
    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if method not in METHODS or not isinstance(operation, dict):
                continue
            media = (
                operation.get("requestBody", {}).get("content", {}).get(JSON_MEDIA_TYPE)
            )
            if not media or "schema" not in media:
                continue
            examples = (media.get("examples"), media.get("example"))
            bodies[f"{method.upper()} {path}"] = (
                resolve(media["schema"], components),
                examples,
            )
    return bodies


def field_paths(schema: Any, prefix: str = "") -> dict[str, Any]:
    """Field path -> its schema, for naming what changed in a payload."""
    out: dict[str, Any] = {}
    if not isinstance(schema, dict):
        return out
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(key, []):
            out.update(field_paths(branch, prefix))
    if isinstance(schema.get("items"), dict):
        out.update(field_paths(schema["items"], f"{prefix}[]"))
    for name, sub in (schema.get("properties") or {}).items():
        path = f"{prefix}.{name}" if prefix else name
        out[path] = sub
        out.update(field_paths(sub, path))
    return out


def describe(base_schema: Any, head_schema: Any) -> str:
    base, head = field_paths(base_schema), field_paths(head_schema)
    added = sorted(head.keys() - base.keys())
    removed = sorted(base.keys() - head.keys())
    differing = {p for p in base.keys() & head.keys() if base[p] != head[p]}
    reported = differing | set(added) | set(removed)
    # A parent differs whenever a child does; name only the innermost field.
    changed = sorted(
        path
        for path in differing
        if not any(
            other.startswith(path + ".") or other.startswith(path + "[")
            for other in reported
        )
    )
    parts = [
        f"{label}: {', '.join(paths)}"
        for label, paths in (
            ("added", added),
            ("removed", removed),
            ("changed", changed),
        )
        if paths
    ]
    return "; ".join(parts) or "schema constraints changed"


def main() -> int:
    base_spec, head_spec = load(sys.argv[1]), load(sys.argv[2])
    base, head = request_bodies(base_spec), request_bodies(head_spec)

    stale = []
    for operation, (head_schema, head_examples) in sorted(head.items()):
        if operation not in base:
            continue
        base_schema, base_examples = base[operation]
        if head_schema != base_schema and head_examples == base_examples:
            stale.append((operation, describe(base_schema, head_schema)))

    if not stale:
        print("Every changed request payload has updated examples.")
        return 0

    for operation, change in stale:
        print(
            f"::error title=Request example not updated::{operation} payload "
            f"changed ({change}) but its request examples did not. Update the "
            "matching constant in app/api/docs/<module>/examples.py."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
