"""Docs-only OpenAPI metadata for pass-through routes.

Nothing in this module validates anything. `openapi_extra` and `responses=`
only shape what Swagger/ReDoc render, which is what a proxy route needs: the
downstream service owns validation, so mirroring its schemas as Pydantic
models here would duplicate rules that drift the moment downstream changes.

Schemas registered through `register_component_schemas` are injected into
`components.schemas` by `customize_openapi_schema`, so `openapi_extra` can
`$ref` them and Swagger still renders a typed body with no local model.
"""

from typing import Any, NamedTuple

JSON_MEDIA_TYPE = "application/json"

_REGISTERED_COMPONENTS: dict[str, dict[str, Any]] = {}


class ResponseDoc(NamedTuple):
    """One documented status code: prose, an example body, optional schema."""

    description: str
    example: Any
    schema: dict[str, Any] | None = None


def register_component_schemas(schemas: dict[str, dict[str, Any]]) -> None:
    """Publish hand-written JSON Schemas under `components.schemas`."""
    _REGISTERED_COMPONENTS.update(schemas)


def inject_component_schemas(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Merge registered schemas into a generated spec.

    Generated components win on a name clash: a real Pydantic model is always
    a better description of a route than a hand-written stand-in.
    """
    components = openapi_schema.setdefault("components", {}).setdefault("schemas", {})
    for name, schema in _REGISTERED_COMPONENTS.items():
        components.setdefault(name, schema)
    return openapi_schema


def ref(name: str) -> dict[str, str]:
    """A `$ref` to a component schema, registered or FastAPI-generated."""
    return {"$ref": f"#/components/schemas/{name}"}


def infer_schema(value: Any) -> dict[str, Any]:
    """Best-effort JSON Schema from an example value."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if value is None:
        return {"nullable": True}
    if isinstance(value, list):
        return {"type": "array", "items": infer_schema(value[0]) if value else {}}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: infer_schema(item) for key, item in value.items()},
        }
    return {}


def doc(
    *,
    request_schema: dict[str, Any] | None = None,
    request: Any = None,
    request_examples: dict[str, tuple[str, Any]] | None = None,
    required: bool = True,
    responses: dict[int, ResponseDoc] | None = None,
) -> dict[str, Any]:
    """Build decorator kwargs: `@router.post("/x", **doc(...))`.

    `request_examples` maps a name to `(summary, body)` and renders as
    Swagger's example dropdown; `request` is the single-example shorthand.
    Pass `request_schema=ref(...)` to keep a typed body in the docs without a
    local Pydantic model.
    """
    kwargs: dict[str, Any] = {}

    if request_schema is not None or request is not None or request_examples:
        content: dict[str, Any] = {"schema": request_schema or infer_schema(request)}
        if request_examples:
            content["examples"] = {
                name: {"summary": summary, "value": body}
                for name, (summary, body) in request_examples.items()
            }
        elif request is not None:
            content["example"] = request
        kwargs["openapi_extra"] = {
            "requestBody": {
                "required": required,
                "content": {JSON_MEDIA_TYPE: content},
            }
        }

    if responses:
        kwargs["responses"] = {
            status: {
                "description": response.description,
                "content": {
                    JSON_MEDIA_TYPE: {
                        "schema": response.schema or infer_schema(response.example),
                        "example": response.example,
                    }
                },
            }
            for status, response in responses.items()
        }

    return kwargs
