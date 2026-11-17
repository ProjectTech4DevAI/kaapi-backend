from typing import Any

import pytest

from app.api import openapi_docs
from app.api.openapi_docs import (
    JSON_MEDIA_TYPE,
    ResponseDoc,
    doc,
    hoist_request_examples,
    infer_schema,
    inject_component_schemas,
    ref,
    register_component_schemas,
)


@pytest.fixture
def clean_registry():
    """Isolate the module-level component registry.

    It is global and populated at import by `openapi_config`, so a test that
    registered into it would otherwise leak schemas into every later test.
    """
    original = dict(openapi_docs._REGISTERED_COMPONENTS)
    openapi_docs._REGISTERED_COMPONENTS.clear()
    yield openapi_docs._REGISTERED_COMPONENTS
    openapi_docs._REGISTERED_COMPONENTS.clear()
    openapi_docs._REGISTERED_COMPONENTS.update(original)


def _body(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The JSON content block of a `doc()` request body."""
    return kwargs["openapi_extra"]["requestBody"]["content"][JSON_MEDIA_TYPE]


class TestInferSchema:
    """Schemas inferred from example values."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (True, {"type": "boolean"}),
            (3, {"type": "integer"}),
            (1.5, {"type": "number"}),
            ("text", {"type": "string"}),
            (None, {"nullable": True}),
        ],
    )
    def test_scalars(self, value: Any, expected: dict[str, Any]) -> None:
        assert infer_schema(value) == expected

    def test_bool_is_not_reported_as_integer(self) -> None:
        """`bool` subclasses `int`, so the order of the checks is load-bearing."""
        assert infer_schema(False) == {"type": "boolean"}

    def test_list_takes_its_item_schema_from_the_first_element(self) -> None:
        assert infer_schema(["a", "b"]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_empty_list_leaves_items_open(self) -> None:
        assert infer_schema([]) == {"type": "array", "items": {}}

    def test_dict_recurses_into_properties(self) -> None:
        assert infer_schema({"name": "pen", "tags": [1]}) == {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "integer"}},
            },
        }

    def test_unsupported_value_yields_an_empty_schema(self) -> None:
        """Better an unconstrained schema than a spec that fails to load."""
        assert infer_schema({"a", "b"}) == {}


class TestRef:
    def test_points_at_the_components_section(self) -> None:
        assert ref("BanList") == {"$ref": "#/components/schemas/BanList"}


class TestComponentRegistry:
    def test_registered_schemas_are_injected(self, clean_registry) -> None:
        register_component_schemas({"Thing": {"type": "object"}})
        spec: dict[str, Any] = {}

        injected = inject_component_schemas(spec)

        assert injected["components"]["schemas"]["Thing"] == {"type": "object"}

    def test_injection_keeps_existing_components(self, clean_registry) -> None:
        register_component_schemas({"Thing": {"type": "object"}})
        spec = {"components": {"schemas": {"Other": {"type": "string"}}}}

        schemas = inject_component_schemas(spec)["components"]["schemas"]

        assert sorted(schemas) == ["Other", "Thing"]

    def test_generated_component_wins_on_a_name_clash(self, clean_registry) -> None:
        """A real Pydantic model always describes a route better than a stand-in."""
        register_component_schemas({"Thing": {"type": "object"}})
        spec = {"components": {"schemas": {"Thing": {"title": "generated"}}}}

        schemas = inject_component_schemas(spec)["components"]["schemas"]

        assert schemas["Thing"] == {"title": "generated"}

    def test_injection_mutates_the_spec_in_place(self, clean_registry) -> None:
        register_component_schemas({"Thing": {"type": "object"}})
        spec: dict[str, Any] = {}

        assert inject_component_schemas(spec) is spec


class TestDoc:
    def test_no_arguments_documents_nothing(self) -> None:
        assert doc() == {}

    def test_request_example_is_documented_with_an_inferred_schema(self) -> None:
        kwargs = doc(request={"name": "pen"})

        assert kwargs["openapi_extra"]["requestBody"]["required"] is True
        assert _body(kwargs) == {
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            "example": {"name": "pen"},
        }

    def test_explicit_schema_overrides_inference(self) -> None:
        kwargs = doc(request={"name": "pen"}, request_schema=ref("BanList"))

        assert _body(kwargs)["schema"] == ref("BanList")

    def test_schema_alone_documents_a_body_without_an_example(self) -> None:
        kwargs = doc(request_schema=ref("BanList"))

        assert _body(kwargs) == {"schema": ref("BanList")}

    def test_examples_render_as_a_named_map(self) -> None:
        kwargs = doc(
            request_schema=ref("BanList"),
            request_examples={"minimal": ("Smallest body", {"name": "pen"})},
        )

        assert _body(kwargs)["examples"] == {
            "minimal": {"summary": "Smallest body", "value": {"name": "pen"}}
        }
        assert "example" not in _body(kwargs)

    def test_examples_without_a_schema_infer_from_the_first_one(self) -> None:
        """Inferring from the absent `request` would document a nullable body."""
        kwargs = doc(request_examples={"minimal": ("Smallest body", {"name": "pen"})})

        assert _body(kwargs)["schema"] == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

    def test_body_can_be_marked_optional(self) -> None:
        kwargs = doc(request={"name": "pen"}, required=False)

        assert kwargs["openapi_extra"]["requestBody"]["required"] is False

    def test_response_schema_is_inferred_from_its_example(self) -> None:
        kwargs = doc(responses={404: ResponseDoc("Not found", {"error": "gone"})})

        assert kwargs["responses"][404] == {
            "description": "Not found",
            "content": {
                JSON_MEDIA_TYPE: {
                    "schema": {
                        "type": "object",
                        "properties": {"error": {"type": "string"}},
                    },
                    "example": {"error": "gone"},
                }
            },
        }

    def test_explicit_response_schema_is_kept(self) -> None:
        kwargs = doc(
            responses={
                200: ResponseDoc("The ban list", {"id": 1}, ref("BanListResponse"))
            }
        )

        content = kwargs["responses"][200]["content"][JSON_MEDIA_TYPE]
        assert content["schema"] == ref("BanListResponse")
        assert content["example"] == {"id": 1}

    def test_request_and_responses_are_documented_together(self) -> None:
        kwargs = doc(
            request={"name": "pen"},
            responses={401: ResponseDoc("Unauthorized", {"error": "nope"})},
        )

        assert sorted(kwargs) == ["openapi_extra", "responses"]

    def test_empty_responses_adds_no_key(self) -> None:
        assert doc(responses={}) == {}


def _spec(body_schema: dict[str, Any], **media: Any) -> dict[str, Any]:
    """A one-route spec whose JSON request body uses `body_schema`."""
    return {
        "paths": {
            "/things": {
                "post": {
                    "requestBody": {
                        "content": {JSON_MEDIA_TYPE: {"schema": body_schema, **media}}
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Thing": {"type": "object", "examples": [{"name": "pen"}, {"x": 1}]},
                "Legacy": {"type": "object", "example": {"name": "legacy"}},
                "Plain": {"type": "object"},
            }
        },
    }


def _hoisted(spec: dict[str, Any]) -> dict[str, Any]:
    return hoist_request_examples(spec)["paths"]["/things"]["post"]["requestBody"][
        "content"
    ][JSON_MEDIA_TYPE]


class TestHoistRequestExamples:
    def test_first_model_example_is_copied_onto_the_body(self) -> None:
        assert _hoisted(_spec(ref("Thing")))["example"] == {"name": "pen"}

    def test_singular_model_example_is_supported(self) -> None:
        assert _hoisted(_spec(ref("Legacy")))["example"] == {"name": "legacy"}

    def test_optional_body_is_unwrapped(self) -> None:
        schema = {"anyOf": [ref("Thing"), {"type": "null"}]}

        assert _hoisted(_spec(schema))["example"] == {"name": "pen"}

    def test_existing_body_examples_are_left_alone(self) -> None:
        body = _hoisted(_spec(ref("Thing"), examples={"a": {"value": {"k": 1}}}))

        assert "example" not in body

    def test_schema_without_example_adds_nothing(self) -> None:
        assert "example" not in _hoisted(_spec(ref("Plain")))

    def test_unknown_ref_is_ignored(self) -> None:
        assert "example" not in _hoisted(_spec(ref("Missing")))


class TestRequestExamplesMatchTheirModels:
    """Every documented request example must pass the route's own validation.

    A stale example ships straight to Swagger and the synced Postman
    collection, so a model change that invalidates one should fail CI.
    """

    def test_every_json_request_example_validates(self) -> None:
        from fastapi.routing import APIRoute

        from app.main import app

        spec = app.openapi()
        failures: list[str] = []
        for route in app.routes:
            if not isinstance(route, APIRoute) or route.body_field is None:
                continue
            for method in route.methods:
                operation = spec["paths"].get(route.path_format, {}).get(method.lower())
                body = (operation or {}).get("requestBody", {}).get("content", {})
                media = body.get(JSON_MEDIA_TYPE)
                if not media:
                    continue
                examples = [("example", media["example"])] if "example" in media else []
                examples += [
                    (name, item.get("value"))
                    for name, item in (media.get("examples") or {}).items()
                ]
                for name, value in examples:
                    _, errors = route.body_field.validate(value, {}, loc=("body",))
                    if errors:
                        failures.append(
                            f"{method} {route.path_format} [{name}]: {errors}"
                        )

        assert not failures, "\n".join(failures)
