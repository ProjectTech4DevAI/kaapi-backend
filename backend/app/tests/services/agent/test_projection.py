import pytest
from pydantic import JsonValue

from app.services.agent.projection import (
    NUMERIC_MAP,
    SCALAR,
    SCALAR_LIST,
    ProjectionSpec,
    make_projector,
    project_record,
)

_ITEM_SPEC: ProjectionSpec = {"id": SCALAR, "label": SCALAR}

_SPEC: ProjectionSpec = {
    "id": SCALAR,
    "tags": SCALAR_LIST,
    "counts": NUMERIC_MAP,
    "owner": _ITEM_SPEC,
    "items": _ITEM_SPEC,
}


class TestUnnamedAndMissingFields:
    def test_unnamed_keys_are_dropped(self) -> None:
        assert project_record(
            {"id": 1, "secret": "s3://bucket/key", "note": "free text"}, _SPEC
        ) == {"id": 1}

    def test_missing_field_stays_missing(self) -> None:
        projected = project_record({"id": 1}, _SPEC)

        assert projected == {"id": 1}
        assert "tags" not in projected
        assert "owner" not in projected

    def test_explicit_null_is_kept(self) -> None:
        assert project_record({"id": None, "owner": None}, _SPEC) == {
            "id": None,
            "owner": None,
        }


class TestScalarLeaf:
    @pytest.mark.parametrize("value", ["run-1", 7, 0.5, True, None])
    def test_scalar_passes_through(self, value: JsonValue) -> None:
        assert project_record({"id": value}, _SPEC) == {"id": value}

    @pytest.mark.parametrize(
        "value",
        [{"nested": "leak"}, ["leak"]],
        ids=["dict", "list"],
    )
    def test_non_scalar_in_scalar_field_becomes_none(self, value: JsonValue) -> None:
        assert project_record({"id": value}, _SPEC) == {"id": None}


class TestScalarListLeaf:
    def test_scalar_items_kept_in_order(self) -> None:
        assert project_record({"tags": ["a", 2, 1.5, False, None]}, _SPEC) == {
            "tags": ["a", 2, 1.5, False, None]
        }

    def test_non_scalar_items_become_none(self) -> None:
        assert project_record(
            {"tags": ["a", {"secret": "x"}, ["nested"], "b"]}, _SPEC
        ) == {"tags": ["a", None, None, "b"]}

    @pytest.mark.parametrize(
        "value", ["a,b", {"a": 1}, 3], ids=["string", "dict", "number"]
    )
    def test_non_list_becomes_none(self, value: JsonValue) -> None:
        assert project_record({"tags": value}, _SPEC) == {"tags": None}


class TestNumericMapLeaf:
    def test_numbers_kept_under_their_keys(self) -> None:
        assert project_record({"counts": {"0-0.5": 3, "0.5-1": 1.5}}, _SPEC) == {
            "counts": {"0-0.5": 3, "0.5-1": 1.5}
        }

    def test_bools_strings_and_nested_values_become_none(self) -> None:
        assert project_record(
            {
                "counts": {
                    "ok": 2,
                    "flag": True,
                    "text": "leak",
                    "nested": {"n": 1},
                    "list": [1],
                    "null": None,
                }
            },
            _SPEC,
        ) == {
            "counts": {
                "ok": 2,
                "flag": None,
                "text": None,
                "nested": None,
                "list": None,
                "null": None,
            }
        }

    @pytest.mark.parametrize("value", [[1, 2], 5, "x"], ids=["list", "int", "str"])
    def test_non_dict_becomes_none(self, value: JsonValue) -> None:
        assert project_record({"counts": value}, _SPEC) == {"counts": None}


class TestNestedSpecs:
    def test_nested_dict_is_projected(self) -> None:
        assert project_record(
            {"owner": {"id": 9, "label": "x", "email": "a@b.c"}}, _SPEC
        ) == {"owner": {"id": 9, "label": "x"}}

    def test_nested_spec_applies_to_each_list_item(self) -> None:
        assert project_record(
            {
                "items": [
                    {"id": 1, "label": "a", "url": "s3://one"},
                    {"id": 2, "trace": {"q": "Q"}},
                ]
            },
            _SPEC,
        ) == {"items": [{"id": 1, "label": "a"}, {"id": 2}]}

    @pytest.mark.parametrize(
        "value", ["free text", 42, True], ids=["string", "number", "bool"]
    )
    def test_scalar_where_record_expected_becomes_none(self, value: JsonValue) -> None:
        assert project_record({"owner": value}, _SPEC) == {"owner": None}

    def test_scalar_item_in_record_list_becomes_none(self) -> None:
        assert project_record({"items": [{"id": 1}, "leak", 3]}, _SPEC) == {
            "items": [{"id": 1}, None, None]
        }


class TestTopLevel:
    def test_top_level_list_projects_each_record(self) -> None:
        assert project_record(
            [{"id": 1, "x": 1}, {"id": 2, "y": 2}], {"id": SCALAR}
        ) == [{"id": 1}, {"id": 2}]

    @pytest.mark.parametrize("value", ["raw text", 7, None], ids=["str", "int", "none"])
    def test_top_level_non_record_becomes_none(self, value: JsonValue) -> None:
        assert project_record(value, _SPEC) is None

    def test_make_projector_applies_its_spec(self) -> None:
        projector = make_projector({"id": SCALAR, "tags": SCALAR_LIST})

        assert projector({"id": 1, "tags": ["a", {"b": 1}], "drop": "me"}) == {
            "id": 1,
            "tags": ["a", None],
        }
