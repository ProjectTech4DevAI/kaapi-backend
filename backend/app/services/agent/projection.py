"""Allowlist projection of tool responses: only fields named in a spec reach the model.
"""

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import TypeAlias

from pydantic import JsonValue


class LeafKindEnum(StrEnum):
    SCALAR = "scalar"
    SCALAR_LIST = "scalar_list"
    NUMERIC_MAP = "numeric_map"


SCALAR = LeafKindEnum.SCALAR
SCALAR_LIST = LeafKindEnum.SCALAR_LIST
NUMERIC_MAP = LeafKindEnum.NUMERIC_MAP

ProjectionSpec: TypeAlias = Mapping[str, "LeafKindEnum | ProjectionSpec"]
Projector = Callable[[JsonValue], JsonValue]


def _is_scalar(value: JsonValue) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_number(value: JsonValue) -> bool:
    # bool is an int subclass; a flag is not a count.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _project_scalar(value: JsonValue) -> JsonValue:
    return value if _is_scalar(value) else None


def _project_scalar_list(value: JsonValue) -> JsonValue:
    if not isinstance(value, list):
        return None
    items: list[JsonValue] = []
    for item in value:
        items.append(_project_scalar(item))
    return items


def _project_numeric_map(value: JsonValue) -> JsonValue:
    if not isinstance(value, dict):
        return None
    numbers: dict[str, JsonValue] = {}
    for key, item in value.items():
        numbers[key] = item if _is_number(item) else None
    return numbers


def _project_leaf(value: JsonValue, kind: LeafKindEnum) -> JsonValue:
    if kind is LeafKindEnum.SCALAR:
        return _project_scalar(value)
    if kind is LeafKindEnum.SCALAR_LIST:
        return _project_scalar_list(value)
    return _project_numeric_map(value)


def _project_dict(record: dict[str, JsonValue], spec: ProjectionSpec) -> JsonValue:
    projected: dict[str, JsonValue] = {}
    for field_name, kind in spec.items():
        if field_name not in record:
            continue
        value = record[field_name]
        if isinstance(kind, LeafKindEnum):
            projected[field_name] = _project_leaf(value, kind)
        else:
            projected[field_name] = project_record(value, kind)
    return projected


def project_record(value: JsonValue, spec: ProjectionSpec) -> JsonValue:
    """Keep only spec-named fields; a list is projected item by item."""
    if isinstance(value, dict):
        return _project_dict(value, spec)
    if isinstance(value, list):
        records: list[JsonValue] = []
        for item in value:
            records.append(project_record(item, spec))
        return records
    return None


def make_projector(spec: ProjectionSpec) -> Projector:
    def _projector(value: JsonValue) -> JsonValue:
        return project_record(value, spec)

    return _projector
