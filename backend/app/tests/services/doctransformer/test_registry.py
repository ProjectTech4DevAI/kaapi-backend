import pytest
from celery.exceptions import SoftTimeLimitExceeded
from gevent import Timeout

from app.services.doctransform.registry import (
    get_file_format,
    get_supported_transformations,
    is_transformation_supported,
    get_available_transformers,
    resolve_transformer,
    convert_document,
    TransformationError,
    TRANSFORMERS,
)


# Fixture for patching supported transformations
@pytest.fixture
def patched_transformations(monkeypatch):
    mapping = {
        ("docx", "pdf"): {"default": "pandoc", "pandoc": "pandoc"},
        ("pdf", "markdown"): {"default": "zerox", "zerox": "zerox"},
    }
    monkeypatch.setattr(
        "app.services.doctransform.registry.SUPPORTED_TRANSFORMATIONS", mapping
    )
    return mapping


def test_get_file_format_valid():
    assert get_file_format("file.pdf") == "pdf"
    assert get_file_format("file.docx") == "docx"
    assert get_file_format("file.md") == "markdown"
    assert get_file_format("file.html") == "html"
    assert get_file_format("file.json") == "json"


def test_get_file_format_json_case_insensitive():
    assert get_file_format("legacy_assistant.JSON") == "json"


def test_get_file_format_invalid():
    with pytest.raises(ValueError):
        get_file_format("file.unknown")


def test_get_supported_transformations(patched_transformations):
    supported = get_supported_transformations()
    assert ("docx", "pdf") in supported
    assert "default" in supported[("docx", "pdf")]
    assert ("pdf", "markdown") in supported
    assert "zerox" in supported[("pdf", "markdown")]


def test_is_transformation_supported(monkeypatch):
    monkeypatch.setattr(
        "app.services.doctransform.registry.SUPPORTED_TRANSFORMATIONS",
        {("docx", "pdf"): {"default": "pandoc"}},
    )
    assert is_transformation_supported("docx", "pdf")
    assert not is_transformation_supported("pdf", "docx")


def test_get_available_transformers(patched_transformations):
    transformers = get_available_transformers("docx", "pdf")
    assert "default" in transformers
    assert "pandoc" in transformers
    assert get_available_transformers("pdf", "docx") == {}


def test_resolve_transformer(patched_transformations):
    assert resolve_transformer("docx", "pdf") == "pandoc"
    assert resolve_transformer("docx", "pdf", "pandoc") == "pandoc"
    with pytest.raises(ValueError):
        resolve_transformer("docx", "pdf", "notfound")
    with pytest.raises(ValueError):
        resolve_transformer("pdf", "docx")


def test_convert_document(tmp_path, monkeypatch):
    class DummyTransformer:
        def transform(self, input_path, output_path):
            output_path.write_text("transformed")
            return output_path

    monkeypatch.setitem(TRANSFORMERS, "dummy", DummyTransformer)
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text("test")
    result = convert_document(input_file, output_file, transformer_name="dummy")
    assert result.read_text() == "transformed"

    # Transformer not found
    with pytest.raises(ValueError):
        convert_document(input_file, output_file, transformer_name="notfound")

    # Transformer raises error
    class FailingTransformer:
        def transform(self, input_path, output_path):
            raise Exception("fail")

    monkeypatch.setitem(TRANSFORMERS, "fail", FailingTransformer)
    with pytest.raises(TransformationError):
        convert_document(input_file, output_file, transformer_name="fail")

    # gevent Timeout propagates without being wrapped
    class TimeoutTransformer:
        def transform(self, input_path, output_path):
            raise Timeout()

    monkeypatch.setitem(TRANSFORMERS, "timeout", TimeoutTransformer)
    with pytest.raises(Timeout):
        convert_document(input_file, output_file, transformer_name="timeout")

    # Celery SoftTimeLimitExceeded propagates without being wrapped
    class SoftLimitTransformer:
        def transform(self, input_path, output_path):
            raise SoftTimeLimitExceeded()

    monkeypatch.setitem(TRANSFORMERS, "softlimit", SoftLimitTransformer)
    with pytest.raises(SoftTimeLimitExceeded):
        convert_document(input_file, output_file, transformer_name="softlimit")
