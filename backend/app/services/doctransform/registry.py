from pathlib import Path
from gevent import Timeout
from celery.exceptions import SoftTimeLimitExceeded

from app.services.doctransform.transformer import Transformer
from app.services.doctransform.zerox_transformer import ZeroxTransformer


class TransformationError(Exception):
    """Raised when a document transformation fails."""


# Map transformer names to their classes
TRANSFORMERS: dict[str, type[Transformer]] = {
    "default": ZeroxTransformer,
    "zerox": ZeroxTransformer,
}

# Define supported transformations: (source_format, target_format) -> [available_transformers]
SUPPORTED_TRANSFORMATIONS: dict[tuple[str, str], dict[str, str]] = {
    ("pdf", "markdown"): {
        "default": "zerox",
        "zerox": "zerox",
    },
    # Future transformations can be added here
    # ("docx", "markdown"): {"default": "pandoc", "pandoc": "pandoc"},
    # ("html", "markdown"): {"default": "pandoc", "pandoc": "pandoc"},
}

# Map file extensions to format names
EXTENSION_TO_FORMAT: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".csv": "csv",
    ".json": "json",
}

# Map format names to file extensions
FORMAT_TO_EXTENSION: dict[str, str] = {
    "pdf": ".pdf",
    "docx": ".docx",
    "doc": ".doc",
    "html": ".html",
    "text": ".txt",
    "markdown": ".md",
    "csv": ".csv",
    "json": ".json",
}


def get_file_format(filename: str) -> str:
    """Extract format from filename extension."""
    ext = Path(filename).suffix.lower()
    format_name = EXTENSION_TO_FORMAT.get(ext)
    if not format_name:
        raise ValueError(f"Unsupported file extension: {ext}")
    return format_name


def get_supported_transformations() -> dict[tuple[str, str], set[str]]:
    """Get all supported transformation combinations."""
    return {
        key: set(transformers.keys())
        for key, transformers in SUPPORTED_TRANSFORMATIONS.items()
    }


def is_transformation_supported(source_format: str, target_format: str) -> bool:
    """Check if a transformation from source_format to target_format is supported."""
    return (source_format, target_format) in SUPPORTED_TRANSFORMATIONS


def get_available_transformers(
    source_format: str, target_format: str
) -> dict[str, str]:
    """Get available transformers for a specific transformation."""
    return SUPPORTED_TRANSFORMATIONS.get((source_format, target_format), {})


def resolve_transformer(
    source_format: str, target_format: str, transformer_name: str | None = None
) -> str:
    """
    Resolve the actual transformer to use for a transformation.
    Returns the transformer name to use.
    """
    available_transformers = get_available_transformers(source_format, target_format)

    if not available_transformers:
        raise ValueError(
            f"Transformation from {source_format} to {target_format} is not supported"
        )

    if transformer_name is None:
        transformer_name = "default"

    if transformer_name not in available_transformers:
        available = ", ".join(available_transformers.keys())
        raise ValueError(
            f"Transformer '{transformer_name}' not available for {source_format} to {target_format}. "
            f"Available: {available}"
        )

    return available_transformers[transformer_name]


def convert_document(
    input_path: Path, output_path: Path, transformer_name: str = "default"
) -> Path:
    """
    Select and run the specified transformer on the input_path, writing to output_path.
    Returns the path to the transformed file.
    """
    try:
        transformer_cls = TRANSFORMERS[transformer_name]
    except KeyError:
        available = ", ".join(TRANSFORMERS.keys())
        raise ValueError(
            f"Transformer '{transformer_name}' not found. Available: {available}"
        )

    transformer = transformer_cls()
    try:
        return transformer.transform(input_path, output_path)
    except (Timeout, SoftTimeLimitExceeded):
        raise
    except Exception as e:
        raise TransformationError(
            f"Error applying transformer '{transformer_name}': {e}"
        ) from e
