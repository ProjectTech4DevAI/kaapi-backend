"""Export the FastAPI app's OpenAPI schema to a YAML file.

Usage:
    uv run python -m scripts.export_openapi [output_path]

Defaults to ./openapi.yaml relative to the current working directory.
"""

import sys
from pathlib import Path

import yaml

from app.main import app


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("openapi.yaml")
    schema = app.openapi()
    output.write_text(yaml.safe_dump(schema, sort_keys=False))
    print(f"[export_openapi] Wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
