import logging

from asyncio import Runner, wait_for
from pathlib import Path
from pyzerox import zerox

from app.services.doctransform.transformer import Transformer

logger = logging.getLogger(__name__)


class ZeroxTransformer(Transformer):
    """
    Transformer that uses zerox to extract content from PDFs.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def transform(self, input_path: Path, output_path: Path) -> Path:
        logger.info(f"ZeroxTransformer Started: (model={self.model})")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            with Runner() as runner:
                result = runner.run(
                    wait_for(
                        zerox(
                            file_path=str(input_path),
                            model=self.model,
                        ),
                        timeout=10 * 60,  # 10 minutes
                    )
                )
        except TimeoutError:
            logger.error(
                f"ZeroxTransformer timed out for {input_path} (model={self.model})"
            )
            raise RuntimeError(
                f"ZeroxTransformer PDF extraction timed out after {10*60} seconds for {input_path}"
            )
        except Exception as e:
            logger.error(
                f"ZeroxTransformer failed for {input_path}: {e}\n"
                "This may be due to a missing Poppler installation or a corrupt PDF file.",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to extract content from PDF. "
                f"Check that Poppler is installed and in your PATH. Original error: {e}"
            ) from e

        if result is None or not hasattr(result, "pages") or result.pages is None:
            raise RuntimeError(
                "Zerox returned no pages. This may indicate a PDF/image conversion failure "
                "(is Poppler installed and in PATH?)"
            )

        with output_path.open("w", encoding="utf-8") as output_file:
            for page in result.pages:
                if not getattr(page, "content", None):
                    continue
                output_file.write(page.content)
                output_file.write("\n\n")

        logger.info(
            f"[ZeroxTransformer.transform] Transformation completed, output written to: {output_path}"
        )
        return output_path
