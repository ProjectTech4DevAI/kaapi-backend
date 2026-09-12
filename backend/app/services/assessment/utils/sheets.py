"""Excel sheet cleaning shared by the upload count, the preview and the run loader."""

from collections.abc import Iterable


def _cell(value: object) -> str:
    return "" if value is None else str(value).strip()


def clean_sheet(
    header: Iterable[object], rows: Iterable[Iterable[object]]
) -> tuple[list[str], list[list[str]]]:
    """Drop blank rows, then columns with no header or no values; cells come back stripped.

    Sheets pad with formatted-but-empty cells, so 1000 rows x 17 columns with 100 x 10
    filled comes back as exactly 100 x 10.
    """
    headers = [_cell(name) for name in header]
    width = len(headers)

    filled: list[list[str]] = []
    for row in rows:
        cells = [_cell(value) for value in row][:width]
        cells += [""] * (width - len(cells))
        if any(cells):
            filled.append(cells)

    keep = [
        idx
        for idx, name in enumerate(headers)
        if name and any(row[idx] for row in filled)
    ]
    return [headers[idx] for idx in keep], [
        [row[idx] for idx in keep] for row in filled
    ]
