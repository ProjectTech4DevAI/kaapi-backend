from unittest.mock import MagicMock

from app.crud.stats import get_daily_stats

EXPECTED_SECTIONS = {
    "LLM Calls",
    "LLM Tokens",
    "LLM Modality",
    "Jobs by Type",
    "Evaluation Runs",
    "STT Results",
    "TTS Results",
    "Assessments",
}


def test_get_daily_stats_runs_every_section_and_maps_rows():
    row = {"organization": "Acme", "project": "Alpha", "calls_7d": 5}
    session = MagicMock()
    execute = session.connection.return_value.execute
    execute.return_value.mappings.return_value.all.return_value = [row]

    stats = get_daily_stats(session=session)

    assert set(stats) == EXPECTED_SECTIONS
    assert execute.call_count == len(EXPECTED_SECTIONS)  # one query per section
    assert stats["LLM Calls"] == [row]  # _rows unpacks each mapping into a dict
