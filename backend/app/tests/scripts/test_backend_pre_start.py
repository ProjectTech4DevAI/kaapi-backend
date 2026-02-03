from unittest.mock import MagicMock, patch
from app.backend_pre_start import init


def test_init_success():
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.exec.return_value = None

    fake_select = MagicMock()
    with patch("app.backend_pre_start.Session", return_value=mock_session), patch(
        "app.backend_pre_start.select", return_value=fake_select
    ):
        try:
            init(MagicMock())
            connection_successful = True
        except Exception:
            connection_successful = False

        assert (
            connection_successful
        ), "The database connection should be successful and not raise an exception."
        mock_session.exec.assert_called_once_with(fake_select)
