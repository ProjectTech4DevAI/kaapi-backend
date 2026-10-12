from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.tests import conftest

# the fixture decorator hides the generator function; drive the raw one
seed_baseline_fn = conftest.seed_baseline.__wrapped__


def _tmp_path_factory(root: Path) -> MagicMock:
    # fixture reads getbasetemp().parent, so hand back a child of root
    factory = MagicMock(spec=pytest.TempPathFactory)
    factory.getbasetemp.return_value = root / "popen-gw0"
    return factory


class TestSeedBaselineMaster:
    def test_seeds_once_without_filelock(self, tmp_path: Path) -> None:
        with (
            patch.object(conftest, "Session"),
            patch.object(conftest, "seed_database") as seed_database,
            patch.object(conftest, "FileLock") as file_lock,
        ):
            gen = seed_baseline_fn(_tmp_path_factory(tmp_path), "master")
            next(gen)
            with suppress(StopIteration):
                next(gen)

        assert seed_database.call_count == 1
        file_lock.assert_not_called()


class TestSeedBaselineXdistWorker:
    def test_seeds_and_touches_flag_when_absent(self, tmp_path: Path) -> None:
        with (
            patch.object(conftest, "Session"),
            patch.object(conftest, "seed_database") as seed_database,
            patch.object(conftest, "FileLock") as file_lock,
        ):
            gen = seed_baseline_fn(_tmp_path_factory(tmp_path), "gw0")
            next(gen)
            with suppress(StopIteration):
                next(gen)

        assert seed_database.call_count == 1
        assert (tmp_path / "seeded").exists()
        file_lock.assert_called_once_with(tmp_path / "seed.lock")

    def test_skips_seeding_when_flag_exists(self, tmp_path: Path) -> None:
        (tmp_path / "seeded").touch()

        with (
            patch.object(conftest, "Session"),
            patch.object(conftest, "seed_database") as seed_database,
            patch.object(conftest, "FileLock") as file_lock,
        ):
            gen = seed_baseline_fn(_tmp_path_factory(tmp_path), "gw1")
            next(gen)
            with suppress(StopIteration):
                next(gen)

        seed_database.assert_not_called()
        file_lock.assert_called_once_with(tmp_path / "seed.lock")
