"""Chunk-artifact cleanup (`fast_chunks.delete_response_chunk_artifacts`).

Cleanup runs after the completed transition but before `save_score`, so it must
swallow everything — including a failure resolving cloud storage. A raise here
would flip a completed run to failed *and* cost it its score unit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlmodel import Session

from app.crud.evaluations.fast import _cleanup_response_chunks
from app.crud.evaluations.fast_chunks import delete_response_chunk_artifacts

_FAST = "app.crud.evaluations.fast"
_CHUNKS = "app.crud.evaluations.fast_chunks"


def _eval_run() -> SimpleNamespace:
    return SimpleNamespace(id=1, project_id=2, organization_id=3)


class TestCleanupResponseChunks:
    def test_storage_resolution_failure_never_escapes(self) -> None:
        with patch(
            f"{_FAST}.get_cloud_storage", side_effect=RuntimeError("no credentials")
        ):
            _cleanup_response_chunks(
                session=MagicMock(spec=Session), eval_run=_eval_run()
            )

    def test_storage_failure_skips_the_delete_pass_entirely(self) -> None:
        with (
            patch(f"{_FAST}.get_cloud_storage", side_effect=RuntimeError("boom")),
            patch(f"{_FAST}.delete_response_chunk_artifacts") as mock_delete,
        ):
            _cleanup_response_chunks(
                session=MagicMock(spec=Session), eval_run=_eval_run()
            )
        mock_delete.assert_not_called()

    def test_resolved_storage_is_handed_to_the_delete_pass(self) -> None:
        storage = MagicMock()
        with (
            patch(f"{_FAST}.get_cloud_storage", return_value=storage),
            patch(f"{_FAST}.delete_response_chunk_artifacts") as mock_delete,
        ):
            session = MagicMock(spec=Session)
            _cleanup_response_chunks(session=session, eval_run=_eval_run())
        mock_delete.assert_called_once_with(
            session=session, storage=storage, eval_run_id=1
        )


class TestDeleteResponseChunkArtifacts:
    def test_deletes_each_chunk_file_and_row(self) -> None:
        storage = MagicMock()
        jobs = [
            SimpleNamespace(raw_output_url="s3://a", id=1),
            SimpleNamespace(raw_output_url="s3://b", id=2),
        ]
        with (
            patch(f"{_CHUNKS}.list_response_chunk_jobs", return_value=jobs),
            patch(f"{_CHUNKS}.delete_batch_job") as mock_delete_job,
        ):
            delete_response_chunk_artifacts(
                session=MagicMock(spec=Session), storage=storage, eval_run_id=1
            )
        assert storage.delete.call_count == 2
        assert mock_delete_job.call_count == 2

    def test_a_row_without_an_uploaded_unit_is_still_removed(self) -> None:
        storage = MagicMock()
        jobs = [SimpleNamespace(raw_output_url=None, id=1)]
        with (
            patch(f"{_CHUNKS}.list_response_chunk_jobs", return_value=jobs),
            patch(f"{_CHUNKS}.delete_batch_job") as mock_delete_job,
        ):
            delete_response_chunk_artifacts(
                session=MagicMock(spec=Session), storage=storage, eval_run_id=1
            )
        storage.delete.assert_not_called()
        mock_delete_job.assert_called_once()

    def test_a_failing_delete_never_escapes(self) -> None:
        storage = MagicMock()
        storage.delete.side_effect = RuntimeError("s3 down")
        jobs = [SimpleNamespace(raw_output_url="s3://a", id=1)]
        with (
            patch(f"{_CHUNKS}.list_response_chunk_jobs", return_value=jobs),
            patch(f"{_CHUNKS}.delete_batch_job"),
        ):
            delete_response_chunk_artifacts(
                session=MagicMock(spec=Session), storage=storage, eval_run_id=1
            )
