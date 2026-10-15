"""Run-level latency rollup (`attach_latency`) onto `eval_run.latency`.

Pure in-memory: `attach_latency` takes no session, so a transient
`EvaluationRun` is enough; persistence is covered by the fast-pipeline tests.
"""

import logging

import pytest

from app.crud.evaluations.latency import attach_latency
from app.models import EvaluationRun

LOG_PREFIX = "[test]"


def _run(latency: dict | None = None) -> EvaluationRun:
    return EvaluationRun(latency=latency)


class TestStageAverage:
    def test_response_stage_mean_and_overall(self) -> None:
        run = _run()
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            response_results=[{"latency_ms": 100.0}, {"latency_ms": 300.0}],
        )
        assert run.latency == {
            "response": {"avg_latency_ms": 200.0, "call_count": 2},
            "avg_latency_ms": 200.0,
            "call_count": 2,
        }

    def test_failed_and_untimed_rows_are_excluded(self) -> None:
        run = _run()
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            response_results=[
                {"latency_ms": 100.0},
                {"failed": True, "latency_ms": 900.0},
                {},
                {"failed": False, "latency_ms": None},
            ],
        )
        assert run.latency == {
            "response": {"avg_latency_ms": 100.0, "call_count": 1},
            "avg_latency_ms": 100.0,
            "call_count": 1,
        }

    def test_embedding_stage_read_from_embedding_result_dicts(self) -> None:
        run = _run()
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            embedding_results=[
                {
                    "item_id": "item_0_0",
                    "output_embedding": [0.1],
                    "ground_truth_embedding": [0.2],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                    "failed": False,
                    "latency_ms": 40.0,
                },
                {
                    "item_id": "item_1_0",
                    "output_embedding": None,
                    "ground_truth_embedding": None,
                    "usage": {},
                    "failed": True,
                    "error": "boom",
                    "latency_ms": None,
                },
            ],
        )
        assert run.latency == {
            "embedding": {"avg_latency_ms": 40.0, "call_count": 1},
            "avg_latency_ms": 40.0,
            "call_count": 1,
        }

    def test_rounds_to_one_decimal(self) -> None:
        run = _run()
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            response_results=[{"latency_ms": 100.04}, {"latency_ms": 100.06}],
        )
        assert run.latency["response"]["avg_latency_ms"] == 100.1


class TestNoQualifyingRows:
    @pytest.mark.parametrize(
        "rows",
        [[], [{"failed": True, "latency_ms": 50.0}], [{}, {"latency_ms": None}]],
    )
    def test_latency_stays_null(self, rows: list[dict]) -> None:
        run = _run()
        attach_latency(eval_run=run, log_prefix=LOG_PREFIX, response_results=rows)
        assert run.latency is None

    def test_no_lists_passed_leaves_null(self) -> None:
        run = _run()
        attach_latency(eval_run=run, log_prefix=LOG_PREFIX)
        assert run.latency is None


class TestOverall:
    def test_overall_is_call_weighted_not_mean_of_means(self) -> None:
        run = _run()
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            response_results=[{"latency_ms": 100.0}, {"latency_ms": 300.0}],
            judge_results=[{"latency_ms": 500.0}],
        )
        # A mean of stage means would give 350.0.
        assert run.latency == {
            "response": {"avg_latency_ms": 200.0, "call_count": 2},
            "judge": {"avg_latency_ms": 500.0, "call_count": 1},
            "avg_latency_ms": 300.0,
            "call_count": 3,
        }

    def test_overall_uses_unrounded_stage_sums(self) -> None:
        run = _run()
        # (1.04 + 1.48) / 2 = 1.26 -> 1.3. Averaging the rounded stage values
        # instead would give (1.0 + 1.5) / 2 = 1.25, and (1.0 + 1.48) / 2 = 1.24 -> 1.2.
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            response_results=[{"latency_ms": 1.04}],
            embedding_results=[{"latency_ms": 1.48}],
        )
        assert run.latency == {
            "response": {"avg_latency_ms": 1.0, "call_count": 1},
            "embedding": {"avg_latency_ms": 1.5, "call_count": 1},
            "avg_latency_ms": 1.3,
            "call_count": 2,
        }


class TestPartialReattach:
    def test_judge_only_preserves_response_and_recomputes_overall(self) -> None:
        run = _run(
            latency={
                "response": {"avg_latency_ms": 200.0, "call_count": 2},
                "avg_latency_ms": 200.0,
                "call_count": 2,
            }
        )
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            judge_results=[{"latency_ms": 500.0}],
        )
        assert run.latency == {
            "response": {"avg_latency_ms": 200.0, "call_count": 2},
            "judge": {"avg_latency_ms": 500.0, "call_count": 1},
            "avg_latency_ms": 300.0,
            "call_count": 3,
        }

    def test_preserved_stage_avg_is_emitted_verbatim(self) -> None:
        # 33.33 has more decimals than the rollup emits: a preserved stage must
        # not be re-rounded, only fed into the overall as avg * count.
        run = _run(
            latency={
                "response": {"avg_latency_ms": 33.33, "call_count": 3},
                "avg_latency_ms": 33.33,
                "call_count": 3,
            }
        )
        attach_latency(
            eval_run=run,
            log_prefix=LOG_PREFIX,
            judge_results=[{"latency_ms": 100.0}],
        )
        # (33.33 * 3 + 100.0) / 4 = 49.9975 -> 50.0
        assert run.latency == {
            "response": {"avg_latency_ms": 33.33, "call_count": 3},
            "judge": {"avg_latency_ms": 100.0, "call_count": 1},
            "avg_latency_ms": 50.0,
            "call_count": 4,
        }

    def test_rerun_with_same_rows_is_idempotent(self) -> None:
        rows = [{"latency_ms": 100.0}, {"latency_ms": 300.0}]
        run = _run()
        attach_latency(eval_run=run, log_prefix=LOG_PREFIX, response_results=rows)
        attach_latency(eval_run=run, log_prefix=LOG_PREFIX, judge_results=[])
        attach_latency(eval_run=run, log_prefix=LOG_PREFIX, response_results=rows)
        assert run.latency == {
            "response": {"avg_latency_ms": 200.0, "call_count": 2},
            "avg_latency_ms": 200.0,
            "call_count": 2,
        }

    def test_reassigns_rather_than_mutating_in_place(self) -> None:
        # In-place mutation of a JSONB dict isn't flushed by SQLAlchemy.
        before = {
            "response": {"avg_latency_ms": 200.0, "call_count": 2},
            "avg_latency_ms": 200.0,
            "call_count": 2,
        }
        run = _run(latency=before)
        attach_latency(
            eval_run=run, log_prefix=LOG_PREFIX, judge_results=[{"latency_ms": 500.0}]
        )
        assert run.latency is not before
        assert "judge" not in before


class TestNeverRaises:
    def test_bad_row_logs_warning_and_leaves_latency_unchanged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        prior = {
            "response": {"avg_latency_ms": 200.0, "call_count": 2},
            "avg_latency_ms": 200.0,
            "call_count": 2,
        }
        run = _run(latency=prior)
        with caplog.at_level(logging.WARNING):
            attach_latency(
                eval_run=run,
                log_prefix=LOG_PREFIX,
                judge_results=[{"latency_ms": "x"}],
            )
        assert run.latency == {
            "response": {"avg_latency_ms": 200.0, "call_count": 2},
            "avg_latency_ms": 200.0,
            "call_count": 2,
        }
        assert any(
            r.levelno == logging.WARNING
            and r.getMessage().startswith("[attach_latency]")
            for r in caplog.records
        )
