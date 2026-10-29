from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.util import is_constraint_violation

CONSTRAINT = "uq_example_active"


class TestIsConstraintViolation:
    # psycopg always exposes diag.constraint_name; the message fallback guards
    # drivers/wrappers that don't, so it's exercised with stand-in errors.
    @pytest.mark.parametrize(
        ("orig", "expected"),
        [
            (SimpleNamespace(diag=SimpleNamespace(constraint_name=CONSTRAINT)), True),
            (
                SimpleNamespace(diag=SimpleNamespace(constraint_name="other_fkey")),
                False,
            ),
            (
                Exception(
                    f'duplicate key value violates unique constraint "{CONSTRAINT}"'
                ),
                True,
            ),
            (Exception("violates foreign key constraint"), False),
        ],
        ids=["diag-match", "diag-other", "message-match", "message-other"],
    )
    def test_matches_only_the_named_constraint(
        self, orig: object, expected: bool
    ) -> None:
        error = IntegrityError("INSERT ...", {}, orig)  # type: ignore[arg-type]

        assert is_constraint_violation(error, CONSTRAINT) is expected
