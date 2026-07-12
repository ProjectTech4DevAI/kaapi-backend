"""Re-encrypt every stored credential through the current encryption scheme.

Atomic: all rows are re-encrypted in a single transaction and committed once. Any
failure rolls the whole thing back and re-raises, so the Celery task fails and
Sentry reports it. Nothing is ever left half-converted.
"""

import logging
from typing import Any

from sqlmodel import Session

from app.core.db import engine
from app.core.security import decrypt_credentials, encrypt_credentials
from app.core.util import now
from app.crud.credentials import list_all_credentials

logger = logging.getLogger(__name__)


def execute_credential_reencrypt(
    *,
    session: Session | None = None,
    task_instance: Any | None = None,
) -> dict[str, int]:
    if session is not None:
        return _reencrypt(session, task_instance)
    with Session(engine) as owned_session:
        return _reencrypt(owned_session, task_instance)


def _reencrypt(session: Session, task_instance: Any | None) -> dict[str, int]:
    rows = list_all_credentials(session=session)
    total = len(rows)
    logger.info(f"[execute_credential_reencrypt] Starting | total: {total}")

    converted = 0
    try:
        for row in rows:
            plaintext = decrypt_credentials(row.credential)
            new_ciphertext = encrypt_credentials(plaintext)
            if decrypt_credentials(new_ciphertext) != plaintext:
                raise ValueError(f"roundtrip mismatch for credential id {row.id}")

            row.credential = new_ciphertext
            row.updated_at = now()
            session.add(row)
            converted += 1

            if task_instance is not None:
                task_instance.update_state(
                    state="PROGRESS",
                    meta={"total": total, "converted": converted},
                )

        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(
            f"[execute_credential_reencrypt] Failed, rolled back all rows | "
            f"converted-before-abort: {converted}/{total}, error: {e}",
            exc_info=True,
        )
        raise

    logger.info(
        f"[execute_credential_reencrypt] Done | total: {total}, converted: {converted}"
    )
    return {"total": total, "converted": converted}
