"""Re-encrypt every stored credential through the current encryption scheme.

Atomic: one transaction, all-or-nothing — any failure rolls back and re-raises.
Sole caller is the one-shot Alembic migration 073; no route/CRUD/Celery path.
"""

import logging

from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine
from app.core.security import _use_kms, decrypt_credentials, encrypt_credentials
from app.core.util import now
from app.crud.credentials import list_all_credentials

logger = logging.getLogger(__name__)


def execute_credential_reencrypt(*, session: Session | None = None) -> dict[str, int]:
    if session is not None:
        return _reencrypt(session)
    with Session(engine) as owned_session:
        return _reencrypt(owned_session)


def _reencrypt(session: Session) -> dict[str, int]:
    if not _use_kms():
        logger.info(
            f"[execute_credential_reencrypt] Skipped, KMS inactive | "
            f"environment: {settings.ENVIRONMENT}"
        )
        return {"total": 0, "converted": 0}

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
