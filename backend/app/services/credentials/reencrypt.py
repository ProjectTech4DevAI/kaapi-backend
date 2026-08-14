"""Re-encrypt every stored credential through the current encryption scheme.

Atomic: one transaction, all-or-nothing — any failure rolls back and re-raises.
Sole caller is the one-shot Alembic migration 073; no route/CRUD/Celery path.
"""

import logging

from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine
from app.core.security import (
    KMS_CIPHERTEXT_PREFIX,
    KMS_ENVELOPE_PREFIX,
    _use_kms,
    decrypt_credentials,
    encrypt_credentials,
    encrypt_fernet,
)
from app.core.util import now
from app.crud.credentials import list_all_credentials

logger = logging.getLogger(__name__)


def execute_credential_reencrypt(*, session: Session | None = None) -> dict[str, int]:
    if session is not None:
        return _reencrypt(session)
    with Session(engine) as owned_session:
        return _reencrypt(owned_session)


def execute_credential_reencrypt_fernet(
    *, session: Session | None = None
) -> dict[str, int]:
    if session is not None:
        return _reencrypt_fernet(session)
    with Session(engine) as owned_session:
        return _reencrypt_fernet(owned_session)


def _reencrypt(session: Session) -> dict[str, int]:
    """
    DO NOT USE MIGRATION BEYOND 076
    THIS IS ONLY MEANT TO IMPLEMENT KMS ENVELOPE MIGRATION,
    AND SHOULD NOT BE USED FOR ROTATION OF KMS CMK.
    """
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
            if row.credential.startswith(KMS_ENVELOPE_PREFIX):
                # Already in the new envelope format; skip.
                continue
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


def _reencrypt_fernet(session: Session) -> dict[str, int]:
    """Downgrade KMS credentials (kms.v1/kms.v2) back to legacy Fernet — nothing else.

    Only KMS rows are converted; Fernet rows are left untouched. Needs KMS access to
    decrypt the existing rows; output decrypts only under the current SECRET_KEY.
    One-shot, atomic — any failure rolls back and re-raises.
    """
    if not _use_kms():
        logger.info(
            f"[_reencrypt_fernet] Skipped, KMS inactive | "
            f"environment: {settings.ENVIRONMENT}"
        )
        return {"total": 0, "converted": 0}

    rows = list_all_credentials(session=session)
    total = len(rows)
    logger.info(f"[_reencrypt_fernet] Starting | total: {total}")

    converted = 0
    try:
        for row in rows:
            # Convert only KMS rows; anything without a KMS prefix is already Fernet.
            if not row.credential.startswith(
                (KMS_ENVELOPE_PREFIX, KMS_CIPHERTEXT_PREFIX)
            ):
                continue
            plaintext = decrypt_credentials(row.credential)
            new_ciphertext = encrypt_fernet(plaintext)
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
            f"[_reencrypt_fernet] Failed, rolled back all rows | "
            f"converted-before-abort: {converted}/{total}, error: {e}",
            exc_info=True,
        )
        raise

    logger.info(f"[_reencrypt_fernet] Done | total: {total}, converted: {converted}")
    return {"total": total, "converted": converted}
