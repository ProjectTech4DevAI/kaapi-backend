"""re-encrypt stored credentials to the kms.v2 envelope format

Revision ID: 076
Revises: 075
Create Date: 2026-08-07

One-shot data backfill onto envelope encryption (KMS-wrapped data key + AES-GCM).
Empty table (fresh DB) and Fernet-only (KMS inactive) environments are no-ops, so
replaying is safe. Requires KMS/IAM access in the migrate runtime — the migrate
container needs the same role/creds the app uses, else the upgrade aborts.
"""
from alembic import op
from sqlmodel import Session

from app.services.credentials.reencrypt import execute_credential_reencrypt


# revision identifiers, used by Alembic.
revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    with Session(bind=bind) as session:
        execute_credential_reencrypt(session=session)


def downgrade():
    # Re-encryption is not reversible — the prior ciphertext is gone. No-op.
    pass
