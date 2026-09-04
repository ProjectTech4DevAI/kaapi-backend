"""re-encrypt all stored credentials through the current encryption scheme

Revision ID: 073
Revises: 072
Create Date: 2026-07-14

One-shot data backfill: runs the app's re-encryption over every credential row.
Empty table (fresh DB) is a no-op, so replaying on a new environment is safe.
Requires KMS access in the migrate runtime — the migrate container needs the
same IAM role/creds the app uses, else encrypt/decrypt fails and the upgrade aborts.
"""
from alembic import op
from sqlmodel import Session

from app.services.credentials.reencrypt import execute_credential_reencrypt


# revision identifiers, used by Alembic.
revision = "073"
down_revision = "072"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    with Session(bind=bind) as session:
        execute_credential_reencrypt(session=session)


def downgrade():
    # Re-encryption is not reversible — old ciphertext is gone. No-op.
    pass
