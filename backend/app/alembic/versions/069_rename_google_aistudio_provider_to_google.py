"""rename credential.provider 'google-aistudio' back to 'google'

Revision ID: 069
Revises: 068
Create Date: 2026-06-19 00:00:00.000000

"""

from alembic import op

revision = "069"
down_revision = "068"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "UPDATE credential SET provider = 'google' WHERE provider = 'google-aistudio'"
    )


def downgrade():
    op.execute(
        "UPDATE credential SET provider = 'google-aistudio' WHERE provider = 'google'"
    )
