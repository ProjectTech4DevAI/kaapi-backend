"""rename providertype enum value google_aistudio to google

Revision ID: 076
Revises: 075
Create Date: 2026-07-31 00:00:00.000000

SQLAlchemy stores the native enum by member NAME; renaming the ProviderType
member google_aistudio -> google requires the DB label to match.
"""

from alembic import op

revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE providertype RENAME VALUE 'google_aistudio' TO 'google'")


def downgrade() -> None:
    op.execute("ALTER TYPE providertype RENAME VALUE 'google' TO 'google_aistudio'")
