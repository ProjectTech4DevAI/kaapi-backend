"""align global.languages.id to INTEGER

Revision ID: 061
Revises: 060
Create Date: 2026-05-07 13:00:00.000000

Migration 043 originally created `global.languages.id` as BIGINT, but every
FK column referencing it (evaluation_dataset, evaluation_run, stt_sample) is
INTEGER. Migration 043's source has been edited to use INTEGER for fresh
setups; this migration aligns already-deployed databases.

The underlying IDENTITY sequence stays BIGINT (PostgreSQL doesn't change it
on ALTER COLUMN TYPE). This is harmless — values would have to exceed
2^31 - 1 to cause an INSERT failure, and the table holds ~13 seeded rows.

Languages table is small (≤100 rows in practice), so the AccessExclusiveLock
taken by ALTER COLUMN TYPE is sub-second.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "061"
down_revision = "060"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE global.languages ALTER COLUMN id SET DATA TYPE INTEGER")


def downgrade():
    op.execute("ALTER TABLE global.languages ALTER COLUMN id SET DATA TYPE BIGINT")
