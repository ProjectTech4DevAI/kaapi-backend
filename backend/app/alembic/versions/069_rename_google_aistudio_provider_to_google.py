"""rename 'google-aistudio' provider references back to 'google'

Mirrors the writes that 066 made under the 'google → google-aistudio' flip:
credential, batch_job, and config_version. Skips global.model_config — those
rows are duplicates that the routing swap doesn't read.

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
    op.execute(
        "UPDATE batch_job SET provider = 'google' WHERE provider = 'google-aistudio'"
    )
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google"')
        WHERE config_blob->'completion'->>'provider' = 'google-aistudio'
        """
    )
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google-native"')
        WHERE config_blob->'completion'->>'provider' = 'google-aistudio-native'
        """
    )


def downgrade():
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google-aistudio-native"')
        WHERE config_blob->'completion'->>'provider' = 'google-native'
        """
    )
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google-aistudio"')
        WHERE config_blob->'completion'->>'provider' = 'google'
        """
    )
    op.execute(
        "UPDATE batch_job SET provider = 'google-aistudio' WHERE provider = 'google'"
    )
    op.execute(
        "UPDATE credential SET provider = 'google-aistudio' WHERE provider = 'google'"
    )
