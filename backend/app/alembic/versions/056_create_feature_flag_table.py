"""create feature flag table

Revision ID: 056
Revises: 055
Create Date: 2026-04-22 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "056"
down_revision = "055"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "feature_flag",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for feature flag",
        ),
        sa.Column(
            "key",
            sa.Enum("ASSESSMENT", name="featureflagkey"),
            nullable=False,
            comment="Feature flag key",
        ),
        sa.Column(
            "organization_id",
            sa.Integer(),
            nullable=False,
            comment="Organization scope for this feature flag",
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=False,
            comment="Project scope for this feature flag",
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            comment="Whether the feature flag is enabled",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the flag row was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the flag row was last updated",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organization.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["project.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_feature_flag_key"),
        "feature_flag",
        ["key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_feature_flag_organization_id"),
        "feature_flag",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "uq_feature_flag_key_org_project",
        "feature_flag",
        ["key", "organization_id", "project_id"],
        unique=True,
    )


def downgrade():
    op.drop_index("uq_feature_flag_key_org_project", table_name="feature_flag")
    op.drop_index(op.f("ix_feature_flag_organization_id"), table_name="feature_flag")
    op.drop_index(op.f("ix_feature_flag_key"), table_name="feature_flag")
    op.drop_table("feature_flag")
    sa.Enum(name="featureflagkey").drop(op.get_bind())
