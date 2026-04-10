"""Add userproject table

Revision ID: 050
Revises: 049
Create Date: 2026-04-01 12:17:42.165482

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_project",
        sa.Column(
            "user_id", sa.Integer(), nullable=False, comment="Reference to the user"
        ),
        sa.Column(
            "organization_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the organization",
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the project",
        ),
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for the user-project mapping",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the mapping was created",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organization.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "project_id", name="uq_user_project"),
    )


def downgrade():
    op.drop_table("user_project")
