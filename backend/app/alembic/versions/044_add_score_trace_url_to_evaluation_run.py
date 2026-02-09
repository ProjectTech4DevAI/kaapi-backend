"""Add score_trace_url to evaluation_run

Revision ID: 044
Revises: 043
Create Date: 2026-01-24 19:34:46.763908

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "score_trace_url",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="S3 URL where per-trace evaluation scores are stored",
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "score_trace_url")
