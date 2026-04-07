"""add project id to job table

Revision ID: 050
Revises: 049
Create Date: 2026-04-07 14:23:00.938901

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None

chain_status_enum = postgresql.ENUM(
    "PENDING",
    "RUNNING",
    "FAILED",
    "COMPLETED",
    name="chainstatus",
    create_type=False,
)


def upgrade():
    chain_status_enum.create(op.get_bind())
    op.add_column(
        "job",
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=True,
            comment="Project ID of the job's project",
        ),
    )
    op.alter_column(
        "llm_call",
        "chain_id",
        existing_type=sa.UUID(),
        comment="Reference to the parent chain (NULL for standalone llm_call requests)",
        existing_comment="Reference to the parent chain (NULL for standalone /llm/call requests)",
        existing_nullable=True,
    )
    op.alter_column(
        "llm_call",
        "input_type",
        existing_type=sa.VARCHAR(),
        comment="Input type: text, audio, image, pdf, multimodal",
        existing_comment="Input type: text, audio, image",
        existing_nullable=False,
    )
    op.execute("ALTER TABLE llm_chain ALTER COLUMN status DROP DEFAULT")
    op.alter_column(
        "llm_chain",
        "status",
        existing_type=sa.VARCHAR(),
        type_=chain_status_enum,
        existing_comment="Chain execution status (pending, running, failed, completed)",
        existing_nullable=False,
        postgresql_using="UPPER(status)::chainstatus",
    )
    op.execute(
        "ALTER TABLE llm_chain ALTER COLUMN status SET DEFAULT 'PENDING'::chainstatus"
    )
    op.alter_column(
        "llm_chain",
        "error",
        existing_type=sa.TEXT(),
        type_=sqlmodel.sql.sqltypes.AutoString(),
        existing_comment="Error message if the chain execution failed",
        existing_nullable=True,
    )


def downgrade():
    op.alter_column(
        "llm_chain",
        "error",
        existing_type=sqlmodel.sql.sqltypes.AutoString(),
        type_=sa.TEXT(),
        existing_comment="Error message if the chain execution failed",
        existing_nullable=True,
    )
    op.execute("ALTER TABLE llm_chain ALTER COLUMN status DROP DEFAULT")
    op.alter_column(
        "llm_chain",
        "status",
        existing_type=sa.Enum(
            "PENDING", "RUNNING", "FAILED", "COMPLETED", name="chainstatus"
        ),
        type_=sa.VARCHAR(),
        existing_comment="Chain execution status (pending, running, failed, completed)",
        existing_nullable=False,
    )
    op.execute("ALTER TABLE llm_chain ALTER COLUMN status SET DEFAULT 'pending'")
    op.execute("DROP TYPE IF EXISTS chainstatus")
    op.alter_column(
        "llm_call",
        "input_type",
        existing_type=sa.VARCHAR(),
        comment="Input type: text, audio, image",
        existing_comment="Input type: text, audio, image, pdf, multimodal",
        existing_nullable=False,
    )
    op.alter_column(
        "llm_call",
        "chain_id",
        existing_type=sa.UUID(),
        comment="Reference to the parent chain (NULL for standalone /llm/call requests)",
        existing_comment="Reference to the parent chain (NULL for standalone llm_call requests)",
        existing_nullable=True,
    )
    op.drop_column("job", "project_id")
