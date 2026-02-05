"""extend collection table for provider agnostic support

Revision ID: 042
Revises: 041
Create Date: 2026-01-15 16:53:19.495583

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None

provider_type = postgresql.ENUM(
    "openai",
    # aws
    # gemini
    name="providertype",
    create_type=False,
)


def upgrade():
    provider_type.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "collection",
        sa.Column(
            "provider",
            provider_type,
            nullable=True,
            comment="LLM provider used for this collection",
        ),
    )
    op.execute("UPDATE collection SET provider = 'openai' WHERE provider IS NULL")
    op.alter_column("collection", "provider", nullable=False)
    op.add_column(
        "collection",
        sa.Column(
            "name",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Name of the collection",
        ),
    )
    op.add_column(
        "collection",
        sa.Column(
            "description",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Description of the collection",
        ),
    )
    op.alter_column(
        "collection",
        "llm_service_name",
        existing_type=sa.VARCHAR(),
        comment="Name of the LLM service",
        existing_comment="Name of the LLM service provider",
        existing_nullable=False,
    )
    op.create_index(
        "uq_collection_project_id_name_active",
        "collection",
        ["project_id", "name"],
        unique=True,
        postgresql_where="deleted_at IS NULL",
    )
    op.drop_constraint(
        op.f("collection_organization_id_fkey"), "collection", type_="foreignkey"
    )
    op.drop_column("collection", "organization_id")


def downgrade():
    op.add_column(
        "collection",
        sa.Column(
            "organization_id",
            sa.INTEGER(),
            autoincrement=False,
            nullable=True,
            comment="Reference to the organization",
        ),
    )
    op.execute(
        """UPDATE collection SET organization_id = (SELECT organization_id FROM project
               WHERE project.id = collection.project_id)"""
    )
    op.alter_column("collection", "organization_id", nullable=False)
    op.create_foreign_key(
        op.f("collection_organization_id_fkey"),
        "collection",
        "organization",
        ["organization_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_index("uq_collection_project_id_name_active", "collection", type_="unique")
    op.alter_column(
        "collection",
        "llm_service_name",
        existing_type=sa.VARCHAR(),
        comment="Name of the LLM service provider",
        existing_comment="Name of the LLM service",
        existing_nullable=False,
    )
    op.drop_column("collection", "description")
    op.drop_column("collection", "name")
    op.drop_column("collection", "provider")
