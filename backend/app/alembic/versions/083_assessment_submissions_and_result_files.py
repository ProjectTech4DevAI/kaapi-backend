"""Assessment submissions table, submission/result-file pointers, provider error file id

Revision ID: 083
Revises: 082
Create Date: 2026-09-09 00:00:00.000000

Assessment submissions leave `evaluation_dataset`, whose type-agnostic name uniqueness
let an eval dataset block an assessment one. Multi-MB payloads leave Postgres too:
`submission_input` and `result_files` hold s3:// urls, and `provider_error_file_id`
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None

RESULT_FILES_CHECK = "ck_assessment_result_files_is_object"


def upgrade() -> None:
    op.create_table(
        "assessment_submission",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            comment="Unique identifier for the submission",
        ),
        sa.Column(
            "name",
            sa.String(),
            nullable=False,
            comment="Sanitized name; the object key is derived from it",
        ),
        sa.Column(
            "description", sa.String(), nullable=True, comment="Optional description"
        ),
        sa.Column(
            "object_store_url",
            sa.String(),
            nullable=False,
            comment="Object-store url of the uploaded file; its suffix gives the format",
        ),
        sa.Column(
            "total_items",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Row count, excluding the header",
        ),
        sa.Column(
            "organization_id",
            sa.Integer(),
            sa.ForeignKey("organization.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("project.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("inserted_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "name",
            "organization_id",
            "project_id",
            name="uq_assessment_submission_name_org_project",
        ),
    )
    op.create_index("ix_assessment_submission_name", "assessment_submission", ["name"])

    op.add_column(
        "assessment",
        sa.Column(
            "result_files",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment=(
                "Result-file kind (results / errors / <stage>_results) to "
                "{object_store_url} for every provider batch dump held; raw s3:// in the "
                "column, presigned per delivery in the BATCH callback"
            ),
        ),
    )
    op.add_column(
        "assessment",
        sa.Column(
            "submission_input",
            sa.String(),
            nullable=True,
            comment=(
                "Object-store url of the API-client BATCH submission rows "
                "(submission.jsonl); the rows are never stored in this table"
            ),
        ),
    )
    op.drop_column("assessment", "dataset_id")
    op.add_column(
        "assessment",
        sa.Column(
            "submission_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("assessment_submission.id", ondelete="SET NULL"),
            nullable=True,
            comment=(
                "Uploaded submission the rows came from; set by RUN and by a BATCH "
                "submitted with `submission_doc_id`. NULL when BATCH sent rows inline"
            ),
        ),
    )
    op.create_index("ix_assessment_submission_id", "assessment", ["submission_id"])

    op.add_column(
        "batch_job",
        sa.Column(
            "provider_error_file_id",
            sa.String(),
            nullable=True,
            comment=(
                "Provider's error file ID (OpenAI only; Anthropic and Gemini report "
                "per-item errors inline)"
            ),
        ),
    )
    op.create_check_constraint(
        RESULT_FILES_CHECK,
        "assessment",
        "jsonb_typeof(result_files) = 'object'",
    )


def downgrade() -> None:
    op.drop_constraint(RESULT_FILES_CHECK, "assessment", type_="check")
    op.drop_column("batch_job", "provider_error_file_id")

    op.drop_index("ix_assessment_submission_id", table_name="assessment")
    op.drop_column("assessment", "submission_id")
    op.add_column(
        "assessment",
        sa.Column(
            "dataset_id",
            sa.Integer(),
            sa.ForeignKey("evaluation_dataset.id", ondelete="SET NULL"),
            nullable=True,
            comment="External dataset (RUN); binding lives in `input`",
        ),
    )

    op.drop_column("assessment", "submission_input")
    op.drop_column("assessment", "result_files")

    op.drop_index("ix_assessment_submission_name", table_name="assessment_submission")
    op.drop_table("assessment_submission")
