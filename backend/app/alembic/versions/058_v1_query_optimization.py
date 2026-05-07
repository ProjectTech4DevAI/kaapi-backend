"""v1.0 query optimization: project_id + composite indexes, drop is_deleted

Revision ID: 058
Revises: 057
Create Date: 2026-05-05 12:00:00.000000

Bundles three coordinated changes for v1.0 lock:

1. Single-column `project_id` btree indexes on every table-mapped model
   that filters by project_id (the dominant tenant filter).
   organization_id-only access is rare and intentionally deferred.
   Tables already covered by a leading-column index are skipped:
     - openai_assistant: UNIQUE(project_id, assistant_id) leads with project_id
     - batch_job: ix_batch_job_project_id (migration 036)

2. Composite + partial indexes for hot list/pagination paths matching:
       WHERE project_id = ? [AND deleted_at IS NULL] ORDER BY <ts> DESC

3. Drop the redundant `is_deleted` boolean from every table that also
   carries `deleted_at`. `deleted_at IS NULL` becomes the single source
   of truth for soft-delete: same query cost when paired with a partial
   index, preserves audit timestamp, no dual-write drift.
   Affected tables: openai_assistant, apikey, document,
   openai_conversation, fine_tuning, model_evaluation.

Execution model:
  Phase A (transactional): backfill deleted_at where is_deleted was true
  but deleted_at was never set, then drop the is_deleted columns.
  Phase B (autocommit_block): CREATE INDEX CONCURRENTLY for every index
  so no AccessExclusiveLock is taken on hot tables.
"""

import sqlalchemy as sa
from alembic import op


revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


# Tables that currently carry both `is_deleted` and `deleted_at`.
IS_DELETED_TABLES = [
    "openai_assistant",
    "apikey",
    "document",
    "openai_conversation",
    "fine_tuning",
    "model_evaluation",
]


# Single-column FK / multi-tenant filter indexes (P0).
# (table_name, column_name, index_name)
FK_INDEXES: list[tuple[str, str, str]] = [
    # project_id across tables that filter by tenant
    ("apikey", "project_id", "ix_apikey_project_id"),
    ("credential", "project_id", "ix_credential_project_id"),
    ("collection", "project_id", "ix_collection_project_id"),
    ("collection_jobs", "project_id", "ix_collection_jobs_project_id"),
    ("document", "project_id", "ix_document_project_id"),
    ("evaluation_dataset", "project_id", "ix_evaluation_dataset_project_id"),
    ("evaluation_run", "project_id", "ix_evaluation_run_project_id"),
    ("file", "project_id", "ix_file_project_id"),
    ("fine_tuning", "project_id", "ix_fine_tuning_project_id"),
    ("job", "project_id", "ix_job_project_id"),
    ("llm_call", "project_id", "ix_llm_call_project_id"),
    ("llm_chain", "project_id", "ix_llm_chain_project_id"),
    ("model_evaluation", "project_id", "ix_model_evaluation_project_id"),
    ("openai_conversation", "project_id", "ix_openai_conversation_project_id"),
    ("stt_result", "project_id", "ix_stt_result_project_id"),
    ("stt_sample", "project_id", "ix_stt_sample_project_id"),
    ("tts_result", "project_id", "ix_tts_result_project_id"),
    ("user_project", "project_id", "ix_user_project_project_id"),
    # Other un-indexed FKs surfaced by the audit
    ("apikey", "user_id", "ix_apikey_user_id"),
    ("collection_jobs", "collection_id", "ix_collection_jobs_collection_id"),
    (
        "doc_transformation_job",
        "source_document_id",
        "ix_doc_transformation_job_source_document_id",
    ),
    (
        "doc_transformation_job",
        "transformed_document_id",
        "ix_doc_transformation_job_transformed_document_id",
    ),
    ("evaluation_run", "dataset_id", "ix_evaluation_run_dataset_id"),
]


# Composite + partial indexes (P1). (index_name, body_after_INDEX_NAME, schema)
# `schema` is the unquoted PG schema for downgrade DROP INDEX, or None for
# the default (public) schema. The upgrade body already names the schema
# inline in its ON clause; the field exists so downgrade doesn't have to
# string-sniff it back out.
COMPOSITE_INDEXES: list[tuple[str, str, str | None]] = [
    (
        "ix_document_project_inserted_at_active",
        'ON "document" ("project_id", "inserted_at" DESC) WHERE "deleted_at" IS NULL',
        None,
    ),
    (
        "ix_openai_conversation_project_inserted_at_active",
        'ON "openai_conversation" ("project_id", "inserted_at" DESC) WHERE "deleted_at" IS NULL',
        None,
    ),
    (
        "ix_openai_conversation_ancestor_project_inserted_at_active",
        'ON "openai_conversation" ("ancestor_response_id", "project_id", "inserted_at" DESC) WHERE "deleted_at" IS NULL',
        None,
    ),
    (
        "ix_openai_conversation_response_project_active",
        'ON "openai_conversation" ("response_id", "project_id") WHERE "deleted_at" IS NULL',
        None,
    ),
    (
        "ix_collection_jobs_project_status_inserted_at",
        'ON "collection_jobs" ("project_id", "status", "inserted_at" DESC)',
        None,
    ),
    (
        "ix_evaluation_run_org_project_type_inserted_at",
        'ON "evaluation_run" ("organization_id", "project_id", "type", "inserted_at" DESC)',
        None,
    ),
    (
        "ix_evaluation_dataset_org_project_type_inserted_at",
        'ON "evaluation_dataset" ("organization_id", "project_id", "type", "inserted_at" DESC)',
        None,
    ),
    (
        "ix_model_evaluation_document_project_updated_at",
        'ON "model_evaluation" ("document_id", "project_id", "updated_at" DESC) WHERE "deleted_at" IS NULL',
        None,
    ),
    (
        "ix_model_config_active_provider_name",
        'ON "global"."model_config" ("is_active", "provider", "model_name")',
        "global",
    ),
    (
        "ix_collection_project_active",
        'ON "collection" ("project_id") WHERE "deleted_at" IS NULL',
        None,
    ),
    # Composite FK indexes that match the actual query shape
    (
        "ix_fine_tuning_document_project",
        'ON "fine_tuning" ("document_id", "project_id")',
        None,
    ),
    (
        "ix_model_evaluation_fine_tuning_project",
        'ON "model_evaluation" ("fine_tuning_id", "project_id")',
        None,
    ),
    # Partial index for active-key listing on apikey
    (
        "ix_apikey_project_active",
        'ON "apikey" ("project_id") WHERE "deleted_at" IS NULL',
        None,
    ),
]


def upgrade():
    # Phase A (transactional): preserve audit timestamp, drop redundant column.
    for table in IS_DELETED_TABLES:
        op.execute(
            f"UPDATE {table} "
            f"SET deleted_at = NOW() "
            f"WHERE is_deleted = TRUE AND deleted_at IS NULL"
        )
        op.drop_column(table, "is_deleted")

    # Phase B (autocommit): CONCURRENTLY index creation. Each statement
    # runs in its own implicit transaction, required by the CONCURRENTLY
    # variant.
    with op.get_context().autocommit_block():
        for table, column, index in FK_INDEXES:
            op.execute(
                f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{index}" '
                f'ON "{table}" ("{column}")'
            )
        for index, body, _schema in COMPOSITE_INDEXES:
            op.execute(f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{index}" {body}')


def downgrade():
    with op.get_context().autocommit_block():
        for index, _body, schema in COMPOSITE_INDEXES:
            qualified = f'"{schema}"."{index}"' if schema else f'"{index}"'
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
        for _table, _column, index in FK_INDEXES:
            op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{index}"')

    for table in IS_DELETED_TABLES:
        op.add_column(
            table,
            sa.Column(
                "is_deleted",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
                comment="Soft delete flag",
            ),
        )
        op.execute(f"UPDATE {table} SET is_deleted = TRUE WHERE deleted_at IS NOT NULL")
