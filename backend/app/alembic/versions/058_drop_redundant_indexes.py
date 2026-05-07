"""drop redundant indexes superseded by 057 composites

Revision ID: 058
Revises: 057
Create Date: 2026-05-05 14:00:00.000000

Drops indexes that are now redundant after migration 057 added the
real composite/partial indexes that match actual query shapes:

  ix_project_name
    Subsumed by uq_project_name_org_id (name is leading column).
    No code path queries Project.name without organization_id.

  ix_credential_provider
    Subsumed by uq_credential_org_project_provider. All four CRUD
    paths in crud/credentials.py filter (org, project, provider) — never
    provider alone.

  ix_openai_conversation_previous_response_id
    Zero query consumers; previous_response_id is read but never
    filtered on in any WHERE clause.

  ix_openai_conversation_response_id
    Superseded by ix_openai_conversation_response_project_active
    (project-scoped partial), which exactly matches CRUD predicates
    in crud/openai_conversation.py:get_conversation_by_response_id.

  ix_openai_conversation_ancestor_response_id
    Superseded by
    ix_openai_conversation_ancestor_project_inserted_at_active, which
    matches the (ancestor_response_id, project_id) + ORDER BY shape
    used in crud/openai_conversation.py:get_conversation_by_ancestor_id
    and the /responses thread reconstruction path.

  idx_file_type
    Low cardinality (4 enum values) and the only consumer in
    crud/file.py:147 always pairs file_type with (organization_id,
    project_id). idx_file_org_project covers the query; an extra
    in-memory filter on file_type is cheaper than a second index hit.

  idx_eval_run_status_org / idx_eval_run_status_project
    Both lead with low-cardinality status. Real CRUD queries lead with
    (organization_id, project_id, type), now covered by
    ix_evaluation_run_org_project_type_inserted_at.

Uses DROP INDEX CONCURRENTLY so no AccessExclusiveLock is taken.
Downgrade recreates the original indexes (also concurrently) so the
schema can be restored bit-for-bit if needed.
"""

from alembic import op


revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


# (index_name, recreate_sql_body)
# recreate_sql_body is "ON \"<table>\" (<columns>)" used by downgrade only.
INDEXES_TO_DROP: list[tuple[str, str]] = [
    ("ix_project_name", 'ON "project" ("name")'),
    ("ix_credential_provider", 'ON "credential" ("provider")'),
    (
        "ix_openai_conversation_previous_response_id",
        'ON "openai_conversation" ("previous_response_id")',
    ),
    (
        "ix_openai_conversation_response_id",
        'ON "openai_conversation" ("response_id")',
    ),
    (
        "ix_openai_conversation_ancestor_response_id",
        'ON "openai_conversation" ("ancestor_response_id")',
    ),
    ("idx_file_type", 'ON "file" ("file_type")'),
    (
        "idx_eval_run_status_org",
        'ON "evaluation_run" ("status", "organization_id")',
    ),
    (
        "idx_eval_run_status_project",
        'ON "evaluation_run" ("status", "project_id")',
    ),
]


def upgrade():
    with op.get_context().autocommit_block():
        for index_name, _body in INDEXES_TO_DROP:
            op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{index_name}"')


def downgrade():
    with op.get_context().autocommit_block():
        for index_name, body in INDEXES_TO_DROP:
            op.execute(f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{index_name}" {body}')
