"""add agent_type to chatbot_session

Revision ID: 090
Revises: 089
Create Date: 2026-10-28 00:00:00.000000

Sessions created before this revision all ran the onboarding agent, so the
server default backfills them as 'onboarding' and keeps the column NOT NULL.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "090"
down_revision = "089"
branch_labels = None
depends_on = None

_AGENT_TYPE_ENUM_NAME = "chatbotagenttypeenum"
_AGENT_TYPE_VALUES = ("onboarding", "rag")
_DEFAULT_AGENT_TYPE = "onboarding"


def upgrade():
    agent_type_enum = postgresql.ENUM(
        *_AGENT_TYPE_VALUES,
        name=_AGENT_TYPE_ENUM_NAME,
        create_type=False,
    )
    agent_type_enum.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "chatbot_session",
        sa.Column(
            "agent_type",
            agent_type_enum,
            nullable=False,
            server_default=_DEFAULT_AGENT_TYPE,
            comment="Agent that runs each turn: onboarding (collects the fields the system_prompt asks for) or rag (answers questions from the config's knowledge bases)",
        ),
    )


def downgrade():
    op.drop_column("chatbot_session", "agent_type")
    sa.Enum(name=_AGENT_TYPE_ENUM_NAME).drop(op.get_bind(), checkfirst=True)
