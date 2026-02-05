"""create global schema and languages table

Revision ID: 043
Revises: 042
Create Date: 2026-02-05 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


def upgrade():
    # Create the global schema
    op.execute("CREATE SCHEMA IF NOT EXISTS global")

    # Create the languages table in the global schema
    op.create_table(
        "languages",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=False),
            primary_key=True,
            comment="Unique identifier for the language",
        ),
        sa.Column(
            "label",
            sa.String(255),
            nullable=False,
            comment="Language name in English (e.g., 'Hindi')",
        ),
        sa.Column(
            "label_locale",
            sa.String(255),
            nullable=False,
            comment="Language name in its native script (e.g., 'हिंदी')",
        ),
        sa.Column(
            "description",
            sa.Text(),
            nullable=True,
            comment="Optional description of the language",
        ),
        sa.Column(
            "locale",
            sa.String(255),
            nullable=False,
            comment="ISO 639-1 language code (e.g., 'en', 'hi')",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("true"),
            comment="Flag indicating if the language is available",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the language was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the language was last updated",
        ),
        sa.Column(
            "localized",
            sa.Boolean(),
            nullable=True,
            server_default=sa.text("false"),
            comment="Flag indicating if the platform is localized for this language",
        ),
        schema="global",
    )

    # Create index on locale for faster lookups
    op.create_index(
        "ix_global_languages_locale",
        "languages",
        ["locale"],
        schema="global",
    )

    # Seed default languages
    op.execute(
        """
        INSERT INTO global.languages (label, label_locale, description, locale, is_active, inserted_at, updated_at, localized)
        VALUES
            ('English', 'English', NULL, 'en', true, NOW(), NOW(), true),
            ('Hindi', 'हिंदी', NULL, 'hi', true, NOW(), NOW(), true),
            ('Tamil', 'தமிழ்', NULL, 'ta', true, NOW(), NOW(), false),
            ('Kannada', 'ಕನ್ನಡ', NULL, 'kn', true, NOW(), NOW(), false),
            ('Malayalam', 'മലയാളം', NULL, 'ml', true, NOW(), NOW(), false),
            ('Telugu', 'తెలుగు', NULL, 'te', true, NOW(), NOW(), false),
            ('Odia', 'ଓଡ଼ିଆ', NULL, 'or', true, NOW(), NOW(), false),
            ('Assamese', 'অসমীয়া', NULL, 'as', true, NOW(), NOW(), false),
            ('Gujarati', 'ગુજરાતી', NULL, 'gu', true, NOW(), NOW(), false),
            ('Bengali', 'বাংলা', NULL, 'bn', true, NOW(), NOW(), false),
            ('Punjabi', 'ਪੰਜਾਬੀ', NULL, 'pa', true, NOW(), NOW(), false),
            ('Marathi', 'मराठी', NULL, 'mr', true, NOW(), NOW(), false),
            ('Urdu', 'اردو', NULL, 'ur', true, NOW(), NOW(), false)
        """
    )


def downgrade():
    op.drop_index("ix_global_languages_locale", table_name="languages", schema="global")
    op.drop_table("languages", schema="global")
    op.execute("DROP SCHEMA IF EXISTS global")
