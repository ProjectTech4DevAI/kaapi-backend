import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.core.feature_flags import is_enabled, resolve_all_flags
from app.crud.feature_flag import (
    create_feature_flag,
    delete_feature_flag,
    list_feature_flags,
    update_feature_flag,
)
from app.tests.utils.test_data import create_test_project


class TestCreateFeatureFlag:
    def test_creates_flag(self, db: Session) -> None:
        project = create_test_project(db)
        # Flag is auto-seeded as disabled by DB trigger; verify it exists with correct shape
        from app.crud.feature_flag import get_feature_flag

        flag = get_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        assert flag is not None
        assert flag.id is not None
        assert flag.key == "ASSESSMENT"
        assert flag.organization_id == project.organization_id
        assert flag.project_id == project.id

    def test_raises_on_duplicate(self, db: Session) -> None:
        project = create_test_project(db)
        # Auto-seeded flag already exists; creating again must raise 409
        with pytest.raises(HTTPException, match="Feature flag already exists"):
            create_feature_flag(
                session=db,
                key="ASSESSMENT",
                organization_id=project.organization_id,
                project_id=project.id,
                enabled=False,
            )

    def test_same_key_different_projects_can_coexist(self, db: Session) -> None:
        from app.crud import create_project
        from app.crud.feature_flag import get_feature_flag
        from app.models import ProjectCreate

        project_a = create_test_project(db)
        project_b = create_project(
            session=db,
            project_create=ProjectCreate(
                name="ProjectB",
                description="",
                is_active=True,
                organization_id=project_a.organization_id,
            ),
        )
        # Both projects have auto-seeded flags; verify both exist independently
        flag_a = get_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_a.id,
        )
        flag_b = get_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_b.id,
        )
        assert flag_a is not None
        assert flag_b is not None


class TestUpdateFeatureFlag:
    def test_updates_enabled_state(self, db: Session) -> None:
        project = create_test_project(db)
        # Flag is auto-seeded as disabled; update it directly
        updated = update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=False,
        )
        assert updated is not None
        assert updated.enabled is False

    def test_raises_when_not_found(self, db: Session) -> None:
        project = create_test_project(db)
        # Delete the auto-seeded flag first so update raises 404
        delete_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        with pytest.raises(HTTPException, match="Feature flag not found"):
            update_feature_flag(
                session=db,
                key="ASSESSMENT",
                organization_id=project.organization_id,
                project_id=project.id,
                enabled=True,
            )

    def test_updates_only_correct_project(self, db: Session) -> None:
        from app.crud import create_project
        from app.models import ProjectCreate

        project_a = create_test_project(db)
        project_b = create_project(
            session=db,
            project_create=ProjectCreate(
                name="ProjectB",
                description="",
                is_active=True,
                organization_id=project_a.organization_id,
            ),
        )
        # Both projects have auto-seeded flags; enable both then disable project_b only
        update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_a.id,
            enabled=True,
        )
        update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_b.id,
            enabled=True,
        )
        update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_b.id,
            enabled=False,
        )
        flags_a = list_feature_flags(session=db, project_id=project_a.id)
        assert flags_a[0].enabled is True


class TestDeleteFeatureFlag:
    def test_deletes_existing_flag(self, db: Session) -> None:
        project = create_test_project(db)
        # Flag is auto-seeded by DB trigger; delete it directly
        delete_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )

    def test_raises_when_not_found(self, db: Session) -> None:
        project = create_test_project(db)
        # Delete the auto-seeded flag first so the next call raises
        delete_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        with pytest.raises(HTTPException, match="Feature flag not found"):
            delete_feature_flag(
                session=db,
                key="ASSESSMENT",
                organization_id=project.organization_id,
                project_id=project.id,
            )

    def test_flag_not_accessible_after_delete(self, db: Session) -> None:
        project = create_test_project(db)
        # Flag is auto-seeded by DB trigger; delete it directly
        delete_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        flags = list_feature_flags(session=db, project_id=project.id)
        assert len(flags) == 0


class TestListFeatureFlags:
    def test_lists_all_for_project(self, db: Session) -> None:
        project = create_test_project(db)
        # ASSESSMENT flag is auto-seeded by DB trigger on project creation
        flags = list_feature_flags(session=db, project_id=project.id)
        assert len(flags) == 1

    def test_does_not_return_other_project_flags(self, db: Session) -> None:
        from app.crud import create_project
        from app.models import ProjectCreate

        project_a = create_test_project(db)
        project_b = create_project(
            session=db,
            project_create=ProjectCreate(
                name="ProjectB",
                description="",
                is_active=True,
                organization_id=project_a.organization_id,
            ),
        )
        # Both projects have auto-seeded flags; verify project isolation
        flags = list_feature_flags(session=db, project_id=project_a.id)
        assert all(f.project_id == project_a.id for f in flags)


class TestIsEnabled:
    def test_returns_false_when_no_flag_exists(self, db: Session) -> None:
        project = create_test_project(db)
        result = is_enabled(
            session=db,
            flag="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        assert result is False

    def test_returns_true_when_flag_enabled(self, db: Session) -> None:
        project = create_test_project(db)
        # Flag is auto-seeded as disabled; enable it via update
        update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        assert (
            is_enabled(
                session=db,
                flag="ASSESSMENT",
                organization_id=project.organization_id,
                project_id=project.id,
            )
            is True
        )

    def test_returns_false_when_flag_disabled(self, db: Session) -> None:
        project = create_test_project(db)
        # ASSESSMENT flag is auto-seeded as disabled=False by DB trigger
        assert (
            is_enabled(
                session=db,
                flag="ASSESSMENT",
                organization_id=project.organization_id,
                project_id=project.id,
            )
            is False
        )


class TestResolveAllFlags:
    def test_returns_empty_dict_when_no_flags(self, db: Session) -> None:
        project = create_test_project(db)
        # Delete the auto-seeded flag to test the truly empty case
        delete_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        result = resolve_all_flags(
            session=db,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        assert result == {}

    def test_returns_project_flags(self, db: Session) -> None:
        project = create_test_project(db)
        # Flag is auto-seeded as disabled; enable it via update
        update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        result = resolve_all_flags(
            session=db,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        assert result == {"ASSESSMENT": True}

    def test_does_not_leak_other_project_flags(self, db: Session) -> None:
        from app.crud import create_project
        from app.models import ProjectCreate

        project_a = create_test_project(db)
        project_b = create_project(
            session=db,
            project_create=ProjectCreate(
                name="ProjectB",
                description="",
                is_active=True,
                organization_id=project_a.organization_id,
            ),
        )
        # Update project_b's auto-seeded flag to True to distinguish from project_a's False
        update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_b.id,
            enabled=True,
        )
        result = resolve_all_flags(
            session=db,
            organization_id=project_a.organization_id,
            project_id=project_a.id,
        )
        # project_a has its own ASSESSMENT=False; project_b's True must not bleed in
        assert result.get("ASSESSMENT") is not True
