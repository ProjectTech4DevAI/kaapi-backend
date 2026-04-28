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
        flag = create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        assert flag is not None
        assert flag.id is not None
        assert flag.key == "ASSESSMENT"
        assert flag.organization_id == project.organization_id
        assert flag.project_id == project.id
        assert flag.enabled is True

    def test_returns_none_on_duplicate(self, db: Session) -> None:
        project = create_test_project(db)
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        duplicate = create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=False,
        )
        assert duplicate is None

    def test_same_key_different_projects_can_coexist(self, db: Session) -> None:
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
        flag_a = create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_a.id,
            enabled=True,
        )
        flag_b = create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_b.id,
            enabled=False,
        )
        assert flag_a is not None
        assert flag_b is not None


class TestUpdateFeatureFlag:
    def test_updates_enabled_state(self, db: Session) -> None:
        project = create_test_project(db)
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        updated = update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=False,
        )
        assert updated is not None
        assert updated.enabled is False

    def test_returns_none_when_not_found(self, db: Session) -> None:
        project = create_test_project(db)
        result = update_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        assert result is None

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
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_a.id,
            enabled=True,
        )
        create_feature_flag(
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
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        deleted = delete_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        assert deleted is True

    def test_returns_false_when_not_found(self, db: Session) -> None:
        project = create_test_project(db)
        result = delete_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        )
        assert result is False

    def test_flag_not_accessible_after_delete(self, db: Session) -> None:
        project = create_test_project(db)
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
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
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
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
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project_a.organization_id,
            project_id=project_b.id,
            enabled=True,
        )
        flags = list_feature_flags(session=db, project_id=project_a.id)
        assert len(flags) == 0


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
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=True,
        )
        assert is_enabled(
            session=db,
            flag="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        ) is True

    def test_returns_false_when_flag_disabled(self, db: Session) -> None:
        project = create_test_project(db)
        create_feature_flag(
            session=db,
            key="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
            enabled=False,
        )
        assert is_enabled(
            session=db,
            flag="ASSESSMENT",
            organization_id=project.organization_id,
            project_id=project.id,
        ) is False


class TestResolveAllFlags:
    def test_returns_empty_dict_when_no_flags(self, db: Session) -> None:
        project = create_test_project(db)
        result = resolve_all_flags(
            session=db,
            organization_id=project.organization_id,
            project_id=project.id,
        )
        assert result == {}

    def test_returns_project_flags(self, db: Session) -> None:
        project = create_test_project(db)
        create_feature_flag(
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
        create_feature_flag(
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
        assert "ASSESSMENT" not in result
