from uuid import uuid4

from sqlmodel import Session, select

from app.core.util import now
from app.crud.evaluations.core import (
    create_evaluation_run,
    get_evaluation_run_by_id,
    list_evaluation_runs,
)
from app.crud.evaluations.dataset import create_evaluation_dataset
from app.models import EvaluationRun, Organization, Project
from app.models.stt_evaluation import EvaluationType


def _create_config(db: Session, project_id: int) -> tuple:
    """Helper to create a config and config_version for evaluation runs."""
    from app.models.config import Config, ConfigVersion

    config = Config(
        name="test_config",
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(config)
    db.commit()
    db.refresh(config)

    config_version = ConfigVersion(
        config_id=config.id,
        version=1,
        blob={"completion": {"params": {"model": "gpt-4o"}}},
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(config_version)
    db.commit()
    db.refresh(config_version)

    return config.id, config_version.version


class TestCreateEvaluationRun:
    """Test creating evaluation runs."""

    def test_create_evaluation_run_sets_text_type(self, db: Session) -> None:
        """Test that create_evaluation_run sets type to TEXT."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset_run_type",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        config_id, config_version = _create_config(db, project.id)

        eval_run = create_evaluation_run(
            session=db,
            run_name="test_run",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config_id,
            config_version=config_version,
            organization_id=org.id,
            project_id=project.id,
        )

        assert eval_run.id is not None
        assert eval_run.type == EvaluationType.TEXT.value
        assert eval_run.status == "pending"
        assert eval_run.run_name == "test_run"


class TestGetEvaluationRunById:
    """Test fetching evaluation runs by ID."""

    def test_get_evaluation_run_by_id_success(self, db: Session) -> None:
        """Test fetching an existing evaluation run by ID."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset_get_run",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        config_id, config_version = _create_config(db, project.id)

        eval_run = create_evaluation_run(
            session=db,
            run_name="test_get_run",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config_id,
            config_version=config_version,
            organization_id=org.id,
            project_id=project.id,
        )

        fetched = get_evaluation_run_by_id(
            session=db,
            evaluation_id=eval_run.id,
            organization_id=org.id,
            project_id=project.id,
        )

        assert fetched is not None
        assert fetched.id == eval_run.id
        assert fetched.run_name == "test_get_run"

    def test_get_evaluation_run_by_id_not_found(self, db: Session) -> None:
        """Test fetching a non-existent evaluation run."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        fetched = get_evaluation_run_by_id(
            session=db,
            evaluation_id=99999,
            organization_id=org.id,
            project_id=project.id,
        )

        assert fetched is None

    def test_get_evaluation_run_by_id_excludes_non_text_type(self, db: Session) -> None:
        """Test that get_evaluation_run_by_id excludes runs with non-text type."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset_exclude_run",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        config_id, config_version = _create_config(db, project.id)

        eval_run = create_evaluation_run(
            session=db,
            run_name="test_stt_run",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config_id,
            config_version=config_version,
            organization_id=org.id,
            project_id=project.id,
        )

        # Manually update type to STT to simulate a non-text run
        eval_run.type = EvaluationType.STT.value
        db.add(eval_run)
        db.commit()

        fetched = get_evaluation_run_by_id(
            session=db,
            evaluation_id=eval_run.id,
            organization_id=org.id,
            project_id=project.id,
        )

        assert fetched is None


class TestListEvaluationRuns:
    """Test listing evaluation runs."""

    def test_list_evaluation_runs_empty(self, db: Session) -> None:
        """Test listing evaluation runs when none exist."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        runs = list_evaluation_runs(
            session=db, organization_id=org.id, project_id=project.id
        )

        assert len(runs) == 0

    def test_list_evaluation_runs_excludes_non_text_type(self, db: Session) -> None:
        """Test that list_evaluation_runs only returns text type runs."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        dataset = create_evaluation_dataset(
            session=db,
            name="test_dataset_list_runs",
            dataset_metadata={"original_items_count": 10},
            organization_id=org.id,
            project_id=project.id,
        )

        config_id, config_version = _create_config(db, project.id)

        # Create text evaluation runs
        for i in range(3):
            create_evaluation_run(
                session=db,
                run_name=f"text_run_{i}",
                dataset_name=dataset.name,
                dataset_id=dataset.id,
                config_id=config_id,
                config_version=config_version,
                organization_id=org.id,
                project_id=project.id,
            )

        # Create a non-text evaluation run by updating type after creation
        stt_run = create_evaluation_run(
            session=db,
            run_name="stt_run",
            dataset_name=dataset.name,
            dataset_id=dataset.id,
            config_id=config_id,
            config_version=config_version,
            organization_id=org.id,
            project_id=project.id,
        )
        stt_run.type = EvaluationType.STT.value
        db.add(stt_run)
        db.commit()

        runs = list_evaluation_runs(
            session=db, organization_id=org.id, project_id=project.id
        )

        assert len(runs) == 3
        assert all(r.type == EvaluationType.TEXT.value for r in runs)
