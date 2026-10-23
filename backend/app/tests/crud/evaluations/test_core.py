from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlmodel import Session, select

from app.core.util import now
from app.crud.evaluations.core import (
    create_evaluation_run,
    get_evaluation_run_by_id,
    get_or_fetch_score,
    list_evaluation_runs,
)
from app.crud.evaluations.dataset import create_evaluation_dataset
from app.models import EvaluationDataset, EvaluationRun, Organization, Project
from app.models.stt_evaluation import EvaluationType
from app.tests.utils.auth import (
    TestAuthContext,
    get_superuser_test_auth_context,
    get_user_test_auth_context,
)
from app.tests.utils.test_data import create_test_evaluation_dataset
from app.tests.utils.utils import random_lower_string


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
        config_blob={"completion": {"params": {"model": "gpt-4o"}}},
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(config_version)
    db.commit()
    db.refresh(config_version)

    return config.id, config_version.version


def _create_run(
    db: Session,
    auth: TestAuthContext,
    dataset: EvaluationDataset,
    config: tuple[UUID, int],
    run_name: str | None = None,
    inserted_at: datetime | None = None,
    run_type: str = EvaluationType.TEXT.value,
) -> EvaluationRun:
    """Helper to create an evaluation run, optionally overriding type/inserted_at."""
    config_id, config_version = config
    run = create_evaluation_run(
        session=db,
        run_name=run_name or f"run_{random_lower_string()}",
        dataset_name=dataset.name,
        dataset_id=dataset.id,
        config_id=config_id,
        config_version=config_version,
        organization_id=auth.organization_id,
        project_id=auth.project_id,
    )

    run.type = run_type
    if inserted_at is not None:
        run.inserted_at = inserted_at
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


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

    def test_list_evaluation_runs_filters_by_dataset_id(self, db: Session) -> None:
        auth = get_user_test_auth_context(db)
        config = _create_config(db, auth.project_id)
        dataset_a = create_test_evaluation_dataset(
            db=db, organization_id=auth.organization_id, project_id=auth.project_id
        )
        dataset_b = create_test_evaluation_dataset(
            db=db, organization_id=auth.organization_id, project_id=auth.project_id
        )

        run_a = _create_run(db, auth, dataset_a, config)
        _create_run(db, auth, dataset_b, config)

        runs = list_evaluation_runs(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset_a.id,
        )

        assert [r.run_name for r in runs] == [run_a.run_name]

    def test_list_evaluation_runs_without_dataset_id_returns_all_datasets(
        self, db: Session
    ) -> None:
        auth = get_user_test_auth_context(db)
        config = _create_config(db, auth.project_id)
        dataset_a = create_test_evaluation_dataset(
            db=db, organization_id=auth.organization_id, project_id=auth.project_id
        )
        dataset_b = create_test_evaluation_dataset(
            db=db, organization_id=auth.organization_id, project_id=auth.project_id
        )

        run_a = _create_run(db, auth, dataset_a, config)
        run_b = _create_run(db, auth, dataset_b, config)

        runs = list_evaluation_runs(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
        )

        assert {r.run_name for r in runs} == {run_a.run_name, run_b.run_name}

    def test_list_evaluation_runs_dataset_from_other_project_excluded(
        self, db: Session
    ) -> None:
        user_auth = get_user_test_auth_context(db)
        other_auth = get_superuser_test_auth_context(db)

        other_dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=other_auth.organization_id,
            project_id=other_auth.project_id,
        )
        other_run = _create_run(
            db, other_auth, other_dataset, _create_config(db, other_auth.project_id)
        )

        user_dataset = create_test_evaluation_dataset(
            db=db,
            organization_id=user_auth.organization_id,
            project_id=user_auth.project_id,
        )
        _create_run(
            db, user_auth, user_dataset, _create_config(db, user_auth.project_id)
        )

        runs = list_evaluation_runs(
            session=db,
            organization_id=user_auth.organization_id,
            project_id=user_auth.project_id,
            dataset_id=other_dataset.id,
        )

        assert runs == []

        # positive control: the same dataset_id resolves in its owning project
        owner_runs = list_evaluation_runs(
            session=db,
            organization_id=other_auth.organization_id,
            project_id=other_auth.project_id,
            dataset_id=other_dataset.id,
        )
        assert [r.run_name for r in owner_runs] == [other_run.run_name]

    def test_list_evaluation_runs_dataset_id_composes_with_type_and_pagination(
        self, db: Session
    ) -> None:
        auth = get_user_test_auth_context(db)
        config = _create_config(db, auth.project_id)
        dataset_a = create_test_evaluation_dataset(
            db=db, organization_id=auth.organization_id, project_id=auth.project_id
        )
        dataset_b = create_test_evaluation_dataset(
            db=db, organization_id=auth.organization_id, project_id=auth.project_id
        )

        base = now()
        oldest = _create_run(
            db, auth, dataset_a, config, inserted_at=base - timedelta(minutes=3)
        )
        middle = _create_run(
            db, auth, dataset_a, config, inserted_at=base - timedelta(minutes=2)
        )
        newest = _create_run(
            db, auth, dataset_a, config, inserted_at=base - timedelta(minutes=1)
        )
        _create_run(
            db,
            auth,
            dataset_a,
            config,
            inserted_at=base,
            run_type=EvaluationType.STT.value,
        )
        _create_run(db, auth, dataset_b, config, inserted_at=base)

        first_page = list_evaluation_runs(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset_a.id,
            limit=2,
        )
        second_page = list_evaluation_runs(
            session=db,
            organization_id=auth.organization_id,
            project_id=auth.project_id,
            dataset_id=dataset_a.id,
            limit=2,
            offset=2,
        )

        assert [r.run_name for r in first_page] == [newest.run_name, middle.run_name]
        assert [r.run_name for r in second_page] == [oldest.run_name]
