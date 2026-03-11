"""Test cases for STT evaluation dataset CRUD operations."""

from sqlmodel import Session, select

from app.core.util import now
from app.crud.language import get_language_by_locale
from app.crud.stt_evaluations.dataset import create_stt_dataset, create_stt_samples
from app.models import EvaluationDataset, Organization, Project, File, FileType
from app.models.stt_evaluation import STTSampleCreate, EvaluationType


def create_test_file(
    db: Session,
    organization_id: int,
    project_id: int,
    filename: str = "test.mp3",
) -> File:
    """Create a test file record."""
    file = File(
        object_store_url=f"s3://test-bucket/audio/{filename}",
        filename=filename,
        size_bytes=1024,
        content_type="audio/mpeg",
        file_type=FileType.AUDIO.value,
        organization_id=organization_id,
        project_id=project_id,
        inserted_at=now(),
        updated_at=now(),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


class TestCreateSTTSamplesLanguageId:
    """Test per-sample language_id override in create_stt_samples."""

    def test_sample_inherits_dataset_language_when_not_specified(
        self, db: Session
    ) -> None:
        """Test that samples inherit dataset language_id when sample language_id is None."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        language = get_language_by_locale(session=db, locale="en")

        file = create_test_file(db, org.id, project.id, filename="audio1.mp3")

        dataset = create_stt_dataset(
            session=db,
            name="dataset_with_lang",
            org_id=org.id,
            project_id=project.id,
            language_id=language.id,
            dataset_metadata={"sample_count": 1, "has_ground_truth_count": 0},
        )

        samples = [
            STTSampleCreate(file_id=file.id, ground_truth="Hello"),
        ]

        created = create_stt_samples(session=db, dataset=dataset, samples=samples)

        assert len(created) == 1
        assert created[0].language_id == language.id

    def test_sample_uses_own_language_when_specified(self, db: Session) -> None:
        """Test that sample uses its own language_id when explicitly provided."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        en_language = get_language_by_locale(session=db, locale="en")
        hi_language = get_language_by_locale(session=db, locale="hi")

        file = create_test_file(db, org.id, project.id, filename="audio2.mp3")

        dataset = create_stt_dataset(
            session=db,
            name="dataset_lang_override",
            org_id=org.id,
            project_id=project.id,
            language_id=en_language.id,
            dataset_metadata={"sample_count": 1, "has_ground_truth_count": 0},
        )

        samples = [
            STTSampleCreate(
                file_id=file.id,
                ground_truth="नमस्ते",
                language_id=hi_language.id,
            ),
        ]

        created = create_stt_samples(session=db, dataset=dataset, samples=samples)

        assert len(created) == 1
        assert created[0].language_id == hi_language.id
        assert created[0].language_id != en_language.id

    def test_mixed_samples_with_and_without_language_id(self, db: Session) -> None:
        """Test mix of samples: some with language_id, some inheriting from dataset."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        en_language = get_language_by_locale(session=db, locale="en")
        hi_language = get_language_by_locale(session=db, locale="hi")

        file1 = create_test_file(db, org.id, project.id, filename="audio3.mp3")
        file2 = create_test_file(db, org.id, project.id, filename="audio4.mp3")
        file3 = create_test_file(db, org.id, project.id, filename="audio5.mp3")

        dataset = create_stt_dataset(
            session=db,
            name="dataset_mixed_lang",
            org_id=org.id,
            project_id=project.id,
            language_id=en_language.id,
            dataset_metadata={"sample_count": 3, "has_ground_truth_count": 0},
        )

        samples = [
            STTSampleCreate(file_id=file1.id, ground_truth="Hello"),
            STTSampleCreate(
                file_id=file2.id,
                ground_truth="नमस्ते",
                language_id=hi_language.id,
            ),
            STTSampleCreate(file_id=file3.id, ground_truth="World"),
        ]

        created = create_stt_samples(session=db, dataset=dataset, samples=samples)

        assert len(created) == 3
        # First sample: no language_id specified -> inherits dataset's en
        assert created[0].language_id == en_language.id
        # Second sample: explicit hi language_id -> uses hi
        assert created[1].language_id == hi_language.id
        # Third sample: no language_id specified -> inherits dataset's en
        assert created[2].language_id == en_language.id

    def test_dataset_defaults_to_none_language(self, db: Session) -> None:
        """Test that dataset defaults to None language_id when not specified."""
        org = db.exec(select(Organization)).first()
        project = db.exec(
            select(Project).where(Project.organization_id == org.id)
        ).first()

        file = create_test_file(db, org.id, project.id, filename="audio6.mp3")

        dataset = create_stt_dataset(
            session=db,
            name="dataset_default_lang",
            org_id=org.id,
            project_id=project.id,
            dataset_metadata={"sample_count": 1, "has_ground_truth_count": 0},
        )

        assert dataset.language_id is None

        samples = [
            STTSampleCreate(file_id=file.id, ground_truth="Hello"),
        ]

        created = create_stt_samples(session=db, dataset=dataset, samples=samples)

        assert len(created) == 1
        assert created[0].language_id is None
