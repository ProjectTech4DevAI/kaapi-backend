from sqlmodel import Session

from app.models import (
    Organization,
    Project,
    APIKeyCreateResponse,
    Credential,
    OrganizationCreate,
    ProjectCreate,
    ConfigBlob,
    CredsCreate,
    FineTuningJobCreate,
    Fine_Tuning,
    ModelEvaluation,
    ModelEvaluationBase,
    ModelEvaluationStatus,
    Config,
    ConfigCreate,
    ConfigVersion,
    ConfigVersionCreate,
    EvaluationDataset,
)
from app.models.llm import KaapiLLMParams, KaapiCompletionConfig, NativeCompletionConfig
from app.crud import (
    create_organization,
    create_project,
    set_creds_for_org,
    create_fine_tuning_job,
    create_model_evaluation,
    APIKeyCrud,
)
from app.crud.config import ConfigCrud, ConfigVersionCrud
from app.core.providers import Provider
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import (
    random_lower_string,
    generate_random_string,
    get_document,
    get_project,
)


def create_test_organization(db: Session) -> Organization:
    """
    Creates and returns a test organization with a unique name.

    Persists the organization to the database.
    """
    name = f"TestOrg-{random_lower_string()}"
    org_in = OrganizationCreate(name=name, is_active=True)
    return create_organization(session=db, org_create=org_in)


def create_test_project(db: Session) -> Project:
    """
    Creates and returns a test project under a newly created test organization.

    Persists both the organization and the project to the database.

    """
    org = create_test_organization(db)
    name = f"TestProject-{random_lower_string()}"
    project_description = "This is a test project description."
    project_in = ProjectCreate(
        name=name,
        description=project_description,
        is_active=True,
        organization_id=org.id,
    )
    return create_project(session=db, project_create=project_in)


def test_credential_data(db: Session) -> CredsCreate:
    """
    Returns credential data for a test project in the form of a CredsCreate schema.

    Use this when you just need credential input data without persisting it to the database.
    """
    api_key = "sk-" + generate_random_string(10)
    creds_data = CredsCreate(
        is_active=True,
        credential={
            Provider.OPENAI.value: {
                "api_key": api_key,
                "model": "gpt-4",
                "temperature": 0.7,
            }
        },
    )
    return creds_data


def create_test_api_key(
    db: Session,
    project_id: int | None = None,
    user_id: int | None = None,
) -> APIKeyCreateResponse:
    """
    Creates and returns a test API key for a specific project and user.

    Persists the API key to the database.
    """
    if user_id is None:
        user = create_random_user(db)
        user_id = user.id

    if project_id is None:
        project = create_test_project(db)
        project_id = project.id

    api_key_crud = APIKeyCrud(session=db, project_id=project_id)
    raw_key, api_key = api_key_crud.create(user_id=user_id, project_id=project_id)
    return APIKeyCreateResponse(key=raw_key, **api_key.model_dump())


def create_test_credential(db: Session) -> tuple[list[Credential], Project]:
    """
    Creates and returns test credentials (OpenAI and Langfuse) for a test project.

    Persists the organization, project, and both credentials to the database.
    This ensures that tests using this helper have both OpenAI and Langfuse credentials available.
    """
    project = create_test_project(db)

    # Create OpenAI credentials
    api_key = "sk-" + generate_random_string(10)
    openai_creds = CredsCreate(
        is_active=True,
        credential={
            Provider.OPENAI.value: {
                "api_key": api_key,
                "model": "gpt-4",
                "temperature": 0.7,
            }
        },
    )
    openai_credentials = set_creds_for_org(
        session=db,
        creds_add=openai_creds,
        organization_id=project.organization_id,
        project_id=project.id,
    )

    # Create Langfuse credentials
    langfuse_creds = CredsCreate(
        is_active=True,
        credential={
            Provider.LANGFUSE.value: {
                "secret_key": "sk-lf-" + generate_random_string(10),
                "public_key": "pk-lf-" + generate_random_string(10),
                "host": "https://cloud.langfuse.com",
            }
        },
    )
    langfuse_credentials = set_creds_for_org(
        session=db,
        creds_add=langfuse_creds,
        organization_id=project.organization_id,
        project_id=project.id,
    )

    # Return both credentials combined
    return (openai_credentials + langfuse_credentials, project)


def create_test_fine_tuning_jobs(
    db: Session,
    ratios: list[float],
) -> tuple[list[Fine_Tuning], bool]:
    project = get_project(db, "Dalgo")
    document = get_document(db, "dalgo_sample.json")
    jobs = []
    any_created = False

    for ratio in ratios:
        job_request = FineTuningJobCreate(
            document_id=document.id,
            base_model="gpt-4",
            split_ratio=[ratio],
            system_prompt="str",
        )
        job, created = create_fine_tuning_job(
            session=db,
            request=job_request,
            split_ratio=ratio,
            project_id=project.id,
            organization_id=project.organization_id,
        )
        jobs.append(job)
        if created:
            any_created = True

    return jobs, any_created


def create_test_finetuning_job_with_extra_fields(
    db: Session,
    ratios: list[float],
) -> tuple[list[Fine_Tuning], bool]:
    jobs, _ = create_test_fine_tuning_jobs(db, ratios)

    if jobs:
        for job in jobs:
            job.test_data_s3_object = "test_data_s3_object_example"
            job.fine_tuned_model = "fine_tuned_model_name"

    return jobs, True


def create_test_model_evaluation(db: Session) -> list[ModelEvaluation]:
    fine_tune_jobs, _ = create_test_finetuning_job_with_extra_fields(db, [0.5, 0.7])

    model_evaluations = []

    for fine_tune in fine_tune_jobs:
        request = ModelEvaluationBase(
            fine_tuning_id=fine_tune.id,
            system_prompt=fine_tune.system_prompt,
            base_model=fine_tune.base_model,
            fine_tuned_model=fine_tune.fine_tuned_model,
            document_id=fine_tune.document_id,
            test_data_s3_object=fine_tune.test_data_s3_object,
        )

        model_eval = create_model_evaluation(
            session=db,
            request=request,
            project_id=fine_tune.project_id,
            organization_id=fine_tune.organization_id,
            status=ModelEvaluationStatus.pending,
        )

        model_evaluations.append(model_eval)

    return model_evaluations


def create_test_config(
    db: Session,
    project_id: int | None = None,
    name: str | None = None,
    description: str | None = None,
    config_blob: ConfigBlob | None = None,
    use_kaapi_schema: bool = False,
) -> Config:
    """
    Creates and returns a test configuration with an initial version.

    Persists the config and version to the database.

    Args:
        db: Database session
        project_id: Project ID (creates new project if None)
        name: Config name (generates random if None)
        description: Config description
        config_blob: Config blob (creates default if None)
        use_kaapi_schema: If True, creates Kaapi-format config; if False, creates native format
    """
    if project_id is None:
        project = create_test_project(db)
        project_id = project.id

    if name is None:
        name = f"test-config-{random_lower_string()}"

    if config_blob is None:
        if use_kaapi_schema:
            # Create Kaapi-format config
            config_blob = ConfigBlob(
                completion=KaapiCompletionConfig(
                    provider="openai",
                    params=KaapiLLMParams(
                        model="gpt-4",
                        temperature=0.7,
                    ),
                )
            )
        else:
            # Create native-format config
            config_blob = ConfigBlob(
                completion=NativeCompletionConfig(
                    provider="openai-native",
                    params={
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    },
                )
            )

    config_create = ConfigCreate(
        name=name,
        description=description or "Test configuration description",
        config_blob=config_blob,
        commit_message="Initial version",
    )

    config_crud = ConfigCrud(session=db, project_id=project_id)
    config, version = config_crud.create_or_raise(config_create)

    return config


def create_test_version(
    db: Session,
    config_id: int,
    project_id: int,
    config_blob: ConfigBlob | None = None,
    commit_message: str | None = None,
) -> ConfigVersion:
    """
    Creates and returns a test version for an existing configuration.

    Persists the version to the database.
    """
    if config_blob is None:
        config_blob = ConfigBlob(
            completion=NativeCompletionConfig(
                provider="openai-native",
                params={
                    "model": "gpt-4",
                    "temperature": 0.8,
                    "max_tokens": 1500,
                },
            )
        )

    version_create = ConfigVersionCreate(
        config_blob=config_blob,
        commit_message=commit_message or "Test version commit",
    )

    version_crud = ConfigVersionCrud(
        session=db, project_id=project_id, config_id=config_id
    )
    version = version_crud.create_or_raise(version_create=version_create)

    return version


def create_test_evaluation_dataset(
    db: Session,
    organization_id: int,
    project_id: int,
    name: str | None = None,
    description: str | None = None,
    original_items_count: int = 3,
    duplication_factor: int = 1,
) -> EvaluationDataset:
    """
    Creates and returns a test evaluation dataset.

    Persists the dataset to the database.
    """
    if name is None:
        name = f"test_dataset_{random_lower_string()}"

    total_items_count = original_items_count * duplication_factor

    dataset = EvaluationDataset(
        name=name,
        description=description or "Test evaluation dataset",
        dataset_metadata={
            "original_items_count": original_items_count,
            "total_items_count": total_items_count,
            "duplication_factor": duplication_factor,
        },
        langfuse_dataset_id=f"langfuse_{random_lower_string()}",
        object_store_url=f"s3://test/{name}.csv",
        organization_id=organization_id,
        project_id=project_id,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset
