"""Config preconditions for a fast run (`validate_fast_evaluation_inputs`).

Generation now goes through `execute_llm_call`, which interpolates the config's
`prompt_template` unconditionally. A template without `{{input}}` would send the
template alone and silently drop every dataset question, so the run is rejected up
front. The passing cases are the regression net: this gate sees configs that ran
fine before the change, and a false positive is an outage for existing eval users.
"""

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.models import Config, EvaluationDataset
from app.models.llm.request import (
    ConfigBlob,
    PromptTemplate,
    build_kaapi_completion_config,
)
from app.services.evaluations.fast import (
    ERR_CONFIG_TEMPLATE_MISSING_INPUT,
    validate_fast_evaluation_inputs,
)
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
)


def _config_with_template(db: Session, project_id: int, template: str | None) -> Config:
    """A text-OpenAI Kaapi config. The model name is absent from ModelConfig on
    purpose: blob validation short-circuits before loading that row, whose
    ARRAY-of-enum column trips a SQLAlchemy result-processor bug in some envs."""
    blob = ConfigBlob(
        completion=build_kaapi_completion_config(
            provider="openai",
            type="text",
            params={"model": "gpt-4o-fast-eval-test", "temperature": 0.7},
        ),
        prompt_template=PromptTemplate(template=template) if template else None,
    )
    return create_test_config(
        db=db, project_id=project_id, use_kaapi_schema=True, config_blob=blob
    )


@pytest.fixture
def dataset(db: Session, user_api_key: TestAuthContext) -> EvaluationDataset:
    return create_test_evaluation_dataset(
        db=db,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
        original_items_count=3,
        duplication_factor=1,
    )


def _validate(
    db: Session,
    user_api_key: TestAuthContext,
    dataset: EvaluationDataset,
    template: str | None,
) -> EvaluationDataset:
    config = _config_with_template(db, user_api_key.project_id, template)
    return validate_fast_evaluation_inputs(
        session=db,
        dataset_id=dataset.id,
        config_id=config.id,
        config_version=1,
        organization_id=user_api_key.organization_id,
        project_id=user_api_key.project_id,
    )


class TestPromptTemplatePlaceholder:
    def test_template_without_the_placeholder_is_rejected(
        self, db: Session, user_api_key: TestAuthContext, dataset: EvaluationDataset
    ) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _validate(db, user_api_key, dataset, "Answer in Hindi. Be concise.")

        assert exc_info.value.status_code == 422
        assert ERR_CONFIG_TEMPLATE_MISSING_INPUT in exc_info.value.detail

    def test_template_with_the_placeholder_passes(
        self, db: Session, user_api_key: TestAuthContext, dataset: EvaluationDataset
    ) -> None:
        validated = _validate(db, user_api_key, dataset, "Answer in Hindi: {{input}}")

        assert validated.id == dataset.id

    def test_config_without_a_template_passes(
        self, db: Session, user_api_key: TestAuthContext, dataset: EvaluationDataset
    ) -> None:
        validated = _validate(db, user_api_key, dataset, None)

        assert validated.id == dataset.id
