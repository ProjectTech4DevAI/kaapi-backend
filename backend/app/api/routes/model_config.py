import logging
from typing import Literal
from collections import defaultdict

from fastapi import APIRouter, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.crud.model_config import (
    estimate_model_cost,
    get_active_models,
    get_model_config,
)
from app.models import ModelConfigPublic, ModelConfigListPublic
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Model Config"])


@router.get(
    "/",
    response_model=APIResponse[ModelConfigListPublic],
    description=load_description("model_config/list_models.md"),
)
def list_models(
    session: SessionDep,
    auth_context: AuthContextDep,
    provider: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> APIResponse[ModelConfigListPublic]:
    models = get_active_models(
        session=session, provider=provider, skip=skip, limit=limit
    )
    return APIResponse.success_response(
        ModelConfigListPublic(data=models, count=len(models))
    )


@router.get(
    "/grouped",
    response_model=APIResponse[dict[str, list[ModelConfigPublic]]],
    description=load_description("model_config/list_models_grouped.md"),
)
def list_models_grouped(
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[dict[str, list[ModelConfigPublic]]]:
    models = get_active_models(session=session, skip=0, limit=1000)
    grouped: dict[str, list[ModelConfigPublic]] = defaultdict(list)
    for model in models:
        grouped[model.provider].append(model)

    return APIResponse.success_response(dict(grouped))


@router.get(
    "/providers",
    response_model=APIResponse[list[str]],
    description=load_description("model_config/list_providers.md"),
)
def list_providers(
    session: SessionDep,
    auth_context: AuthContextDep,
) -> APIResponse[list[str]]:
    models = get_active_models(session=session, skip=0, limit=1000)
    providers = sorted({model.provider for model in models})
    return APIResponse.success_response(providers)


@router.get(
    "/{provider}/{model_name}",
    response_model=APIResponse[ModelConfigPublic],
    description=load_description("model_config/get_model.md"),
)
def get_model(
    session: SessionDep, auth_context: AuthContextDep, provider: str, model_name: str
) -> APIResponse[ModelConfigPublic]:
    model = get_model_config(session=session, provider=provider, model_name=model_name)

    if model is None:
        logger.error(
            f"[get_model] Model not found | provider={provider}, model_name={model_name}"
        )
        raise HTTPException(status_code=404, detail="Model not found")

    return APIResponse.success_response(model)
