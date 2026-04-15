import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException

from app.api.deps import SessionDep
from app.crud.model_config import get_model_config, list_active_model_configs
from app.models import ModelConfigListPublic, ModelConfigPublic
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
    provider: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> APIResponse[ModelConfigListPublic]:
    models = list_active_model_configs(
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
) -> APIResponse[dict[str, list[ModelConfigPublic]]]:
    models = list_active_model_configs(session=session, skip=0, limit=1000)
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
) -> APIResponse[list[str]]:
    models = list_active_model_configs(session=session, skip=0, limit=1000)
    providers = sorted({model.provider for model in models})
    return APIResponse.success_response(providers)


@router.get(
    "/{provider}/{model_name}",
    response_model=APIResponse[ModelConfigPublic],
    description=load_description("model_config/get_model.md"),
)
def get_model(
    session: SessionDep, provider: str, model_name: str
) -> APIResponse[ModelConfigPublic]:
    model = get_model_config(session=session, provider=provider, model_name=model_name)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return APIResponse.success_response(model)
