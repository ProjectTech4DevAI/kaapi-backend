import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import SessionDep
from app.models.llm.constants import Provider
from app.crud.model_config import (
    bulk_create_model_configs,
    bulk_update_model_configs,
    delete_model_config,
    get_model_config,
    list_active_model_configs,
    list_all_active_model_configs,
    update_model_config,
)
from app.models import (
    ModelConfigBulkUpdateItem,
    ModelConfigCreate,
    ModelConfigListPublic,
    ModelConfigPublic,
    ModelConfigUpdate,
)
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Model Config"])


@router.get(
    "",
    response_model=APIResponse[ModelConfigListPublic],
    description=load_description("model_config/list_models.md"),
)
def list_models(
    session: SessionDep,
    provider: Provider | None = None,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum records to return"),
) -> APIResponse[ModelConfigListPublic]:
    models, has_more = list_active_model_configs(
        session=session, provider=provider, skip=skip, limit=limit
    )
    return APIResponse.success_response(
        ModelConfigListPublic(data=models, count=len(models)),
        metadata={"has_more": has_more},
    )


@router.get(
    "/grouped",
    response_model=APIResponse[dict[str, list[ModelConfigPublic]]],
    description=load_description("model_config/list_models_grouped.md"),
)
def list_models_grouped(
    session: SessionDep,
    skip: int = Query(0, ge=0, description="Number of model records to skip"),
    limit: int = Query(
        100, ge=1, le=100, description="Maximum model records to return before grouping"
    ),
) -> APIResponse[dict[str, list[ModelConfigPublic]]]:
    models, has_more = list_active_model_configs(
        session=session, skip=skip, limit=limit
    )
    grouped: dict[str, list[ModelConfigPublic]] = defaultdict(list)
    for model in models:
        grouped[model.provider].append(model)

    return APIResponse.success_response(dict(grouped), metadata={"has_more": has_more})


@router.get(
    "/providers",
    response_model=APIResponse[list[str]],
    description=load_description("model_config/list_providers.md"),
)
def list_providers(
    session: SessionDep,
) -> APIResponse[list[str]]:
    models = list_all_active_model_configs(session=session)
    providers = sorted({model.provider for model in models})
    return APIResponse.success_response(providers)


@router.get(
    "/{provider}/{model_name}",
    response_model=APIResponse[ModelConfigPublic],
    description=load_description("model_config/get_model.md"),
)
def get_model(
    session: SessionDep, provider: Provider, model_name: str
) -> APIResponse[ModelConfigPublic]:
    model = get_model_config(session=session, provider=provider, model_name=model_name)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return APIResponse.success_response(model)


@router.post(
    "",
    response_model=APIResponse[list[ModelConfigPublic]],
    description=load_description("model_config/create_models.md"),
    status_code=201,
)
def create_models(
    session: SessionDep, data: ModelConfigCreate | list[ModelConfigCreate]
) -> APIResponse[list[ModelConfigPublic]]:
    items = data if isinstance(data, list) else [data]
    models = bulk_create_model_configs(session=session, items=items)
    return APIResponse.success_response(models)


@router.patch(
    "/{provider}/{model_name}",
    response_model=APIResponse[ModelConfigPublic],
    description=load_description("model_config/update_model.md"),
)
def update_model(
    session: SessionDep, provider: Provider, model_name: str, data: ModelConfigUpdate
) -> APIResponse[ModelConfigPublic]:
    model = update_model_config(
        session=session, provider=provider, model_name=model_name, data=data
    )
    return APIResponse.success_response(model)


@router.patch(
    "",
    response_model=APIResponse[list[ModelConfigPublic]],
    description=load_description("model_config/update_models.md"),
    summary="Update Models",
)
def bulk_update_models(
    session: SessionDep, items: list[ModelConfigBulkUpdateItem]
) -> APIResponse[list[ModelConfigPublic]]:
    models = bulk_update_model_configs(session=session, items=items)
    return APIResponse.success_response(models)


@router.delete(
    "/{provider}/{model_name}",
    response_model=APIResponse[None],
    description=load_description("model_config/delete_model.md"),
)
def delete_model(
    session: SessionDep, provider: Provider, model_name: str
) -> APIResponse[None]:
    delete_model_config(session=session, provider=provider, model_name=model_name)
    return APIResponse.success_response(None)
