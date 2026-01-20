from fastapi import APIRouter

from app.api.routes import (
    api_keys,
    assistants,
    collections,
    config,
    doc_transformation_job,
    documents,
    login,
    llm,
    organization,
    openai_conversation,
    project,
    responses,
    private,
    threads,
    users,
    utils,
    onboarding,
    credentials,
    cron,
    fine_tuning,
    model_evaluation,
    collection_job,
)
from app.api.routes.evaluations import dataset as evaluation_dataset, evaluation
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(api_keys.router)
api_router.include_router(assistants.router)
api_router.include_router(collections.router)
api_router.include_router(collection_job.router)
api_router.include_router(config.router)
api_router.include_router(credentials.router)
api_router.include_router(cron.router)
api_router.include_router(documents.router)
api_router.include_router(doc_transformation_job.router)
api_router.include_router(evaluation_dataset.router)
api_router.include_router(evaluation.router)
api_router.include_router(llm.router)
api_router.include_router(login.router)
api_router.include_router(onboarding.router)
api_router.include_router(openai_conversation.router)
api_router.include_router(organization.router)
api_router.include_router(project.router)
api_router.include_router(responses.router)
api_router.include_router(threads.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(fine_tuning.router)
api_router.include_router(model_evaluation.router)


if settings.ENVIRONMENT in ["development", "testing"]:
    api_router.include_router(private.router)
