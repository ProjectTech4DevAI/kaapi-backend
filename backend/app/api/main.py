from fastapi import APIRouter

from app.api.routes import (
    analytics,
    api_keys,
    assessment as assessment_routes,
    assistants,
    auth,
    collection_job,
    collections,
    config,
    credentials,
    cron,
    doc_transformation_job,
    documents,
    evaluations,
    features,
    fine_tuning,
    languages,
    llm,
    llm_chain,
    llm_sts,
    login,
    model_config,
    model_evaluation,
    onboarding,
    openai_conversation,
    organization,
    private,
    project,
    responses,
    threads,
    user_project,
    users,
    utils,
)
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(analytics.router)
api_router.include_router(api_keys.router)
api_router.include_router(assessment_routes.router)
api_router.include_router(assistants.router)
api_router.include_router(auth.router)
api_router.include_router(collection_job.router)
api_router.include_router(collections.router)
api_router.include_router(config.router)
api_router.include_router(credentials.router)
api_router.include_router(cron.router)
api_router.include_router(doc_transformation_job.router)
api_router.include_router(documents.router)
api_router.include_router(evaluations.router)
api_router.include_router(features.router)
api_router.include_router(fine_tuning.router)
api_router.include_router(languages.router)
api_router.include_router(llm.router)
api_router.include_router(llm_chain.router)
api_router.include_router(llm_sts.router)
api_router.include_router(login.router)
api_router.include_router(model_config.router)
api_router.include_router(model_evaluation.router)
api_router.include_router(onboarding.router)
api_router.include_router(openai_conversation.router)
api_router.include_router(organization.router)
api_router.include_router(project.router)
api_router.include_router(responses.router)
api_router.include_router(threads.router)
api_router.include_router(user_project.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(private.router)
# if settings.ENVIRONMENT in ["development", "testing"]:
#     api_router.include_router(private.router)
