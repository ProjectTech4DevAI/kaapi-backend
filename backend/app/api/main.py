from fastapi import APIRouter

from app.api.routes import (
    analytics,
    api_keys,
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
    guardrails,
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
from app.api.routes import (
    assessment as assessment_routes,
)
from app.api.routes.assessment import api as assessment_api_routes
from app.api.routes.evaluations.dataset_v2 import (
    router as evaluations_dataset_v2_router,
)
from app.api.routes.evaluations.evaluation_v2 import router as evaluations_v2_router

api_router = APIRouter()
api_router.include_router(analytics.router)
api_router.include_router(api_keys.router)
api_router.include_router(assessment_routes.router)
api_router.include_router(assessment_api_routes.router)
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
api_router.include_router(guardrails.router)
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


# v2 API surface (mounted at settings.API_V2_STR). Only the endpoints that differ
# from v1 live here — currently the judged run trigger. Everything else stays v1.
api_v2_router = APIRouter()
api_v2_router.include_router(evaluations_v2_router)
api_v2_router.include_router(evaluations_dataset_v2_router)
