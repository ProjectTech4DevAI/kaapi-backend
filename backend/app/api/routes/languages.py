import logging

from fastapi import APIRouter, HTTPException

from app.api.deps import AuthContextDep, SessionDep
from app.crud.language import get_language_by_id, get_languages
from app.models import LanguagePublic, LanguagesPublic
from app.utils import APIResponse, load_description

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/languages", tags=["Languages"])


@router.get(
    "/",
    response_model=APIResponse[LanguagesPublic],
    description=load_description("languages/list.md"),
)
def list_languages(
    session: SessionDep, auth_context: AuthContextDep, skip: int = 0, limit: int = 100
):
    """
    Retrieve all active languages.
    """
    languages = get_languages(session=session, skip=skip, limit=limit)
    return APIResponse.success_response(
        LanguagesPublic(data=languages, count=len(languages))
    )


@router.get(
    "/{language_id}",
    response_model=APIResponse[LanguagePublic],
    description=load_description("languages/get.md"),
)
def get_language(session: SessionDep, auth_context: AuthContextDep, language_id: int):
    """
    Retrieve a language by ID.
    """
    language = get_language_by_id(session=session, language_id=language_id)
    if language is None:
        logger.error(f"[get_language] Language not found | language_id={language_id}")
        raise HTTPException(status_code=404, detail="Language not found")
    return APIResponse.success_response(language)
