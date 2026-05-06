import base64
import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import col, select

from app.api.deps import SessionDep
from app.core.cloud.storage import get_cloud_storage
from app.core.security import get_password_hash
from app.core.storage_utils import upload_audio_bytes_to_s3
from app.core.util import now
from app.models import (
    LlmCall,
    User,
    UserPublic,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["private"], prefix="/private")


class PrivateUserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    is_verified: bool = False


@router.post("/migrate/tts-base64-to-s3", include_in_schema=False)
def migrate_tts_base64_to_s3(session: SessionDep) -> dict:
    """
    One-shot migration: find all llm_call rows with input_type=text / output_type=audio
    whose content still holds raw base64, upload the audio to S3, and replace with a URI.
    """
    processed = skipped = failed = 0
    errors: list[dict] = []

    # Storage instances are cached per project_id to avoid redundant DB lookups.
    storage_cache: dict[int, Any] = {}

    statement = (
        select(LlmCall)
        .where(
            LlmCall.input_type == "text",
            LlmCall.output_type == "audio",
            col(LlmCall.deleted_at).is_(None),
        )
        .order_by(col(LlmCall.created_at).desc())
        .execution_options(yield_per=100)
    )

    for call in session.exec(statement):
        content = call.content
        if not content:
            skipped += 1
            continue

        audio_content = content.get("content", {})
        if audio_content.get("format") != "base64":
            skipped += 1
            continue

        b64_value = audio_content.get("value")
        if not b64_value:
            skipped += 1
            continue

        try:
            if call.project_id not in storage_cache:
                storage_cache[call.project_id] = get_cloud_storage(
                    session, call.project_id
                )
            storage = storage_cache[call.project_id]

            audio_bytes = base64.b64decode(b64_value)
            s3_url = upload_audio_bytes_to_s3(
                storage,
                audio_bytes,
                call.id,
                audio_content.get("mime_type"),
                "llm/tts/audio",
            )

            if not s3_url:
                raise RuntimeError("upload returned None")

            call.content = {
                "type": "audio",
                "content": {
                    "format": "uri",
                    "value": s3_url,
                    "mime_type": audio_content.get("mime_type"),
                },
            }
            call.updated_at = now()
            session.add(call)
            processed += 1

        except Exception as e:
            failed += 1
            errors.append({"call_id": str(call.id), "error": str(e)})
            logger.warning(
                f"[migrate_tts_base64_to_s3] Failed | call_id={call.id}, error={e}"
            )

    session.commit()

    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors[:50],
    }


@router.post("/users", response_model=UserPublic, include_in_schema=False)
def create_user(user_in: PrivateUserCreate, session: SessionDep) -> Any:
    """
    Create a new user.
    """

    user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
    )

    session.add(user)
    session.commit()

    return user
