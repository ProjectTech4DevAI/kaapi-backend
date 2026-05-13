import base64
import logging
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import col, func, select

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


MIGRATION_BATCH_SIZE = 50
MIGRATION_LOG_INTERVAL = 100


@router.post("/migrate/tts-base64-to-s3", include_in_schema=False)
def migrate_tts_base64_to_s3(session: SessionDep) -> dict:
    """
    One-shot migration: find all llm_call rows with input_type=text / output_type=audio
    whose content still holds raw base64, upload the audio to S3, and replace with a URI.

    Commits in batches so that partial progress is preserved on failure.
    """
    fn = "migrate_tts_base64_to_s3"
    start_time = time.monotonic()

    processed = skipped = failed = 0
    committed = 0
    pending_in_batch = 0
    errors: list[dict] = []

    # Storage instances are cached per project_id to avoid redundant DB lookups.
    storage_cache: dict[int, Any] = {}

    # --- count total candidates for progress logging ---
    count_stmt = (
        select(func.count())
        .select_from(LlmCall)
        .where(
            LlmCall.input_type == "text",
            LlmCall.output_type == "audio",
            col(LlmCall.deleted_at).is_(None),
        )
    )
    total_candidates = session.exec(count_stmt).one()
    logger.info(f"[{fn}] Starting migration | total_candidates={total_candidates}")

    statement = (
        select(LlmCall)
        .where(
            LlmCall.input_type == "text",
            LlmCall.output_type == "audio",
            col(LlmCall.deleted_at).is_(None),
        )
        .order_by(col(LlmCall.inserted_at).desc())
        .execution_options(yield_per=100)
    )

    for idx, call in enumerate(session.exec(statement), start=1):
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
            b64_size_kb = len(b64_value) / 1024
            audio_size_kb = len(audio_bytes) / 1024

            prefix = f"orgs/{call.organization_id}/{call.project_id}/audio/tts"
            s3_url = upload_audio_bytes_to_s3(
                storage,
                audio_bytes,
                call.id,
                audio_content.get("mime_type"),
                prefix,
            )

            if not s3_url:
                raise RuntimeError("upload_audio_bytes_to_s3 returned None")

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
            pending_in_batch += 1

            logger.debug(
                f"[{fn}] Uploaded | call_id={call.id}, "
                f"project_id={call.project_id}, "
                f"b64_kb={b64_size_kb:.1f}, audio_kb={audio_size_kb:.1f}, "
                f"s3_url={s3_url}"
            )

        except Exception as e:
            failed += 1
            errors.append(
                {
                    "call_id": str(call.id),
                    "project_id": str(call.project_id),
                    "error": str(e),
                }
            )
            logger.warning(
                f"[{fn}] Row failed | call_id={call.id}, "
                f"project_id={call.project_id}, error={e}",
                exc_info=True,
            )
            # Expunge the dirty object so the failed row doesn't poison the batch
            session.expunge(call)

        # --- batch commit for partial progress ---
        if pending_in_batch >= MIGRATION_BATCH_SIZE:
            try:
                session.commit()
                committed += pending_in_batch
                logger.info(
                    f"[{fn}] Batch committed | "
                    f"batch_size={pending_in_batch}, total_committed={committed}"
                )
            except Exception as e:
                logger.error(
                    f"[{fn}] Batch commit failed, rolling back | "
                    f"pending={pending_in_batch}, error={e}",
                    exc_info=True,
                )
                session.rollback()
                failed += pending_in_batch
                processed -= pending_in_batch
            pending_in_batch = 0

        # --- periodic progress log ---
        if idx % MIGRATION_LOG_INTERVAL == 0:
            elapsed = time.monotonic() - start_time
            logger.info(
                f"[{fn}] Progress | "
                f"scanned={idx}/{total_candidates}, "
                f"processed={processed}, skipped={skipped}, failed={failed}, "
                f"elapsed={elapsed:.1f}s"
            )

    # --- final batch ---
    if pending_in_batch > 0:
        try:
            session.commit()
            committed += pending_in_batch
            logger.info(
                f"[{fn}] Final batch committed | "
                f"batch_size={pending_in_batch}, total_committed={committed}"
            )
        except Exception as e:
            logger.error(
                f"[{fn}] Final batch commit failed, rolling back | "
                f"pending={pending_in_batch}, error={e}",
                exc_info=True,
            )
            session.rollback()
            failed += pending_in_batch
            processed -= pending_in_batch

    elapsed = time.monotonic() - start_time
    summary = {
        "processed": processed,
        "committed": committed,
        "skipped": skipped,
        "failed": failed,
        "total_candidates": total_candidates,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors[:50],
    }
    logger.info(f"[{fn}] Migration complete | {summary}")

    return summary


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
