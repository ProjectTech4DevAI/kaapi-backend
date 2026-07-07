import logging
from datetime import timedelta
from typing import Any

from sqlmodel import Session

from app.core.util import now
from app.crud.stats import get_daily_stats

logger = logging.getLogger(__name__)

DAILY_WINDOW = timedelta(hours=24)


def collect_daily_stats(*, session: Session) -> dict[str, Any]:
    end_at = now()
    start_at = end_at - DAILY_WINDOW
    logger.info(
        f"[collect_daily_stats] Starting | start_at: {start_at.isoformat()}, "
        f"end_at: {end_at.isoformat()}"
    )
    stats = get_daily_stats(session=session, start_at=start_at, end_at=end_at)
    logger.info(
        f"[collect_daily_stats] Completed | sections: {len(stats)}, "
        f"llm_calls: {len(stats['llm_call_token_summary'])}"
    )
    return {
        "window": {
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
        },
        "stats": stats,
    }
