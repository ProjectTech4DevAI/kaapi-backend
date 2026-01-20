from fastapi import APIRouter, Depends
from pydantic.networks import EmailStr

from app.models import Message
from app.utils import generate_test_email, send_email
from app.api.permissions import Permission, require_permission

router = APIRouter(prefix="/utils", tags=["utils"])


@router.post(
    "/test-email/",
    dependencies=[Depends(require_permission(Permission.SUPERUSER))],
    status_code=201,
    include_in_schema=False,
)
def test_email(email_to: EmailStr) -> Message:
    """
    Test emails.
    """
    email_data = generate_test_email(email_to=email_to)
    send_email(
        email_to=email_to,
        subject=email_data.subject,
        html_content=email_data.html_content,
    )
    return Message(message="Test email sent")


@router.get("/health/", include_in_schema=False)
async def health_check() -> bool:
    return True
