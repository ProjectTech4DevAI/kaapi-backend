from sqlmodel import Session, create_engine, select

from app import crud
from app.models import User, UserCreate


def get_engine():
    """Get database engine with current settings."""
    # Import settings dynamically to get the current instance
    from app.core.config import settings

    # Configure connection pool settings
    # For testing, we need more connections since tests run in parallel
    pool_size = 5
    max_overflow = 10

    if settings.ENVIRONMENT == "development":
        pool_size = 20
        max_overflow = 30
    elif settings.ENVIRONMENT == "staging":
        pool_size = 12
        # test it out
        max_overflow = 0
    return create_engine(
        str(settings.SQLALCHEMY_DATABASE_URI),
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=300,  # Recycle connections after 5 minutes
    )


# Create a default engine for backward compatibility
engine = get_engine()


# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/fastapi/full-stack-fastapi-template/issues/28


def init_db(session: Session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next lines
    # from sqlmodel import SQLModel

    # This works because the models are already imported and registered from app.models
    # SQLModel.metadata.create_all(engine)

    user = session.exec(
        select(User).where(User.email == settings.FIRST_SUPERUSER)
    ).first()
    if not user:
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud.create_user(session=session, user_create=user_in)
