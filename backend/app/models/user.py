from pydantic import EmailStr
from sqlmodel import Field, SQLModel


# Shared properties
class UserBase(SQLModel):
    """Base model for users with common data fields."""

    email: EmailStr = Field(
        unique=True,
        index=True,
        max_length=255,
        sa_column_kwargs={"comment": "User's email address"},
    )
    is_active: bool = Field(
        default=True,
        sa_column_kwargs={"comment": "Flag indicating if the user account is active"},
    )
    is_superuser: bool = Field(
        default=False,
        sa_column_kwargs={
            "comment": "Flag indicating if user has superuser privileges"
        },
    )
    full_name: str | None = Field(
        default=None,
        max_length=255,
        sa_column_kwargs={"comment": "User's full name"},
    )


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=40)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=40)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(UserBase):
    email: EmailStr | None = Field(default=None, max_length=255)  # type: ignore
    password: str | None = Field(default=None, min_length=8, max_length=40)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=40)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=40)
    new_password: str = Field(min_length=8, max_length=40)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    """Database model for users."""

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the user"},
    )
    hashed_password: str = Field(
        sa_column_kwargs={"comment": "Bcrypt hash of the user's password"},
    )


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: int


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int
