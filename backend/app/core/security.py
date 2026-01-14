"""
Security module for handling authentication, encryption, and password management.
This module provides utilities for:
- JWT token generation and validation
- Password hashing and verification
- API key encryption/decryption
- Credentials encryption/decryption
"""

import base64
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Tuple

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from passlib.context import CryptContext
from sqlmodel import Session, and_, select

from app.models import APIKey, User, Organization, Project, AuthContext
from app.core.config import settings


logger = logging.getLogger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = "HS256"

# Fernet instance for encryption/decryption
_fernet = None


def get_encryption_key() -> bytes:
    """
    Generate a key for API key encryption using the app's secret key.

    Returns:
        bytes: A URL-safe base64 encoded encryption key derived from the app's secret key.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=settings.SECRET_KEY.encode(),
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(settings.SECRET_KEY.encode()))


def get_fernet() -> Fernet:
    """
    Get a Fernet instance with the encryption key.
    Uses singleton pattern to avoid creating multiple instances.

    Returns:
        Fernet: A Fernet instance initialized with the encryption key.
    """
    global _fernet
    if _fernet is None:
        _fernet = Fernet(get_encryption_key())
    return _fernet


def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
    """
    Create a JWT access token.

    Args:
        subject: The subject of the token (typically user ID)
        expires_delta: Token expiration time delta

    Returns:
        str: Encoded JWT token
    """
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {"exp": expire, "sub": str(subject)}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to check against

    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generate a password hash.

    Args:
        password: The plain text password to hash

    Returns:
        str: The hashed password
    """
    return pwd_context.hash(password)


def encrypt_credentials(credentials: dict) -> str:
    """
    Encrypt the entire credentials object before storage.

    Args:
        credentials: Dictionary containing credentials to encrypt

    Returns:
        str: The encrypted credentials

    Raises:
        ValueError: If encryption fails
    """
    try:
        credentials_str = json.dumps(credentials)
        return get_fernet().encrypt(credentials_str.encode()).decode()
    except Exception as e:
        raise ValueError(f"Failed to encrypt credentials: {e}")


def decrypt_credentials(encrypted_credentials: str) -> dict:
    """
    Decrypt the entire credentials object when retrieving it.

    Args:
        encrypted_credentials: The encrypted credentials string to decrypt

    Returns:
        dict: The decrypted credentials dictionary

    Raises:
        ValueError: If decryption fails
    """
    try:
        decrypted_str = get_fernet().decrypt(encrypted_credentials.encode()).decode()
        return json.loads(decrypted_str)
    except Exception as e:
        raise ValueError(f"Failed to decrypt credentials: {e}")


class APIKeyManager:
    """
    Handles secure API key generation and verification.

    Overview:
    - **Old Format (Legacy)**: 43 chars after "ApiKey ", with 12-char prefix and 31-char secret.
    - **New Format (Current)**: 65 chars after "ApiKey ", with 22-char prefix and 43-char secret.
    - Generates cryptographically secure API keys with fixed lengths,
    storing only the hashed secret using bcrypt while keeping the prefix in plaintext for quick lookup.
    Raw keys are displayed only once during creation for security.
    The system automatically verifies both old and new key formats to ensure backward compatibility.

    Compatibility:
    Both old and new formats are supported automatically during verification.
    """

    # Configuration constants
    PREFIX_NAME = "ApiKey "
    PREFIX_BYTES = 16  # Generates 22 chars in urlsafe base64
    SECRET_BYTES = 32  # Generates 43 chars in urlsafe base64
    PREFIX_LENGTH = 22
    KEY_LENGTH = 65  # Total length: 22 (prefix) + 43 (secret)
    HASH_ALGORITHM = "bcrypt"

    pwd_context = CryptContext(schemes=[HASH_ALGORITHM], deprecated="auto")

    @classmethod
    def generate(cls) -> Tuple[str, str, str]:
        """
        Generate a new API key with prefix and hashed value.
        Ensures exact lengths: prefix=22 chars, secret=43 chars.

        Returns:
            Tuple of (raw_key, key_prefix, key_hash)
        """
        # Generate tokens and ensure exact length
        secret_length = cls.KEY_LENGTH - cls.PREFIX_LENGTH
        key_prefix = secrets.token_urlsafe(cls.PREFIX_BYTES)[: cls.PREFIX_LENGTH].ljust(
            cls.PREFIX_LENGTH, "A"
        )
        secret_key = secrets.token_urlsafe(cls.SECRET_BYTES)[:secret_length].ljust(
            secret_length, "A"
        )

        # Construct raw key: "ApiKey {prefix}{secret}"
        raw_key = f"{cls.PREFIX_NAME}{key_prefix}{secret_key}"

        key_hash = cls.pwd_context.hash(secret_key)

        return raw_key, key_prefix, key_hash

    @classmethod
    def _extract_key_parts(cls, raw_key: str) -> Tuple[str, str] | None:
        """
        Extract prefix and secret from an API key based on its format.

        Supports:
        - New format: "ApiKey {22-char-prefix}{43-char-secret}"
        - Old format: "ApiKey {12-char-prefix}{31-char-secret}"

        Returns:
            Tuple[str, str] -> (key_prefix, secret_to_verify)
            or None if invalid
        """
        if not raw_key.startswith(cls.PREFIX_NAME):
            return None

        key_part = raw_key[len(cls.PREFIX_NAME) :]

        if len(key_part) == cls.KEY_LENGTH:
            key_prefix = key_part[: cls.PREFIX_LENGTH]
            secret_key = key_part[cls.PREFIX_LENGTH :]
            return key_prefix, secret_key

        old_key_length = 43
        old_prefix_length = 12
        if len(key_part) == old_key_length:
            key_prefix = key_part[:old_prefix_length]
            secret_key = key_part[old_prefix_length:]
            return key_prefix, secret_key

        # Invalid format
        return None

    @classmethod
    def verify(cls, session: Session, raw_key: str) -> AuthContext | None:
        """
        Verify an API key by checking its prefix and hashed value.
        Supports both old (43 chars) and new ("ApiKey " + 65 chars) formats.

        Eagerly loads User, Organization, and Project in a single query.

        Args:
            session: Database session
            raw_key: The raw API key to verify

        Returns:
            AuthContext if valid, None otherwise
        """
        try:
            key_parts = cls._extract_key_parts(raw_key)

            if not key_parts:
                return None

            key_prefix, secret = key_parts

            # Single query to fetch APIKey with User, Organization, and Project
            statement = (
                select(APIKey, User, Organization, Project)
                .where(
                    and_(
                        APIKey.key_prefix == key_prefix,
                        APIKey.is_deleted.is_(False),
                    )
                )
                .join(User, User.id == APIKey.user_id)
                .join(Organization, Organization.id == APIKey.organization_id)
                .join(Project, Project.id == APIKey.project_id)
            )

            result = session.exec(statement).first()

            if not result:
                return None
            api_key_record, user, organization, project = result
            auth_context = AuthContext(
                user=user,
                project=project,
                organization=organization,
            )

            # Verify the secret hash
            if cls.pwd_context.verify(secret, api_key_record.key_hash):
                return auth_context

            return None

        except Exception as e:
            logger.error(
                f"[APIKeyManager.verify] Error verifying API key: {str(e)}",
                exc_info=True,
            )
            return None


api_key_manager = APIKeyManager()
