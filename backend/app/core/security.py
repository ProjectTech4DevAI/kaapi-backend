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
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
import jwt
from botocore.client import BaseClient
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from sqlmodel import Session, and_, select

from app.core.config import settings
from app.core.exceptions import UpstreamError
from app.models import APIKey, AuthContext, Organization, Project, User

logger = logging.getLogger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = "HS256"

# Fernet instance for encryption/decryption
_fernet = None

# Marks KMS-encrypted credentials; rows without it are legacy Fernet.
KMS_CIPHERTEXT_PREFIX = "kms.v1:"

_kms_client: BaseClient | None = None


def _use_kms() -> bool:
    """KMS is used everywhere except development, and only when a key is set."""
    return settings.ENVIRONMENT != "development" and bool(settings.AWS_KMS_KEY_ID)


def get_kms_client() -> BaseClient:
    """Singleton boto3 KMS client. Empty AWS_* settings are omitted so boto3
    falls back to the task role / instance profile credential chain."""
    global _kms_client
    if _kms_client is None:
        cred_params = (
            ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
            ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
            ("region_name", "AWS_DEFAULT_REGION"),
        )
        kwargs = {}
        for param, env_var in cred_params:
            value = os.environ.get(env_var, getattr(settings, env_var))
            if value:
                kwargs[param] = value
        _kms_client = boto3.client("kms", **kwargs)
    return _kms_client


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


def encode_jwt_token(
    subject: str | Any,
    token_type: str,
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Encode a JWT with standard `exp`, `nbf`, `sub`, and `type` claims.

    Any additional claims (e.g. `org_id`, `project_id`) can be passed via
    `extra_claims` and are merged into the payload before signing.
    """
    now = datetime.now(UTC)
    to_encode: dict[str, Any] = {
        "exp": now + expires_delta,
        "nbf": now,
        "sub": str(subject),
        "type": token_type,
    }
    if extra_claims:
        to_encode.update({k: v for k, v in extra_claims.items() if v is not None})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_jwt_token(
    token: str, expected_type: str | None = None
) -> dict[str, Any] | None:
    """Decode and verify a JWT. Returns the payload or None if invalid.

    If `expected_type` is given, the token's `type` claim must match.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except InvalidTokenError:
        return None
    if expected_type is not None and payload.get("type") != expected_type:
        return None
    return payload


def create_access_token(
    subject: str | Any,
    expires_delta: timedelta,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> str:
    """Create a JWT access token."""
    return encode_jwt_token(
        subject=subject,
        token_type="access",
        expires_delta=expires_delta,
        extra_claims={"org_id": organization_id, "project_id": project_id},
    )


def create_refresh_token(
    subject: str | Any,
    expires_delta: timedelta,
    organization_id: int | None = None,
    project_id: int | None = None,
) -> str:
    """Create a JWT refresh token."""
    return encode_jwt_token(
        subject=subject,
        token_type="refresh",
        expires_delta=expires_delta,
        extra_claims={"org_id": organization_id, "project_id": project_id},
    )


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


def encrypt_credentials(credentials: dict[str, Any]) -> str:
    """Encrypt credentials for storage. KMS outside dev, Fernet otherwise.

    KMS Encrypt caps plaintext at 4096 bytes, so payloads must stay under that.
    """
    try:
        credentials_str = json.dumps(credentials)
        if _use_kms():
            response = get_kms_client().encrypt(
                KeyId=settings.AWS_KMS_KEY_ID,
                Plaintext=credentials_str.encode(),
            )
            encoded = base64.b64encode(response["CiphertextBlob"]).decode()
            return f"{KMS_CIPHERTEXT_PREFIX}{encoded}"
        return get_fernet().encrypt(credentials_str.encode()).decode()
    except Exception as e:
        # Log the real cause (may carry AWS ARNs); never surface it to callers.
        logger.error(
            f"[encrypt_credentials] Encryption failed | error: {e}", exc_info=True
        )
        raise UpstreamError(
            "Failed to encrypt credentials. Retry shortly.", provider="kms"
        )


def decrypt_credentials(encrypted_credentials: str) -> dict[str, Any]:
    """Decrypt stored credentials. Routing is by ciphertext prefix, not the
    active mode, so legacy Fernet rows always decrypt even after KMS cutover.
    """
    try:
        if encrypted_credentials.startswith(KMS_CIPHERTEXT_PREFIX):
            blob = base64.b64decode(encrypted_credentials[len(KMS_CIPHERTEXT_PREFIX) :])
            response = get_kms_client().decrypt(CiphertextBlob=blob)
            decrypted_str = response["Plaintext"].decode()
        else:
            decrypted_str = (
                get_fernet().decrypt(encrypted_credentials.encode()).decode()
            )
        return json.loads(decrypted_str)
    except Exception as e:
        # Log the real cause (may carry AWS ARNs); never surface it to callers.
        logger.error(
            f"[decrypt_credentials] Decryption failed | error: {e}", exc_info=True
        )
        raise UpstreamError(
            "Failed to decrypt credentials. Retry shortly.", provider="kms"
        )


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
    def generate(cls) -> tuple[str, str, str]:
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
    def _extract_key_parts(cls, raw_key: str) -> tuple[str, str] | None:
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
                        APIKey.deleted_at.is_(None),
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
