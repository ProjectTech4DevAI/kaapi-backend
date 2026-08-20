import base64
import json
from datetime import timedelta
from unittest.mock import MagicMock

import boto3
import jwt
import pytest
from fastapi import HTTPException
from moto import mock_aws
from sqlmodel import Session

import app.core.security as security
from app.core.config import settings
from app.core.security import (
    ALGORITHM,
    KMS_CIPHERTEXT_PREFIX,
    KMS_ENVELOPE_PREFIX,
    APIKeyManager,
    create_access_token,
    create_refresh_token,
    decrypt_credentials,
    encrypt_credentials,
    encrypt_fernet,
    get_encryption_key,
)
from app.core.util import now
from app.models import APIKey, AuthContext, Organization, Project, User
from app.tests.utils.test_data import create_test_api_key


def test_get_encryption_key():
    """Test that encryption key generation works correctly."""
    # Get the encryption key
    key = get_encryption_key()

    # Verify the key
    assert key is not None
    assert isinstance(key, bytes)
    # The key is base64 encoded, so it should be 44 bytes
    assert len(key) == 44  # Base64 encoded Fernet key length is 44 bytes


@pytest.fixture
def kms_key(monkeypatch):
    """Stand up a mocked KMS key and switch security.py onto the KMS path."""
    with mock_aws():
        client = boto3.client("kms", region_name="ap-south-1")
        key_id = client.create_key()["KeyMetadata"]["KeyId"]

        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "AWS_KMS_KEY_ID", key_id)
        monkeypatch.setattr(security, "_kms_client", client)
        yield key_id


class TestCredentialEncryption:
    """Credential encrypt/decrypt across Fernet (dev) and KMS (non-dev)."""

    def test_fernet_roundtrip_in_development(self, monkeypatch):
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")
        creds = {"api_key": "sk-fernet-123"}

        encrypted = encrypt_credentials(creds)

        assert not encrypted.startswith(KMS_CIPHERTEXT_PREFIX)
        assert decrypt_credentials(encrypted) == creds

    def test_kms_roundtrip(self, kms_key):
        creds = {"openai": {"api_key": "sk-kms-123"}}

        encrypted = encrypt_credentials(creds)

        assert encrypted.startswith(KMS_ENVELOPE_PREFIX)
        assert decrypt_credentials(encrypted) == creds

    def test_dual_read_fernet_row_with_kms_active(self, monkeypatch, kms_key):
        """A legacy Fernet ciphertext must still decrypt after KMS cutover."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")
        creds = {"api_key": "sk-legacy"}
        fernet_encrypted = encrypt_credentials(creds)
        assert not fernet_encrypted.startswith(KMS_CIPHERTEXT_PREFIX)

        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        assert decrypt_credentials(fernet_encrypted) == creds

    def test_kms_encrypt_failure_raises_502_without_leaking_cause(
        self, monkeypatch, kms_key
    ):
        broken = MagicMock()
        broken.generate_data_key.side_effect = Exception(
            "arn:aws:kms:ap-south-1:secret"
        )
        monkeypatch.setattr(security, "_kms_client", broken)

        with pytest.raises(HTTPException) as exc:
            encrypt_credentials({"api_key": "sk-1"})

        assert exc.value.status_code == 502
        assert "arn:aws:kms" not in exc.value.detail

    def test_kms_decrypt_failure_raises_502(self, monkeypatch, kms_key):
        encrypted = encrypt_credentials({"api_key": "sk-1"})
        broken = MagicMock()
        broken.decrypt.side_effect = Exception("arn:aws:kms:ap-south-1:secret")
        monkeypatch.setattr(security, "_kms_client", broken)

        with pytest.raises(HTTPException) as exc:
            decrypt_credentials(encrypted)

        assert exc.value.status_code == 502
        assert "arn:aws:kms" not in exc.value.detail

    def test_kms_v2_envelope_roundtrip(self, kms_key):
        creds = {"openai": {"api_key": "sk-envelope-123"}}

        encrypted = encrypt_credentials(creds)

        assert encrypted.startswith(KMS_ENVELOPE_PREFIX)
        segments = encrypted[len(KMS_ENVELOPE_PREFIX) :].split(":")
        assert len(segments) == 3
        for seg in segments:
            base64.b64decode(seg)  # each segment must be valid base64
        assert decrypt_credentials(encrypted) == creds

    def test_kms_v2_large_payload_over_4096_bytes(self, kms_key):
        # Direct KMS encrypt caps at 4096 bytes; envelope encryption has no such limit.
        creds = {
            "service_account": {"private_key": "k" * 5000, "client_email": "svc@x"}
        }
        assert len(json.dumps(creds).encode()) > 4096

        encrypted = encrypt_credentials(creds)

        assert encrypted.startswith(KMS_ENVELOPE_PREFIX)
        assert decrypt_credentials(encrypted) == creds

    def test_v1_row_still_decrypts_with_v2_active(self, kms_key):
        creds = {"api_key": "sk-v1-legacy"}
        blob = security._kms_client.encrypt(
            KeyId=kms_key, Plaintext=json.dumps(creds).encode()
        )["CiphertextBlob"]
        v1_ciphertext = KMS_CIPHERTEXT_PREFIX + base64.b64encode(blob).decode()

        assert decrypt_credentials(v1_ciphertext) == creds

    def test_new_writes_produce_v2(self, kms_key):
        encrypted = encrypt_credentials({"api_key": "sk-new"})

        assert encrypted.startswith(KMS_ENVELOPE_PREFIX)
        assert not encrypted.startswith(KMS_CIPHERTEXT_PREFIX)

    def test_v2_tampered_ciphertext_raises(self, kms_key):
        encrypted = encrypt_credentials({"api_key": "sk-tamper"})
        wrapped_b64, nonce_b64, ct_b64 = encrypted[len(KMS_ENVELOPE_PREFIX) :].split(
            ":"
        )
        ct = bytearray(base64.b64decode(ct_b64))
        ct[0] ^= 0xFF
        tampered = (
            f"{KMS_ENVELOPE_PREFIX}{wrapped_b64}:{nonce_b64}:"
            f"{base64.b64encode(bytes(ct)).decode()}"
        )

        with pytest.raises(ValueError, match="Failed to decrypt credentials"):
            decrypt_credentials(tampered)

    def test_v2_wrong_segment_count_raises(self, kms_key):
        with pytest.raises(ValueError, match="Failed to decrypt credentials"):
            decrypt_credentials(f"{KMS_ENVELOPE_PREFIX}onlyoneseg")

    def test_encrypt_fernet_forces_fernet_even_with_kms_active(self, kms_key):
        creds = {"api_key": "sk-force-fernet"}

        encrypted = encrypt_fernet(creds)

        # KMS is active, yet the output is Fernet (no KMS prefix) and roundtrips.
        assert not encrypted.startswith(KMS_ENVELOPE_PREFIX)
        assert not encrypted.startswith(KMS_CIPHERTEXT_PREFIX)
        assert decrypt_credentials(encrypted) == creds

    def test_encrypt_fernet_wraps_errors(self, monkeypatch):
        def boom() -> None:
            raise RuntimeError("fernet unavailable")

        monkeypatch.setattr(security, "get_fernet", boom)
        with pytest.raises(ValueError, match="Failed to encrypt credentials"):
            encrypt_fernet({"api_key": "x"})


class TestAPIKeyManager:
    """Test suite for APIKeyManager class."""

    def test_generate_returns_correct_tuple(self):
        """Test that generate returns a tuple of (raw_key, key_prefix, key_hash)."""
        raw_key, key_prefix, key_hash = APIKeyManager.generate()

        assert isinstance(raw_key, str)
        assert isinstance(key_prefix, str)
        assert isinstance(key_hash, str)

    def test_generate_raw_key_format(self):
        """Test that generated raw key has correct format."""
        raw_key, key_prefix, key_hash = APIKeyManager.generate()

        # Should start with "ApiKey "
        assert raw_key.startswith(APIKeyManager.PREFIX_NAME)

        # Should have correct length (7 for "ApiKey " + 22 for prefix + 43 for secret)
        expected_length = len(APIKeyManager.PREFIX_NAME) + APIKeyManager.KEY_LENGTH
        assert len(raw_key) == expected_length

    def test_generate_key_prefix_length(self):
        """Test that generated key prefix has correct length."""
        raw_key, key_prefix, key_hash = APIKeyManager.generate()

        assert len(key_prefix) == APIKeyManager.PREFIX_LENGTH

    def test_generate_unique_keys(self):
        """Test that generate creates unique keys on each call."""
        raw_key1, prefix1, hash1 = APIKeyManager.generate()
        raw_key2, prefix2, hash2 = APIKeyManager.generate()

        assert raw_key1 != raw_key2
        assert prefix1 != prefix2
        assert hash1 != hash2

    def test_generate_hash_is_bcrypt(self):
        """Test that the generated hash uses bcrypt format."""
        raw_key, key_prefix, key_hash = APIKeyManager.generate()

        # bcrypt hashes start with $2b$ (or $2a$ or $2y$)
        assert key_hash.startswith("$2")

    def test_extract_key_parts_new_format(self):
        """Test extracting key parts from new format (65 chars)."""
        raw_key, expected_prefix, _ = APIKeyManager.generate()

        result = APIKeyManager._extract_key_parts(raw_key)

        assert result is not None
        extracted_prefix, secret = result
        assert extracted_prefix == expected_prefix
        assert len(secret) == APIKeyManager.KEY_LENGTH - APIKeyManager.PREFIX_LENGTH

    def test_extract_key_parts_old_format(self):
        """Test extracting key parts from old format (43 chars)."""
        old_prefix = "a" * 12
        old_secret = "b" * 31
        raw_key = f"{APIKeyManager.PREFIX_NAME}{old_prefix}{old_secret}"

        result = APIKeyManager._extract_key_parts(raw_key)

        assert result is not None
        extracted_prefix, secret = result
        assert extracted_prefix == old_prefix
        assert secret == old_secret

    def test_extract_key_parts_invalid_prefix(self):
        """Test that invalid prefix returns None."""
        invalid_key = "InvalidPrefix abcdefghij1234567890"

        result = APIKeyManager._extract_key_parts(invalid_key)

        assert result is None

    def test_extract_key_parts_invalid_length(self):
        """Test that invalid length returns None."""
        invalid_key = f"{APIKeyManager.PREFIX_NAME}tooshort"

        result = APIKeyManager._extract_key_parts(invalid_key)

        assert result is None

    def test_verify_valid_key(self, db: Session):
        """Test verifying a valid API key."""
        api_key = create_test_api_key(db)

        auth_context = APIKeyManager.verify(db, api_key.key)

        user = db.get(User, api_key.user_id)
        organization = db.get(Organization, api_key.organization_id)
        project = db.get(Project, api_key.project_id)

        assert auth_context is not None
        assert isinstance(auth_context, AuthContext)
        assert auth_context.user.id == api_key.user_id
        assert auth_context.organization.id == api_key.organization_id
        assert auth_context.project.id == api_key.project_id
        assert auth_context.user == user
        assert auth_context.organization == organization
        assert auth_context.project == project

    def test_verify_invalid_key(self, db: Session):
        """Test verifying an invalid API key."""
        # Generate a key but don't store it
        raw_key, _, _ = APIKeyManager.generate()

        auth_context = APIKeyManager.verify(db, raw_key)

        assert auth_context is None

    def test_verify_wrong_secret(self, db: Session):
        """Test verifying with correct prefix but wrong secret."""
        create_test_api_key(db)

        # Generate a different key to try verification
        raw_key2, _, _ = APIKeyManager.generate()

        # Try to verify with key2 (wrong secret)
        auth_context = APIKeyManager.verify(db, raw_key2)

        assert auth_context is None

    def test_verify_deleted_key(self, db: Session):
        """Test that deleted API keys cannot be verified."""
        api_key_response = create_test_api_key(db)
        raw_key = api_key_response.key

        api_key = db.get(APIKey, api_key_response.id)
        api_key.deleted_at = now()
        db.commit()

        auth_context = APIKeyManager.verify(db, raw_key)

        assert auth_context is None

    def test_verify_malformed_key(self, db: Session):
        """Test verifying with malformed key format."""
        malformed_keys = [
            "not_an_api_key",
            "",
            "ApiKey",
            "ApiKey ",
            None,
        ]

        for malformed_key in malformed_keys:
            if malformed_key is not None:
                auth_context = APIKeyManager.verify(db, malformed_key)
                assert auth_context is None

    def test_prefix_name_constant(self):
        """Test that PREFIX_NAME is correct."""
        assert APIKeyManager.PREFIX_NAME == "ApiKey "

    def test_key_length_constants(self):
        """Test that key length constants are correct."""
        assert APIKeyManager.PREFIX_LENGTH == 22
        assert APIKeyManager.KEY_LENGTH == 65
        assert APIKeyManager.KEY_LENGTH == APIKeyManager.PREFIX_LENGTH + 43

    def test_generate_creates_verifiable_key(self, db: Session):
        """Integration test: generated key can be verified."""
        api_key_response = create_test_api_key(db)

        auth_context = APIKeyManager.verify(db, api_key_response.key)

        assert auth_context is not None
        assert auth_context.user.id == api_key_response.user_id


class TestCreateAccessToken:
    """Test suite for create_access_token function."""

    def test_creates_valid_jwt(self):
        """Test that a valid JWT is created."""
        token = create_access_token(subject="42", expires_delta=timedelta(minutes=30))
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["sub"] == "42"
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_includes_org_and_project(self):
        """Test that org_id and project_id are embedded in the token."""
        token = create_access_token(
            subject="1",
            expires_delta=timedelta(minutes=30),
            organization_id=10,
            project_id=20,
        )
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["org_id"] == 10
        assert payload["project_id"] == 20

    def test_omits_org_and_project_when_none(self):
        """Test that org_id and project_id are omitted when not provided."""
        token = create_access_token(subject="1", expires_delta=timedelta(minutes=30))
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])

        assert "org_id" not in payload
        assert "project_id" not in payload


class TestCreateRefreshToken:
    """Test suite for create_refresh_token function."""

    def test_creates_valid_refresh_jwt(self):
        """Test that a valid refresh JWT is created."""
        token = create_refresh_token(subject="42", expires_delta=timedelta(days=7))
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["sub"] == "42"
        assert payload["type"] == "refresh"
        assert "exp" in payload

    def test_includes_org_and_project(self):
        """Test that org_id and project_id are embedded in the refresh token."""
        token = create_refresh_token(
            subject="1",
            expires_delta=timedelta(days=7),
            organization_id=10,
            project_id=20,
        )
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["org_id"] == 10
        assert payload["project_id"] == 20
        assert payload["type"] == "refresh"

    def test_omits_org_and_project_when_none(self):
        """Test that org_id and project_id are omitted when not provided."""
        token = create_refresh_token(subject="1", expires_delta=timedelta(days=7))
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])

        assert "org_id" not in payload
        assert "project_id" not in payload
