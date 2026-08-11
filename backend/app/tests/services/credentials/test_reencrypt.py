import base64
import json

import boto3
import pytest
from moto import mock_aws
from sqlmodel import Session

import app.core.security as security
from app.core.config import settings
from app.core.security import (
    KMS_CIPHERTEXT_PREFIX,
    KMS_ENVELOPE_PREFIX,
    decrypt_credentials,
)
from app.models import Credential
from app.services.credentials.reencrypt import (
    execute_credential_reencrypt,
    execute_credential_reencrypt_fernet,
)
from app.tests.utils.test_data import create_test_credential


def test_reencrypt_converts_fernet_rows_to_kms(db: Session, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    creds, _ = create_test_credential(db)
    expected = {c.id: decrypt_credentials(c.credential) for c in creds}
    for c in creds:
        assert not c.credential.startswith(KMS_CIPHERTEXT_PREFIX)

    with mock_aws():
        client = boto3.client("kms", region_name="ap-south-1")
        key_id = client.create_key()["KeyMetadata"]["KeyId"]
        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "AWS_KMS_KEY_ID", key_id)
        monkeypatch.setattr(security, "_kms_client", client)

        result = execute_credential_reencrypt(session=db)

        assert result["converted"] >= len(expected)
        for cid, plain in expected.items():
            row = db.get(Credential, cid)
            assert row.credential.startswith(KMS_ENVELOPE_PREFIX)
            assert decrypt_credentials(row.credential) == plain


def test_reencrypt_skips_already_v2_rows(db: Session, monkeypatch):
    creds, _ = create_test_credential(db)

    with mock_aws():
        client = boto3.client("kms", region_name="ap-south-1")
        key_id = client.create_key()["KeyMetadata"]["KeyId"]
        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "AWS_KMS_KEY_ID", key_id)
        monkeypatch.setattr(security, "_kms_client", client)

        execute_credential_reencrypt(session=db)  # -> kms.v2
        snapshot = {c.id: db.get(Credential, c.id).credential for c in creds}

        # Second pass: already-v2 rows hit the skip branch and stay byte-identical.
        execute_credential_reencrypt(session=db)
        for cid, ciphertext in snapshot.items():
            assert db.get(Credential, cid).credential == ciphertext


def test_reencrypt_fernet_downgrades_kms_to_fernet(db: Session, monkeypatch):
    creds, _ = create_test_credential(db)
    expected = {c.id: decrypt_credentials(c.credential) for c in creds}

    with mock_aws():
        client = boto3.client("kms", region_name="ap-south-1")
        key_id = client.create_key()["KeyMetadata"]["KeyId"]
        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "AWS_KMS_KEY_ID", key_id)
        monkeypatch.setattr(security, "_kms_client", client)

        execute_credential_reencrypt(session=db)  # -> kms.v2
        for c in creds:
            assert db.get(Credential, c.id).credential.startswith(KMS_ENVELOPE_PREFIX)

        result = execute_credential_reencrypt_fernet(session=db)  # kms.v2 -> Fernet

        assert result["converted"] >= len(expected)
        for cid, plain in expected.items():
            row = db.get(Credential, cid)
            assert not row.credential.startswith(KMS_ENVELOPE_PREFIX)
            assert not row.credential.startswith(KMS_CIPHERTEXT_PREFIX)
            assert decrypt_credentials(row.credential) == plain


def test_reencrypt_fernet_leaves_fernet_rows_untouched(db: Session, monkeypatch):
    creds, _ = create_test_credential(db)  # Fernet rows
    before = {c.id: db.get(Credential, c.id).credential for c in creds}

    with mock_aws():
        client = boto3.client("kms", region_name="ap-south-1")
        key_id = client.create_key()["KeyMetadata"]["KeyId"]
        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "AWS_KMS_KEY_ID", key_id)
        monkeypatch.setattr(security, "_kms_client", client)

        execute_credential_reencrypt_fernet(session=db)

        for cid, original in before.items():
            assert db.get(Credential, cid).credential == original


def test_reencrypt_fernet_rolls_back_on_roundtrip_mismatch(db: Session, monkeypatch):
    import app.services.credentials.reencrypt as reencrypt_mod
    from app.core.security import encrypt_fernet as real_encrypt_fernet

    creds, _ = create_test_credential(db)

    with mock_aws():
        client = boto3.client("kms", region_name="ap-south-1")
        key_id = client.create_key()["KeyMetadata"]["KeyId"]
        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "AWS_KMS_KEY_ID", key_id)
        monkeypatch.setattr(security, "_kms_client", client)

        execute_credential_reencrypt(session=db)  # -> kms.v2 rows to process

        # Force the verify step to see a different plaintext -> roundtrip mismatch.
        monkeypatch.setattr(
            reencrypt_mod,
            "encrypt_fernet",
            lambda _creds: real_encrypt_fernet({"tampered": True}),
        )

        with pytest.raises(ValueError, match="roundtrip mismatch"):
            execute_credential_reencrypt_fernet(session=db)


def test_reencrypt_fernet_skipped_when_kms_inactive(db: Session, monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")

    result = execute_credential_reencrypt_fernet(session=db)

    assert result == {"total": 0, "converted": 0}


def test_execute_reencrypt_fernet_owns_session_when_none(monkeypatch):
    # session=None -> opens its own Session(engine); guard returns before any query.
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")

    result = execute_credential_reencrypt_fernet()

    assert result == {"total": 0, "converted": 0}


def test_reencrypt_converts_v1_row_to_v2(db: Session, monkeypatch):
    creds, _ = create_test_credential(db)
    expected = {c.id: decrypt_credentials(c.credential) for c in creds}

    with mock_aws():
        client = boto3.client("kms", region_name="ap-south-1")
        key_id = client.create_key()["KeyMetadata"]["KeyId"]
        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "AWS_KMS_KEY_ID", key_id)
        monkeypatch.setattr(security, "_kms_client", client)

        # Rewrite each row as a legacy direct-KMS (v1) ciphertext.
        for c in creds:
            plain = expected[c.id]
            blob = client.encrypt(KeyId=key_id, Plaintext=json.dumps(plain).encode())[
                "CiphertextBlob"
            ]
            c.credential = KMS_CIPHERTEXT_PREFIX + base64.b64encode(blob).decode()
            db.add(c)
        db.commit()

        result = execute_credential_reencrypt(session=db)

        assert result["converted"] >= len(expected)
        for cid, plain in expected.items():
            row = db.get(Credential, cid)
            assert row.credential.startswith(KMS_ENVELOPE_PREFIX)
            assert decrypt_credentials(row.credential) == plain
