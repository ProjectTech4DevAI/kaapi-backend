import boto3
from moto import mock_aws
from sqlmodel import Session

import app.core.security as security
from app.core.config import settings
from app.core.security import KMS_CIPHERTEXT_PREFIX, decrypt_credentials
from app.models import Credential
from app.services.credentials.reencrypt import execute_credential_reencrypt
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
            assert row.credential.startswith(KMS_CIPHERTEXT_PREFIX)
            assert decrypt_credentials(row.credential) == plain
