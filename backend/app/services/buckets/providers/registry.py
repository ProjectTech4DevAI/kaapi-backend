"""Global bucket-provider registry and resolver."""

import logging

from sqlmodel import Session

from app.services.buckets.providers.base import BaseBucketProvider
from app.services.buckets.providers.gcs import GCSBucketProvider

logger = logging.getLogger(__name__)


class BucketProvider:
    GCS = "gcs"

    _registry: dict[str, type[BaseBucketProvider]] = {
        GCS: GCSBucketProvider,
    }

    # Bucket-provider key -> credential-provider key. GCS reuses google-gcp.
    _credential_provider: dict[str, str] = {
        GCS: "google-gcp",
    }

    @classmethod
    def get_provider_class(cls, provider_type: str) -> type[BaseBucketProvider]:
        """Return the bucket-provider class for a given name."""
        provider = cls._registry.get(provider_type)
        if not provider:
            raise ValueError(
                f"Bucket provider '{provider_type}' is not supported. "
                f"Supported providers: {', '.join(cls._registry.keys())}"
            )
        return provider

    @classmethod
    def supported_providers(cls) -> list[str]:
        """Return a list of supported bucket-provider names."""
        return list(cls._registry.keys())

    @classmethod
    def get_credential_provider(cls, provider_type: str) -> str:
        """Return the credential-provider key backing a bucket provider."""
        credential_provider = cls._credential_provider.get(provider_type)
        if not credential_provider:
            raise ValueError(
                f"Bucket provider '{provider_type}' has no credential mapping."
            )
        return credential_provider


def get_bucket_provider(
    session: Session, provider_type: str, project_id: int, organization_id: int
) -> BaseBucketProvider:
    # Lazy import to avoid the crud <-> provider-registry cycle.
    from app.crud.credentials import get_provider_credential

    provider_class = BucketProvider.get_provider_class(provider_type)
    credential_provider = BucketProvider.get_credential_provider(provider_type)

    credentials = get_provider_credential(
        session=session,
        provider=credential_provider,
        project_id=project_id,
        org_id=organization_id,
    )

    if not credentials:
        raise ValueError(
            f"Credentials for provider '{credential_provider}' not configured "
            f"for this project."
        )

    # Default fetch yields the decrypted dict create_client needs.
    if not isinstance(credentials, dict):
        raise ValueError(
            f"Expected decrypted credentials dict for provider "
            f"'{credential_provider}', got {type(credentials).__name__}."
        )

    try:
        client = provider_class.create_client(credentials=credentials)
        return provider_class(client=client)
    except ValueError:
        # Credential/config errors are the caller's to fix; surface as-is.
        raise
    except Exception as e:
        logger.error(
            f"[get_bucket_provider] Failed to initialize {provider_type} client: {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Could not connect to {provider_type} bucket services.")
