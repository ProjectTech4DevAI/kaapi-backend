"""Attachment utilities for LLM providers: gs:// path selection, URL resolution, etc."""

from enum import Enum

from sqlmodel import Session

from app.models.llm.constants import KaapiProvider, Provider
from app.services.buckets.providers.registry import get_bucket_provider

GCS_URI_SCHEME = "gs"
DEFAULT_BUCKET_PROVIDER = "gcs"

# LLM providers that read gs:// URIs natively (Vertex/google-gcp, incl. native key).
NATIVE_PROVIDERS: frozenset[str] = frozenset(
    {Provider.GOOGLE_GCP, f"{Provider.GOOGLE_GCP}-native"}
)


class BucketPathStrategyEnum(str, Enum):
    NATIVE = "native"  # Path A: pass the gs:// URI straight to the provider.
    SIGNED_URL = "signed_url"  # Path B: convert to a signed HTTPS URL.


def is_gcs_uri(uri: str) -> bool:
    return uri.startswith(f"{GCS_URI_SCHEME}://")


def resolve_bucket_path_strategy(
    *,
    llm_provider: KaapiProvider,
    source_uri: str,
    credential: dict | None = None,
) -> BucketPathStrategyEnum:
    """Native only for a gs:// source the provider reads directly; else signed."""
    if is_gcs_uri(source_uri) and llm_provider in NATIVE_PROVIDERS:
        return BucketPathStrategyEnum.NATIVE
    return BucketPathStrategyEnum.SIGNED_URL


def resolve_attachments(
    *,
    session: Session,
    source: str | list[str],
    llm_provider: KaapiProvider,
    project_id: int,
    organization_id: int,
    expires_in: int = 3600,
    bucket_provider_type: str = DEFAULT_BUCKET_PROVIDER,
) -> str | dict[str, str]:
    """Make attachment(s) LLM-reachable. Accepts a single URL or a list.

    Per URI: non-``gs://`` (http(s)/direct) is returned by reference; ``gs://`` on
    a native provider passes through (Path A); otherwise it is signed (Path B).
    Path-B URIs are bulk-signed with a single bucket-provider client per call.
    Returns a ``str`` for a single input, a ``uri -> url`` dict for a list.
    """
    is_single = isinstance(source, str)
    uris = [source] if is_single else list(source)

    resolved: dict[str, str] = {}
    to_sign: list[str] = []
    for uri in uris:
        if not is_gcs_uri(uri):
            resolved[uri] = uri
            continue
        strategy = resolve_bucket_path_strategy(
            llm_provider=llm_provider, source_uri=uri
        )
        if strategy is BucketPathStrategyEnum.NATIVE:
            resolved[uri] = uri
        else:
            to_sign.append(uri)

    if to_sign:
        provider = get_bucket_provider(
            session=session,
            provider_type=bucket_provider_type,
            project_id=project_id,
            organization_id=organization_id,
        )
        resolved.update(provider.get_bulk_signed_urls(to_sign, expires_in=expires_in))

    return resolved[uris[0]] if is_single else resolved
