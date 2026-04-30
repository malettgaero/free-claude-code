"""Dependency injection for FastAPI."""

from fastapi import Depends, HTTPException, Request
from loguru import logger
from starlette.applications import Starlette

from config.settings import Settings
from core.anthropic import get_user_facing_error_message
from providers.base import BaseProvider
from providers.exceptions import (
    AuthenticationError,
    ServiceUnavailableError,
    UnknownProviderTypeError,
)
from providers.registry import PROVIDER_DESCRIPTORS, ProviderRegistry


def get_request_settings(request: Request) -> Settings:
    """Return app-owned settings installed by the application factory."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise ServiceUnavailableError(
            "Settings are not configured. Ensure create_app/AppRuntime startup ran."
        )
    return settings


def resolve_provider(
    provider_type: str,
    *,
    app: Starlette,
    settings: Settings,
) -> BaseProvider:
    """Resolve a provider using the app-scoped registry.

    The app-owned :attr:`app.state.provider_registry` must exist (installed by
    :class:`~api.runtime.AppRuntime` during startup). Callers that construct a
    bare ``FastAPI`` without lifespan must set ``app.state.provider_registry``
    explicitly.
    """
    reg = getattr(app.state, "provider_registry", None)
    if reg is None:
        raise ServiceUnavailableError(
            "Provider registry is not configured. Ensure AppRuntime startup ran "
            "or assign app.state.provider_registry for test apps."
        )
    return _resolve_with_registry(reg, provider_type, settings)


def _resolve_with_registry(
    registry: ProviderRegistry, provider_type: str, settings: Settings
) -> BaseProvider:
    should_log_init = not registry.is_cached(provider_type)
    try:
        provider = registry.get(provider_type, settings)
    except AuthenticationError as e:
        # Provider :class:`~providers.exceptions.AuthenticationError` messages are
        # curated configuration hints (env var names, docs links), not upstream noise.
        detail = str(e).strip() or get_user_facing_error_message(e)
        raise HTTPException(status_code=503, detail=detail) from e
    except UnknownProviderTypeError:
        logger.error(
            "Unknown provider_type: '{}'. Supported: {}",
            provider_type,
            ", ".join(f"'{key}'" for key in PROVIDER_DESCRIPTORS),
        )
        raise
    if should_log_init:
        logger.info("Provider initialized: {}", provider_type)
    return provider


def require_api_key(
    request: Request, settings: Settings = Depends(get_request_settings)
) -> None:
    """Require a server API key (Anthropic-style).

    Checks `x-api-key` header or `Authorization: Bearer ...` against
    `Settings.anthropic_auth_token`. If `ANTHROPIC_AUTH_TOKEN` is empty, this is a no-op.
    """
    anthropic_auth_token = settings.anthropic_auth_token
    if not anthropic_auth_token:
        # No API key configured -> allow
        return

    header = (
        request.headers.get("x-api-key")
        or request.headers.get("authorization")
        or request.headers.get("anthropic-auth-token")
    )
    if not header:
        raise HTTPException(status_code=401, detail="Missing API key")

    # Support both raw key in X-API-Key and Bearer token in Authorization
    token = header
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1]

    # Strip anything after the first colon to handle tokens with appended model names
    if token and ":" in token:
        token = token.split(":", 1)[0]

    if token != anthropic_auth_token:
        raise HTTPException(status_code=401, detail="Invalid API key")
