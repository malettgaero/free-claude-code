from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.datastructures import State
from starlette.requests import Request

from api.dependencies import get_request_settings, resolve_provider
from config.nim import NimSettings
from providers.deepseek import DeepSeekProvider
from providers.exceptions import ServiceUnavailableError, UnknownProviderTypeError
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.ollama import OllamaProvider
from providers.open_router import OpenRouterProvider
from providers.registry import ProviderRegistry


def _make_mock_settings(**overrides):
    mock = MagicMock()
    mock.model = "nvidia_nim/meta/llama3"
    mock.provider_type = "nvidia_nim"
    mock.nvidia_nim_api_key = "test_key"
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.open_router_api_key = "test_openrouter_key"
    mock.deepseek_api_key = "test_deepseek_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.llamacpp_base_url = "http://localhost:8080/v1"
    mock.ollama_base_url = "http://localhost:11434"
    mock.nvidia_nim_proxy = ""
    mock.open_router_proxy = ""
    mock.lmstudio_proxy = ""
    mock.llamacpp_proxy = ""
    mock.nim = NimSettings()
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.enable_model_thinking = True
    mock.log_raw_sse_events = False
    mock.log_api_error_tracebacks = False
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def _app_with_registry(settings=None):
    app = SimpleNamespace(state=State())
    app.state.settings = settings or _make_mock_settings()
    app.state.provider_registry = ProviderRegistry()
    return cast(Starlette, app)


def test_get_request_settings_reads_app_state() -> None:
    settings = _make_mock_settings()
    app = SimpleNamespace(state=State())
    app.state.settings = settings
    request = Request({"type": "http", "app": app, "headers": []})

    assert get_request_settings(request) is settings


def test_get_request_settings_missing_state_raises() -> None:
    app = SimpleNamespace(state=State())
    request = Request({"type": "http", "app": app, "headers": []})

    with pytest.raises(ServiceUnavailableError, match="Settings are not configured"):
        get_request_settings(request)


def test_resolve_provider_caches_per_app_registry() -> None:
    settings = _make_mock_settings()
    app = _app_with_registry(settings)

    p1 = resolve_provider("nvidia_nim", app=app, settings=settings)
    p2 = resolve_provider("nvidia_nim", app=app, settings=settings)

    assert isinstance(p1, NvidiaNimProvider)
    assert p1 is p2


def test_resolve_provider_per_app_uses_separate_registries() -> None:
    settings = _make_mock_settings()
    app1 = _app_with_registry(settings)
    app2 = _app_with_registry(settings)

    p1 = resolve_provider("nvidia_nim", app=app1, settings=settings)
    p2 = resolve_provider("nvidia_nim", app=app2, settings=settings)

    assert isinstance(p1, NvidiaNimProvider)
    assert isinstance(p2, NvidiaNimProvider)
    assert p1 is not p2
    assert p1._rate_limiter is not p2._rate_limiter


@pytest.mark.parametrize(
    ("provider_type", "provider_cls", "expected_base", "expected_key"),
    [
        (
            "open_router",
            OpenRouterProvider,
            "https://openrouter.ai/api/v1",
            "test_openrouter_key",
        ),
        ("lmstudio", LMStudioProvider, "http://localhost:1234/v1", "lm-studio"),
        ("ollama", OllamaProvider, "http://localhost:11434", "ollama"),
        (
            "deepseek",
            DeepSeekProvider,
            "https://api.deepseek.com/anthropic",
            "test_deepseek_key",
        ),
    ],
)
def test_resolve_provider_builtin_types(
    provider_type, provider_cls, expected_base, expected_key
) -> None:
    settings = _make_mock_settings(provider_type=provider_type)
    app = _app_with_registry(settings)

    provider = resolve_provider(provider_type, app=app, settings=settings)

    assert isinstance(provider, provider_cls)
    assert provider._base_url == expected_base
    assert provider._api_key == expected_key


def test_resolve_provider_passes_enable_model_thinking() -> None:
    settings = _make_mock_settings(
        provider_type="deepseek", enable_model_thinking=False
    )
    app = _app_with_registry(settings)

    provider = resolve_provider("deepseek", app=app, settings=settings)

    assert isinstance(provider, DeepSeekProvider)
    assert provider._config.enable_thinking is False


def test_resolve_provider_passes_http_timeouts_from_settings() -> None:
    settings = _make_mock_settings(
        http_read_timeout=600.0,
        http_write_timeout=20.0,
        http_connect_timeout=5.0,
    )
    app = _app_with_registry(settings)

    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = resolve_provider("nvidia_nim", app=app, settings=settings)

    assert isinstance(provider, NvidiaNimProvider)
    timeout = mock_openai.call_args.kwargs["timeout"]
    assert timeout.read == 600.0
    assert timeout.write == 20.0
    assert timeout.connect == 5.0


def test_resolve_provider_passes_proxy_from_settings() -> None:
    settings = _make_mock_settings(nvidia_nim_proxy="http://proxy.example:8080")
    app = _app_with_registry(settings)

    with (
        patch("providers.openai_compat.httpx.AsyncClient") as mock_http_client,
        patch("providers.openai_compat.AsyncOpenAI") as mock_openai,
    ):
        provider = resolve_provider("nvidia_nim", app=app, settings=settings)

    assert isinstance(provider, NvidiaNimProvider)
    mock_http_client.assert_called_once()
    assert mock_http_client.call_args.kwargs["proxy"] == "http://proxy.example:8080"
    assert mock_openai.call_args.kwargs["http_client"] is mock_http_client.return_value


def test_resolve_provider_ignores_non_string_proxy_value() -> None:
    settings = _make_mock_settings(nvidia_nim_proxy=MagicMock(name="proxy"))
    app = _app_with_registry(settings)

    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = resolve_provider("nvidia_nim", app=app, settings=settings)

    assert isinstance(provider, NvidiaNimProvider)
    assert mock_openai.call_args.kwargs["http_client"] is None


@pytest.mark.parametrize(
    ("provider_type", "settings_override", "expected"),
    [
        ("nvidia_nim", {"nvidia_nim_api_key": ""}, "NVIDIA_NIM_API_KEY"),
        ("nvidia_nim", {"nvidia_nim_api_key": "   "}, "NVIDIA_NIM_API_KEY"),
        ("open_router", {"open_router_api_key": ""}, "OPENROUTER_API_KEY"),
        ("deepseek", {"deepseek_api_key": ""}, "DEEPSEEK_API_KEY"),
    ],
)
def test_resolve_provider_missing_key_raises_503(
    provider_type, settings_override, expected
) -> None:
    settings = _make_mock_settings(provider_type=provider_type, **settings_override)
    app = _app_with_registry(settings)

    with pytest.raises(HTTPException) as exc_info:
        resolve_provider(provider_type, app=app, settings=settings)

    assert exc_info.value.status_code == 503
    assert expected in exc_info.value.detail


def test_resolve_provider_unknown_type() -> None:
    settings = _make_mock_settings()
    app = _app_with_registry(settings)

    with pytest.raises(UnknownProviderTypeError, match="Unknown provider_type"):
        resolve_provider("unknown", app=app, settings=settings)


@pytest.mark.asyncio
async def test_provider_registry_cleanup_cleans_all_cached_providers() -> None:
    settings = _make_mock_settings()
    registry = ProviderRegistry()
    app = SimpleNamespace(state=State())
    app.state.provider_registry = registry

    nim = resolve_provider("nvidia_nim", app=cast(Starlette, app), settings=settings)
    lmstudio = resolve_provider("lmstudio", app=cast(Starlette, app), settings=settings)
    assert isinstance(nim, NvidiaNimProvider)
    assert isinstance(lmstudio, LMStudioProvider)
    nim._client = AsyncMock()
    lmstudio._client = AsyncMock()

    await registry.cleanup()

    nim._client.aclose.assert_called_once()
    lmstudio._client.aclose.assert_called_once()


def test_resolve_provider_missing_registry_raises_service_unavailable() -> None:
    settings = _make_mock_settings()
    app = SimpleNamespace(state=State())

    with pytest.raises(
        ServiceUnavailableError, match="Provider registry is not configured"
    ):
        resolve_provider("nvidia_nim", app=cast(Starlette, app), settings=settings)


def test_resolve_provider_unrelated_value_error_is_not_unknown_provider_log() -> None:
    import api.dependencies as deps

    settings = _make_mock_settings()
    app = _app_with_registry(settings)
    with (
        patch.object(
            ProviderRegistry,
            "get",
            side_effect=ValueError("unrelated config"),
        ),
        patch.object(deps.logger, "error") as log_err,
        pytest.raises(ValueError, match="unrelated config"),
    ):
        deps.resolve_provider("nvidia_nim", app=app, settings=settings)
    log_err.assert_not_called()
