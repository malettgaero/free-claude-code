"""LM Studio provider implementation."""

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import LMSTUDIO_DEFAULT_BASE
from providers.rate_limit import ProviderRateLimiter


class LMStudioProvider(AnthropicMessagesTransport):
    """LM Studio provider using native Anthropic Messages endpoint."""

    def __init__(
        self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter | None = None
    ):
        super().__init__(
            config,
            provider_name="LMSTUDIO",
            default_base_url=LMSTUDIO_DEFAULT_BASE,
            rate_limiter=rate_limiter,
        )
