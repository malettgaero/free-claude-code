"""Llama.cpp provider implementation."""

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import LLAMACPP_DEFAULT_BASE
from providers.rate_limit import ProviderRateLimiter


class LlamaCppProvider(AnthropicMessagesTransport):
    """Llama.cpp provider using native Anthropic Messages endpoint."""

    def __init__(
        self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter | None = None
    ):
        super().__init__(
            config,
            provider_name="LLAMACPP",
            default_base_url=LLAMACPP_DEFAULT_BASE,
            rate_limiter=rate_limiter,
        )
