from .grok_provider import GrokProvider
from .openai_generator import OpenAIGenerator


def create_llm_provider(settings, openai_client=None, xai_client=None, provider_name=None):
    provider_name = (provider_name or settings.llm_provider).strip().casefold()
    if provider_name == "openai":
        return OpenAIGenerator(settings, openai_client)
    if provider_name == "grok":
        return GrokProvider(settings, xai_client)
    raise ValueError(f"Unsupported LLM provider: {provider_name}")


def create_llm_providers(settings, openai_client=None, xai_client=None):
    return {
        "grok": create_llm_provider(settings, xai_client=xai_client, provider_name="grok"),
        "openai": create_llm_provider(settings, openai_client=openai_client, provider_name="openai"),
    }
