import os

from .responses_provider import ResponsesLLMProvider


class GrokProvider(ResponsesLLMProvider):
    """xAI Responses API adapter. Persona and memory stay outside this class."""

    provider_label = "Grok"

    def __init__(self, settings, client=None):
        self.settings = settings
        self._client = client

    def __repr__(self):
        return (
            f"GrokProvider(model={self.settings.xai_model!r}, "
            f"base_url={self.settings.xai_base_url!r}, available={self.available})"
        )

    @property
    def available(self):
        return bool(self._client or os.getenv("XAI_API_KEY"))

    @property
    def unavailable_reason(self):
        return "XAI_API_KEY is not configured"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url=self.settings.xai_base_url,
                timeout=self.settings.xai_timeout,
            )
        return self._client

    def _response_kwargs(self, request):
        return {
            "model": self.settings.xai_model,
            "instructions": request.instructions,
            "input": request.input,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }


__all__ = ["GrokProvider"]
