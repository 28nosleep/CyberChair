import os

from .responses_provider import ResponsesLLMProvider


class OpenAIGenerator(ResponsesLLMProvider):
    provider_label = "OpenAI"
    provider_key = "openai"

    def __init__(self, settings, client=None):
        self.settings = settings
        self._client = client

    @property
    def available(self):
        return self.settings.openai_enabled and bool(
            self._client or os.getenv("OPENAI_API_KEY")
        )

    @property
    def unavailable_reason(self):
        if not self.settings.openai_enabled:
            return "OPENAI_ENABLED=false"
        return "OPENAI_API_KEY is not configured"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(timeout=self.settings.openai_timeout)
        return self._client

    def _response_kwargs(self, request):
        return {
            "model": self.settings.openai_model,
            "instructions": request.instructions,
            "input": request.input,
            "max_output_tokens": request.max_output_tokens,
            "reasoning": {"effort": "none"},
            "text": {"verbosity": "low"},
            "safety_identifier": request.safety_identifier,
            "store": False,
        }

    def _model_for_request(self, request):
        return self.settings.openai_model


__all__ = ["OpenAIGenerator"]
