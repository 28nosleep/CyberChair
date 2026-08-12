import os

from .responses_provider import ResponsesLLMProvider


class GrokProvider(ResponsesLLMProvider):
    """xAI Responses API adapter. Persona and memory stay outside this class."""

    provider_label = "Grok"
    provider_key = "grok"

    def __init__(self, settings, client=None):
        self.settings = settings
        self._client = client

    def __repr__(self):
        return (
            f"GrokProvider(reply_model={self.settings.xai_reply_model!r}, "
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
        call_type = (request.metadata or {}).get("call_type", "reply")
        kwargs = {
            "model": self._model_for_request(request),
            "instructions": request.instructions,
            "input": request.input,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            # xAI-specific Responses parameter, intentionally stable per
            # prompt family rather than per message/chat.
            "extra_body": {
                "prompt_cache_key": (
                    "cyberchair:summary:v1" if call_type == "summary"
                    else "cyberchair:persona:v2"
                )
            },
        }
        effort = self._reasoning_for_request(request)
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        if request.safety_identifier:
            kwargs["safety_identifier"] = request.safety_identifier
        if call_type == "summary":
            kwargs["text"] = {"format": _SUMMARY_JSON_SCHEMA}
        return kwargs

    def _model_for_request(self, request):
        return (
            self.settings.xai_summary_model
            if (request.metadata or {}).get("call_type") == "summary"
            else self.settings.xai_reply_model
        )

    def _reasoning_for_request(self, request):
        effort = (
            self.settings.xai_summary_reasoning_effort
            if (request.metadata or {}).get("call_type") == "summary"
            else self.settings.xai_reply_reasoning_effort
        )
        if effort in {"low", "medium", "high", "xhigh"}:
            return effort
        # Grok 4.3 supports an explicit none mode; omitting it leaves the
        # provider free to reason. Grok 4.5 rejects none, so preserve its safe
        # low/minimum mode by omitting an invalid override only for that model.
        if effort == "none" and not self._model_for_request(request).startswith("grok-4.5"):
            return "none"
        return None


_SUMMARY_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "cyberchair_memory_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "main_topics": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "current_mood": {"type": "string", "maxLength": 200},
            "active_conflicts": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "inside_jokes": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "frequently_mentioned_people": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "notable_events": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "repeated_phrases": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "callback_jokes": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "memory_candidates": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        },
        "required": [
            "main_topics", "current_mood", "active_conflicts", "inside_jokes",
            "frequently_mentioned_people", "notable_events", "repeated_phrases",
            "callback_jokes", "memory_candidates",
        ],
    },
}


__all__ = ["GrokProvider"]
