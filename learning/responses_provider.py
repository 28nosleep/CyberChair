import json
import logging
import re

from .preprocessing import FOREIGN_BOT_COMMAND_RE, normalize_spaces, strip_mentions


log = logging.getLogger(__name__)


class ResponsesLLMProvider:
    """Shared Responses API mechanics; contains no persona or chat policy."""

    provider_label = "LLM"

    def _get_client(self):
        raise NotImplementedError

    def _response_kwargs(self, request):
        raise NotImplementedError

    def _create(self, request):
        return self._get_client().responses.create(**self._response_kwargs(request))

    @staticmethod
    def _usage_value(value, *path):
        for part in path:
            if value is None:
                return None
            value = value.get(part) if isinstance(value, dict) else getattr(value, part, None)
        return value

    def _record_usage(self, request, response):
        """Emit usage only; the recorder never receives prompts or API keys."""
        recorder = getattr(self, "_usage_recorder", None)
        metadata = request.metadata or {}
        chat_id = metadata.get("chat_id")
        if not recorder or chat_id is None:
            return
        usage = getattr(response, "usage", None)
        payload = {
            "input_tokens": self._usage_value(usage, "input_tokens"),
            "cached_input_tokens": self._usage_value(usage, "input_tokens_details", "cached_tokens"),
            "output_tokens": self._usage_value(usage, "output_tokens"),
            "reasoning_tokens": self._usage_value(usage, "output_tokens_details", "reasoning_tokens"),
            "cost_usd_ticks": self._usage_value(usage, "cost_in_usd_ticks"),
        }
        try:
            recorder(
                chat_id,
                getattr(self, "provider_key", self.provider_label.casefold()),
                self._model_for_request(request),
                metadata.get("call_type", "reply"),
                payload,
            )
        except Exception:
            # Metering must never make the live response unavailable.
            log.warning("%s usage accounting failed", self.provider_label)

    def generate(self, request):
        if not self.available:
            log.warning("%s unavailable: %s", self.provider_label, self.unavailable_reason)
            return None
        try:
            response = self._create(request)
            self._record_usage(request, response)
            cleaned_lines = []
            for raw_line in response.output_text.splitlines():
                line = re.sub(r"\[[^\]\n]{1,60}\]\s*", "", raw_line)
                line = re.sub(
                    r"\b(?:ACCESS_DENIED|CHAIR_PROCESS)\b", "", line, flags=re.I
                )
                line = line.translate(str.maketrans("", "", "░▒▓"))
                if FOREIGN_BOT_COMMAND_RE.search(line):
                    continue
                line = re.sub(r"0x[0-9A-Fa-f]+\s*//\s*", "", line)
                line = normalize_spaces(line)
                if line:
                    cleaned_lines.append(line.lower().rstrip(".!…"))
            purpose = (request.metadata or {}).get("purpose")
            lines = cleaned_lines[:5] if purpose == "voice_story" else cleaned_lines[:2]
            if not self.settings.allow_user_mentions:
                lines = [strip_mentions(line) for line in lines]
            return "\n".join(lines).strip()
        except Exception as error:
            log.warning("%s generation failed: %s", self.provider_label, type(error).__name__)
            return None

    def summarize(self, request):
        if not self.available or not request.input:
            if not self.available:
                log.warning("%s unavailable: %s", self.provider_label, self.unavailable_reason)
            return None
        try:
            response = self._create(request)
            self._record_usage(request, response)
            raw = response.output_text.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
            data = json.loads(raw)
            keys = (
                "main_topics", "current_mood", "active_conflicts", "inside_jokes",
                "frequently_mentioned_people", "notable_events", "repeated_phrases",
                "callback_jokes", "memory_candidates", "topics", "mood",
                "local_memes", "people", "events", "stable_memory_candidates",
            )
            return {
                key: (
                    str(data.get(key, "")).strip()[:200]
                    if key == "current_mood"
                    else [
                        str(item).strip()[:200]
                        for item in data.get(key, [])[:6]
                        if str(item).strip()
                    ]
                )
                for key in keys
                if key in data
            }
        except (TypeError, ValueError) as error:
            log.warning("%s memory summary invalid: %s", self.provider_label, type(error).__name__)
            return None
        except Exception as error:
            log.warning("%s memory summary failed: %s", self.provider_label, type(error).__name__)
            return None
