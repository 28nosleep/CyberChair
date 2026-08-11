import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    GenerateRequest,
    GrokProvider,
    LLMProvider,
    LearningService,
    LearningSettings,
    OpenAIGenerator,
    PersonaBuilder,
    SummarizeRequest,
    create_llm_provider,
)


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(output_text=output)


class FakeClient:
    def __init__(self, *outputs):
        self.responses = FakeResponses(outputs)


class GrokProviderTests(unittest.TestCase):
    def settings(self, **overrides):
        values = dict(llm_provider="grok", xai_model="grok-4.5")
        values.update(overrides)
        return LearningSettings(**values)

    def test_grok_implements_provider_and_generate_uses_neutral_request(self):
        client = FakeClient("КОРОТКИЙ ОТВЕТ.")
        provider = GrokProvider(self.settings(), client)
        self.assertIsInstance(provider, LLMProvider)
        request = GenerateRequest("prepared persona", "prepared context", 77)
        self.assertEqual(provider.generate(request), "короткий ответ")
        self.assertEqual(client.responses.calls[0]["instructions"], "prepared persona")
        self.assertEqual(client.responses.calls[0]["input"], "prepared context")
        self.assertEqual(client.responses.calls[0]["model"], "grok-4.5")
        self.assertFalse(client.responses.calls[0]["store"])
        self.assertNotIn("tools", client.responses.calls[0])

    def test_grok_summarize_uses_responses_api(self):
        payload = {"main_topics": ["релиз"], "callback_jokes": ["принтер"]}
        client = FakeClient(json.dumps(payload, ensure_ascii=False))
        provider = GrokProvider(self.settings(), client)
        result = provider.summarize(SummarizeRequest("summary", "fragment"))
        self.assertEqual(result["main_topics"], ["релиз"])
        self.assertEqual(result["callback_jokes"], ["принтер"])
        self.assertFalse(client.responses.calls[0]["store"])

    def test_key_never_appears_in_repr_log_or_error(self):
        secret = "xai-secret-do-not-print"
        client = FakeClient(RuntimeError(f"network failed {secret}"))
        with patch.dict(os.environ, {"XAI_API_KEY": secret}):
            provider = GrokProvider(self.settings(), client)
            with self.assertLogs("learning.responses_provider", level="WARNING") as logs:
                self.assertIsNone(provider.generate(GenerateRequest("i", "q", 10)))
        self.assertNotIn(secret, repr(provider))
        self.assertNotIn(secret, "\n".join(logs.output))

    def test_factory_and_global_default_are_grok(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = LearningSettings()
        self.assertEqual(settings.llm_provider, "grok")
        self.assertIsInstance(create_llm_provider(settings, xai_client=FakeClient()), GrokProvider)
        self.assertIsInstance(
            create_llm_provider(settings, openai_client=FakeClient(), provider_name="openai"),
            OpenAIGenerator,
        )

    def test_per_chat_override_and_restart_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(data_dir=Path(directory), openai_chat_id=-1)
            service = LearningService(
                settings, openai_client=FakeClient(), xai_client=FakeClient()
            )
            self.assertEqual(service.llm_provider_name(-1), "grok")
            self.assertTrue(service.set_llm_provider(-1, "openai"))
            reopened = LearningService(
                settings, openai_client=FakeClient(), xai_client=FakeClient()
            )
            self.assertEqual(reopened.llm_provider_name(-1), "openai")

    def test_generate_routes_to_selected_chat_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            grok = FakeClient("ответ именно выбранного грока")
            openai = FakeClient("ответ именно выбранного опенаи")
            settings = self.settings(data_dir=Path(directory), openai_chat_id=-1)
            service = LearningService(settings, openai_client=openai, xai_client=grok)
            self.assertEqual(service.generate_llm(-1, "проверка", "reply"), "ответ именно выбранного грока")
            self.assertEqual(len(grok.responses.calls), 1)
            self.assertEqual(len(openai.responses.calls), 0)
            self.assertTrue(service.set_llm_provider(-1, "openai"))
            self.assertEqual(service.generate_llm(-1, "ещё проверка", "reply"), "ответ именно выбранного опенаи")
            self.assertEqual(len(openai.responses.calls), 1)

    def test_unavailable_grok_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"XAI_API_KEY": "", "OPENAI_API_KEY": "configured"}
        ):
            settings = self.settings(
                data_dir=Path(directory), llm_provider="openai", openai_chat_id=-1
            )
            service = LearningService(settings)
            self.assertFalse(service.set_llm_provider(-1, "grok"))
            self.assertEqual(service.llm_provider_name(-1), "openai")

    def test_persona_request_is_identical_for_openai_and_grok(self):
        settings = self.settings()
        selection = PersonaBuilder(settings).build_request(
            chat_id=-1, context="что с релизом", purpose="reply", troll_mode=True
        )
        grok_client = FakeClient("одинаковая персона")
        openai_client = FakeClient("одинаковая персона")
        GrokProvider(settings, grok_client).generate(selection.request)
        OpenAIGenerator(settings, openai_client).generate(selection.request)
        grok_call = grok_client.responses.calls[0]
        openai_call = openai_client.responses.calls[0]
        self.assertEqual(grok_call["instructions"], openai_call["instructions"])
        self.assertEqual(grok_call["input"], openai_call["input"])

    def test_old_database_gets_non_destructive_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(data_dir=Path(directory))
            service = LearningService(
                settings, openai_client=FakeClient(), xai_client=FakeClient()
            )
            self.assertEqual(service.llm_provider_name(-5), "grok")
            self.assertTrue(service.autonomous_enabled(-5))
            self.assertTrue(service.media_enabled(-5))


if __name__ == "__main__":
    unittest.main()
