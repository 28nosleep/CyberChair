import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from learning import (
    ChatState,
    ConversationDecision,
    GrokProvider,
    LearningService,
    LearningSettings,
    MemeLexicon,
    PersonaBuilder,
)
from learning.direct_address import SOCIAL
from learning.local_responder import LocalResponder
from learning.markov import MarkovModel
from learning.meme_renderer import MemeRenderer
from learning.repository import ChatRepository


class FixedRandom:
    def __init__(self, value=0.99):
        self.value = value

    def random(self):
        return self.value


class RecordingProvider:
    available = True

    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self.result

    def summarize(self, request):
        return None


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output)


def state(kind="humor"):
    return ChatState(
        "high", 10, kind, "релиз", .8, .8, .2, .1, .1, .4, 4, 123, 7, .85,
    )


def decision(intensity=.75):
    return ConversationDecision(
        "reply", .3, intensity, 30, "direct_mocking", 123, 7, "synthetic", 0, .3,
    )


def message(text):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=1, text=text,
        date=datetime.now(timezone.utc).timestamp(),
        from_user=SimpleNamespace(id=7, username="tester", is_bot=False),
        reply_to_message=None,
    )


class ProfanityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = LearningSettings(
            data_dir=Path(self.temp.name), openai_chat_id=-1,
            addressed_cooldown=0, generated_cooldown=0,
            direct_social_markov_share=0,
        )

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **overrides):
        values = dict(
            chat_id=-1, context="стул как починить ебаный релиз?", purpose="question",
            conversation_decision=decision(), chat_state=state(), troll_mode=True,
        )
        values.update(overrides)
        return PersonaBuilder(self.settings).build_request(**values).request

    def test_troll_on_explicitly_allows_unmasked_profanity_without_moralizing(self):
        instructions = self.build().instructions.casefold()
        self.assertIn("без звёздочек", instructions)
        self.assertIn("без морализаторства", instructions)
        self.assertNotIn("avoid profanity", instructions)
        self.assertNotIn("maintain professional respectful tone", instructions)

    def test_useful_answer_keeps_profanity_and_persona_layer(self):
        request = self.build()
        self.assertEqual(request.metadata["behavior_mode"], "useful_answer")
        self.assertTrue(request.metadata["troll_mode"])
        self.assertIn("полезность не отключает разрешённый мат", request.input)
        self.assertIn("CyberChair", request.instructions)

    def test_provider_payload_adds_no_conflicting_style_wrapper(self):
        client = SimpleNamespace(responses=FakeResponses("ебать, релиз починен"))
        request = self.build()
        result = GrokProvider(self.settings, client).generate(request)
        payload = client.responses.calls[0]
        self.assertEqual(payload["instructions"], request.instructions)
        self.assertEqual(payload["input"], request.input)
        self.assertEqual(result, "ебать, релиз починен")

    def test_local_responder_troll_pool_is_not_sterile(self):
        responder = LocalResponder(MemeLexicon(), FixedRandom())
        result, _ = responder.respond(
            -1, "стул ты охуел", SOCIAL, ChatRepository(Path(self.temp.name), -1),
            troll_mode=True, troll_intensity=.8,
        )
        self.assertRegex(result, r"(?:ебать|нахуй|нихуя|хуйня)")

    def test_local_responder_low_intensity_does_not_jump_to_hard_profanity(self):
        responder = LocalResponder(MemeLexicon(), FixedRandom())
        result, _ = responder.respond(
            -1, "стул ты охуел", SOCIAL, ChatRepository(Path(self.temp.name), -3),
            troll_mode=True, troll_intensity=.3,
        )
        self.assertNotRegex(result, r"(?:ебать|нахуй|нихуя|хуйня)")

    def test_meme_caption_normalization_does_not_mask_profanity(self):
        self.assertEqual(
            MemeRenderer.normalize_text("ну это пиздец блять"),
            "ну это пиздец блять",
        )
        self.assertIn("мат разрешён без маскировки", self.build(purpose="meme_caption").input)

    def test_markov_keeps_profanity_in_training_corpus(self):
        model = MarkovModel().train(["этот ебаный релиз опять проебал сроки"])
        self.assertIn("релиз", model.transitions[("этот", "ебаный")])
        self.assertIn("сроки", model.transitions[("опять", "проебал")])
        generated = model.generate(min_words=3, max_words=12, rng=random.Random(1))
        self.assertIn("ебаный", generated)
        self.assertIn("проебал", generated)

    def test_troll_off_keeps_neutral_behavior(self):
        request = self.build(troll_mode=False)
        self.assertIn("Не используй мат", request.instructions)
        self.assertNotIn("без звёздочек", request.instructions)
        responder = LocalResponder(MemeLexicon(), FixedRandom())
        result, _ = responder.respond(
            -1, "стул ты охуел", SOCIAL, ChatRepository(Path(self.temp.name), -2),
            troll_mode=False,
        )
        self.assertNotRegex(result, r"(?:ебать|нахуй|нихуя|хуйня)")

    def test_provider_refusal_uses_local_fallback_without_retry(self):
        provider = RecordingProvider("я не могу поддержать подобный тон")
        service = LearningService(self.settings, llm_provider=provider, rng=FixedRandom(0))
        service.set_media_enabled(-1, False)
        result = service.maybe_direct_reply(
            message("стул как восстановить postgres backup"), explicit_address=True,
        )
        self.assertTrue(result)
        self.assertNotIn("не могу поддержать", result)
        self.assertEqual(len(provider.calls), 1)

    def test_thirty_synthetic_requests_preserve_one_call_per_event(self):
        outputs = tuple(
            "63 градуса, потом обжарь — не усложняй эту хуйню" if index % 3 == 0
            else "рабочий ответ по теме, chairOS всё понял"
            for index in range(30)
        )
        provider = RecordingProvider("")
        service = LearningService(self.settings, llm_provider=provider)
        results = []
        for index, output in enumerate(outputs):
            provider.result = output
            results.append(service.generate_llm(
                -1, f"синтетический сценарий {index}", purpose="question",
                conversation_decision=decision(.3 + (index % 7) / 10),
                chat_state=state("casual"),
            ))
        self.assertEqual(len(provider.calls), 30)
        self.assertEqual(tuple(results), outputs)
        self.assertTrue(any("хуйню" in output for output in results))
        self.assertTrue(any("хуйню" not in output for output in results))
        self.assertTrue(all("***" not in output for output in results))


if __name__ == "__main__":
    unittest.main()
