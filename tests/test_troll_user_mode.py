import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from learning import LearningService, LearningSettings


class FixedRandom:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


class Provider:
    available = True

    def __init__(self, response="полезный ответ по существу"):
        self.response = response
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self.response

    def summarize(self, request):
        return None


def message(text, message_id=1):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-42), message_id=message_id, text=text, date=0,
        from_user=SimpleNamespace(id=7, username="tester", is_bot=False),
        reply_to_message=None,
    )


class TrollUserModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, roll, response="полезный ответ по существу", **settings):
        values = dict(data_dir=Path(self.temp.name), openai_chat_id=-42,
                      min_training_messages=1)
        values.update(settings)
        provider = Provider(response)
        service = LearningService(LearningSettings(**values), llm_provider=provider,
                                  rng=FixedRandom(roll))
        service.set_media_enabled(-42, False)
        return service, provider

    def test_rng_boundary_selects_one_mode_before_one_call(self):
        troll, provider = self.service(.49)
        troll.maybe_direct_reply(message("стул почему docker падает?"), explicit_address=True)
        self.assertEqual(provider.calls[0].metadata["purpose"], "troll_user")
        self.assertEqual(provider.calls[0].metadata["behavior_mode"], "troll_user")
        self.assertEqual(len(provider.calls), 1)

        useful, provider = self.service(.50)
        useful.maybe_direct_reply(message("стул почему docker падает?", 2), explicit_address=True)
        self.assertEqual(provider.calls[0].metadata["behavior_mode"], "useful_answer")
        self.assertEqual(len(provider.calls), 1)

    def test_every_substantive_question_kind_uses_the_same_gate(self):
        cases = ("стул как набрать вес?", "стул почему docker падает?",
                 "стул посоветуй SSD", "стул айфон или пиксель?",
                 "стул что думаешь про альбом?")
        for text in cases:
            with self.subTest(text=text):
                service, provider = self.service(.0)
                service.maybe_direct_reply(message(text), explicit_address=True)
                self.assertEqual(provider.calls[0].metadata["behavior_mode"], "troll_user")

    def test_troll_off_never_selects_troll_user(self):
        service, provider = self.service(.0)
        service.set_troll_mode(-42, False)
        service.maybe_direct_reply(message("стул как накопить денег?"), explicit_address=True)
        self.assertEqual(provider.calls[0].metadata["behavior_mode"], "useful_answer")

    def test_troll_provider_failure_uses_roast_and_never_pending(self):
        service, provider = self.service(.0, response=None)
        result = service.maybe_direct_reply(message("стул что делать если болит голова?"), explicit_address=True)
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn("обратись", result.casefold())
        self.assertIsNone(service.pending_conversation(-42, 7))

    def test_useful_provider_failure_keeps_useful_fallback(self):
        service, provider = self.service(.9, response=None)
        result = service.maybe_direct_reply(message("стул как накопить денег?"), explicit_address=True)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("внешний мозг", result)

    def test_choice_troll_does_not_create_hard_pending(self):
        service, provider = self.service(.0)
        service.maybe_direct_reply(message("стул айфон или пиксель?"), explicit_address=True)
        self.assertEqual(provider.calls[0].metadata["behavior_mode"], "troll_user")
        self.assertIsNone(service.pending_conversation(-42, 7))

    def test_troll_prompt_prioritizes_relevant_stable_callback(self):
        service, provider = self.service(.0)
        service.repository(-42).remember_stable(["пользователь удаляет все игры каждую неделю"])
        service.maybe_direct_reply(message("стул какую игру купить?"), explicit_address=True)
        request = provider.calls[0]
        self.assertEqual(request.metadata["behavior_mode"], "troll_user")
        self.assertIn("удаляет все игры", request.input)
        self.assertIn("не выдумывай прошлые события", request.input)


if __name__ == "__main__":
    unittest.main()
