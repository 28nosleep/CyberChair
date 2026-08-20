import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import LearningService, LearningSettings, LocalIntentClassifier, MediaDecision


class FixedRandom:
    def __init__(self, value=0.5):
        self.value = value

    def random(self):
        return self.value


class CountingProvider:
    available = True

    def __init__(self, result="полезный ответ по существу вопроса"):
        self.result = result
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self.result

    def summarize(self, request):
        return None


def message(text, message_id=1, reply=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1),
        message_id=message_id,
        text=text,
        date=0,
        from_user=SimpleNamespace(id=7, username="tester", is_bot=False),
        reply_to_message=reply,
    )


def bot_reply(text="исходная реплика стула"):
    return SimpleNamespace(
        message_id=900,
        text=text,
        from_user=SimpleNamespace(id=99, username="chair", is_bot=True),
    )


class DirectAddressRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, provider=None, rng=None, **overrides):
        values = dict(
            data_dir=Path(self.temp.name), openai_chat_id=-1,
            min_training_messages=20,
        )
        values.update(overrides)
        provider = provider or CountingProvider()
        instance = LearningService(
            LearningSettings(**values), llm_provider=provider,
            rng=rng or FixedRandom(),
        )
        instance.set_media_enabled(-1, False)
        return instance, provider

    def test_classifier_regression_matrix(self):
        classifier = LocalIntentClassifier()
        cases = (
            ("", False, "trivial"),
            ("ты тут", False, "trivial"),
            ("иди нахуй", False, "social"),
            ("кто тут соя", False, "social"),
            ("что думаешь", False, "substantive"),
            ("как приготовить курицу", False, "substantive"),
            ("почему docker падает", False, "substantive"),
            ("почему", True, "substantive"),
            ("а если наоборот?", True, "substantive"),
        )
        for text, reply, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    classifier.classify(text, direct_reply=reply), expected
                )

    def test_bare_chair_is_guaranteed_and_free(self):
        service, provider = self.service()
        result = service.maybe_direct_reply(message("стул"), explicit_address=True)
        self.assertTrue(result)
        self.assertEqual(provider.calls, [])
        self.assertEqual(service.foreground._last_direct_decision[-1].intent, "trivial")

    def test_short_summon_is_guaranteed_and_free(self):
        service, provider = self.service()
        result = service.maybe_direct_reply(message("стул ты тут"), explicit_address=True)
        self.assertTrue(result)
        self.assertEqual(provider.calls, [])

    def test_simple_social_is_local(self):
        service, provider = self.service()
        result = service.maybe_direct_reply(message("стул иди нахуй"), explicit_address=True)
        self.assertTrue(result)
        self.assertEqual(provider.calls, [])
        self.assertEqual(service.foreground._last_direct_decision[-1].priority, "P2")

    def test_substantive_uses_exactly_one_llm_call(self):
        service, provider = self.service()
        result = service.maybe_direct_reply(
            message("стул почему docker падает"), explicit_address=True
        )
        self.assertIn("полезный", result)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(service.foreground._last_direct_decision[-1].priority, "P3")

    def test_how_to_runtime_pipeline_uses_grok_and_accepts_useful_long_answer(self):
        answer = (
            "сначала добейся узнаваемого звука, затем регулярно выпускай треки и сниппеты, "
            "делай коллабы с артистами своего размера, играй лайвы и собирай вокруг имени "
            "понятный образ: один сильный трек помогает, но без повторяемой системы внимания "
            "он обычно тонет в рекомендациях быстрее, чем chairOS успевает заскрипеть колёсиками"
        )
        service, provider = self.service(CountingProvider(answer))
        with self.assertLogs("learning.service", level="INFO") as logs:
            result = service.maybe_direct_reply(
                message("так и как по итогу прославиться в рэпе, стул?"),
                explicit_address=True,
            )
        self.assertEqual(result, answer)
        self.assertEqual(len(provider.calls), 1)
        decision = service.foreground._last_direct_decision[-1]
        self.assertEqual(decision.intent, "substantive")
        self.assertEqual(decision.priority, "P3")
        self.assertEqual(decision.producer, "llm")
        rendered_logs = "\n".join(logs.output)
        self.assertIn("DIRECT_ROUTE", rendered_logs)
        self.assertIn("intent=how_to", rendered_logs)
        self.assertIn("LLM_RESULT", rendered_logs)
        self.assertIn("accepted=true", rendered_logs)
        self.assertNotIn("дай предмет", result)

    def test_how_to_provider_failure_is_graceful_not_a_phantom_clarification(self):
        service, provider = self.service(CountingProvider(None))
        result = service.maybe_direct_reply(
            message("стул как прославиться в рэпе"), explicit_address=True,
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("вопрос", result)
        self.assertNotIn("дай предмет", result)
        self.assertNotIn("уточни", result)

    def test_normal_useful_answer_is_not_a_provider_refusal(self):
        service, provider = self.service(CountingProvider(
            "сделай пять треков с узнаваемой подачей, режь из них сниппеты и регулярно выпускай лучшее"
        ))
        result = service.maybe_direct_reply(
            message("стул как прославиться в рэпе"), explicit_address=True,
        )
        self.assertEqual(result, provider.result)

    def test_reply_to_chair_is_always_answered(self):
        service, provider = self.service()
        result = service.maybe_direct_reply(
            message("ахах", reply=bot_reply()), bot_id=99,
        )
        self.assertTrue(result)
        self.assertEqual(provider.calls, [])

    def test_dependent_reply_uses_target_context(self):
        service, provider = self.service()
        result = service.maybe_direct_reply(
            message("почему", reply=bot_reply("серёга уже полностью кукд")),
            bot_id=99,
        )
        self.assertTrue(result)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("серёга уже полностью кукд", provider.calls[0].input)

    def test_llm_failure_falls_back_locally_without_second_call(self):
        service, provider = self.service(CountingProvider(None))
        result = service.maybe_direct_reply(
            message("стул как восстановить postgres backup"),
            explicit_address=True,
        )
        self.assertTrue(result)
        self.assertEqual(len(provider.calls), 1)
        report = service.direct_response_diagnostics(-1)
        self.assertEqual(report["llm_fallback_local"], 1)
        self.assertEqual(report["routes"]["local"], 1)

    def test_soft_budget_never_blocks_p1_or_p3(self):
        service, provider = self.service(xai_daily_chat_budget_usd=0.000001)
        service.repository(-1).record_llm_call(
            "grok", "grok-test", "reply", {"cost_usd_ticks": 100_000}
        )
        local = service.maybe_direct_reply(message("стул", 1), explicit_address=True)
        useful = service.maybe_direct_reply(
            message("стул объясни как работает dns", 2), explicit_address=True
        )
        self.assertTrue(local)
        self.assertTrue(useful)
        self.assertEqual(len(provider.calls), 1)

    def test_media_and_grok_are_mutually_exclusive(self):
        service, provider = self.service()
        service.set_media_enabled(-1, True)
        gif = MediaDecision(
            action="gif", asset_id="file-id", reason="test",
            asset_key="gif:key", cooldown_group="gif_reaction", archetype="gif",
        )
        with patch.object(service.media, "decide", return_value=gif):
            result = service.maybe_direct_reply(
                message("стул ты долбоёб"), explicit_address=True
            )
        self.assertIs(result, gif)
        self.assertEqual(provider.calls, [])

    def test_free_social_and_grok_are_mutually_exclusive(self):
        service, provider = self.service(rng=FixedRandom(0.0))
        result = service.maybe_direct_reply(
            message("стул чё ты несёшь"), explicit_address=True
        )
        self.assertTrue(result)
        self.assertEqual(provider.calls, [])

    def test_local_responder_avoids_immediate_repeats(self):
        service, _ = self.service(rng=FixedRandom(0.0))
        outputs = [
            service.maybe_direct_reply(message("стул", item), explicit_address=True)
            for item in range(1, 5)
        ]
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertNotIn("минус аура", " ".join(outputs))

    def test_non_direct_message_keeps_old_probabilistic_policy(self):
        service, provider = self.service(rng=FixedRandom(0.99))
        with patch.object(service, "_policy_quiet_hours", return_value=False):
            result = service.maybe_reply(message("обычное сообщение"))
        self.assertIsNone(result)
        self.assertEqual(provider.calls, [])

    def test_private_routing_report_has_counts_not_message_text(self):
        service, _ = self.service()
        service.maybe_direct_reply(message("стул секретный текст"), explicit_address=True)
        report = service.direct_response_diagnostics(
            -1, current=datetime.now(timezone.utc)
        )
        self.assertEqual(report["received"], 1)
        self.assertEqual(report["answered"], 1)
        self.assertEqual(report["response_rate"], 1.0)
        self.assertNotIn("секретный", repr(report))


if __name__ == "__main__":
    unittest.main()
