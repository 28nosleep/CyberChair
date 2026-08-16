import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from learning import (
    ChatState,
    ConversationPolicy,
    LearningService,
    LearningSettings,
)


class LocalProvider:
    available = True

    def __init__(self):
        self.generate_calls = []
        self.summarize_calls = []

    def generate(self, request):
        self.generate_calls.append(request)
        return "нормальный короткий ответ стула"

    def summarize(self, request):
        self.summarize_calls.append(request)
        return None


class ZeroRandom:
    def random(self):
        return 0.0


def incoming(message_id, text, user_id=1, reply=None, created_at=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1),
        message_id=message_id,
        text=text,
        date=created_at or 0,
        from_user=SimpleNamespace(
            id=user_id,
            username=f"user{user_id}",
            first_name=f"User {user_id}",
            is_bot=False,
        ),
        reply_to_message=reply,
    )


class ConversationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, **overrides):
        values = {
            "data_dir": self.data_dir,
            "openai_chat_id": -1,
            "summary_message_interval": 50,
            "timezone_name": "Europe/Moscow",
        }
        values.update(overrides)
        return LearningSettings(**values)

    def service(self, provider=None, rng=None, **settings):
        provider = provider or LocalProvider()
        service = LearningService(
            self.settings(**settings),
            llm_provider=provider,
            rng=rng or ZeroRandom(),
        )
        service.memory._clock = lambda: self.now
        service.chat_state_analyzer._clock = lambda: self.now
        return service, provider

    def add(self, repository, message_id, user_id, text, seconds_ago, reply_to=None):
        repository.add_message(
            message_id,
            user_id,
            f"user{user_id}",
            text,
            self.now - timedelta(seconds=seconds_ago),
            reply_to_message_id=reply_to,
            is_reply=reply_to is not None,
        )

    def analyze(self, rows, last_target_user_id=None):
        service, _ = self.service()
        repository = service.repository(-1)
        repository.clear()
        for row in rows:
            self.add(repository, *row)
        return service.chat_state_analyzer.analyze(
            repository,
            last_target_user_id=last_target_user_id,
            now=self.now,
        )

    def state(self, **overrides):
        values = {
            "activity_level": "normal",
            "silence_seconds": 10,
            "conversation_type": "casual",
            "dominant_topic": None,
            "topic_strength": 0.0,
            "humor_score": 0.0,
            "argument_score": 0.0,
            "serious_score": 0.0,
            "work_score": 0.0,
            "reply_density": 0.0,
            "participant_count": 2,
            "target_message_id": 1,
            "target_user_id": 1,
            "confidence": 0.6,
        }
        values.update(overrides)
        return ChatState(**values)

    def test_silence_is_measured_from_latest_activity(self):
        state = self.analyze([(1, 1, "последняя реплика", 125, None)])
        self.assertEqual(state.silence_seconds, 125)

    def test_low_normal_high_and_burst_activity_are_distinct(self):
        low = self.analyze([(1, 1, "редкая реплика", 20, None)])
        normal = self.analyze(
            [(index, index % 2 + 1, f"обычная реплика {index}", 240 - index * 50, None)
             for index in range(1, 5)]
        )
        high = self.analyze(
            [(index, index % 2 + 1, f"активная реплика {index}", 290 - index * 30, None)
             for index in range(1, 9)]
        )
        burst = self.analyze(
            [(index, index % 3 + 1, f"быстрая реплика {index}", 50 - index * 5, None)
             for index in range(1, 9)]
        )
        self.assertEqual(
            [low.activity_level, normal.activity_level, high.activity_level, burst.activity_level],
            ["low", "normal", "high", "burst"],
        )

    def test_plain_profanity_without_reply_conflict_is_not_argument(self):
        state = self.analyze(
            [
                (1, 1, "бля опять дождь", 120, None),
                (2, 2, "охренеть какая погода", 80, None),
                (3, 1, "пиздец промок", 40, None),
            ]
        )
        self.assertNotEqual(state.conversation_type, "argument")
        self.assertLess(state.argument_score, 0.42)

    def test_active_reciprocal_exchange_is_argument(self):
        state = self.analyze(
            [
                (1, 1, "я считаю иначе", 100, None),
                (2, 2, "нет ты не понял, это бред", 80, 1),
                (3, 1, "нет, перечитай аргументы", 60, 2),
                (4, 2, "чушь, ты ошибаешься", 40, 3),
            ]
        )
        self.assertEqual(state.conversation_type, "argument")
        self.assertGreater(state.argument_score, 0.5)

    def test_joke_stream_is_humor(self):
        state = self.analyze(
            [
                (1, 1, "ахахаха вот это релиз", 100, None),
                (2, 2, "лол, лучший баг", 80, None),
                (3, 3, "ору с этого 😂", 60, None),
                (4, 1, "кек", 40, None),
                (5, 2, "🤣🤣", 20, None),
            ]
        )
        self.assertIn(state.conversation_type, {"humor", "mixed"})
        self.assertGreater(state.humor_score, 0.5)

    def test_long_sequential_discussion_is_serious(self):
        texts = [
            "важно понять причину проблемы потому что текущее решение создаёт дополнительные риски для всей команды и пользователей продукта",
            "предлагаю сначала проверить несколько вариантов решения затем сравнить последствия и только после этого принимать окончательное решение",
            "необходимо подробно проанализировать причины потому что прежний вариант не учитывает важные ограничения и возможные последствия",
        ]
        state = self.analyze(
            [(index, index, text, 120 - index * 30, None) for index, text in enumerate(texts, 1)]
        )
        self.assertEqual(state.conversation_type, "serious")
        self.assertGreater(state.serious_score, 0.5)

    def test_work_discussion_is_work(self):
        state = self.analyze(
            [
                (1, 1, "задача по релизу готова", 100, None),
                (2, 2, "баг на сервере записал в тикет", 70, None),
                (3, 3, "созвон с клиентом после ревью", 40, None),
            ]
        )
        self.assertEqual(state.conversation_type, "work")
        self.assertGreater(state.work_score, 0.5)

    def test_strong_work_humor_combination_is_mixed(self):
        state = self.analyze(
            [
                (1, 1, "ахаха релиз опять упал", 100, None),
                (2, 2, "лол баг на сервере", 80, None),
                (3, 3, "ору с этого деплоя 😂", 60, None),
                (4, 1, "кек клиент ждёт отчёт", 40, None),
            ]
        )
        self.assertEqual(state.conversation_type, "mixed")

    def test_dominant_topic_requires_obvious_repetition(self):
        state = self.analyze(
            [
                (1, 1, "релиз перенесли", 100, None),
                (2, 2, "релиз будет завтра", 70, None),
                (3, 3, "проверяем релиз", 40, None),
            ]
        )
        self.assertEqual(state.dominant_topic, "релиз")
        self.assertGreater(state.topic_strength, 0.5)

    def test_dominant_topic_is_none_for_weak_signal(self):
        state = self.analyze(
            [
                (1, 1, "сегодня дождь", 100, None),
                (2, 2, "завтра отпуск", 70, None),
                (3, 3, "вечером кино", 40, None),
            ]
        )
        self.assertIsNone(state.dominant_topic)

    def test_humor_increases_troll_intensity(self):
        policy = ConversationPolicy(self.settings())
        casual = policy.decide(self.state(), local_allowed=True, llm_allowed=True)
        humor = policy.decide(
            self.state(conversation_type="humor", humor_score=0.9),
            local_allowed=True,
            llm_allowed=True,
        )
        self.assertGreater(humor.troll_intensity, casual.troll_intensity)

    def test_serious_keeps_nonzero_troll_intensity(self):
        decision = ConversationPolicy(self.settings()).decide(
            self.state(conversation_type="serious", serious_score=0.9)
        )
        self.assertGreater(decision.troll_intensity, 0.0)
        self.assertGreater(decision.reply_probability, 0.0)

    def test_burst_cannot_bypass_cooldown_and_hard_limits(self):
        policy = ConversationPolicy(self.settings())
        decision = policy.decide(
            self.state(activity_level="burst"),
            local_allowed=False,
            llm_allowed=False,
        )
        self.assertEqual(decision.action, "none")
        self.assertEqual(decision.reply_probability, 0.0)

    def test_target_selection_never_chooses_bot_message(self):
        service, _ = self.service()
        repository = service.repository(-1)
        self.add(repository, 1, 1, "ахаха интересная реплика", 40)
        repository.record_generated("ахаха ответ самого бота", "random")
        state = service.chat_state_analyzer.analyze(repository, now=self.now)
        self.assertEqual(state.target_message_id, 1)
        self.assertEqual(state.target_user_id, 1)

    def test_target_selection_avoids_same_user_when_alternative_exists(self):
        state = self.analyze(
            [
                (1, 1, "ахаха очень смешная реплика", 60, None),
                (2, 2, "лол другая смешная реплика", 30, None),
            ],
            last_target_user_id=1,
        )
        self.assertEqual(state.target_user_id, 2)

    def test_policy_can_return_none(self):
        decision = ConversationPolicy(self.settings()).decide(
            self.state(), local_allowed=False, llm_allowed=False
        )
        self.assertEqual(decision.action, "none")

    def test_analysis_does_not_add_an_llm_request(self):
        provider = LocalProvider()
        service, _ = self.service(provider, rng=Mock(random=Mock(return_value=0.99)))
        repository = service.repository(-1)
        self.add(repository, 1, 1, "обычное входящее сообщение", 0)
        with patch.object(service, "_policy_quiet_hours", return_value=False):
            self.assertIsNone(service.maybe_reply(incoming(1, "обычное входящее сообщение")))
        self.assertEqual(provider.generate_calls, [])
        self.assertEqual(provider.summarize_calls, [])

    def test_old_trigger_probability_is_not_applied_over_policy(self):
        rng = Mock(random=Mock(return_value=0.0))
        service, _ = self.service(rng=rng)
        repository = service.repository(-1)
        self.add(repository, 1, 1, "обычное входящее сообщение", 0)
        with (
            patch.object(service, "_policy_quiet_hours", return_value=False),
            patch.object(service.triggers, "decide_user_reply") as old_decision,
            patch.object(service, "generate_local", return_value="локальный ответ"),
        ):
            self.assertEqual(
                service.maybe_reply(incoming(1, "обычное входящее сообщение")),
                "локальный ответ",
            )
        old_decision.assert_not_called()
        self.assertEqual(rng.random.call_count, 1)

    def test_generation_request_carries_state_and_policy_metadata(self):
        provider = LocalProvider()
        service, _ = self.service(provider, rng=ZeroRandom())
        repository = service.repository(-1)
        self.add(repository, 1, 1, "почему контейнер падает", 0)
        bot_reply = SimpleNamespace(from_user=SimpleNamespace(id=99), message_id=900)
        message = incoming(1, "почему контейнер падает", reply=bot_reply)
        self.assertIsNotNone(service.maybe_reply(message, bot_id=99))
        metadata = provider.generate_calls[0].metadata
        self.assertIn("troll_intensity", metadata["conversation_decision"])
        self.assertIn("conversation_type", metadata["chat_state"])


if __name__ == "__main__":
    unittest.main()
