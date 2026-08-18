import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from learning.autonomous_policy import AutonomousPolicy
from learning.chat_state import ChatState
from learning.conversation_policy import ConversationPolicy
from learning.media_service import MediaDecision
from learning.response_plan import DeliveryReceipt
from learning.service import LearningService
from learning.settings import LearningSettings


NOW = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)


def state(silence=600, kind="humor", activity="normal"):
    return ChatState(
        activity_level=activity, silence_seconds=silence, conversation_type=kind,
        dominant_topic="релиз", topic_strength=.7, humor_score=.8 if kind == "humor" else .1,
        argument_score=.1, serious_score=.1, work_score=.1, reply_density=.2,
        participant_count=3, target_message_id=1, target_user_id=2, confidence=.8,
    )


class AutonomousPolicyTests(unittest.TestCase):
    def settings(self, **overrides):
        values = dict(
            autonomous_min_silence=300, autonomous_max_silence=21600,
            autonomous_cooldown=7200, autonomous_bot_pause=0,
            autonomous_no_response_cooldown=21600, autonomous_daily_limit=3,
            autonomous_probability_cap=.5, autonomous_active_message_count=6,
        )
        values.update(overrides)
        return LearningSettings(**values)

    def decide(self, chat_state, **kwargs):
        settings = self.settings()
        return AutonomousPolicy(settings, ConversationPolicy(settings)).decide(
            chat_state, current=NOW, troll_mode=True, **kwargs
        )

    def test_short_silence_is_not_an_autonomous_trigger(self):
        decision = self.decide(state(299))
        self.assertEqual(decision.action, "none")
        self.assertEqual(decision.reason, "silence_too_short")

    def test_probability_falls_after_long_dead_chat(self):
        recent = self.decide(state(15 * 60), prior_activity=9)
        long_pause = self.decide(state(2 * 60 * 60), prior_activity=1)
        self.assertGreater(recent.probability, long_pause.probability)
        self.assertEqual(self.decide(state(7 * 60 * 60)).reason, "chat_too_quiet")

    def test_work_hours_are_more_conservative_than_evening(self):
        settings = self.settings()
        policy = AutonomousPolicy(settings, ConversationPolicy(settings))
        daytime = policy.decide(state(15 * 60), current=NOW, troll_mode=True, prior_activity=8)
        evening = policy.decide(
            state(15 * 60), current=NOW.replace(hour=20), troll_mode=True, prior_activity=8
        )
        self.assertLess(daytime.probability, evening.probability)

    def test_autonomous_cooldown_and_unanswered_guard(self):
        last = NOW - timedelta(minutes=30)
        self.assertEqual(
            self.decide(state(), last_autonomous_at=last, last_human_at=NOW - timedelta(hours=1)).reason,
            "autonomous_cooldown",
        )
        self.assertEqual(
            self.decide(state(), last_autonomous_at=NOW - timedelta(hours=3), last_human_at=NOW - timedelta(hours=4)).reason,
            "awaiting_human_reaction",
        )

    def test_daily_limit_and_troll_mode_are_hard_gates(self):
        self.assertEqual(self.decide(state(), daily_count=3).reason, "daily_limit")
        settings = self.settings()
        policy = AutonomousPolicy(settings, ConversationPolicy(settings))
        self.assertEqual(
            policy.decide(state(), current=NOW, troll_mode=False).reason,
            "troll_mode_off",
        )


class AutonomousServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = LearningSettings(
            data_dir=Path(self.temp.name), openai_chat_id=-1,
            autonomous_min_silence=300, autonomous_bot_pause=0,
            generated_cooldown=0, max_generated_per_hour=10,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_rejected_decision_does_not_call_llm(self):
        provider = Mock(available=True)
        service = LearningService(self.settings, llm_provider=provider)
        with patch.object(service.chat_state_analyzer, "analyze", return_value=state(120)):
            self.assertIsNone(service.prepare_autonomous(-1, NOW))
        provider.generate.assert_not_called()

    def test_selected_text_calls_llm_once_after_local_policy(self):
        provider = Mock(available=True)
        events = []

        def generate(request):
            events.append("llm")
            return "ну релиз конечно кукд"

        provider.generate.side_effect = generate
        service = LearningService(self.settings, llm_provider=provider)

        @contextmanager
        def activity(chat_id, action, producer=None):
            events.append(action)
            yield None

        service.response_activity = activity
        repository = service.repository(-1)
        repository.add_message(1, 2, "u2", "релиз опять упал", NOW - timedelta(minutes=10))
        service.rng = SimpleNamespace(random=lambda: 0.0)
        with (
            patch.object(service.chat_state_analyzer, "analyze", return_value=state(600)),
            patch.object(service, "activity_allows", return_value=True),
            patch.object(service.media, "decide", return_value=MediaDecision()),
        ):
            plan = service.prepare_autonomous(-1, NOW)
        self.assertEqual(plan.payload.text, "ну релиз конечно кукд")
        self.assertEqual(events, ["typing", "llm"])
        provider.generate.assert_called_once()
        service.finalize_response(
            plan,
            DeliveryReceipt(
                plan.event_id, True, plan.delivery_type, telegram_message_id=42
            ),
        )
        self.assertEqual(repository.latest_generated(("autonomous",))["kind"], "autonomous")
