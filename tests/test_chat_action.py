import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import ChatActionManager, LearningService, LearningSettings, MediaDecision


class FakeBot:
    def __init__(self, events=None, fail=False):
        self.events = events if events is not None else []
        self.fail = fail

    def send_chat_action(self, chat_id, action):
        self.events.append(("action", chat_id, action))
        if self.fail:
            raise RuntimeError("telegram unavailable")


class FixedRandom:
    def __init__(self, value=0.5):
        self.value = value

    def random(self):
        return self.value


class RecordingProvider:
    available = True

    def __init__(self, events, result="полезный ответ по существу вопроса"):
        self.events = events
        self.result = result
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        self.events.append(("llm", request.metadata["purpose"]))
        return self.result

    def summarize(self, request):
        return None


def message(text, message_id=1, reply=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=message_id, text=text, date=0,
        from_user=SimpleNamespace(id=7, username="tester", is_bot=False),
        reply_to_message=reply,
    )


class ChatActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, provider, rng=None):
        service = LearningService(
            LearningSettings(
                data_dir=Path(self.temp.name), openai_chat_id=-1,
                min_training_messages=20,
            ),
            llm_provider=provider,
            rng=rng or FixedRandom(),
        )
        service.set_media_enabled(-1, False)
        return service

    def test_grok_starts_typing_before_llm_call(self):
        events = []
        provider = RecordingProvider(events)
        manager = ChatActionManager(FakeBot(events), refresh_interval=10)
        service = self.service(provider)
        service.response_activity = manager.activity

        result = service.maybe_direct_reply(
            message("стул почему docker падает"), explicit_address=True
        )

        self.assertTrue(result)
        self.assertEqual(events[0], ("action", -1, "typing"))
        self.assertEqual(events[1], ("llm", "reply"))
        self.assertEqual(provider.calls, 1)
        self.assertEqual(manager.active_count(), 0)

    def test_local_responder_typing_has_short_jitter(self):
        events, sleeps = [], []
        provider = RecordingProvider(events)
        manager = ChatActionManager(
            FakeBot(events), refresh_interval=10, rng=FixedRandom(0.5),
            clock=lambda: 100.0, sleeper=sleeps.append,
        )
        service = self.service(provider)
        service.response_activity = manager.activity

        result = service.maybe_direct_reply(message("стул"), explicit_address=True)

        self.assertTrue(result)
        self.assertEqual(events, [("action", -1, "typing")])
        self.assertEqual(provider.calls, 0)
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 0.4)
        self.assertLessEqual(sleeps[0], 1.2)

    def test_long_generation_refreshes_typing_and_cleans_up(self):
        events = []
        manager = ChatActionManager(FakeBot(events), refresh_interval=0.03)
        with manager.activity(-1, "typing", "llm"):
            time.sleep(0.13)
        self.assertGreaterEqual(len(events), 3)
        self.assertTrue(all(event[2] == "typing" for event in events))
        self.assertEqual(manager.active_count(-1), 0)
        count = len(events)
        time.sleep(0.05)
        self.assertEqual(len(events), count)

    def test_grok_error_keeps_one_call_and_local_fallback_cleanup(self):
        events, sleeps = [], []
        provider = RecordingProvider(events, result=None)
        manager = ChatActionManager(
            FakeBot(events), refresh_interval=10, rng=FixedRandom(0),
            clock=lambda: 100.0, sleeper=sleeps.append,
        )
        service = self.service(provider)
        service.response_activity = manager.activity

        result = service.maybe_direct_reply(
            message("стул как восстановить postgres backup"),
            explicit_address=True,
        )

        self.assertTrue(result)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(
            [event for event in events if event[0] == "action"],
            [("action", -1, "typing")],
        )
        self.assertEqual(manager.active_count(), 0)
        self.assertEqual(len(sleeps), 1)

    def test_ignored_ordinary_message_has_no_typing(self):
        events = []
        provider = RecordingProvider(events)
        manager = ChatActionManager(FakeBot(events), refresh_interval=10)
        service = self.service(provider, rng=FixedRandom(0.99))
        service.response_activity = manager.activity
        with patch.object(service, "_policy_quiet_hours", return_value=False):
            self.assertIsNone(service.maybe_reply(message("обычная реплика")))
        self.assertEqual(events, [])

    def test_chat_action_error_does_not_cancel_response(self):
        provider = RecordingProvider([])
        manager = ChatActionManager(
            FakeBot(fail=True), refresh_interval=10,
            clock=lambda: 100.0, sleeper=lambda _: None,
        )
        service = self.service(provider)
        service.response_activity = manager.activity
        self.assertTrue(
            service.maybe_direct_reply(message("стул"), explicit_address=True)
        )
        self.assertEqual(manager.active_count(), 0)

    def test_same_chat_reuses_one_refresh_loop(self):
        manager = ChatActionManager(FakeBot(), refresh_interval=10)
        with manager.activity(-1, "typing"):
            first_thread = manager._active[-1].thread
            with manager.activity(-1, "typing"):
                self.assertEqual(manager.active_count(-1), 1)
                self.assertIs(manager._active[-1].thread, first_thread)
            self.assertEqual(manager.active_count(-1), 1)
        self.assertEqual(manager.active_count(-1), 0)

    def test_contextual_media_uses_matching_action(self):
        import bot as bot_module

        incoming = SimpleNamespace(chat=SimpleNamespace(id=-1), message_id=10)
        captured = []

        @contextmanager
        def activity(chat_id, action, producer=None):
            captured.append(action)
            yield None

        cases = (
            (MediaDecision(action="gif", asset_id="gif-file"), "upload_video", "send_animation"),
            (MediaDecision(action="sticker", asset_id="sticker-file"), "choose_sticker", "send_sticker"),
        )
        for decision, expected, sender_name in cases:
            with self.subTest(action=decision.action):
                with (
                    patch.object(bot_module.chat_action_manager, "activity", side_effect=activity),
                    patch.object(bot_module.bot, sender_name),
                ):
                    self.assertTrue(bot_module.send_contextual_response(incoming, decision))
                    self.assertEqual(captured.pop(), expected)

    def test_manual_meme_uses_upload_photo_without_typing(self):
        import bot as bot_module

        path = Path(self.temp.name) / "meme.png"
        path.write_bytes(b"png")
        rendered = SimpleNamespace(path=path)
        incoming = message("с м стул")
        decision = MediaDecision(action="meme", template_id="test")
        actions = []

        @contextmanager
        def activity(chat_id, action, producer=None):
            actions.append(action)
            yield None

        with (
            patch.object(bot_module.chat_action_manager, "activity", side_effect=activity),
            patch.object(bot_module.learning_service, "render_meme", return_value=rendered),
            patch.object(bot_module.learning_service, "mark_command_meme_sent"),
            patch.object(bot_module.learning_service, "cleanup_rendered_meme"),
            patch.object(bot_module.bot, "send_photo"),
        ):
            self.assertTrue(bot_module.send_manual_meme(incoming, decision))
        self.assertEqual(actions, ["upload_photo"])
        self.assertNotIn("typing", actions)


if __name__ == "__main__":
    unittest.main()
