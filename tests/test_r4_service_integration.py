import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bot as bot_module
from learning import (
    ConcurrencyController,
    LearningService,
    LearningSettings,
    MediaDecision,
)
from learning.event_context import RuntimeConcurrencyTelemetry


def message(message_id, chat_id=-1, text="стул почему docker падает?"):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        text=text,
        caption=None,
        content_type="text",
        date=1_776_000_000 + message_id,
        from_user=SimpleNamespace(
            id=7, username="tester", first_name="Tester", is_bot=False
        ),
        reply_to_message=None,
    )


class BlockingProvider:
    available = True
    provider_key = "test-provider"
    _usage_recorder = None

    def __init__(
        self, barrier=None, hold_first=None, release_first=None, results=None
    ):
        self.barrier = barrier
        self.hold_first = hold_first
        self.release_first = release_first
        self.results = list(results or ())
        self.requests = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def generate(self, request):
        with self.lock:
            self.requests.append(request)
            index = len(self.requests)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if index == 1 and self.hold_first is not None:
                self.hold_first.set()
                self.release_first.wait(timeout=3)
            if self.barrier is not None:
                self.barrier.wait(timeout=3)
            return (
                self.results[min(index - 1, len(self.results) - 1)]
                if self.results else
                "сначала проверь логи и откати только последний сломанный релиз"
            )
        finally:
            with self.lock:
                self.active -= 1

    def summarize(self, request):
        return None


class FixedRandom:
    def random(self):
        return 0.9

    @staticmethod
    def choice(values):
        return list(values)[0]


class R4ServiceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, provider, controller):
        service = LearningService(
            LearningSettings(
                data_dir=Path(self.temp.name),
                openai_chat_id=-1,
                min_training_messages=1,
                summary_message_interval=50,
                generated_cooldown=0,
                addressed_cooldown=0,
                max_generated_per_hour=100,
            ),
            llm_provider=provider,
            rng=FixedRandom(),
            concurrency_controller=controller,
        )
        return service

    @staticmethod
    def bot_patches(service):
        return (
            patch.object(bot_module, "learning_service", service),
            patch.object(bot_module, "remember_user"),
            patch.object(
                bot_module, "get_bot_identity",
                return_value={"id": 99, "username": "chair"},
            ),
            patch.object(
                bot_module.bot, "reply_to",
                return_value=SimpleNamespace(message_id=900),
            ),
        )

    def test_production_handler_serializes_same_chat_through_commit(self):
        first_inside = threading.Event()
        release = threading.Event()
        provider = BlockingProvider(hold_first=first_inside, release_first=release)
        telemetry = RuntimeConcurrencyTelemetry()
        control = ConcurrencyController(2, 1, 1, 1, telemetry)
        service = self.service(provider, control)
        service.set_media_enabled(-1, False)
        first = message(100)
        second = message(101)

        patches = self.bot_patches(service)
        with patches[0], patches[1], patches[2], patches[3] as send:
            threads = [
                threading.Thread(target=bot_module.handle_message, args=(incoming,))
                for incoming in (first, second)
            ]
            threads[0].start()
            self.assertTrue(first_inside.wait(timeout=2))
            threads[1].start()
            time.sleep(0.05)
            self.assertEqual(len(provider.requests), 1)
            self.assertEqual(send.call_count, 0)
            release.set()
            for thread in threads:
                thread.join(timeout=4)
                self.assertFalse(thread.is_alive())

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.peak, 1)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(len(service.repository(-1).recent_generated(10)), 2)
        self.assertEqual(control.snapshot()["chat_gate_registry_size"], 0)

    def test_production_handler_keeps_different_chats_parallel(self):
        barrier = threading.Barrier(2)
        provider = BlockingProvider(barrier=barrier)
        telemetry = RuntimeConcurrencyTelemetry()
        control = ConcurrencyController(2, 1, 1, 1, telemetry)
        service = self.service(provider, control)
        service.llm_allowed = lambda chat_id: True
        for chat_id in (-1, -2):
            service.set_media_enabled(chat_id, False)

        patches = self.bot_patches(service)
        with patches[0], patches[1], patches[2], patches[3] as send:
            threads = [
                threading.Thread(
                    target=bot_module.handle_message,
                    args=(message(200 + index, chat_id=chat_id),),
                )
                for index, chat_id in enumerate((-1, -2))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=4)
                self.assertFalse(thread.is_alive())

        self.assertEqual(provider.peak, 2)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(telemetry.snapshot()["peak_active_llm_calls"], 2)

    def test_direct_admission_timeout_uses_local_plan_without_provider_call(self):
        provider = BlockingProvider()
        telemetry = RuntimeConcurrencyTelemetry()
        control = ConcurrencyController(1, 1, 0.03, 1, telemetry)
        service = self.service(provider, control)
        service.set_media_enabled(-1, False)
        occupied = threading.Event()
        release = threading.Event()

        def holder():
            with control.llm_slot("occupied", -2):
                occupied.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(occupied.wait(timeout=1))
        patches = self.bot_patches(service)
        with patches[0], patches[1], patches[2], patches[3] as send:
            bot_module.handle_message(message(300))
        release.set()
        thread.join(timeout=2)

        self.assertEqual(provider.requests, [])
        send.assert_called_once()
        report = service.repository(-1).routing_report("2020-01-01T00:00:00+00:00")
        self.assertEqual(report.get("llm_admission_timeout"), 1)
        self.assertEqual(report.get("route_selected_local"), 1)
        self.assertEqual(telemetry.snapshot()["active_llm_calls"], 0)

    def test_two_matching_continuations_do_not_consume_pending_twice(self):
        first_inside = threading.Event()
        release = threading.Event()
        provider = BlockingProvider(hold_first=first_inside, release_first=release)
        telemetry = RuntimeConcurrencyTelemetry()
        control = ConcurrencyController(2, 1, 1, 1, telemetry)
        service = self.service(provider, control)
        service.set_media_enabled(-1, False)
        repository = service.repository(-1)
        repository.save_pending_conversation(
            user_id=7,
            original_message_id=1,
            original_question="что выбрать",
            clarification_question="между чем?",
            intent="choice",
            context="между чем выбираешь?",
            expected_type="choices",
            pending_mode="hard",
            bot_message_id=400,
        )
        incoming = (message(401, text="айфон или пиксель"),
                    message(402, text="самсунг или пиксель"))
        patches = self.bot_patches(service)
        with patches[0], patches[1], patches[2], patches[3] as send:
            threads = [
                threading.Thread(target=bot_module.handle_message, args=(item,))
                for item in incoming
            ]
            threads[0].start()
            self.assertTrue(first_inside.wait(timeout=2))
            threads[1].start()
            time.sleep(0.05)
            self.assertEqual(len(provider.requests), 1)
            release.set()
            for thread in threads:
                thread.join(timeout=4)
                self.assertFalse(thread.is_alive())

        self.assertEqual(len(provider.requests), 1)
        send.assert_called_once()
        self.assertIsNone(repository.pending_conversation(7, 1200))

    def test_rapid_followup_sees_pending_committed_by_first_delivery(self):
        first_inside = threading.Event()
        release = threading.Event()
        provider = BlockingProvider(
            hold_first=first_inside,
            release_first=release,
            results=(
                "между чем выбираешь?",
                "бери пиксель: камера ровнее и обновления приходят без цирка",
            ),
        )
        telemetry = RuntimeConcurrencyTelemetry()
        control = ConcurrencyController(2, 1, 1, 1, telemetry)
        service = self.service(provider, control)
        service.set_media_enabled(-1, False)
        patches = self.bot_patches(service)
        with patches[0], patches[1], patches[2], patches[3] as send:
            first = threading.Thread(
                target=bot_module.handle_message,
                args=(message(500, text="стул что выбрать?"),),
            )
            second = threading.Thread(
                target=bot_module.handle_message,
                args=(message(501, text="айфон или пиксель"),),
            )
            first.start()
            self.assertTrue(first_inside.wait(timeout=2))
            second.start()
            time.sleep(0.05)
            self.assertEqual(len(provider.requests), 1)
            release.set()
            first.join(timeout=4)
            second.join(timeout=4)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(send.call_count, 2)
        self.assertIn("айфон или пиксель", provider.requests[1].input)
        self.assertIsNone(service.repository(-1).pending_conversation(7, 1200))

    def test_autonomous_same_chat_does_not_enter_busy_user_lifecycle(self):
        provider = BlockingProvider()
        telemetry = RuntimeConcurrencyTelemetry()
        control = ConcurrencyController(2, 1, 1, 1, telemetry)
        service = self.service(provider, control)
        user_event = service._normalized_event(message(600))
        current = SimpleNamespace(isoformat=lambda: "2026-08-18T12:00:00+00:00")
        with (
            patch.object(bot_module, "learning_service", service),
            patch.object(service, "prepare_autonomous") as prepare,
            service.chat_event_slot(user_event),
        ):
            result = bot_module.run_autonomous_response(-1, current, True)
        self.assertIsNone(result)
        prepare.assert_not_called()
        self.assertEqual(
            control.snapshot()["autonomous_skipped_chat_busy"], 1
        )

    def test_explicit_meme_media_timeout_stays_single_special_route(self):
        provider = BlockingProvider()
        telemetry = RuntimeConcurrencyTelemetry()
        control = ConcurrencyController(2, 1, 1, 0.02, telemetry)
        service = self.service(provider, control)
        ready = threading.Event()
        release = threading.Event()

        def holder():
            with control.media_slot("other-render", -2):
                ready.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(ready.wait(timeout=1))
        decision = MediaDecision(
            action="meme", template_id="synthetic", caption_text="caption"
        )
        with (
            patch.object(bot_module, "learning_service", service),
            patch.object(bot_module, "remember_user") as remember,
            patch.object(service, "maybe_command_meme", return_value=decision),
            patch.object(service, "render_meme") as render,
            patch.object(bot_module.bot, "reply_to") as text_send,
            patch.object(bot_module.bot, "send_photo") as photo_send,
        ):
            bot_module.handle_message(message(700, text="с м стул"))
        release.set()
        thread.join(timeout=2)

        render.assert_not_called()
        text_send.assert_not_called()
        photo_send.assert_not_called()
        remember.assert_not_called()
        self.assertEqual(provider.requests, [])
        self.assertEqual(control.snapshot()["media_admission_timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
