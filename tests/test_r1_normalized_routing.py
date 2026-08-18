import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    LearningService,
    LearningSettings,
    normalize_callback_event,
    normalize_telegram_event,
)
from learning.event_context import current_event, runtime_concurrency


def user(user_id=7, username="user", is_bot=False):
    return SimpleNamespace(
        id=user_id, username=username, first_name="User", is_bot=is_bot
    )


def media(file_id="photo", *, mime_type=None, width=640, height=480):
    return SimpleNamespace(
        file_id=file_id, file_unique_id=f"{file_id}-unique",
        mime_type=mime_type, width=width, height=height, file_size=1000,
    )


def message(message_id, text="обычный текст", *, reply=None, caption=None,
            content_type="text", photo=None, document=None, animation=None,
            sticker=None, user_id=7):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-100, type="supergroup"),
        message_id=message_id, text=text, caption=caption,
        content_type=content_type, date=1_776_000_000 + message_id,
        from_user=user(user_id), reply_to_message=reply, photo=photo,
        document=document, animation=animation, sticker=sticker,
    )


def chair_message(message_id=900, *, photo=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-100), message_id=message_id,
        text=None if photo else "ответ", caption="картинка" if photo else None,
        content_type="photo" if photo else "text", date=1_776_000_000,
        from_user=user(99, "chair", True), reply_to_message=None,
        photo=photo, document=None, animation=None, sticker=None,
    )


class RecordingProvider:
    available = True
    provider_key = "r1-provider"
    _usage_recorder = None

    def __init__(self):
        self.generate_requests = []
        self.summarize_requests = []
        self._lock = threading.Lock()

    def generate(self, request):
        with self._lock:
            self.generate_requests.append(request)
        return "конкретный ответ по событию без второго производителя"

    def summarize(self, request):
        with self._lock:
            self.summarize_requests.append(request)
        return None


class R1NormalizedRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, provider=None, **overrides):
        values = dict(
            data_dir=Path(self.temp.name), openai_chat_id=-100,
            min_training_messages=1, summary_message_interval=1,
            summary_time_interval=0, addressed_cooldown=0,
            generated_cooldown=0, max_generated_per_hour=100,
            manual_meme_cooldown=0, chat_image_background_chance=1.0,
        )
        values.update(overrides)
        provider = provider or RecordingProvider()
        service = LearningService(
            LearningSettings(**values), llm_provider=provider,
            rng=SimpleNamespace(random=lambda: 0.0, choice=lambda values: values[0]),
        )
        return service, provider

    def test_normalized_event_and_runtime_context_share_one_identity(self):
        service, _ = self.service()
        event = normalize_telegram_event(message(1))
        with service.telegram_user_event(event) as context:
            self.assertEqual(event.event_id, context.event_id)
            self.assertEqual(event.event_id, current_event().event_id)

        call = SimpleNamespace(
            id="callback-1", data="chair:status", from_user=user(),
            message=SimpleNamespace(
                chat=SimpleNamespace(id=-100), message_id=50
            ),
        )
        callback = normalize_callback_event(call)
        with service.telegram_user_event(callback) as callback_context:
            self.assertEqual(callback.event_id, callback_context.event_id)
            self.assertEqual(callback.event_id, current_event().event_id)

    def test_summary_due_plus_normalized_foreground_call_uses_one_permit(self):
        service, provider = self.service()
        event = normalize_telegram_event(message(2))
        with service.telegram_user_event(event) as context:
            inserted, reason = service.ingest_event(event)
            self.assertTrue(service.generate_llm(-100, "ответь по релизу", "reply"))
        self.assertTrue(inserted)
        self.assertIsNone(reason)
        self.assertEqual(context.permit.call_count, 1)
        self.assertEqual(len(provider.generate_requests), 1)
        self.assertEqual(provider.summarize_requests, [])
        self.assertEqual(
            provider.generate_requests[0].metadata["event_id"], event.event_id
        )
        summary = service.repository(-100).summary_state()
        self.assertEqual(summary["last_message_row_id"], 0)
        self.assertIsNotNone(summary["pending_since"])

    def test_pending_hard_soft_direct_and_reply_facts_keep_semantics(self):
        service, _ = self.service(summary_message_interval=100)
        repository = service.repository(-100)
        cases = (
            ("hard", "choices", "айфон и пиксель", None, True),
            ("soft", "measurements", "181/68", None, True),
            ("soft", "measurements", "серёга опять уснул", None, False),
            ("hard", "choices", "новая тема", chair_message(501), True),
        )
        for index, (mode, expected_type, text, reply, expected) in enumerate(cases, 10):
            with self.subTest(mode=mode, text=text, reply=bool(reply)):
                repository.clear_pending_conversation(7)
                repository.save_pending_conversation(
                    7, 1, "что выбрать", "уточни данные", "substantive",
                    expected_type=expected_type, pending_mode=mode,
                    bot_message_id=501 if reply else None,
                )
                event = normalize_telegram_event(message(index, text, reply=reply))
                self.assertEqual(
                    service.is_pending_continuation(event, bot_id=99), expected
                )

    def test_reply_meme_uses_explicit_normalized_image_target(self):
        service, _ = self.service(summary_message_interval=100)
        service.repository(-100).add_message(
            1, 7, "user", "реальная локальная подпись для мема"
        )
        service.repository(-100).record_generated("cooldown", "manual_meme")
        reply = chair_message(700, photo=[media("reply-photo")])
        reply.from_user = user(8, "other", False)
        event = normalize_telegram_event(message(701, "с м стул", reply=reply))
        decision = service.maybe_command_meme(event)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.background_file_id, "reply-photo")
        self.assertEqual(decision.background_message_id, 700)
        self.assertTrue(decision.background_explicit)

        current = normalize_telegram_event(message(
            702, None, caption="с м стул", content_type="photo",
            photo=[media("current-photo")],
        ))
        current_decision = service.maybe_command_meme(current)
        self.assertIsNotNone(current_decision)
        self.assertEqual(current_decision.background_file_id, "current-photo")
        self.assertEqual(current_decision.background_message_id, 702)
        self.assertTrue(current_decision.background_explicit)

    def test_concurrent_normalized_events_keep_independent_permits(self):
        barrier = threading.Barrier(2)

        class ConcurrentProvider(RecordingProvider):
            def generate(self, request):
                with self._lock:
                    self.generate_requests.append(request)
                barrier.wait(timeout=5)
                return "конкретный независимый ответ по текущему событию"

        service, provider = self.service(ConcurrentProvider())
        runtime_concurrency.reset_peaks_for_test()
        results = {}

        def worker(message_id):
            event = normalize_telegram_event(message(message_id))
            with service.telegram_user_event(event) as context:
                first = service.generate_llm(-100, "первый", "reply")
                second = service.generate_llm(-100, "второй", "reply")
            results[message_id] = (
                event.event_id, context.permit.call_count, bool(first), second
            )

        threads = [threading.Thread(target=worker, args=(value,)) for value in (80, 81)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[80][0], results[81][0])
        self.assertEqual(results[80][1:], (1, True, None))
        self.assertEqual(results[81][1:], (1, True, None))
        self.assertEqual(
            {request.metadata["event_id"] for request in provider.generate_requests},
            {results[80][0], results[81][0]},
        )
        self.assertGreaterEqual(
            service.concurrency_diagnostics()["peak_active_llm_calls"], 2
        )


if __name__ == "__main__":
    unittest.main()
