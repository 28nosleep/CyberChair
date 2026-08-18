import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bot as bot_module
from learning import LearningService, LearningSettings
from learning.response_plan import (
    DeliveryReceipt,
    GeneratedCommit,
    Producer,
)
from test_r2_response_plan import event
from test_r0_safety_baseline import RecordingProvider, message


class DeliveryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = LearningService(
            LearningSettings(data_dir=Path(self.temp.name))
        )

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, message_id=1):
        return self.service.prepare_text_response(
            event(message_id), "готовый ответ", "direct", producer=Producer.LLM,
            actions=(GeneratedCommit("готовый ответ", "direct_grok"),),
        )

    def test_success_returns_telegram_message_id_and_commits_once(self):
        plan = self.plan(10)
        with (
            patch.object(bot_module, "learning_service", self.service),
            patch.object(
                bot_module.bot, "send_message",
                return_value=SimpleNamespace(message_id=808),
            ) as send,
        ):
            receipt = bot_module.execute_response_plan(plan)
        self.assertTrue(receipt.success)
        self.assertEqual(receipt.telegram_message_id, 808)
        send.assert_called_once()
        self.assertEqual(len(self.service.repository(-1).recent_generated(10)), 1)

    def test_timeout_and_generic_transport_failure_abort_without_retry(self):
        for message_id, error, category in (
            (20, TimeoutError("secret timeout"), "telegram_timeout"),
            (21, RuntimeError("secret transport"), "unknown_transport"),
        ):
            with self.subTest(category=category):
                plan = self.plan(message_id)
                with (
                    patch.object(bot_module, "learning_service", self.service),
                    patch.object(bot_module.bot, "send_message", side_effect=error) as send,
                ):
                    receipt = bot_module.execute_response_plan(plan)
                self.assertFalse(receipt.success)
                self.assertEqual(receipt.error_category, category)
                send.assert_called_once()
        self.assertEqual(self.service.repository(-1).recent_generated(10), [])

    def test_post_delivery_db_failure_never_repeats_telegram_send(self):
        plan = self.plan(30)
        repository = self.service.repository(-1)
        with (
            patch.object(bot_module, "learning_service", self.service),
            patch.object(
                bot_module.bot, "send_message",
                return_value=SimpleNamespace(message_id=909),
            ) as send,
            patch.object(
                repository, "record_generated", side_effect=OSError("db unavailable")
            ),
        ):
            receipt = bot_module.execute_response_plan(plan)
        self.assertTrue(receipt.success)
        send.assert_called_once()
        report = repository.routing_report("2020-01-01T00:00:00+00:00")
        self.assertEqual(report.get("post_delivery_commit_failed"), 1)

    def routed_service(self, provider):
        service = LearningService(
            LearningSettings(
                data_dir=Path(self.temp.name), openai_chat_id=-1,
                min_training_messages=1, summary_message_interval=50,
                generated_cooldown=0, addressed_cooldown=0,
                max_generated_per_hour=100,
            ),
            llm_provider=provider,
            rng=SimpleNamespace(random=lambda: 0.9, choice=lambda values: values[0]),
        )
        service.set_media_enabled(-1, False)
        return service

    def test_direct_llm_send_failure_has_one_call_one_delivery_and_no_commit(self):
        provider = RecordingProvider(
            "сначала проверь логи контейнера и откати последний релиз"
        )
        service = self.routed_service(provider)
        incoming = message(60, "стул почему docker падает?")
        with (
            patch.object(bot_module, "learning_service", service),
            patch.object(bot_module, "remember_user"),
            patch.object(
                bot_module, "get_bot_identity",
                return_value={"id": 99, "username": "chair"},
            ),
            patch.object(bot_module.bot, "reply_to", side_effect=TimeoutError()) as send,
        ):
            bot_module.handle_message(incoming)
        self.assertEqual(len(provider.generate_requests), 1)
        send.assert_called_once()
        self.assertEqual(service.repository(-1).recent_generated(10), [])
        report = service.llm_event_invariant_diagnostics(-1, hours=24 * 365)
        self.assertEqual(report["events_with_2plus_llm"], 0)
        self.assertLessEqual(report["max_calls_per_user_event"], 1)

    def test_provider_failure_resolves_to_one_local_final_plan(self):
        provider = RecordingProvider(None)
        service = self.routed_service(provider)
        incoming = message(61, "стул почему docker падает?")
        normalized = service._normalized_event(incoming)
        with service.telegram_user_event(normalized) as context:
            plan = service.prepare_direct_reply(
                normalized, explicit_address=True
            )
        self.assertEqual(len(provider.generate_requests), 1)
        self.assertEqual(context.permit.call_count, 1)
        self.assertEqual(plan.producer, Producer.LOCAL)
        self.assertEqual(plan.delivery_type.value, "text")


if __name__ == "__main__":
    unittest.main()
