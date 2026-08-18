import sqlite3
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    ConcurrencyController,
    LearningService,
    LearningSettings,
    SummaryJob,
)


EMPTY_SUMMARY = {
    "main_topics": [],
    "current_mood": "",
    "active_conflicts": [],
    "inside_jokes": [],
    "frequently_mentioned_people": [],
    "notable_events": [],
    "repeated_phrases": [],
    "callback_jokes": [],
    "memory_candidates": [],
}


class SummaryProvider:
    available = True
    provider_key = "test"

    def __init__(self, responses=None, callback=None):
        self.responses = list(responses or [])
        self.callback = callback
        self.summarize_requests = []
        self.generate_requests = []

    def summarize(self, request):
        self.summarize_requests.append(request)
        if self.callback:
            return self.callback(request)
        return self.responses.pop(0) if self.responses else dict(EMPTY_SUMMARY)

    def generate(self, request):
        self.generate_requests.append(request)
        return "готовый ответ Киберстула"


def telegram_message(message_id, text="обычная реплика", chat_id=-1):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        date=datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc),
        from_user=SimpleNamespace(
            id=7, username="tester", first_name="Test", is_bot=False
        ),
        text=text,
        caption=None,
        content_type="text",
        reply_to_message=None,
        photo=None,
        document=None,
        animation=None,
        sticker=None,
    )


class R5MemoryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.now = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def service(self, provider=None, controller=None, **overrides):
        values = {
            "data_dir": self.data_dir,
            "openai_chat_id": -1,
            "summary_message_interval": 1,
            "summary_time_interval": 1200,
            "summary_batch_messages": 50,
            "summary_batch_chars": 20000,
            "summary_claim_lease_seconds": 30,
            "summary_failure_backoff_base_seconds": 30,
            "summary_failure_backoff_cap_seconds": 300,
            "max_messages_per_chat": 50,
            "max_unsummarized_messages": 500,
            "timezone_name": "Europe/Moscow",
        }
        values.update(overrides)
        controller = controller or ConcurrencyController(
            llm_max_concurrency=2,
            media_max_concurrency=1,
            llm_timeout_seconds=0.1,
            media_timeout_seconds=0.1,
        )
        service = LearningService(
            LearningSettings(**values),
            llm_provider=provider or SummaryProvider(),
            concurrency_controller=controller,
        )
        service.memory._clock = lambda: self.now
        return service

    def add(self, service, message_id, text=None, created_at=None):
        return service.repository(-1).add_message(
            message_id,
            7,
            "tester",
            text or f"реплика номер {message_id}",
            created_at or self.now,
        )

    def test_foreground_ingest_never_calls_summary_provider(self):
        provider = SummaryProvider()
        service = self.service(provider)
        samples = (
            "обычная реплика",
            "стул как сварить рис?",
            "стул",
            "с м стул",
            "продолжение ответа",
        )
        for index, text in enumerate(samples, 1):
            incoming = telegram_message(index, text)
            with service.telegram_user_event(incoming) as event:
                service.ingest(incoming)
            self.assertEqual(event.permit.call_count, 0)
        self.assertEqual(provider.summarize_requests, [])
        self.assertIsNotNone(service.repository(-1).summary_state()["pending_since"])

    def test_summary_job_is_frozen_bounded_and_has_memory_identity(self):
        service = self.service(summary_batch_messages=2)
        for index in range(1, 4):
            self.add(service, index)
        job, status = service.memory.prepare_summary_job(
            service.repository(-1), -1, self.now
        )
        self.assertEqual(status, "claimed")
        self.assertIsInstance(job, SummaryJob)
        self.assertTrue(job.event_id.startswith("mem_"))
        self.assertEqual([item.row_id for item in job.messages], [1, 2])
        self.assertEqual(job.end_message_row_id, 2)
        with self.assertRaises(FrozenInstanceError):
            job.end_message_row_id = 3

    def test_summary_batch_character_limit_is_bounded(self):
        service = self.service(summary_batch_messages=50, summary_batch_chars=1000)
        for index in range(1, 6):
            self.add(service, index, "я" * 800)
        job, status = service.memory.prepare_summary_job(
            service.repository(-1), -1, self.now
        )
        self.assertEqual(status, "claimed")
        self.assertEqual(len(job.messages), 2)

    def test_success_uses_one_memory_permit_and_advances_exact_range(self):
        provider = SummaryProvider()
        service = self.service(provider, summary_batch_messages=2)
        for index in range(1, 4):
            self.add(service, index)
        result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.status, "committed")
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.finalize.cursor_after, 2)
        self.assertEqual(service.repository(-1).summary_state()["last_message_row_id"], 2)
        self.assertEqual(len(provider.summarize_requests), 1)
        self.assertEqual(
            provider.summarize_requests[0].metadata["event_id"], result.event_id
        )

    def test_provider_failure_keeps_cursor_and_persists_backoff(self):
        provider = SummaryProvider([None])
        service = self.service(provider)
        self.add(service, 1)
        result = service.run_memory_maintenance(-1, self.now)
        state = service.repository(-1).summary_state()
        self.assertEqual(result.status, "provider_failure")
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(state["last_message_row_id"], 0)
        self.assertIsNone(state["claim_token"])
        self.assertEqual(state["failure_count"], 1)
        self.assertIsNotNone(state["next_attempt_at"])
        self.assertEqual(service.repository(-1).count(), 1)

    def test_invalid_summary_is_failure_and_never_retried_in_same_event(self):
        provider = SummaryProvider([{"unexpected": "shape"}])
        service = self.service(provider)
        self.add(service, 1)
        result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.status, "provider_failure")
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(len(provider.summarize_requests), 1)
        self.assertEqual(service.repository(-1).summary_state()["last_message_row_id"], 0)

    def test_provider_timeout_releases_global_slot_and_keeps_backlog(self):
        def timeout(_request):
            raise TimeoutError("synthetic timeout")

        provider = SummaryProvider(callback=timeout)
        service = self.service(provider)
        self.add(service, 1)
        result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.status, "provider_failure")
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(service.concurrency.snapshot()["active_llm_calls"], 0)
        self.assertEqual(service.repository(-1).summary_state()["last_message_row_id"], 0)
        self.assertEqual(service.repository(-1).count(), 1)

    def test_backoff_and_due_state_survive_repository_restart(self):
        failed = self.service(SummaryProvider([None]))
        self.add(failed, 1)
        failed.run_memory_maintenance(-1, self.now)
        recovered_provider = SummaryProvider()
        restarted = self.service(recovered_provider)
        self.assertEqual(
            restarted.run_memory_maintenance(-1, self.now + timedelta(seconds=10)).status,
            "backoff",
        )
        self.assertEqual(recovered_provider.summarize_requests, [])
        after = self.now + timedelta(seconds=31)
        self.assertEqual(
            restarted.run_memory_maintenance(-1, after).status, "committed"
        )

    def test_expired_claim_is_recoverable_with_new_event_identity(self):
        service = self.service()
        self.add(service, 1)
        first, _ = service.memory.prepare_summary_job(
            service.repository(-1), -1, self.now
        )
        blocked, status = service.memory.prepare_summary_job(
            service.repository(-1), -1, self.now + timedelta(seconds=10)
        )
        self.assertIsNone(blocked)
        self.assertEqual(status, "claimed")
        second, status = service.memory.prepare_summary_job(
            service.repository(-1), -1, self.now + timedelta(seconds=31)
        )
        self.assertEqual(status, "claimed")
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertEqual(first.start_cursor, second.start_cursor)
        self.assertEqual(first.end_message_row_id, second.end_message_row_id)

    def test_duplicate_finalize_is_idempotent(self):
        summary = {**EMPTY_SUMMARY, "memory_candidates": ["Серёга опаздывает"]}
        provider = SummaryProvider([summary])
        service = self.service(provider)
        self.add(service, 1, "Серёга опаздывает")
        job, _ = service.memory.prepare_summary_job(service.repository(-1), -1, self.now)
        first = service.memory.execute_summary_job(
            service.repository(-1), job, provider, self.now
        )
        count_before = service.repository(-1).memory_candidates()[0][
            "observation_count"
        ]
        second = service.repository(-1).finalize_summary_job(
            job,
            EMPTY_SUMMARY,
            [("Серёга опаздывает", 1)],
            self.now,
        )
        self.assertTrue(first.committed)
        self.assertEqual(second.status, "already_finalized")
        self.assertEqual(
            service.repository(-1).memory_candidates()[0]["observation_count"],
            count_before,
        )

    def test_late_stale_job_cannot_overwrite_newer_claim(self):
        provider = SummaryProvider([EMPTY_SUMMARY, EMPTY_SUMMARY])
        service = self.service(provider)
        self.add(service, 1)
        old, _ = service.memory.prepare_summary_job(service.repository(-1), -1, self.now)
        newer, _ = service.memory.prepare_summary_job(
            service.repository(-1), -1, self.now + timedelta(seconds=31)
        )
        stale = service.memory.execute_summary_job(
            service.repository(-1), old, provider, self.now + timedelta(seconds=32)
        )
        committed = service.memory.execute_summary_job(
            service.repository(-1), newer, provider, self.now + timedelta(seconds=33)
        )
        self.assertEqual(stale.status, "stale")
        self.assertTrue(committed.committed)
        self.assertEqual(service.repository(-1).summary_state()["last_message_row_id"], 1)

    def test_messages_arriving_during_provider_are_not_added_to_fixed_job(self):
        service = None

        def during_call(_request):
            self.add(service, 3, "появилась во время summary")
            return dict(EMPTY_SUMMARY)

        provider = SummaryProvider(callback=during_call)
        service = self.service(provider, summary_batch_messages=2)
        self.add(service, 1)
        self.add(service, 2)
        result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.finalize.cursor_after, 2)
        self.assertEqual(result.finalize.remaining_messages, 1)
        self.assertEqual(service.repository(-1).summary_state()["last_message_row_id"], 2)
        self.assertEqual(service.repository(-1).summary_backlog_state(self.now)["backlog_messages"], 1)

    def test_day_rollover_never_mixes_daily_summary_fragment(self):
        provider = SummaryProvider([EMPTY_SUMMARY, EMPTY_SUMMARY])
        service = self.service(provider, timezone_name="UTC")
        day_one = datetime(2026, 8, 10, 23, 59, tzinfo=timezone.utc)
        day_two = datetime(2026, 8, 11, 0, 1, tzinfo=timezone.utc)
        self.add(service, 1, "хвост первого дня", day_one)
        self.add(service, 2, "начало второго дня", day_two)
        first = service.run_memory_maintenance(-1, day_two)
        second = service.run_memory_maintenance(-1, day_two + timedelta(seconds=1))
        self.assertEqual(first.finalize.cursor_after, 1)
        self.assertEqual(second.finalize.cursor_after, 2)
        self.assertIn("хвост первого дня", provider.summarize_requests[0].input)
        self.assertNotIn("начало второго дня", provider.summarize_requests[0].input)
        self.assertIsNotNone(service.repository(-1).summary_for_day("2026-08-10"))
        self.assertIsNotNone(service.repository(-1).summary_for_day("2026-08-11"))

    def test_hard_cap_rejects_explicitly_without_deleting_backlog(self):
        service = self.service(
            max_messages_per_chat=3, max_unsummarized_messages=5
        )
        for index in range(1, 6):
            self.assertTrue(self.add(service, index))
        inserted, reason = service.repository(-1).add_message(
            6, 7, "tester", "за пределами envelope", self.now,
            return_reason=True,
        )
        self.assertFalse(inserted)
        self.assertEqual(reason, "summary_backlog_hard_cap")
        self.assertEqual(service.repository(-1).count(), 5)
        self.assertEqual(
            [row["message_id"] for row in service.repository(-1).recent_messages()],
            [1, 2, 3, 4, 5],
        )

    def test_background_resource_busy_releases_claim_without_spending_permit(self):
        controller = ConcurrencyController(
            llm_max_concurrency=1,
            media_max_concurrency=1,
            llm_timeout_seconds=0.1,
            media_timeout_seconds=0.1,
        )
        provider = SummaryProvider()
        service = self.service(provider, controller)
        self.add(service, 1)
        with controller.llm_slot("foreground", -2):
            result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.status, "resource_deferred")
        self.assertEqual(result.llm_calls, 0)
        self.assertEqual(provider.summarize_requests, [])
        self.assertIsNone(service.repository(-1).summary_state()["claim_token"])
        self.assertEqual(service.repository(-1).summary_state()["last_message_row_id"], 0)

    def test_active_foreground_chat_defers_before_claim(self):
        controller = ConcurrencyController(2, 1, 0.1, 0.1)
        provider = SummaryProvider()
        service = self.service(provider, controller)
        self.add(service, 1)
        with controller.chat_event_slot(-1, "tg_foreground"):
            result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.status, "chat_busy")
        self.assertEqual(provider.summarize_requests, [])
        self.assertIsNone(service.repository(-1).summary_state()["claim_token"])
        self.assertEqual(controller.snapshot()["chat_gate_registry_size"], 0)

    def test_user_chat_gate_is_free_during_summary_network_call(self):
        entered = threading.Event()
        release = threading.Event()

        def blocked(_request):
            entered.set()
            self.assertTrue(release.wait(2))
            return dict(EMPTY_SUMMARY)

        controller = ConcurrencyController(2, 1, 0.2, 0.2)
        service = self.service(SummaryProvider(callback=blocked), controller)
        self.add(service, 1)
        thread = threading.Thread(
            target=lambda: service.run_memory_maintenance(-1, self.now)
        )
        thread.start()
        self.assertTrue(entered.wait(2))
        with controller.chat_event_slot(-1, "tg_user") as admission:
            self.assertTrue(admission)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(controller.snapshot()["chat_gate_registry_size"], 0)

    def test_provider_call_holds_no_repository_transaction(self):
        service = None

        def inspect_database(_request):
            # A second immediate writer succeeds while provider network work is
            # in progress; the claim read transaction is already closed.
            with sqlite3.connect(service.repository(-1).path, timeout=0.1) as db:
                db.execute(
                    "INSERT INTO settings(key, value) VALUES('r5_probe', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value='1'"
                )
            return dict(EMPTY_SUMMARY)

        service = self.service(SummaryProvider(callback=inspect_database))
        self.add(service, 1)
        self.assertEqual(
            service.run_memory_maintenance(-1, self.now).status, "committed"
        )

    def test_preparation_db_failure_releases_chat_gate(self):
        service = self.service()
        self.add(service, 1)
        with patch.object(
            service.memory, "prepare_summary_job", side_effect=sqlite3.OperationalError
        ):
            result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.status, "preparation_failure")
        self.assertEqual(service.concurrency.snapshot()["chat_gate_registry_size"], 0)
        self.assertEqual(service.concurrency.snapshot()["active_llm_calls"], 0)

    def test_finalize_db_failure_does_not_retry_and_lease_recovers(self):
        provider = SummaryProvider([EMPTY_SUMMARY, EMPTY_SUMMARY])
        service = self.service(provider)
        self.add(service, 1)
        original = service.repository(-1).finalize_summary_job
        with patch.object(
            service.repository(-1),
            "finalize_summary_job",
            side_effect=sqlite3.OperationalError,
        ):
            result = service.run_memory_maintenance(-1, self.now)
        self.assertEqual(result.status, "storage_failure")
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(len(provider.summarize_requests), 1)
        self.assertEqual(service.concurrency.snapshot()["active_llm_calls"], 0)
        self.assertIsNotNone(service.repository(-1).summary_state()["claim_token"])
        with patch.object(
            service.repository(-1), "finalize_summary_job", wraps=original
        ):
            recovered = service.run_memory_maintenance(
                -1, self.now + timedelta(seconds=31)
            )
        self.assertEqual(recovered.status, "committed")
        self.assertEqual(len(provider.summarize_requests), 2)

    def test_memory_diagnostic_proves_no_foreground_summary_calls(self):
        service = self.service()
        incoming = telegram_message(1)
        with service.telegram_user_event(incoming):
            service.ingest(incoming)
        service.run_memory_maintenance(-1, self.now)
        report = service.memory_lifecycle_diagnostics(-1, self.now)
        self.assertEqual(report["foreground_summary_calls"], 0)
        self.assertEqual(report["max_summary_calls_per_memory_event"], 1)
        self.assertEqual(report["summary_success"], 1)
        self.assertIn("MEMORY LIFECYCLE", service.format_memory_lifecycle_diagnostics(-1, self.now))

    def test_existing_scheduler_invokes_at_most_one_maintenance_callback_per_tick(self):
        import scheduler as scheduler_module

        calls = []
        scheduler_module._next_random_at = self.now + timedelta(hours=1)
        with (
            patch.object(scheduler_module, "get_now", return_value=self.now),
            patch.object(scheduler_module, "daily_quote_minutes", return_value=[]),
            patch.object(scheduler_module, "is_workday", return_value=False),
            patch.object(
                scheduler_module.time, "sleep", side_effect=KeyboardInterrupt
            ),
        ):
            with self.assertRaises(KeyboardInterrupt):
                scheduler_module.scheduler(
                    SimpleNamespace(),
                    -1,
                    "UTC",
                    9,
                    0,
                    17,
                    30,
                    memory_maintenance_callback=lambda chat_id, current: calls.append(
                        (chat_id, current)
                    ),
                )
        self.assertEqual(calls, [(-1, self.now)])


if __name__ == "__main__":
    unittest.main()
