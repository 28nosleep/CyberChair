import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from learning import ConcurrencyController, LearningService, LearningSettings

from test_r5_memory_lifecycle import EMPTY_SUMMARY, SummaryProvider


class R5MemoryDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def service(self, provider, **overrides):
        values = {
            "data_dir": self.data_dir,
            "openai_chat_id": -1,
            "timezone_name": "UTC",
            "summary_message_interval": 1,
            "summary_batch_messages": 50,
            "summary_batch_chars": 20000,
            "summary_failure_backoff_base_seconds": 30,
            "summary_failure_backoff_cap_seconds": 300,
            "max_messages_per_chat": 50,
            "max_unsummarized_messages": 500,
            "max_memory_candidates": 20,
            "daily_summary_retention_days": 90,
        }
        values.update(overrides)
        service = LearningService(
            LearningSettings(**values),
            llm_provider=provider,
            concurrency_controller=ConcurrencyController(2, 1, 0.1, 0.1),
        )
        service.memory._clock = lambda: self.now
        return service

    @staticmethod
    def add(repository, message_id, current, text=None):
        return repository.add_message(
            message_id, 1, "tester", text or f"message {message_id}", current
        )

    def test_120_message_outage_restart_and_recovery_loses_nothing(self):
        failing = self.service(SummaryProvider([None]))
        repository = failing.repository(-1)
        for index in range(1, 121):
            self.assertTrue(self.add(repository, index, self.now))
        self.assertEqual(failing.run_memory_maintenance(-1, self.now).status, "provider_failure")
        self.assertEqual(repository.count(), 120)
        self.assertEqual(repository.summary_state()["last_message_row_id"], 0)

        provider = SummaryProvider()
        recovered = self.service(provider)
        current = self.now + timedelta(seconds=31)
        cursors = []
        while recovered.repository(-1).summary_backlog_state(current)["backlog_messages"]:
            result = recovered.run_memory_maintenance(-1, current)
            self.assertEqual(result.status, "committed")
            cursors.append(result.finalize.cursor_after)
            current += timedelta(seconds=1)
        self.assertEqual(cursors, [50, 100, 120])
        self.assertEqual(recovered.repository(-1).count(), 50)
        self.assertEqual(len(provider.summarize_requests), 3)
        covered = "\n".join(request.input for request in provider.summarize_requests)
        for index in range(1, 121):
            self.assertIn(f"message {index}", covered)

    def test_500_message_outage_envelope_recovers_in_bounded_batches(self):
        provider = SummaryProvider()
        service = self.service(provider)
        # The production deployment is configured for one primary chat, while
        # the runner itself remains correctly isolated across repositories.
        service.memory_maintenance.provider_allowed = lambda _chat_id: True
        repository = service.repository(-1)
        for index in range(1, 501):
            self.assertTrue(self.add(repository, index, self.now))
        self.assertEqual(repository.count(), 500)
        cursors = []
        current = self.now
        for _ in range(10):
            result = service.run_memory_maintenance(-1, current)
            self.assertEqual(result.status, "committed")
            self.assertEqual(result.llm_calls, 1)
            self.assertLessEqual(len(provider.summarize_requests[-1].input), 22000)
            cursors.append(result.finalize.cursor_after)
            current += timedelta(seconds=1)
        self.assertEqual(cursors, list(range(50, 501, 50)))
        self.assertEqual(repository.summary_backlog_state(current)["backlog_messages"], 0)
        self.assertEqual(repository.count(), 50)
        self.assertEqual(len(provider.summarize_requests), 10)

        for chat_id, total in ((-2, 40), (-3, 60)):
            other = service.repository(chat_id)
            for index in range(1, total + 1):
                self.assertTrue(other.add_message(
                    index, 1, "tester", f"chat {chat_id} message {index}", self.now
                ))
            while other.summary_backlog_state(current)["backlog_messages"]:
                result = service.run_memory_maintenance(chat_id, current)
                self.assertEqual(result.status, "committed")
                self.assertEqual(result.llm_calls, 1)
                current += timedelta(seconds=1)
            self.assertLessEqual(other.count(), 50)
        self.assertEqual(len(provider.summarize_requests), 13)
        self.assertEqual(service.concurrency.snapshot()["active_llm_calls"], 0)
        self.assertLessEqual(
            service.concurrency.snapshot()["peak_active_llm_calls"], 2
        )

    def test_candidate_lifecycle_bounds_active_and_preserves_stable_cap(self):
        responses = []
        for index in range(30):
            responses.append({
                **EMPTY_SUMMARY,
                "memory_candidates": [f"уникальный факт номер {index}"],
            })
        provider = SummaryProvider(responses)
        service = self.service(provider, max_long_memories=40)
        repository = service.repository(-1)
        current = self.now
        for index in range(30):
            self.add(repository, index + 1, current, f"уникальный факт номер {index}")
            self.assertEqual(
                service.run_memory_maintenance(-1, current).status, "committed"
            )
            current += timedelta(seconds=1)
        candidates = repository.memory_candidates()
        self.assertLessEqual(len(candidates), 20)
        self.assertLessEqual(len(repository.stable_memories(100)), 40)

    def test_promoted_candidate_audit_row_expires_but_stable_fact_remains(self):
        fact = "Серёга постоянно опаздывает"
        responses = [
            {**EMPTY_SUMMARY, "memory_candidates": [fact]},
            {**EMPTY_SUMMARY, "memory_candidates": [fact]},
            dict(EMPTY_SUMMARY),
        ]
        service = self.service(
            SummaryProvider(responses),
            memory_candidate_promoted_retention_days=7,
        )
        repository = service.repository(-1)
        for index in (1, 2):
            self.add(repository, index, self.now + timedelta(seconds=index), fact)
            service.run_memory_maintenance(-1, self.now + timedelta(seconds=index))
        self.assertEqual(repository.stable_memories(), [fact])
        self.assertEqual(len(repository.memory_candidates()), 1)
        later = self.now + timedelta(days=8)
        self.add(repository, 3, later, "другая тема")
        service.run_memory_maintenance(-1, later)
        self.assertEqual(repository.memory_candidates(), [])
        self.assertEqual(repository.stable_memories(), [fact])

    def test_unconfirmed_candidate_eventually_expires(self):
        service = self.service(
            SummaryProvider([
                {**EMPTY_SUMMARY, "memory_candidates": ["случайный неподтверждённый факт"]},
                dict(EMPTY_SUMMARY),
            ]),
            memory_candidate_stale_days=30,
        )
        repository = service.repository(-1)
        self.add(repository, 1, self.now, "случайный неподтверждённый факт")
        service.run_memory_maintenance(-1, self.now)
        self.assertEqual(len(repository.memory_candidates()), 1)
        later = self.now + timedelta(days=31)
        self.add(repository, 2, later, "совсем другая тема")
        service.run_memory_maintenance(-1, later)
        self.assertEqual(repository.memory_candidates(), [])
        self.assertEqual(repository.stable_memories(), [])

    def test_ninety_day_daily_summary_retention_is_bounded(self):
        provider = SummaryProvider()
        service = self.service(provider)
        repository = service.repository(-1)
        for offset in range(100):
            current = self.now + timedelta(days=offset)
            self.add(repository, offset + 1, current)
            result = service.run_memory_maintenance(-1, current)
            self.assertEqual(result.status, "committed")
        self.assertLessEqual(len(repository.recent_summaries(1000)), 91)
        self.assertIsNotNone(repository.summary_for_day("2026-04-10"))

    def test_long_run_messages_failures_restarts_and_retention(self):
        provider = SummaryProvider()
        service = self.service(provider)
        repository = service.repository(-1)
        current = self.now
        message_id = 0
        for day in range(90):
            current = self.now + timedelta(days=day)
            for _ in range(12):
                message_id += 1
                self.assertTrue(self.add(repository, message_id, current))
            # One bounded event is enough for this daily 12-row fragment.
            self.assertEqual(
                service.run_memory_maintenance(-1, current).status, "committed"
            )
            if day in {20, 50}:
                service = self.service(provider)
                repository = service.repository(-1)
        state = repository.summary_state()
        self.assertEqual(state["last_message_row_id"], message_id)
        self.assertEqual(repository.summary_backlog_state(current)["backlog_messages"], 0)
        self.assertLessEqual(repository.count(), 50)
        self.assertLessEqual(len(repository.recent_summaries(1000)), 91)
        self.assertLessEqual(len(repository.stable_memories(100)), 40)
        self.assertEqual(len(provider.summarize_requests), 90)


if __name__ == "__main__":
    unittest.main()
