import dataclasses
import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    LearningService,
    LearningSettings,
    format_context_snapshot_read_diagnostic,
)
from learning.normalized_event import normalize_telegram_event
from learning.repository import repository_query_profile


NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


class FixedRandom:
    def __init__(self, value=.8):
        self.value = value

    def random(self):
        return self.value

    def choice(self, values):
        return list(values)[0]


class RecordingProvider:
    available = True
    provider_key = "test"

    def __init__(self, result="сначала проверь логи контейнера и DNS, затем перезапусти только упавший сервис"):
        self.result = result
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.result

    def summarize(self, request):
        raise AssertionError("ContextSnapshotBuilder must never call an LLM")


def telegram_event(message_id=100, text="стул почему docker теряет dns?"):
    return normalize_telegram_event(SimpleNamespace(
        chat=SimpleNamespace(id=-1),
        message_id=message_id,
        text=text,
        caption=None,
        content_type="text",
        date=NOW.timestamp() + message_id,
        from_user=SimpleNamespace(
            id=7, username="tester", first_name="Tester", is_bot=False,
        ),
        reply_to_message=None,
    ))


class ContextSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, name="service", provider=None, **overrides):
        values = dict(
            data_dir=Path(self.temp.name) / name,
            openai_chat_id=-1,
            min_training_messages=1,
            direct_social_markov_share=0.0,
            timezone_name="Europe/Moscow",
        )
        values.update(overrides)
        provider = provider or RecordingProvider()
        service = LearningService(
            LearningSettings(**values), llm_provider=provider,
            rng=FixedRandom(),
        )
        service.memory._clock = lambda: NOW
        return service, provider

    def seed(self, service, rows=12):
        repository = service.repository(-1)
        for index in range(rows):
            repository.add_message(
                index + 1, index % 3 + 1, f"user{index % 3}",
                f"обсуждаем docker dns и релиз номер {index}",
                NOW - timedelta(minutes=rows - index),
            )
        repository.record_generated(
            "предыдущий ответ про контейнер", "reply", NOW - timedelta(minutes=2)
        )
        repository.save_daily_summary(service.memory.logical_day(NOW), {
            "main_topics": ["docker и релиз"],
            "current_mood": "деловой",
            "callback_jokes": ["контейнер опять убежал"],
        })
        repository.remember_stable([
            "команда выкатывает релиз по пятницам",
            "рэп это хуйня, больше писать не буду",
        ])
        repository.set_setting("media_enabled", "0")
        return repository

    def build(self, service, event):
        with service.telegram_user_event(event):
            return service.context_snapshot(event, current=NOW)

    def test_snapshot_is_frozen_event_local_and_builder_is_zero_llm(self):
        provider = RecordingProvider()
        service, _ = self.service(provider=provider)
        self.seed(service)
        event = telegram_event()
        with service.telegram_user_event(event) as context:
            snapshot = service.context_snapshot(event, current=NOW)
            self.assertEqual(snapshot.event_id, event.event_id)
            self.assertEqual(snapshot.event_id, context.event_id)
            self.assertIs(snapshot, service.context_snapshot(event, current=NOW))
            self.assertEqual(snapshot.metrics.db_connections, 1)
            self.assertNotIn("media", snapshot.sections_loaded)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                snapshot.chat_id = 9
            with self.assertRaises(TypeError):
                snapshot.current_summary["main_topics"] = []
            self.assertIsInstance(snapshot.current_summary["main_topics"], tuple)
        self.assertEqual(provider.requests, [])

    def test_snapshot_matches_compatibility_reads_and_chat_state(self):
        service, _ = self.service()
        repository = self.seed(service)
        event = telegram_event()
        cutoff = (NOW - timedelta(minutes=service.settings.short_memory_minutes)).isoformat()
        expected_dialogue = repository.short_term_dialogue(cutoff, 50)
        expected_summary = service.memory.relevant_memory(
            repository, event.effective_text
        )
        expected_generated = repository.recent_generated(50)
        old_state = service.chat_state_analyzer.analyze(
            repository, incoming_message=event, now=NOW
        )
        with service.telegram_user_event(event):
            snapshot = service.context_snapshot(event, current=NOW)
            new_state = service.chat_state_analyzer.analyze(
                repository, incoming_message=event, now=NOW, snapshot=snapshot
            )
            actual_memory = service.memory.relevant_memory_from_values(
                snapshot.current_summary, snapshot.stable_memories,
                event.effective_text,
            )
        self.assertEqual([dict(row) for row in snapshot.recent_dialogue], expected_dialogue)
        self.assertEqual([dict(row) for row in snapshot.recent_generated], expected_generated)
        self.assertEqual(actual_memory, expected_summary)
        self.assertEqual(new_state, old_state)

    def test_direct_path_equivalence_and_read_reduction(self):
        old_provider, new_provider = RecordingProvider(), RecordingProvider()
        old, _ = self.service("old", old_provider)
        new, _ = self.service("new", new_provider)
        self.seed(old, 30)
        self.seed(new, 30)
        event = telegram_event(800)

        with repository_query_profile() as before:
            old_result = old.maybe_direct_reply(event, explicit_address=True)
        with new.telegram_user_event(event):
            with repository_query_profile() as after:
                snapshot = new.context_snapshot(event, current=NOW)
                plan = new.prepare_direct_reply(event, explicit_address=True)

        self.assertEqual(plan.payload.text, old_result)
        self.assertEqual(len(old_provider.requests), 1)
        self.assertEqual(len(new_provider.requests), 1)
        old_request, new_request = old_provider.requests[0], new_provider.requests[0]
        self.assertEqual(old_request.metadata["response_purpose"], new_request.metadata["response_purpose"])
        self.assertEqual(old_request.metadata.get("model"), new_request.metadata.get("model"))
        self.assertEqual(
            old_request.metadata.get("reasoning_effort"),
            new_request.metadata.get("reasoning_effort"),
        )
        self.assertIn("docker", new_request.input.casefold())
        self.assertLessEqual(len(new_request.input), int(len(old_request.input) * 1.05))
        self.assertLessEqual(after["connections"], int(before["connections"] * .70))
        self.assertEqual(after["summary_reads"], 1)
        self.assertEqual(after["dialogue_reads"], 1)
        self.assertEqual(after["stable_memory_reads"], 1)
        self.assertEqual(after["generated_history_reads"], 1)
        self.assertEqual(snapshot.metrics.db_connections, 1)
        diagnostic = format_context_snapshot_read_diagnostic([before], [after])
        self.assertIn("avg_db_connections_before: 22.00", diagnostic)
        self.assertIn("avg_db_connections_after: 7.00", diagnostic)
        self.assertIn("summary_reads_per_event: before=4.00 after=1.00", diagnostic)

    def test_media_section_is_conditional_and_loaded_once(self):
        service, _ = self.service()
        repository = self.seed(service)
        repository.set_setting("media_enabled", "1")
        event = telegram_event(900, "стул ахаха контейнер опять убежал")
        with service.telegram_user_event(event):
            with repository_query_profile() as profile:
                snapshot = service.context_snapshot(event, current=NOW)
                self.assertIsNone(snapshot.media)
                enriched = service.media_context_snapshot(snapshot)
                again = service.media_context_snapshot(enriched)
        self.assertIs(enriched, again)
        self.assertIn("media", enriched.sections_loaded)
        self.assertEqual(enriched.metrics.db_connections, 2)
        self.assertEqual(profile["connections"], 2)

    def test_empty_snapshot_and_connection_cleanup_after_error(self):
        service, _ = self.service()
        event = telegram_event(1000)
        snapshot = self.build(service, event)
        self.assertEqual(snapshot.recent_dialogue, ())
        self.assertEqual(snapshot.stable_memories, ())
        self.assertEqual(snapshot.recent_generated, ())
        # A successful query after the scoped transaction proves it closed and
        # did not leave an open transaction/lock behind.
        self.assertEqual(service.repository(-1).count(), 0)

    def test_repository_failure_does_not_publish_partial_snapshot_or_call_llm(self):
        service, provider = self.service()
        event = telegram_event(1100)
        repository = service.repository(-1)
        with service.telegram_user_event(event):
            with patch.object(
                repository, "load_context_snapshot_inputs",
                side_effect=OSError("synthetic read failure"),
            ):
                with self.assertRaises(OSError):
                    service.context_snapshot(event, current=NOW)
            self.assertIsNone(service.current_context_snapshot())
        self.assertEqual(provider.requests, [])
        self.assertEqual(service.context_snapshot_diagnostics()["events"], 0)

    def test_concurrent_events_do_not_share_snapshots(self):
        service, _ = self.service()
        self.seed(service)

        def worker(index):
            event = telegram_event(2000 + index, f"сообщение события {index}")
            with service.telegram_user_event(event):
                snapshot = service.context_snapshot(event, current=NOW)
                return event.event_id, snapshot, service.current_context_snapshot()

        with ThreadPoolExecutor(max_workers=2) as pool:
            left, right = list(pool.map(worker, (1, 2)))
        self.assertNotEqual(left[0], right[0])
        self.assertIsNot(left[1], right[1])
        self.assertEqual(left[0], left[2].event_id)
        self.assertEqual(right[0], right[2].event_id)

    def test_memory_relevance_twenty_scenarios(self):
        service, _ = self.service()
        summary = {"main_topics": ["рэп и альбом"]}
        stable = ("рэп это хуйня, больше писать не буду",)
        relevant = [f"как прославиться в рэпе номер {index}?" for index in range(10)]
        irrelevant = [f"как сварить рис номер {index}?" for index in range(10)]
        for text in relevant:
            with self.subTest(kind="relevant", text=text):
                result = service.memory.relevant_memory_from_values(
                    summary, stable, text
                )
                self.assertTrue(result["stable_chat_memory"])
        for text in irrelevant:
            with self.subTest(kind="irrelevant", text=text):
                result = service.memory.relevant_memory_from_values(
                    summary, stable, text
                )
                self.assertEqual(result["stable_chat_memory"], [])
        self.assertTrue(service.persona.select_callbacks(
            summary, stable, "как прославиться в рэпе?", None
        ))
        self.assertEqual(
            service.persona.select_callbacks(
                summary, stable, "как сварить рис?", None
            ),
            (),
        )

    def test_long_context_remains_bounded_and_prompt_size_is_neutral(self):
        old_provider, new_provider = RecordingProvider(), RecordingProvider()
        old, _ = self.service("long_old", old_provider, max_messages_per_chat=50)
        new, _ = self.service("long_new", new_provider, max_messages_per_chat=50)
        for service in (old, new):
            repository = service.repository(-1)
            for index in range(50):
                repository.add_message(
                    index + 1, index % 4 + 1, f"u{index % 4}",
                    f"длинный контекст docker релиз реплика {index}",
                    NOW - timedelta(seconds=50 - index),
                )
                repository.record_generated(
                    f"исторический ответ стула {index}", "reply",
                    NOW - timedelta(seconds=100 - index),
                )
            repository.remember_stable(
                [
                    hashlib.sha256(f"stable-{index}".encode()).hexdigest()[:24]
                    for index in range(40)
                ]
            )
            repository.save_daily_summary(service.memory.logical_day(NOW), {
                "main_topics": ["docker релиз"],
                "current_mood": "рабочий",
            })
            repository.set_setting("media_enabled", "0")
        event = telegram_event(9800, "стул почему docker релиз опять падает?")
        old.maybe_direct_reply(event, explicit_address=True)
        with new.telegram_user_event(event):
            snapshot = new.context_snapshot(event, current=NOW)
            new.prepare_direct_reply(event, explicit_address=True)
        self.assertLessEqual(len(snapshot.recent_dialogue), 50)
        self.assertEqual(len(snapshot.recent_generated), 50)
        self.assertEqual(len(snapshot.stable_memories), 20)
        before_chars = len(old_provider.requests[0].input)
        after_chars = len(new_provider.requests[0].input)
        self.assertLessEqual(after_chars, int(before_chars * 1.05))
        self.assertEqual(after_chars, before_chars)

    def test_one_hundred_event_snapshot_smoke(self):
        service, provider = self.service()
        self.seed(service, 50)
        snapshots = []
        categories = (
            ["direct_useful"] * 20 + ["direct_troll"] * 15
            + ["social_local"] * 10 + ["pending"] * 10
            + ["ordinary_ai"] * 10 + ["ordinary_markov"] * 10
            + ["media"] * 10 + ["manual_meme"] * 5
            + ["autonomous"] * 5 + ["no_response"] * 5
        )
        for index, category in enumerate(categories):
            event = telegram_event(3000 + index, f"{category} событие {index}")
            with self.subTest(index=index, category=category):
                with service.telegram_user_event(event) as context:
                    snapshot = service.context_snapshot(event, current=NOW)
                    state = service.chat_state_analyzer.analyze(
                        service.repository(-1), incoming_message=event,
                        now=NOW, snapshot=snapshot,
                    )
                    service.memory.short_term_context_from_snapshot(
                        snapshot, event.effective_text,
                        dominant_topic=state.dominant_topic,
                        conversation_type=state.conversation_type,
                    )
                    self.assertEqual(context.permit.call_count, 0)
                    self.assertEqual(snapshot.event_id, event.event_id)
                    snapshots.append(snapshot)
        self.assertEqual(len({item.event_id for item in snapshots}), 100)
        self.assertEqual(provider.requests, [])
        report = service.context_snapshot_diagnostics()
        self.assertEqual(report["events"], 100)
        self.assertEqual(report["peak_db_connections"], 1)


if __name__ == "__main__":
    unittest.main()
