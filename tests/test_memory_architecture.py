import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from learning import LearningService, LearningSettings
from learning.repository import ChatRepository


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

    def __init__(self, summaries):
        self.summaries = list(summaries)
        self.requests = []

    def generate(self, request):
        return "короткий нормальный ответ стула"

    def summarize(self, request):
        self.requests.append(request)
        if not self.summaries:
            return None
        return self.summaries.pop(0)


class MemoryArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.now = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, **overrides):
        values = {
            "data_dir": self.data_dir,
            "openai_chat_id": -1,
            "summary_message_interval": 1,
            "timezone_name": "Europe/Moscow",
        }
        values.update(overrides)
        return LearningSettings(**values)

    def service(self, provider=None, **settings):
        provider = provider or SummaryProvider([EMPTY_SUMMARY])
        service = LearningService(self.settings(**settings), llm_provider=provider)
        service.memory._clock = lambda: self.now
        return service, provider

    def add(self, repository, message_id, text, created_at=None):
        repository.add_message(
            message_id,
            1,
            "tester",
            text,
            created_at or self.now,
        )

    def test_message_older_than_thirty_minutes_is_not_in_short_term_context(self):
        service, _ = self.service()
        repository = service.repository(-1)
        self.add(repository, 1, "старое сообщение вне окна", self.now - timedelta(minutes=31))
        self.add(repository, 2, "свежее сообщение внутри окна", self.now - timedelta(minutes=5))
        context = service._dialogue_context(-1)
        self.assertNotIn("старое сообщение вне окна", context)
        self.assertIn("свежее сообщение внутри окна", context)

    def test_generation_context_selects_at_most_twenty_from_allowed_window(self):
        service, _ = self.service(context_message_limit=40, max_messages_per_chat=50)
        repository = service.repository(-1)
        for index in range(25):
            self.add(
                repository,
                index + 1,
                f"допустимая релевантная реплика номер {index}",
                self.now - timedelta(minutes=25 - index),
            )
        context = service._dialogue_context(-1, "релевантная реплика")
        dialogue = context.partition("Последние релевантные реплики:\n")[2]
        self.assertEqual(len(dialogue.splitlines()), 20)

    def test_summary_receives_only_rows_after_successful_cursor(self):
        provider = SummaryProvider([EMPTY_SUMMARY, EMPTY_SUMMARY])
        service, _ = self.service(provider)
        repository = service.repository(-1)
        self.add(repository, 1, "первый новый фрагмент")
        self.assertTrue(service._maybe_refresh_memory(-1))
        self.add(repository, 2, "второй отдельный фрагмент")
        self.assertTrue(service._maybe_refresh_memory(-1))
        self.assertIn("первый новый фрагмент", provider.requests[0].input)
        self.assertIn("второй отдельный фрагмент", provider.requests[1].input)
        self.assertNotIn("первый новый фрагмент", provider.requests[1].input)

    def test_successful_summary_advances_cursor(self):
        service, _ = self.service()
        repository = service.repository(-1)
        self.add(repository, 1, "сообщение успешно обработано")
        row_id = repository.recent_messages()[0]["id"]
        self.assertTrue(service._maybe_refresh_memory(-1))
        self.assertEqual(repository.summary_state()["last_message_row_id"], row_id)

    def test_failed_summary_does_not_advance_cursor(self):
        service, _ = self.service(SummaryProvider([None]))
        repository = service.repository(-1)
        self.add(repository, 1, "сообщение осталось необработанным")
        self.assertFalse(service._maybe_refresh_memory(-1))
        state = repository.summary_state()
        self.assertEqual(state["last_message_row_id"], 0)
        self.assertIsNotNone(state["pending_since"])

    def test_previous_summary_is_used_for_next_increment(self):
        first = {**EMPTY_SUMMARY, "main_topics": ["первый релиз"]}
        second = {**EMPTY_SUMMARY, "main_topics": ["второй релиз"]}
        provider = SummaryProvider([first, second])
        service, _ = self.service(provider)
        repository = service.repository(-1)
        self.add(repository, 1, "обсудили первый релиз")
        service._maybe_refresh_memory(-1)
        self.add(repository, 2, "обсудили второй релиз")
        service._maybe_refresh_memory(-1)
        self.assertIn("Предыдущее резюме дня", provider.requests[1].input)
        self.assertIn("первый релиз", provider.requests[1].input)

    def test_summary_persists_active_conflicts_and_callback_jokes(self):
        summary = {
            **EMPTY_SUMMARY,
            "active_conflicts": ["спор о релизе"],
            "callback_jokes": ["шутка про пятничный деплой"],
        }
        service, _ = self.service(SummaryProvider([summary]))
        repository = service.repository(-1)
        self.add(repository, 1, "снова спорим о пятничном деплое")
        service._maybe_refresh_memory(-1)
        stored = repository.summary_for_day("2026-08-11")
        self.assertEqual(stored["active_conflicts"], ["спор о релизе"])
        self.assertEqual(stored["callback_jokes"], ["шутка про пятничный деплой"])

    def test_logical_day_uses_configured_timezone_instead_of_utc(self):
        self.now = datetime(2026, 8, 10, 21, 30, tzinfo=timezone.utc)
        service, _ = self.service()
        repository = service.repository(-1)
        self.add(repository, 1, "сообщение после полуночи по Москве")
        service._maybe_refresh_memory(-1)
        self.assertIsNotNone(repository.summary_for_day("2026-08-11"))
        self.assertIsNone(repository.summary_for_day("2026-08-10"))

    def test_first_candidate_observation_is_not_stable(self):
        summary = {**EMPTY_SUMMARY, "memory_candidates": ["Серёга постоянно опаздывает"]}
        service, _ = self.service(SummaryProvider([summary]))
        repository = service.repository(-1)
        self.add(repository, 1, "Серёга постоянно опаздывает")
        service._maybe_refresh_memory(-1)
        self.assertEqual(repository.stable_memories(), [])
        self.assertEqual(repository.memory_candidates()[0]["observation_count"], 1)

    def test_second_independent_observation_promotes_candidate(self):
        summary = {**EMPTY_SUMMARY, "memory_candidates": ["Серёга постоянно опаздывает"]}
        service, _ = self.service(SummaryProvider([summary, summary]))
        repository = service.repository(-1)
        self.add(repository, 1, "Серёга постоянно опаздывает")
        service._maybe_refresh_memory(-1)
        self.add(repository, 2, "И снова Серёга постоянно опаздывает")
        service._maybe_refresh_memory(-1)
        self.assertEqual(repository.stable_memories(), ["Серёга постоянно опаздывает"])
        self.assertEqual(repository.memory_candidates()[0]["observation_count"], 2)

    def test_normalized_close_duplicates_create_one_stable_memory(self):
        first = {**EMPTY_SUMMARY, "memory_candidates": ["Серёга постоянно опаздывает"]}
        second = {**EMPTY_SUMMARY, "memory_candidates": ["Серега часто опаздывает!!!"]}
        service, _ = self.service(SummaryProvider([first, second]))
        repository = service.repository(-1)
        self.add(repository, 1, "Серёга постоянно опаздывает")
        service._maybe_refresh_memory(-1)
        self.add(repository, 2, "Серега часто опаздывает!!!")
        service._maybe_refresh_memory(-1)
        self.assertEqual(len(repository.memory_candidates()), 1)
        self.assertEqual(len(repository.stable_memories()), 1)

    def test_old_sqlite_schema_is_migrated_additively(self):
        path = self.data_dir / "chat_m77.sqlite3"
        with sqlite3.connect(path) as db:
            db.executescript(
                """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    user_id INTEGER,
                    username TEXT,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reply_to_message_id INTEGER,
                    is_reply INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT,
                    UNIQUE(chat_id, message_id)
                );
                CREATE TABLE daily_summaries (
                    day TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE long_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory TEXT NOT NULL UNIQUE,
                    score INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                """
            )
            db.execute(
                "INSERT INTO messages(chat_id, message_id, user_id, username, text, "
                "created_at) VALUES(-77, 10, 1, 'old', 'старое сообщение', ?)",
                (self.now.isoformat(),),
            )
            db.execute(
                "INSERT INTO daily_summaries VALUES(?, ?, ?)",
                (
                    "2026-08-11",
                    json.dumps({"topics": ["старый релиз"]}, ensure_ascii=False),
                    self.now.isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO long_memories(memory, score, updated_at) VALUES(?, 2, ?)",
                ("старый мем", self.now.isoformat()),
            )
        repository = ChatRepository(self.data_dir, -77, max_messages=50)
        self.assertEqual(repository.recent_messages()[0]["text"], "старое сообщение")
        self.assertEqual(repository.stable_memories(), ["старый мем"])
        self.assertEqual(repository.recent_summaries()[0]["topics"], ["старый релиз"])
        self.assertEqual(repository.summary_state()["last_message_row_id"], 1)
        self.assertEqual(repository.memory_candidates(), [])

    def test_twenty_minute_due_time_requires_new_messages(self):
        service, provider = self.service(
            SummaryProvider([EMPTY_SUMMARY]), summary_message_interval=50
        )
        repository = service.repository(-1)
        self.add(repository, 1, "одно новое сообщение")
        self.assertFalse(service._maybe_refresh_memory(-1))
        self.now += timedelta(minutes=20)
        self.assertTrue(service._maybe_refresh_memory(-1))
        self.assertEqual(len(provider.requests), 1)


if __name__ == "__main__":
    unittest.main()
