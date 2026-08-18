import sqlite3
import inspect
import threading
from pathlib import Path

import pytest

from learning.db_migrations import (
    CURRENT_SCHEMA_VERSION,
    FutureSchemaError,
    SchemaMigrationError,
    migrate_database,
    schema_signature,
    schema_version,
)
from learning.repository import ChatRepository


LEGACY_SCHEMA = """
CREATE TABLE messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
 message_id INTEGER NOT NULL, user_id INTEGER, username TEXT, text TEXT NOT NULL,
 created_at TEXT NOT NULL, reply_to_message_id INTEGER,
 is_reply INTEGER NOT NULL DEFAULT 0, last_used_at TEXT,
 UNIQUE(chat_id, message_id));
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE daily_summaries (
 day TEXT PRIMARY KEY, summary_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE summary_state (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 last_message_row_id INTEGER NOT NULL DEFAULT 0,
 last_summary_at TEXT, pending_since TEXT);
CREATE TABLE llm_calls (
 id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL,
 model TEXT NOT NULL, call_type TEXT NOT NULL, input_tokens INTEGER,
 cached_input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER,
 cost_usd_ticks INTEGER, created_at TEXT NOT NULL);
CREATE TABLE routing_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE TABLE pending_conversations (
 user_id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL,
 bot_message_id INTEGER, original_message_id INTEGER,
 original_question TEXT NOT NULL, clarification_question TEXT NOT NULL,
 intent TEXT NOT NULL, context TEXT NOT NULL DEFAULT '',
 expected_type TEXT NOT NULL DEFAULT 'short_answer', created_at TEXT NOT NULL);
"""


def legacy_db(path):
    db = sqlite3.connect(path)
    db.executescript(LEGACY_SCHEMA)
    db.execute(
        "INSERT INTO messages(chat_id,message_id,user_id,username,text,created_at) "
        "VALUES(-1,10,7,'u','legacy text','2026-01-01T00:00:00+00:00')"
    )
    db.execute("INSERT INTO settings VALUES('talk','0')")
    db.execute(
        "INSERT INTO summary_state(singleton,last_message_row_id,pending_since) "
        "VALUES(1,0,'2026-01-01T00:00:00+00:00')"
    )
    db.execute(
        "INSERT INTO llm_calls(provider,model,call_type,input_tokens,created_at) "
        "VALUES('legacy','m','reply',3,'2026-01-01T00:00:00+00:00')"
    )
    db.commit()
    db.close()


def test_empty_database_bootstraps_latest_and_reopens_idempotently(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    signature = repository.schema_signature()
    assert repository.current_schema_version() == CURRENT_SCHEMA_VERSION
    assert len(repository.migration_status()["migrations"]) == CURRENT_SCHEMA_VERSION

    for _ in range(10):
        repository = ChatRepository(tmp_path, -1, 50, 500)
    assert repository.schema_signature() == signature
    assert len(repository.migration_status()["migrations"]) == CURRENT_SCHEMA_VERSION


def test_legacy_upgrade_preserves_data_and_matches_new_schema(tmp_path):
    legacy_dir = tmp_path / "legacy"
    new_dir = tmp_path / "new"
    legacy_dir.mkdir()
    legacy_path = legacy_dir / "chat_m1.sqlite3"
    legacy_db(legacy_path)

    upgraded = ChatRepository(legacy_dir, -1, 50, 500)
    fresh = ChatRepository(new_dir, -1, 50, 500)
    assert upgraded.current_schema_version() == CURRENT_SCHEMA_VERSION
    assert upgraded.schema_signature() == fresh.schema_signature()
    assert upgraded.recent_messages(10)[0]["text"] == "legacy text"
    assert upgraded.setting("talk") == "0"
    assert upgraded.summary_state()["pending_since"].startswith("2026-01-01")
    with upgraded._connect() as db:
        row = db.execute("SELECT * FROM llm_calls").fetchone()
        assert row["provider"] == "legacy"
        assert row["chat_id"] == 0
        assert row["event_id"] is None


def test_migration_failure_does_not_advance_failed_version(tmp_path):
    path = tmp_path / "partial.sqlite3"
    db = sqlite3.connect(path)

    def fail_second(migration):
        if migration.version == 2:
            raise RuntimeError("injected")

    with pytest.raises(SchemaMigrationError):
        migrate_database(db, before_apply=fail_second)
    assert schema_version(db) == 1
    assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
    assert migrate_database(db) == CURRENT_SCHEMA_VERSION
    db.close()


def test_future_version_fails_before_repository_mutation(tmp_path):
    path = tmp_path / "chat_m1.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE marker(value TEXT)")
    db.execute("INSERT INTO marker VALUES('keep')")
    db.execute("PRAGMA user_version=99")
    db.commit()
    before = path.read_bytes()
    db.close()

    with pytest.raises(FutureSchemaError, match="newer"):
        ChatRepository(tmp_path, -1, 50, 500)
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == 99
    assert db.execute("SELECT value FROM marker").fetchone()[0] == "keep"
    db.close()
    assert path.read_bytes() == before


@pytest.mark.parametrize("variant", range(20))
def test_twenty_supported_legacy_variants_upgrade(variant, tmp_path):
    path = tmp_path / f"variant_{variant}.sqlite3"
    db = sqlite3.connect(path)
    # Programmatic pre-media/pre-pending/pre-R5 shapes. Migration 1 fills all
    # omitted tables; later migrations repair whichever legacy tables exist.
    db.execute("CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    if variant % 2 == 0:
        db.execute(
            "CREATE TABLE summary_state(singleton INTEGER PRIMARY KEY, "
            "last_message_row_id INTEGER NOT NULL DEFAULT 0, "
            "last_summary_at TEXT, pending_since TEXT)"
        )
    if variant % 3 == 0:
        db.execute(
            "CREATE TABLE routing_events(id INTEGER PRIMARY KEY, "
            "event_type TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
    if variant % 5 == 0:
        db.execute(
            "CREATE TABLE llm_calls(id INTEGER PRIMARY KEY, provider TEXT NOT NULL, "
            "model TEXT NOT NULL, call_type TEXT NOT NULL, input_tokens INTEGER, "
            "cached_input_tokens INTEGER, output_tokens INTEGER, "
            "reasoning_tokens INTEGER, cost_usd_ticks INTEGER, created_at TEXT NOT NULL)"
        )
    db.commit()
    assert migrate_database(db) == CURRENT_SCHEMA_VERSION
    assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    db.close()


def test_large_unversioned_r5_database_migrates_without_data_loss(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    stamp = "2026-01-01T00:00:00+00:00"
    with repository._connect() as db, db:
        db.executemany(
            """INSERT INTO llm_calls(
                chat_id,provider,model,call_type,input_tokens,created_at
            ) VALUES(-1,'p','m','reply',1,?)""",
            [(stamp,)] * 3000,
        )
        db.execute("DELETE FROM schema_migrations")
        db.execute("PRAGMA user_version=0")
    reopened = ChatRepository(tmp_path, -1, 50, 500)
    with reopened._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 3000
    assert reopened.current_schema_version() == CURRENT_SCHEMA_VERSION
    assert reopened.quick_check() == ("ok",)


def test_corrupt_database_fails_without_recreation(tmp_path):
    path = tmp_path / "chat_m1.sqlite3"
    payload = b"not-a-sqlite-database-private-data"
    path.write_bytes(payload)
    with pytest.raises(sqlite3.DatabaseError):
        ChatRepository(tmp_path, -1, 50, 500)
    assert path.read_bytes() == payload


def test_unknown_custom_table_survives_migration(tmp_path):
    path = tmp_path / "custom.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE custom_extension(value TEXT)")
    db.execute("INSERT INTO custom_extension VALUES('keep')")
    db.commit()
    assert migrate_database(db) == CURRENT_SCHEMA_VERSION
    assert db.execute("SELECT value FROM custom_extension").fetchone()[0] == "keep"
    db.close()


def test_connection_pragmas_and_journal_initialization_boundary(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    with repository._connect() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    # WAL is persistent and initialized once; hot-path connections only apply
    # connection-scoped pragmas.
    assert "journal_mode" not in inspect.getsource(ChatRepository._connect)


def test_parallel_repository_initialization_applies_each_migration_once(tmp_path):
    errors = []

    def initialize():
        try:
            ChatRepository(tmp_path, -1, 50, 500)
        except Exception as error:  # pragma: no cover - assertion reports it
            errors.append(error)

    threads = [threading.Thread(target=initialize) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert errors == []
    repository = ChatRepository(tmp_path, -1, 50, 500)
    assert len(repository.migration_status()["migrations"]) == CURRENT_SCHEMA_VERSION
