import sqlite3
import threading
from datetime import datetime, timezone

from learning import CURRENT_SCHEMA_VERSION
from learning.repository import ChatRepository


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


def test_online_backup_restore_and_migration(tmp_path):
    source = ChatRepository(tmp_path / "source", -1, 50, 500)
    assert source.add_message(1, 7, "u", "backup text", NOW)
    source.set_setting("talk", "0")
    source.add_chat_image(2, 7, "file", "unique", "photo", created_at=NOW)
    source.record_llm_call(
        "p", "m", "reply", {"input_tokens": 3, "cost_usd_ticks": 7}, NOW
    )

    writer_done = threading.Event()

    def writer():
        source.add_message(3, 7, "u", "concurrent row", NOW)
        writer_done.set()

    thread = threading.Thread(target=writer)
    thread.start()
    backup_path = tmp_path / "restore" / "chat_m1.sqlite3"
    source.backup_to(backup_path)
    thread.join(timeout=2)
    assert writer_done.is_set()

    source.clear()
    restored = ChatRepository(tmp_path / "restore", -1, 50, 500)
    texts = [row["text"] for row in restored.recent_messages(10)]
    assert "backup text" in texts
    assert restored.setting("talk") == "0"
    assert restored.chat_image_count() == 1
    assert restored.llm_usage_report("2000-01-01T00:00:00+00:00")["total"]["calls"] == 1
    assert restored.current_schema_version() == CURRENT_SCHEMA_VERSION
    assert restored.quick_check() == ("ok",)


def test_backup_destination_safety(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    try:
        repository.backup_to(repository.path)
    except ValueError:
        pass
    else:
        raise AssertionError("same-path backup must be rejected")
    target = tmp_path / "copy.sqlite3"
    repository.backup_to(target)
    try:
        repository.backup_to(target)
    except FileExistsError:
        pass
    else:
        raise AssertionError("overwrite must be explicit")
    repository.backup_to(target, overwrite=True)
    db = sqlite3.connect(target)
    assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    db.close()


def test_backup_of_legacy_database_migrates_when_restored(tmp_path):
    legacy = tmp_path / "legacy.sqlite3"
    source = sqlite3.connect(legacy)
    source.execute("CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    source.execute("INSERT INTO settings VALUES('talk','0')")
    source.commit()
    restored_path = tmp_path / "restored" / "chat_m1.sqlite3"
    restored_path.parent.mkdir()
    target = sqlite3.connect(restored_path)
    source.backup(target)
    target.close()
    source.close()

    restored = ChatRepository(restored_path.parent, -1, 50, 500)
    assert restored.setting("talk") == "0"
    assert restored.current_schema_version() == CURRENT_SCHEMA_VERSION
    assert restored.quick_check() == ("ok",)
