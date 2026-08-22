import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from learning import CURRENT_SCHEMA_VERSION
from learning.repository import ChatRepository
from learning.scheduled_delivery import (
    ScheduledDeliveryCoordinator,
    ScheduledEventSpec,
)


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


def test_v4_claim_markers_migrate_as_terminal_history(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    old = (NOW - timedelta(days=100)).isoformat()
    with repository._connect() as db, db:
        db.execute("DROP TABLE scheduled_events")
        db.execute(
            "CREATE TABLE scheduled_events(event_key TEXT PRIMARY KEY, "
            "created_at TEXT NOT NULL)"
        )
        db.execute("INSERT INTO scheduled_events VALUES('workday_start:old',?)", (old,))
        db.execute("DELETE FROM schema_migrations WHERE version=5")
        db.execute("PRAGMA user_version=4")

    upgraded = ChatRepository(tmp_path, -1, 50, 500)
    row = upgraded.scheduled_events()[0]
    assert upgraded.current_schema_version() == CURRENT_SCHEMA_VERSION == 6
    assert row["state"] == "DEAD"
    assert row["payload"] == ""
    assert row["delivered_at"] is None
    assert row["last_failure_category"] == "legacy_unconfirmed"


def test_backup_restore_preserves_pending_unknown_and_sent(tmp_path):
    source = ChatRepository(tmp_path / "source", -1, 50, 500)
    coordinator = ScheduledDeliveryCoordinator(lambda _chat_id: source)
    pending = ScheduledEventSpec("pending", "test", NOW, "p")
    unknown = ScheduledEventSpec("unknown", "test", NOW, "u")
    sent = ScheduledEventSpec("sent", "test", NOW, "s")
    coordinator.ensure_event(-1, pending, NOW)
    coordinator.deliver_event(
        -1, unknown, lambda *_: (_ for _ in ()).throw(TimeoutError()), NOW
    )
    coordinator.deliver_event(
        -1, sent, lambda *_: SimpleNamespace(message_id=4), NOW
    )

    backup = tmp_path / "restore" / "chat_m1.sqlite3"
    source.backup_to(backup)
    restored = ChatRepository(backup.parent, -1, 50, 500)
    assert {row["event_key"]: row["state"] for row in restored.scheduled_events()} == {
        "pending": "PENDING", "unknown": "UNKNOWN", "sent": "SENT",
    }


def test_retention_only_prunes_old_sent_and_dead(tmp_path):
    old = NOW - timedelta(days=100)
    repository = ChatRepository(tmp_path, -1, 50, 500)
    coordinator = ScheduledDeliveryCoordinator(lambda _chat_id: repository)
    pending = ScheduledEventSpec("pending", "test", old, "p")
    unknown = ScheduledEventSpec("unknown", "test", old, "u")
    sent = ScheduledEventSpec("sent", "test", old, "s")
    claimed = ScheduledEventSpec("claimed", "test", old, "c")
    sending = ScheduledEventSpec("sending", "test", old, "g")
    retry_wait = ScheduledEventSpec("retry_wait", "test", old, "r")
    dead = ScheduledEventSpec("dead", "test", old, "d")
    coordinator.ensure_event(-1, pending, old)
    coordinator.deliver_event(
        -1, unknown, lambda *_: (_ for _ in ()).throw(TimeoutError()), old
    )
    coordinator.deliver_event(
        -1, sent, lambda *_: SimpleNamespace(message_id=5), old
    )
    for event in (claimed, sending, retry_wait, dead):
        coordinator.ensure_event(-1, event, old)
    with repository._connect() as db, db:
        for state in ("CLAIMED", "SENDING", "RETRY_WAIT", "DEAD"):
            db.execute(
                "UPDATE scheduled_events SET state=? WHERE event_key=?",
                (state, state.casefold()),
            )

    result = repository.run_persistence_maintenance(NOW, force=True)
    states = {row["event_key"]: row["state"] for row in repository.scheduled_events()}
    assert result["scheduled_events_pruned"] == 2
    assert states == {
        "pending": "PENDING",
        "unknown": "UNKNOWN",
        "claimed": "CLAIMED",
        "sending": "SENDING",
        "retry_wait": "RETRY_WAIT",
    }


@pytest.mark.parametrize(
    ("hour", "minute"), [(10, 0), (12, 16), (18, 0)],
)
def test_missed_exact_minute_events_are_not_backfilled(hour, minute):
    import scheduler as scheduler_module

    current = NOW.replace(hour=hour, minute=minute)
    created = []
    scheduler_module._last_event = None
    scheduler_module._last_quote_event = None
    scheduler_module._next_random_at = current + timedelta(hours=1)
    with (
        patch.object(scheduler_module, "get_now", return_value=current),
        patch.object(scheduler_module, "daily_quote_minutes", return_value=[12 * 60 + 15]),
        patch.object(scheduler_module, "daily_freekucher_minute", return_value=23 * 60),
        patch.object(scheduler_module, "is_workday", return_value=True),
        patch.object(scheduler_module.time, "sleep", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            scheduler_module.scheduler(
                SimpleNamespace(send_message=lambda *_a, **_k: None),
                -1, "Europe/Moscow", 9, 0, 17, 30,
                daily_freekucher_callback=lambda *_: None,
                scheduled_delivery_callback=lambda *args: created.append(args),
            )
    assert created == []


def test_daily_freekucher_keeps_same_day_catchup_behavior():
    import scheduler as scheduler_module

    current = NOW.replace(hour=21, minute=0)
    created = []
    scheduler_module._next_random_at = current + timedelta(hours=1)
    with (
        patch.object(scheduler_module, "get_now", return_value=current),
        patch.object(scheduler_module, "daily_quote_minutes", return_value=[]),
        patch.object(scheduler_module, "daily_freekucher_minute", return_value=12 * 60),
        patch.object(scheduler_module, "is_workday", return_value=False),
        patch.object(scheduler_module.time, "sleep", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            scheduler_module.scheduler(
                SimpleNamespace(), -1, "Europe/Moscow", 9, 0, 17, 30,
                daily_freekucher_callback=lambda *_: None,
                scheduled_delivery_callback=lambda *args: created.append(args),
            )
    assert len(created) == 1
    assert created[0][1] == "freekucher:2026-08-18"
