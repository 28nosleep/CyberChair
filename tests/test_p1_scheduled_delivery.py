import sqlite3
from datetime import timedelta
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from learning.repository import ChatRepository
from learning.scheduled_delivery import (
    ScheduledDeliveryCoordinator,
    ScheduledEventSpec,
)


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


def setup_coordinator(tmp_path):
    repositories = {}

    def repository(chat_id):
        return repositories.setdefault(
            chat_id, ChatRepository(tmp_path, chat_id, 50, 500)
        )

    return ScheduledDeliveryCoordinator(repository, clock=lambda: NOW), repository


def spec(key="quote:2026-08-18:720", payload="fixed payload"):
    return ScheduledEventSpec(key, "movie_quote", NOW, payload, "HTML")


def test_simple_success_is_sent_only_after_telegram_result(tmp_path):
    coordinator, repository = setup_coordinator(tmp_path)
    result = coordinator.deliver_event(
        -1, spec(), lambda *_: SimpleNamespace(message_id=42), NOW
    )
    row = repository(-1).scheduled_event(event_id=result.event_id)

    assert result.state == "SENT"
    assert row["state"] == "SENT"
    assert row["telegram_message_id"] == 42
    assert row["delivered_at"] == NOW.isoformat()
    assert row["attempt_count"] == 1
    assert repository(-1).finalize_scheduled_success(
        result.event_id, "already-finalized", 99, NOW
    ) == "sent"
    assert repository(-1).scheduled_event(
        event_id=result.event_id
    )["telegram_message_id"] == 42


def test_duplicate_ticks_keep_one_row_one_success_and_first_payload(tmp_path):
    coordinator, repository = setup_coordinator(tmp_path)
    sends = []

    for index in range(20):
        coordinator.deliver_event(
            -1,
            spec(payload=f"payload-{index}"),
            lambda _event_id, _chat_id, payload, _parse_mode: (
                sends.append(payload) or SimpleNamespace(message_id=7)
            ),
            NOW,
        )

    rows = repository(-1).scheduled_events()
    assert len(rows) == 1
    assert sends == ["payload-0"]
    assert rows[0]["payload"] == "payload-0"
    assert rows[0]["state"] == "SENT"


def test_no_sqlite_transaction_or_repository_lock_during_network_send(tmp_path):
    coordinator, repository_provider = setup_coordinator(tmp_path)
    repository = repository_provider(-1)

    def sender(*_args):
        # BEGIN IMMEDIATE and this public repository read would both block if
        # the claim transaction/lock leaked across the Telegram callback.
        with sqlite3.connect(repository.path, timeout=0.1) as db:
            db.execute("BEGIN IMMEDIATE")
            db.rollback()
        assert repository.scheduled_events()
        return SimpleNamespace(message_id=9)

    assert coordinator.deliver_event(-1, spec(), sender, NOW).state == "SENT"


def test_diagnostics_and_telemetry_are_content_free(tmp_path):
    coordinator, repository = setup_coordinator(tmp_path)
    coordinator.deliver_event(
        -1, spec(payload="secret scheduled text"),
        lambda *_: SimpleNamespace(message_id=1), NOW,
    )
    report = coordinator.diagnostics(-1, NOW)
    rendered = coordinator.format_diagnostics(-1, NOW)
    event_counts = repository(-1).routing_report("2000-01-01T00:00:00+00:00")

    assert report["sent_recent"] == 1
    assert report["unknown"] == 0
    assert rendered.startswith("SCHEDULED DELIVERY\n")
    assert "secret scheduled text" not in rendered
    for event_type in (
        "scheduled_created", "scheduled_claimed", "scheduled_attempt",
        "scheduled_success",
    ):
        assert event_counts[event_type] == 1


@pytest.mark.parametrize(
    ("current", "quote_minutes", "last_workday", "expected_kinds"),
    (
        (NOW.replace(hour=9), [], False, ["workday_start"]),
        (NOW.replace(hour=17, minute=30), [], False, ["workday_end"]),
        (
            NOW.replace(day=21, hour=17, minute=30), [], True,
            ["workday_end", "weekly_summary"],
        ),
        (NOW.replace(hour=12, minute=15), [12 * 60 + 15], False, ["movie_quote"]),
    ),
)
def test_all_utility_paths_route_through_reliable_callback(
    current, quote_minutes, last_workday, expected_kinds,
):
    import scheduler as scheduler_module

    created = []
    scheduler_module._last_event = None
    scheduler_module._last_quote_event = None
    scheduler_module._next_random_at = current + timedelta(hours=1)
    with (
        patch.object(scheduler_module, "get_now", return_value=current),
        patch.object(scheduler_module, "daily_quote_minutes", return_value=quote_minutes),
        patch.object(scheduler_module, "daily_freekucher_minute", return_value=23 * 60),
        patch.object(scheduler_module, "is_workday", return_value=True),
        patch.object(
            scheduler_module, "is_last_workday_of_week",
            return_value=last_workday,
        ),
        patch.object(scheduler_module.time, "sleep", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            scheduler_module.scheduler(
                SimpleNamespace(send_message=lambda *_args, **_kwargs: None),
                -1, "Europe/Moscow", 9, 0, 17, 30,
                daily_freekucher_callback=lambda *_: None,
                scheduled_delivery_callback=lambda *args: created.append(args),
            )
    assert [item[2] for item in created] == expected_kinds
