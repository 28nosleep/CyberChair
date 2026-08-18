import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from learning.repository import ChatRepository
from learning.scheduled_delivery import (
    ScheduledDeliveryCoordinator,
    ScheduledEventSpec,
)


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


class TelegramError(Exception):
    def __init__(self, code):
        self.error_code = code


def test_concurrent_ticks_have_one_active_claim_and_send(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    coordinator = ScheduledDeliveryCoordinator(lambda _chat_id: repository)
    spec = ScheduledEventSpec("same", "test", NOW, "payload")
    barrier = threading.Barrier(20)
    sends = []
    send_lock = threading.Lock()

    def sender(*_args):
        with send_lock:
            sends.append(1)
        return SimpleNamespace(message_id=1)

    def tick():
        barrier.wait(timeout=3)
        coordinator.deliver_event(-1, spec, sender, NOW)

    threads = [threading.Thread(target=tick) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert sends == [1]
    assert len(repository.scheduled_events()) == 1
    assert repository.scheduled_events()[0]["state"] == "SENT"


def test_deterministic_100_event_workload(tmp_path):
    repositories = {}

    def repository(chat_id):
        return repositories.setdefault(
            chat_id, ChatRepository(tmp_path, chat_id, 50, 500)
        )

    coordinator = ScheduledDeliveryCoordinator(
        repository,
        max_attempts=3,
        backoff_base_seconds=1,
        backoff_cap_seconds=4,
    )
    calls = Counter()
    successes = Counter()

    def sender(event_id, _chat_id, _payload, _parse_mode):
        calls[event_id] += 1
        index = int(_payload)
        if 60 <= index < 75 and calls[event_id] == 1:
            raise TelegramError(500)
        if 75 <= index < 90:
            raise TimeoutError("ambiguous")
        if index >= 90:
            raise TelegramError(403)
        successes[event_id] += 1
        return SimpleNamespace(message_id=index + 1)

    specs = []
    for index in range(100):
        chat_id = -(index % 5 + 1)
        event = ScheduledEventSpec(
            f"stress:{index}", "stress", NOW, str(index)
        )
        specs.append((chat_id, event))
        coordinator.deliver_event(chat_id, event, sender, NOW)
        coordinator.deliver_event(chat_id, event, sender, NOW)

    # Restart and run only eligible safe retries. UNKNOWN and DEAD remain inert.
    restarted = ScheduledDeliveryCoordinator(
        repository,
        max_attempts=3,
        backoff_base_seconds=1,
        backoff_cap_seconds=4,
    )
    for chat_id in repositories:
        restarted.deliver_pending(
            chat_id, sender, NOW + timedelta(seconds=1), limit=100
        )
        restarted.deliver_pending(
            chat_id, sender, NOW + timedelta(days=10), limit=100
        )

    rows = [
        row for repository_item in repositories.values()
        for row in repository_item.scheduled_events()
    ]
    states = Counter(row["state"] for row in rows)
    assert len(rows) == 100
    assert states == Counter({"SENT": 75, "UNKNOWN": 15, "DEAD": 10})
    assert all(count <= 1 for count in successes.values())
    assert all(row["state"] != "CLAIMED" for row in rows)
    assert all(row["state"] != "SENDING" for row in rows)
    assert all(
        calls[row["event_id"]] == 1
        for row in rows if row["state"] == "UNKNOWN"
    )
