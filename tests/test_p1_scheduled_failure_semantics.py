from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from learning.repository import ChatRepository
from learning.scheduled_delivery import (
    ScheduledDeliveryCoordinator,
    ScheduledEventSpec,
)


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


class TelegramError(Exception):
    def __init__(self, code, retry_after=None):
        self.error_code = code
        self.result_json = (
            {"parameters": {"retry_after": retry_after}}
            if retry_after is not None else {}
        )


def setup(tmp_path, max_attempts=3):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    coordinator = ScheduledDeliveryCoordinator(
        lambda _chat_id: repository,
        clock=lambda: NOW,
        lease_seconds=30,
        max_attempts=max_attempts,
        backoff_base_seconds=10,
        backoff_cap_seconds=60,
    )
    spec = ScheduledEventSpec("test:12", "test", NOW, "payload")
    return repository, coordinator, spec


def test_definite_server_failure_has_bounded_backoff_then_succeeds(tmp_path):
    repository, coordinator, spec = setup(tmp_path)
    calls = 0

    def sender(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TelegramError(503)
        return SimpleNamespace(message_id=2)

    first = coordinator.deliver_event(-1, spec, sender, NOW)
    assert first.state == "RETRY_WAIT"
    assert repository.scheduled_events()[0]["next_attempt_at"] == (
        NOW + timedelta(seconds=10)
    ).isoformat()
    assert coordinator.deliver_pending(
        -1, sender, NOW + timedelta(seconds=9)
    ) == ()
    retry = coordinator.deliver_pending(
        -1, sender, NOW + timedelta(seconds=10)
    )
    assert retry[0].state == "SENT"
    assert calls == 2
    assert repository.scheduled_events()[0]["safe_retry_count"] == 1


def test_rate_limit_respects_retry_after(tmp_path):
    repository, coordinator, spec = setup(tmp_path)

    def limited(*_args):
        raise TelegramError(429, retry_after=90)

    assert coordinator.deliver_event(-1, spec, limited, NOW).state == "RETRY_WAIT"
    row = repository.scheduled_events()[0]
    assert row["last_failure_category"] == "telegram_rate_limit"
    assert row["next_attempt_at"] == (NOW + timedelta(seconds=90)).isoformat()


def test_permanent_failure_is_dead_without_retry(tmp_path):
    repository, coordinator, spec = setup(tmp_path)
    calls = 0

    def forbidden(*_args):
        nonlocal calls
        calls += 1
        raise TelegramError(403)

    assert coordinator.deliver_event(-1, spec, forbidden, NOW).state == "DEAD"
    for index in range(100):
        assert coordinator.deliver_pending(
            -1, forbidden, NOW + timedelta(days=index + 1)
        ) == ()
    assert calls == 1
    assert repository.scheduled_events()[0]["state"] == "DEAD"


def test_timeout_is_unknown_and_never_automatically_retried(tmp_path):
    repository, coordinator, spec = setup(tmp_path)
    calls = 0

    def timeout(*_args):
        nonlocal calls
        calls += 1
        raise TimeoutError("payload must not be logged")

    assert coordinator.deliver_event(-1, spec, timeout, NOW).state == "UNKNOWN"
    for index in range(100):
        assert coordinator.deliver_pending(
            -1, timeout, NOW + timedelta(hours=index + 1)
        ) == ()
    assert calls == 1
    row = repository.scheduled_events()[0]
    assert row["state"] == "UNKNOWN"
    assert row["last_failure_category"] == "telegram_timeout"


def test_retry_budget_ends_in_dead(tmp_path):
    repository, coordinator, spec = setup(tmp_path, max_attempts=2)

    def server_error(*_args):
        raise TelegramError(500)

    assert coordinator.deliver_event(-1, spec, server_error, NOW).state == "RETRY_WAIT"
    result = coordinator.deliver_pending(
        -1, server_error, NOW + timedelta(seconds=10)
    )
    assert result[0].state == "DEAD"
    assert repository.scheduled_events()[0]["attempt_count"] == 2
