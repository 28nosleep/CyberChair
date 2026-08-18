from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from learning.repository import ChatRepository
from learning.scheduled_delivery import (
    ScheduledDeliveryCoordinator,
    ScheduledEventSpec,
)


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


def parts(tmp_path, lease=30):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    provider = lambda _chat_id: repository
    coordinator = ScheduledDeliveryCoordinator(
        provider, clock=lambda: NOW, lease_seconds=lease
    )
    spec = ScheduledEventSpec("restart:12", "restart", NOW, "payload")
    return repository, provider, coordinator, spec


def success(*_args):
    return SimpleNamespace(message_id=77)


def test_created_event_survives_restart_and_is_delivered(tmp_path):
    repository, provider, coordinator, spec = parts(tmp_path)
    row, created = coordinator.ensure_event(-1, spec, NOW)
    assert created and row["state"] == "PENDING"

    restarted = ScheduledDeliveryCoordinator(provider, clock=lambda: NOW)
    assert restarted.deliver_pending(-1, success, NOW)[0].state == "SENT"
    assert repository.scheduled_events()[0]["state"] == "SENT"


def test_expired_claim_is_recovered_but_active_claim_is_not(tmp_path):
    repository, provider, coordinator, spec = parts(tmp_path, lease=30)
    row, _ = coordinator.ensure_event(-1, spec, NOW)
    repository.claim_due_scheduled_event(NOW, 30, row["event_id"])
    restarted = ScheduledDeliveryCoordinator(provider, clock=lambda: NOW)

    assert restarted.deliver_pending(
        -1, success, NOW + timedelta(seconds=29)
    ) == ()
    result = restarted.deliver_pending(
        -1, success, NOW + timedelta(seconds=30)
    )
    assert result[0].state == "SENT"
    counts = repository.routing_report("2000-01-01T00:00:00+00:00")
    assert counts["scheduled_claim_recovered"] == 1


def test_sending_crash_becomes_unknown_on_restart(tmp_path):
    repository, provider, coordinator, spec = parts(tmp_path)
    row, _ = coordinator.ensure_event(-1, spec, NOW)
    claim = repository.claim_due_scheduled_event(NOW, 30, row["event_id"])
    repository.mark_scheduled_sending(row["event_id"], claim["claim_token"], NOW)

    sends = []
    restarted = ScheduledDeliveryCoordinator(provider, clock=lambda: NOW)
    assert restarted.deliver_pending(
        -1, lambda *_: sends.append(1), NOW + timedelta(days=1)
    ) == ()
    assert sends == []
    assert repository.scheduled_events()[0]["state"] == "UNKNOWN"


def test_success_then_local_commit_failure_is_fenced_unknown(
    tmp_path, monkeypatch, caplog,
):
    repository, _provider, coordinator, spec = parts(tmp_path)

    def fail_commit(*_args, **_kwargs):
        raise RuntimeError("local commit failed")

    monkeypatch.setattr(repository, "finalize_scheduled_success", fail_commit)
    result = coordinator.deliver_event(-1, spec, success, NOW)
    assert result.state == "UNKNOWN"
    assert repository.scheduled_events()[0]["state"] == "UNKNOWN"
    assert "POST_DELIVERY_SCHEDULE_COMMIT_FAILED" in caplog.text
    assert "payload" not in caplog.text
    assert coordinator.deliver_pending(
        -1, success, NOW + timedelta(days=1)
    ) == ()


def test_sent_remains_terminal_after_restart(tmp_path):
    repository, provider, coordinator, spec = parts(tmp_path)
    assert coordinator.deliver_event(-1, spec, success, NOW).state == "SENT"
    sends = []
    restarted = ScheduledDeliveryCoordinator(provider, clock=lambda: NOW)
    assert restarted.deliver_pending(
        -1, lambda *_: sends.append(1), NOW + timedelta(days=1)
    ) == ()
    assert sends == []
    assert repository.scheduled_events()[0]["state"] == "SENT"


def test_forget_during_send_prevents_late_finalize_resurrection(tmp_path):
    repository, _provider, coordinator, spec = parts(tmp_path)

    def forget_then_success(*_args):
        repository.clear()
        return SimpleNamespace(message_id=8)

    result = coordinator.deliver_event(-1, spec, forget_then_success, NOW)
    assert result.state == "STALE"
    assert repository.scheduled_events() == []
