import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import LearningService, LearningSettings
from learning.concurrency import ConcurrencyController
from learning.repository import ChatRepository
from learning.scheduled_delivery import (
    ScheduledDeliveryCoordinator,
    ScheduledEventSpec,
)
from runtime_shutdown import ShutdownCoordinator


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


def test_scheduler_stop_prevents_callbacks_after_current_work():
    import scheduler as scheduler_module

    stop = threading.Event()
    memory_entered = threading.Event()
    memory_release = threading.Event()
    calls = []

    def memory(*_args):
        calls.append("memory")
        memory_entered.set()
        memory_release.wait(1)

    thread = threading.Thread(
        target=scheduler_module.scheduler,
        args=(
            SimpleNamespace(), -1, "Europe/Moscow", 9, 0, 17, 30,
            lambda *_: calls.append("autonomous"), None, None, None,
            None, None, lambda *_: calls.append("freekucher"), memory,
            lambda *_: calls.append("scheduled"),
            lambda *_: calls.append("retry"), stop,
        ),
    )
    with (
        patch.object(scheduler_module, "get_now", return_value=NOW),
        patch.object(scheduler_module, "is_workday", return_value=False),
    ):
        thread.start()
        assert memory_entered.wait(1)
        stop.set()
        memory_release.set()
        thread.join(timeout=1)

    assert not thread.is_alive()
    assert calls == ["memory"]


def test_scheduler_stop_between_end_of_day_jobs_prevents_second_delivery():
    import scheduler as scheduler_module

    stop = threading.Event()
    calls = []
    end_of_day = NOW.replace(hour=17, minute=30)

    def scheduled(_chat_id, _event_key, event_kind, *_args):
        calls.append(event_kind)
        stop.set()

    with (
        patch.object(scheduler_module, "get_now", return_value=end_of_day),
        patch.object(scheduler_module, "is_workday", return_value=True),
        patch.object(scheduler_module, "is_last_workday_of_week", return_value=True),
        patch.object(scheduler_module, "daily_quote_minutes", return_value=[]),
    ):
        scheduler_module.scheduler(
            SimpleNamespace(), -1, "Europe/Moscow", 9, 0, 17, 30,
            activity_callback=lambda _chat_id: True,
            scheduled_delivery_callback=scheduled,
            stop_event=stop,
        )

    assert calls == ["workday_end"]


def p1_parts(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    provider = lambda _chat_id: repository
    delivery = ScheduledDeliveryCoordinator(provider, lease_seconds=1)
    spec = ScheduledEventSpec("shutdown", "test", NOW, "payload")
    return repository, provider, delivery, spec


def test_p1_claimed_before_send_remains_lease_recoverable(tmp_path):
    repository, provider, delivery, spec = p1_parts(tmp_path)
    row, _ = delivery.ensure_event(-1, spec, NOW)
    repository.claim_due_scheduled_event(NOW, 1, row["event_id"])
    shutdown = ShutdownCoordinator(0.01)
    shutdown.drain("SIGTERM")
    assert repository.scheduled_events()[0]["state"] == "CLAIMED"

    restarted = ScheduledDeliveryCoordinator(provider, lease_seconds=1)
    result = restarted.deliver_pending(
        -1, lambda *_: SimpleNamespace(message_id=1),
        NOW + timedelta(seconds=1),
    )
    assert result[0].state == "SENT"


def test_p1_sending_success_finishes_within_grace(tmp_path):
    repository, _provider, delivery, spec = p1_parts(tmp_path)
    shutdown = ShutdownCoordinator(1)
    entered = threading.Event()
    release = threading.Event()

    def sender(*_args):
        entered.set()
        release.wait(1)
        return SimpleNamespace(message_id=2)

    def active():
        with shutdown.work("scheduled") as admitted:
            assert admitted
            delivery.deliver_event(-1, spec, sender, NOW)

    worker = threading.Thread(target=active)
    worker.start()
    assert entered.wait(1)
    shutdown.request_shutdown("SIGTERM")
    release.set()
    report = shutdown.drain()
    worker.join(timeout=1)

    assert report.drained
    assert repository.scheduled_events()[0]["state"] == "SENT"


def test_p1_sending_at_forced_boundary_recovers_unknown(tmp_path):
    repository, provider, delivery, spec = p1_parts(tmp_path)
    row, _ = delivery.ensure_event(-1, spec, NOW)
    claim = repository.claim_due_scheduled_event(NOW, 1, row["event_id"])
    repository.mark_scheduled_sending(row["event_id"], claim["claim_token"], NOW)
    assert not ShutdownCoordinator(0).drain("SIGTERM").unfinished

    restarted = ScheduledDeliveryCoordinator(provider)
    assert restarted.deliver_pending(-1, lambda *_: None, NOW) == ()
    assert repository.scheduled_events()[0]["state"] == "UNKNOWN"


class SummaryProvider:
    available = True
    provider_key = "fake"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def generate(self, _request):
        return None

    def summarize(self, _request):
        self.calls += 1
        self.entered.set()
        self.release.wait(1)
        return {
            "main_topics": ["release"],
            "current_mood": "calm",
            "memory_candidates": [],
        }


def memory_service(tmp_path, provider, controller):
    service = LearningService(
        LearningSettings(
            data_dir=Path(tmp_path), summary_message_interval=1,
            summary_time_interval=1, openai_chat_id=-1,
        ),
        llm_provider=provider,
        concurrency_controller=controller,
    )
    repository = service.repository(-1)
    repository.add_message(1, 7, "u", "release context", NOW)
    repository.mark_summary_pending(NOW.isoformat())
    return service, repository


def test_r5_claimed_before_provider_is_released_on_shutdown(tmp_path):
    provider = SummaryProvider()
    controller = ConcurrencyController(1, 1, 1, 1)
    service, repository = memory_service(tmp_path, provider, controller)
    job, status = service.memory.prepare_summary_job(repository, -1, NOW)
    assert status == "claimed"
    controller.shutdown()
    result = service.memory.execute_summary_job(repository, job, provider, NOW)

    assert result.status == "resource_deferred"
    assert provider.calls == 0
    assert repository.summary_state()["claim_token"] is None


def test_r5_active_provider_finishes_and_finalizes_within_grace(tmp_path):
    provider = SummaryProvider()
    controller = ConcurrencyController(1, 1, 1, 1)
    service, repository = memory_service(tmp_path, provider, controller)
    shutdown = ShutdownCoordinator(1, diagnostics=controller.snapshot)
    shutdown.register_component("admission", stop=controller.shutdown)
    results = []

    def active():
        with shutdown.work("memory") as admitted:
            assert admitted
            results.append(service.run_memory_maintenance(-1, NOW))

    worker = threading.Thread(target=active)
    worker.start()
    assert provider.entered.wait(1)
    shutdown.request_shutdown("SIGTERM")
    provider.release.set()
    report = shutdown.drain()
    worker.join(timeout=1)

    assert report.drained
    assert results[0].status == "committed"
    assert repository.summary_state()["last_message_row_id"] == 1
