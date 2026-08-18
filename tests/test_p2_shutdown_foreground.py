import threading
from contextlib import contextmanager

import pytest

from learning.concurrency import ConcurrencyController
from learning.event_context import EventContext, bind_event, llm_network_call
from runtime_shutdown import ShutdownCoordinator


class Repository:
    chat_id = -1

    def record_routing_event(self, *_args, **_kwargs):
        return None


def coordinated(controller, grace=1):
    coordinator = ShutdownCoordinator(grace, diagnostics=controller.snapshot)
    coordinator.register_component("admission", stop=controller.shutdown)
    return coordinator


def test_foreground_llm_active_finishes_delivery_and_commit():
    control = ConcurrencyController(1, 1, 1, 1)
    coordinator = coordinated(control)
    provider_entered = threading.Event()
    provider_release = threading.Event()
    lifecycle = []

    def event_a():
        with coordinator.work("foreground") as accepted:
            assert accepted
            with control.chat_event_slot(-1, "event-a") as chat:
                assert chat
                with bind_event(EventContext("event-a", "user", -1)):
                    with llm_network_call(
                        Repository(), "fake", "reply", control
                    ) as allowed:
                        assert allowed
                        provider_entered.set()
                        provider_release.wait(1)
                        lifecycle.append("provider")
                lifecycle.extend(("plan", "delivery", "commit"))

    worker = threading.Thread(target=event_a)
    worker.start()
    assert provider_entered.wait(1)
    coordinator.request_shutdown("SIGTERM")
    with coordinator.work("foreground") as accepted:
        assert not accepted
    provider_release.set()
    report = coordinator.drain()
    worker.join(timeout=1)

    assert lifecycle == ["provider", "plan", "delivery", "commit"]
    assert report.drained
    assert report.resources["active_llm"] == 0
    assert report.resources["chat_gate_registry_size"] == 0


@pytest.mark.parametrize("success", [True, False])
def test_foreground_telegram_send_finishes_without_alternative(success):
    control = ConcurrencyController(1, 1, 1, 1)
    coordinator = coordinated(control)
    sender_entered = threading.Event()
    sender_release = threading.Event()
    events = []

    def lifecycle():
        with coordinator.work("foreground") as accepted:
            assert accepted
            with control.chat_event_slot(-1, "delivery") as chat:
                assert chat
                sender_entered.set()
                sender_release.wait(1)
                events.append("send")
                events.append("commit" if success else "abort")

    worker = threading.Thread(target=lifecycle)
    worker.start()
    assert sender_entered.wait(1)
    coordinator.request_shutdown("SIGTERM")
    sender_release.set()
    report = coordinator.drain()
    worker.join(timeout=1)

    assert events == ["send", "commit" if success else "abort"]
    assert report.drained
    assert events.count("send") == 1


def test_same_chat_waiter_wakes_and_never_enters_substantive_work():
    control = ConcurrencyController(1, 1, 5, 5)
    coordinator = coordinated(control)
    holder_entered = threading.Event()
    holder_release = threading.Event()
    waiter_started = threading.Event()
    waiter_done = threading.Event()
    substantive = []

    def holder():
        with coordinator.work("foreground"):
            with control.chat_event_slot(-1, "holder") as admission:
                assert admission
                holder_entered.set()
                holder_release.wait(1)

    def waiter():
        with coordinator.work("foreground") as accepted:
            assert accepted
            waiter_started.set()
            with control.chat_event_slot(-1, "waiter") as admission:
                if admission:
                    substantive.append("waiter")
        waiter_done.set()

    first = threading.Thread(target=holder)
    second = threading.Thread(target=waiter)
    first.start()
    assert holder_entered.wait(1)
    second.start()
    assert waiter_started.wait(1)
    coordinator.request_shutdown("SIGTERM")
    assert waiter_done.wait(0.5)
    holder_release.set()
    report = coordinator.drain()
    first.join(timeout=1)
    second.join(timeout=1)

    assert substantive == []
    assert report.drained
    assert report.resources["chat_gate_registry_size"] == 0


def test_new_fake_update_is_rejected_after_polling_stop():
    import bot as bot_module

    coordinator = ShutdownCoordinator(0.1)
    previous = bot_module._runtime_shutdown_coordinator
    calls = []

    @bot_module.telegram_user_event_handler
    def handler(_message, _event):
        calls.append("domain")

    try:
        bot_module._runtime_shutdown_coordinator = coordinator
        coordinator.request_shutdown("SIGTERM")
        handler(object())
    finally:
        bot_module._runtime_shutdown_coordinator = previous
    assert calls == []
