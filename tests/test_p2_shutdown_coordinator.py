import signal
import threading
import time
from unittest.mock import Mock, patch

import pytest

from runtime_shutdown import ShutdownCoordinator


def test_repeated_shutdown_is_idempotent_and_stops_components_once():
    coordinator = ShutdownCoordinator(1)
    calls = []
    coordinator.register_component("one", stop=lambda: calls.append("one"))
    coordinator.register_component("two", stop=lambda: calls.append("two"))

    threads = [
        threading.Thread(target=coordinator.request_shutdown, args=("SIGTERM",))
        for _ in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    assert coordinator.state == coordinator.DRAINING
    assert calls == ["one", "two"]
    report = coordinator.drain()
    assert report.drained
    assert coordinator.state == coordinator.STOPPED


def test_active_work_finishes_but_new_work_is_rejected():
    coordinator = ShutdownCoordinator(1)
    entered = threading.Event()
    release = threading.Event()

    def active():
        with coordinator.work("foreground") as admitted:
            assert admitted
            entered.set()
            release.wait(1)

    worker = threading.Thread(target=active)
    worker.start()
    assert entered.wait(1)
    coordinator.request_shutdown("SIGTERM")
    with coordinator.work("foreground") as admitted:
        assert not admitted
    release.set()
    report = coordinator.drain()
    worker.join(timeout=1)

    assert report.drained
    assert report.resources["active_foreground"] == 0
    assert not worker.is_alive()


def test_stop_hook_error_does_not_skip_remaining_components():
    coordinator = ShutdownCoordinator(1)
    calls = []

    def broken():
        calls.append("broken")
        raise RuntimeError("synthetic")

    coordinator.register_component("broken", stop=broken)
    coordinator.register_component("healthy", stop=lambda: calls.append("healthy"))
    report = coordinator.drain("startup_failure")

    assert calls == ["broken", "healthy"]
    assert report.drained
    assert report.errors == ("broken:RuntimeError",)


def test_deadline_expiry_returns_bounded_report():
    coordinator = ShutdownCoordinator(0.03)
    hold = threading.Event()
    entered = threading.Event()

    def never_during_grace():
        with coordinator.work("foreground") as admitted:
            assert admitted
            entered.set()
            hold.wait(1)

    worker = threading.Thread(target=never_during_grace, daemon=True)
    worker.start()
    assert entered.wait(1)
    started = time.monotonic()
    report = coordinator.drain("SIGTERM")
    elapsed = time.monotonic() - started
    hold.set()
    worker.join(timeout=1)

    assert elapsed < 0.25
    assert not report.drained
    assert "active_foreground" in report.unfinished


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_main_signal_uses_same_bounded_lifecycle(signum):
    import bot as bot_module

    handlers = {}
    fake_thread = Mock()
    fake_thread.is_alive.return_value = False

    def install(received_signum, handler):
        if callable(handler):
            handlers[received_signum] = handler

    def polling(**_kwargs):
        handlers[signum](signum, None)

    with (
        patch.object(bot_module, "TOKEN", "123:configured"),
        patch.object(bot_module.learning_service, "provider_available", return_value=True),
        patch.object(bot_module, "send_startup_meme"),
        patch.object(bot_module.threading, "Thread", return_value=fake_thread),
        patch.object(bot_module.signal, "getsignal", return_value=signal.SIG_DFL),
        patch.object(bot_module.signal, "signal", side_effect=install),
        patch.object(bot_module.bot, "infinity_polling", side_effect=polling),
        patch.object(bot_module.bot, "stop_polling") as stop_polling,
        patch.object(bot_module.learning_service.concurrency, "shutdown") as stop_admission,
        patch.object(bot_module.chat_action_manager, "shutdown") as stop_actions,
    ):
        bot_module.main()

    stop_polling.assert_called_once_with()
    stop_admission.assert_called_once_with()
    stop_actions.assert_called_once_with()
    fake_thread.start.assert_called_once_with()


def test_partial_startup_failure_is_safe():
    coordinator = ShutdownCoordinator(0.1)
    coordinator.register_component("created", stop=lambda: None)
    report = coordinator.drain("startup_failure")
    assert report.drained
    assert report.reason == "startup_failure"
