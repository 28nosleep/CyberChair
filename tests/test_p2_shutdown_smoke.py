import threading

from learning.concurrency import ConcurrencyController
from runtime_shutdown import ShutdownCoordinator


def test_production_like_50_operation_shutdown_smoke():
    control = ConcurrencyController(2, 1, 5, 5)
    coordinator = ShutdownCoordinator(2, diagnostics=control.snapshot)
    coordinator.register_component("admission", stop=control.shutdown)
    release = threading.Event()
    active_ready = threading.Barrier(6)
    entered = []
    lock = threading.Lock()

    def operation(index):
        chat_id = index % 5
        with coordinator.work("foreground") as accepted:
            if not accepted:
                return
            with control.chat_event_slot(chat_id, f"event-{index}") as chat:
                if not chat:
                    return
                with lock:
                    entered.append(index)
                if index < 5:
                    active_ready.wait(timeout=1)
                    if index == 0:
                        with control.llm_slot("llm-active", chat_id) as slot:
                            assert slot
                            release.wait(1)
                    elif index == 1:
                        with control.media_slot("media-active", chat_id) as slot:
                            assert slot
                            release.wait(1)
                    else:
                        release.wait(1)

    active = [threading.Thread(target=operation, args=(index,)) for index in range(5)]
    for thread in active:
        thread.start()
    active_ready.wait(timeout=1)
    waiters = [
        threading.Thread(target=operation, args=(index,))
        for index in range(5, 50)
    ]
    for thread in waiters:
        thread.start()

    coordinator.request_shutdown("SIGTERM")
    for kind in ("foreground", "scheduled", "memory", "autonomous"):
        with coordinator.work(kind) as accepted:
            assert not accepted
    release.set()
    report = coordinator.drain()

    for thread in active + waiters:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert report.drained
    assert sorted(entered) == list(range(5))
    assert report.resources["active_llm"] == 0
    assert report.resources["active_media"] == 0
    assert report.resources["chat_gate_registry_size"] == 0
