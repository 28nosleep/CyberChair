import tempfile
import threading
from pathlib import Path

from learning.chat_action import ChatActionManager
from learning.concurrency import ConcurrencyController
from learning.event_context import EventContext, bind_event, llm_network_call
from runtime_shutdown import ShutdownCoordinator


class Repository:
    chat_id = -1

    def record_routing_event(self, *_args, **_kwargs):
        return None


class Bot:
    def __init__(self):
        self.actions = []

    def send_chat_action(self, chat_id, action):
        self.actions.append((chat_id, action))


def test_llm_waiter_wakes_without_spending_event_permit():
    control = ConcurrencyController(1, 1, 5, 5)
    coordinator = ShutdownCoordinator(1, diagnostics=control.snapshot)
    coordinator.register_component("admission", stop=control.shutdown)
    holder_entered = threading.Event()
    holder_release = threading.Event()
    waiter_done = threading.Event()
    waiter_event = EventContext("waiter", "user", -1)
    allowed_values = []

    def holder():
        with control.llm_slot("holder", -1) as admission:
            assert admission
            holder_entered.set()
            holder_release.wait(1)

    def waiter():
        with bind_event(waiter_event):
            with llm_network_call(
                Repository(), "fake", "reply", control
            ) as allowed:
                allowed_values.append(bool(allowed))
        waiter_done.set()

    active = threading.Thread(target=holder)
    waiting = threading.Thread(target=waiter)
    active.start()
    assert holder_entered.wait(1)
    waiting.start()
    coordinator.request_shutdown("SIGTERM")
    assert waiter_done.wait(0.5)
    holder_release.set()
    report = coordinator.drain()
    active.join(timeout=1)
    waiting.join(timeout=1)

    assert allowed_values == [False]
    assert waiter_event.permit.call_count == 0
    assert report.resources["active_llm"] == 0


def test_media_waiter_wakes_while_active_render_cleans_temp_file():
    control = ConcurrencyController(1, 1, 5, 5)
    coordinator = ShutdownCoordinator(1, diagnostics=control.snapshot)
    coordinator.register_component("admission", stop=control.shutdown)
    entered = threading.Event()
    release = threading.Event()
    waiter_done = threading.Event()
    waiter_admitted = []
    raw = tempfile.NamedTemporaryFile(delete=False)
    path = Path(raw.name)
    raw.close()

    def active_render():
        try:
            with control.media_slot("active", -1) as admission:
                assert admission
                entered.set()
                release.wait(1)
        finally:
            path.unlink(missing_ok=True)

    def waiter():
        with control.media_slot("waiter", -2) as admission:
            waiter_admitted.append(bool(admission))
        waiter_done.set()

    active = threading.Thread(target=active_render)
    waiting = threading.Thread(target=waiter)
    active.start()
    assert entered.wait(1)
    waiting.start()
    coordinator.request_shutdown("SIGTERM")
    assert waiter_done.wait(0.5)
    release.set()
    report = coordinator.drain()
    active.join(timeout=1)
    waiting.join(timeout=1)

    assert waiter_admitted == [False]
    assert not path.exists()
    assert report.resources["active_media"] == 0


def test_typing_refresher_stops_during_active_foreground():
    bot = Bot()
    manager = ChatActionManager(bot, refresh_interval=0.02)
    coordinator = ShutdownCoordinator(
        1, diagnostics=lambda: {"typing_refreshers": manager.worker_count()}
    )
    coordinator.register_component(
        "chat_actions", stop=manager.shutdown,
        stopped=lambda: manager.worker_count() == 0,
    )
    entered = threading.Event()
    release = threading.Event()

    def foreground():
        with coordinator.work("foreground"):
            with manager.activity(-1, "typing"):
                entered.set()
                release.wait(1)

    worker = threading.Thread(target=foreground)
    worker.start()
    assert entered.wait(1)
    coordinator.request_shutdown("SIGTERM")
    release.set()
    report = coordinator.drain()
    worker.join(timeout=1)

    assert report.drained
    assert report.resources["typing_refreshers"] == 0
    before = len(bot.actions)
    with manager.activity(-1, "typing"):
        pass
    assert len(bot.actions) == before
