import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from learning.concurrency import ConcurrencyController
from learning.event_context import (
    EventContext,
    RuntimeConcurrencyTelemetry,
    bind_event,
    llm_network_call,
)
from learning.settings import LearningSettings


class FakeRepository:
    def __init__(self, chat_id=1):
        self.chat_id = chat_id
        self.events = []
        self.lock = threading.Lock()

    def record_routing_event(self, kind, **fields):
        with self.lock:
            self.events.append((kind, fields))


def controller(llm=2, media=1, llm_timeout=0.5, media_timeout=0.5):
    telemetry = RuntimeConcurrencyTelemetry()
    return ConcurrencyController(
        llm, media, llm_timeout, media_timeout, telemetry
    ), telemetry


class LLMAdmissionTests(unittest.TestCase):
    def test_operational_defaults_keep_llm_two_and_media_one(self):
        with patch.dict("os.environ", {
            "LLM_MAX_CONCURRENCY": "2",
            "MEDIA_MAX_CONCURRENCY": "1",
            "LLM_ADMISSION_TIMEOUT_SECONDS": "5",
            "MEDIA_ADMISSION_TIMEOUT_SECONDS": "15",
        }):
            settings = LearningSettings()
        self.assertEqual(settings.llm_max_concurrency, 2)
        self.assertEqual(settings.media_max_concurrency, 1)
        self.assertGreater(settings.llm_admission_timeout_seconds, 0)
        self.assertGreater(settings.media_admission_timeout_seconds, 0)

    def test_third_foreground_waits_and_active_never_exceeds_two(self):
        control, telemetry = controller(llm=2, llm_timeout=2)
        release = threading.Event()
        entered = threading.Barrier(3)
        third_entered = threading.Event()

        def first(event_id):
            with control.llm_slot(event_id, int(event_id[-1])) as admitted:
                self.assertTrue(admitted)
                entered.wait(timeout=2)
                release.wait(timeout=2)

        first_threads = [
            threading.Thread(target=first, args=(event_id,))
            for event_id in ("event-1", "event-2")
        ]
        for thread in first_threads:
            thread.start()
        entered.wait(timeout=2)

        def third():
            with control.llm_slot("event-3", 3) as admitted:
                self.assertTrue(admitted)
                third_entered.set()

        third_thread = threading.Thread(target=third)
        third_thread.start()
        time.sleep(0.05)
        self.assertFalse(third_entered.is_set())
        self.assertEqual(telemetry.snapshot()["active_llm_calls"], 2)
        release.set()
        for thread in first_threads + [third_thread]:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        values = control.snapshot()
        self.assertEqual(values["peak_active_llm_calls"], 2)
        self.assertEqual(values["active_llm_calls"], 0)

    def test_admission_timeout_does_not_spend_event_permit(self):
        control, telemetry = controller(llm=1, llm_timeout=0.03)
        holder_ready = threading.Event()
        release = threading.Event()

        def holder():
            with control.llm_slot("holder", 2):
                holder_ready.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(holder_ready.wait(timeout=1))
        repository = FakeRepository(1)
        event = EventContext("user-timeout", "user", 1)
        with bind_event(event):
            with llm_network_call(repository, "xai", "reply", control) as allowed:
                self.assertFalse(allowed)
        self.assertEqual(event.permit.call_count, 0)
        self.assertEqual(telemetry.snapshot()["active_llm_calls"], 1)
        release.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(control.snapshot()["llm_admission_timeouts"], 1)

    def test_autonomous_busy_is_nonblocking_and_does_not_spend_permit(self):
        control, _ = controller(llm=1, llm_timeout=2)
        repository = FakeRepository(9)
        with control.llm_slot("foreground", 1):
            event = EventContext("auto-busy", "autonomous", 9)
            started = time.monotonic()
            with bind_event(event):
                with llm_network_call(
                    repository, "xai", "autonomous", control
                ) as allowed:
                    self.assertFalse(allowed)
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(event.permit.call_count, 0)
        values = control.snapshot()
        self.assertEqual(values["autonomous_skipped_llm_busy"], 1)

    def test_provider_exception_releases_global_slot_but_not_event_permit(self):
        control, telemetry = controller(llm=1)
        repository = FakeRepository(1)
        event = EventContext("provider-error", "user", 1)
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            with bind_event(event):
                with llm_network_call(
                    repository, "xai", "reply", control
                ) as allowed:
                    self.assertTrue(allowed)
                    raise RuntimeError("provider failed")
        self.assertEqual(event.permit.call_count, 1)
        self.assertEqual(telemetry.snapshot()["active_llm_calls"], 0)
        next_event = EventContext("provider-next", "user", 1)
        with bind_event(next_event):
            with llm_network_call(
                repository, "xai", "reply", control
            ) as allowed:
                self.assertTrue(allowed)


class MediaAdmissionTests(unittest.TestCase):
    def test_cross_chat_media_is_serial_and_exception_safe(self):
        control, telemetry = controller(media=1, media_timeout=2)
        first_entered = threading.Event()
        release = threading.Event()
        second_entered = threading.Event()

        def first():
            with self.assertRaises(ValueError):
                with control.media_slot("media-a", 1) as admitted:
                    self.assertTrue(admitted)
                    first_entered.set()
                    release.wait(timeout=2)
                    raise ValueError("renderer")

        def second():
            with control.media_slot("media-b", 2) as admitted:
                self.assertTrue(admitted)
                second_entered.set()

        a = threading.Thread(target=first)
        b = threading.Thread(target=second)
        a.start()
        self.assertTrue(first_entered.wait(timeout=1))
        b.start()
        time.sleep(0.05)
        self.assertFalse(second_entered.is_set())
        release.set()
        a.join(timeout=2)
        b.join(timeout=2)
        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        values = telemetry.snapshot()
        self.assertEqual(values["peak_active_media_jobs"], 1)
        self.assertEqual(values["active_media_jobs"], 0)

    def test_nested_media_slot_is_one_physical_admission(self):
        control, telemetry = controller(media=1)
        with control.media_slot("manual", 1) as outer:
            self.assertTrue(outer)
            with control.media_slot("manual", 1) as inner:
                self.assertTrue(inner)
                self.assertEqual(inner.outcome, "reentrant")
                self.assertEqual(telemetry.snapshot()["active_media_jobs"], 1)
        self.assertEqual(telemetry.snapshot()["active_media_jobs"], 0)

    def test_autonomous_media_busy_skips_without_waiting(self):
        control, _ = controller(media=1, media_timeout=1)
        ready = threading.Event()
        release = threading.Event()

        def holder():
            with control.media_slot("user-media", 1):
                ready.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(ready.wait(timeout=1))
        started = time.monotonic()
        with control.media_slot(
            "auto-media", 2, background=True
        ) as admission:
            self.assertFalse(admission)
        self.assertLess(time.monotonic() - started, 0.1)
        release.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(
            control.snapshot()["autonomous_skipped_media_busy"], 1
        )

    def test_foreground_media_timeout_has_no_usage_and_slot_remains_reusable(self):
        control, telemetry = controller(media=1, media_timeout=0.02)
        ready = threading.Event()
        release = threading.Event()

        def holder():
            with control.media_slot("holder", 1):
                ready.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(ready.wait(timeout=1))
        with control.media_slot("explicit-meme", 2) as admission:
            self.assertFalse(admission)
            self.assertEqual(admission.outcome, "timeout")
        self.assertEqual(control.snapshot()["media_admission_timeouts"], 1)
        self.assertEqual(telemetry.snapshot()["active_media_jobs"], 1)
        release.set()
        thread.join(timeout=2)
        with control.media_slot("next-meme", 2) as admission:
            self.assertTrue(admission)
        self.assertEqual(telemetry.snapshot()["active_media_jobs"], 0)


class ChatArbitrationTests(unittest.TestCase):
    def test_same_chat_fifo_covers_snapshot_through_commit(self):
        control, _ = controller()
        first_in_provider = threading.Event()
        release = threading.Event()
        lifecycle = []
        lock = threading.Lock()

        def event_a():
            with control.chat_event_slot(10, "a") as admission:
                self.assertTrue(admission)
                with lock:
                    lifecycle.extend(("a:snapshot", "a:plan"))
                first_in_provider.set()
                release.wait(timeout=2)
                with lock:
                    lifecycle.extend(("a:delivery", "a:commit"))

        def event_b():
            first_in_provider.wait(timeout=2)
            with control.chat_event_slot(10, "b") as admission:
                self.assertTrue(admission)
                with lock:
                    lifecycle.extend(
                        ("b:snapshot", "b:plan", "b:delivery", "b:commit")
                    )

        a = threading.Thread(target=event_a)
        b = threading.Thread(target=event_b)
        a.start()
        b.start()
        self.assertTrue(first_in_provider.wait(timeout=1))
        time.sleep(0.05)
        self.assertNotIn("b:snapshot", lifecycle)
        release.set()
        a.join(timeout=2)
        b.join(timeout=2)
        self.assertEqual(lifecycle, [
            "a:snapshot", "a:plan", "a:delivery", "a:commit",
            "b:snapshot", "b:plan", "b:delivery", "b:commit",
        ])
        self.assertEqual(control.snapshot()["chat_gate_registry_size"], 0)

    def test_cross_chat_lifecycles_are_parallel(self):
        control, _ = controller()
        barrier = threading.Barrier(2)
        reached = []
        lock = threading.Lock()

        def work(chat_id):
            with control.chat_event_slot(chat_id, f"event-{chat_id}"):
                with lock:
                    reached.append(chat_id)
                barrier.wait(timeout=2)

        threads = [threading.Thread(target=work, args=(value,)) for value in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertCountEqual(reached, [1, 2])
        self.assertGreaterEqual(control.snapshot()["peak_chat_gate_active"], 2)

    def test_pending_can_only_be_consumed_once(self):
        control, _ = controller()
        state = {"pending": True, "consumers": [], "routes": []}
        first_started = threading.Event()
        release = threading.Event()

        def continuation(name):
            with control.chat_event_slot(7, name):
                if state["pending"]:
                    state["consumers"].append(name)
                    if name == "first":
                        first_started.set()
                        release.wait(timeout=2)
                    # Models R2 finalize only after delivery success.
                    state["pending"] = False
                else:
                    state["routes"].append((name, "normal_next_event"))

        first = threading.Thread(target=continuation, args=("first",))
        second = threading.Thread(target=continuation, args=("second",))
        first.start()
        self.assertTrue(first_started.wait(timeout=1))
        second.start()
        time.sleep(0.03)
        release.set()
        first.join(timeout=2)
        second.join(timeout=2)
        self.assertEqual(state["consumers"], ["first"])
        self.assertEqual(state["routes"], [("second", "normal_next_event")])

    def test_failed_delivery_keeps_pending_visible_to_next_event(self):
        control, _ = controller()
        state = {"pending": True, "seen": []}
        with control.chat_event_slot(8, "failed"):
            state["seen"].append(("failed", state["pending"]))
            # R2 abort: no pending mutation.
        with control.chat_event_slot(8, "next"):
            state["seen"].append(("next", state["pending"]))
        self.assertEqual(state["seen"], [("failed", True), ("next", True)])

    def test_cooldown_commit_is_visible_to_next_same_chat_event(self):
        control, _ = controller()
        state = {"cooldown": False, "selected": []}

        def choose(name):
            with control.chat_event_slot(4, name):
                if not state["cooldown"]:
                    state["selected"].append(name)
                    state["cooldown"] = True

        first = threading.Thread(target=choose, args=("first",))
        second = threading.Thread(target=choose, args=("second",))
        first.start()
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)
        self.assertEqual(len(state["selected"]), 1)

    def test_autonomous_same_chat_skips_when_foreground_busy(self):
        control, _ = controller()
        with control.chat_event_slot(3, "user"):
            started = time.monotonic()
            with control.chat_event_slot(
                3, "autonomous", background=True
            ) as admission:
                self.assertFalse(admission)
            self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(
            control.snapshot()["autonomous_skipped_chat_busy"], 1
        )

    def test_gate_is_released_after_lifecycle_exception(self):
        control, _ = controller()
        with self.assertRaises(RuntimeError):
            with control.chat_event_slot(11, "broken"):
                raise RuntimeError("snapshot/provider/delivery/commit")
        with control.chat_event_slot(11, "next") as admission:
            self.assertTrue(admission)
        self.assertEqual(control.snapshot()["chat_gate_registry_size"], 0)

    def test_shutdown_wakes_waiting_gate_and_rejects_new_optional_work(self):
        control, _ = controller()
        holder_ready = threading.Event()
        release = threading.Event()
        waiter_result = []

        def holder():
            with control.chat_event_slot(1, "holder"):
                holder_ready.set()
                release.wait(timeout=2)

        def waiter():
            with control.chat_event_slot(1, "waiter") as admission:
                waiter_result.append(bool(admission))

        first = threading.Thread(target=holder)
        second = threading.Thread(target=waiter)
        first.start()
        self.assertTrue(holder_ready.wait(timeout=1))
        second.start()
        time.sleep(0.03)
        control.shutdown()
        second.join(timeout=1)
        self.assertFalse(second.is_alive())
        self.assertEqual(waiter_result, [False])
        with control.chat_event_slot(2, "auto", background=True) as admission:
            self.assertFalse(admission)
        release.set()
        first.join(timeout=2)
        self.assertEqual(control.snapshot()["chat_gate_registry_size"], 0)


class ProductionLikeConcurrencyStressTests(unittest.TestCase):
    def test_200_events_five_chats_preserve_hard_limits_and_cleanup(self):
        control, telemetry = controller(llm=2, media=1, llm_timeout=2, media_timeout=2)
        categories = (
            ["direct"] * 80
            + ["local"] * 30
            + ["pending"] * 20
            + ["ordinary_ai"] * 20
            + ["autonomous"] * 15
            + ["media"] * 15
            + ["provider_failure"] * 10
            + ["delivery_renderer_failure"] * 10
        )
        active_by_chat = {}
        overlap = []
        per_event = {}
        lock = threading.Lock()

        def run(index, event_type):
            chat_id = index % 5
            event_id = f"stress-{index}"
            counts = {
                "event_type": event_type,
                "llm": 0,
                "media": 0,
                "plans": 0,
                "delivery": 0,
                "commit": 0,
                "abort": 0,
            }
            with control.chat_event_slot(chat_id, event_id) as chat_admission:
                if not chat_admission:
                    return
                with lock:
                    active_by_chat[chat_id] = active_by_chat.get(chat_id, 0) + 1
                    if active_by_chat[chat_id] > 1:
                        overlap.append((chat_id, index))
                try:
                    if event_type in {
                        "direct", "pending", "ordinary_ai", "autonomous",
                        "provider_failure",
                    }:
                        with control.llm_slot(
                            event_id, chat_id,
                            background=event_type == "autonomous",
                        ) as admitted:
                            if admitted:
                                counts["llm"] += 1
                                time.sleep(0.0005)
                    if event_type in {"media", "delivery_renderer_failure"}:
                        with control.media_slot(event_id, chat_id) as admitted:
                            if admitted:
                                counts["media"] += 1
                                time.sleep(0.0005)
                    # Provider failures resolve locally before one final plan;
                    # delivery/renderer failures abort that one plan.
                    optional_skipped = event_type == "autonomous" and counts["llm"] == 0
                    counts["plans"] = 0 if optional_skipped else 1
                    counts["delivery"] = counts["plans"]
                    if counts["plans"]:
                        if event_type == "delivery_renderer_failure":
                            counts["abort"] = 1
                        else:
                            counts["commit"] = 1
                finally:
                    with lock:
                        active_by_chat[chat_id] -= 1
            with lock:
                per_event[event_id] = counts

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [
                pool.submit(run, index, event_type)
                for index, event_type in enumerate(categories)
            ]
            for future in futures:
                future.result(timeout=10)

        self.assertEqual(len(per_event), 200)
        self.assertEqual(overlap, [])
        for event_id, counts in per_event.items():
            with self.subTest(event_id=event_id):
                self.assertLessEqual(counts["llm"], 1)
                self.assertLessEqual(counts["plans"], 1)
                self.assertLessEqual(counts["delivery"], 1)
                self.assertLessEqual(counts["commit"] + counts["abort"], 1)
        values = telemetry.snapshot()
        self.assertLessEqual(values["peak_active_llm_calls"], 2)
        self.assertLessEqual(values["peak_active_media_jobs"], 1)
        self.assertEqual(values["active_llm_calls"], 0)
        self.assertEqual(values["active_media_jobs"], 0)
        self.assertEqual(control.snapshot()["chat_gate_registry_size"], 0)


if __name__ == "__main__":
    unittest.main()
