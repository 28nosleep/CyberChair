"""Process-local admission control for synchronous CyberChair workers.

The outer lifecycle order is deliberately narrow and uniform:

    per-chat gate -> snapshot read session (closed) -> LLM/media slot
    -> Telegram delivery -> short repository commit

No repository connection is held while waiting for admission or during network
I/O. A provider may perform its existing short usage write before returning;
that lock order is chat -> resource -> repository and is never reversed.
Global resource slots cover only the scarce provider/render operation; they are
released before Telegram delivery and post-delivery commit.  ResponsePlan stays
plain immutable data and never carries a lock or semaphore.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic as _monotonic


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Admission:
    acquired: bool
    resource: str
    outcome: str
    wait_ms: float

    def __bool__(self):
        return self.acquired


@dataclass
class _ChatGate:
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.Lock())
    )
    next_ticket: int = 0
    serving: int = 0
    active: bool = False
    references: int = 0
    cancelled: set[int] = field(default_factory=set)


class ConcurrencyController:
    """Provider-neutral admission and FIFO per-chat lifecycle arbitration."""

    def __init__(
        self,
        llm_max_concurrency=2,
        media_max_concurrency=1,
        llm_timeout_seconds=5.0,
        media_timeout_seconds=15.0,
        runtime_telemetry=None,
    ):
        self.llm_max_concurrency = max(1, int(llm_max_concurrency))
        self.media_max_concurrency = max(1, int(media_max_concurrency))
        self.llm_timeout_seconds = max(0.0, float(llm_timeout_seconds))
        self.media_timeout_seconds = max(0.0, float(media_timeout_seconds))
        self._llm = threading.BoundedSemaphore(self.llm_max_concurrency)
        self._media = threading.BoundedSemaphore(self.media_max_concurrency)
        self._runtime = runtime_telemetry
        self._shutdown = threading.Event()
        self._registry_lock = threading.Lock()
        self._chat_gates: dict[int, _ChatGate] = {}
        self._metrics_lock = threading.Lock()
        self._llm_waits = deque(maxlen=4096)
        self._media_waits = deque(maxlen=4096)
        self._chat_waits = deque(maxlen=4096)
        self._llm_timeouts = 0
        self._media_timeouts = 0
        self._active_llm = 0
        self._peak_llm = 0
        self._active_media = 0
        self._peak_media = 0
        self._chat_active = 0
        self._peak_chat_active = 0
        self._autonomous_skipped_chat_busy = 0
        self._autonomous_skipped_llm_busy = 0
        self._autonomous_skipped_media_busy = 0
        self._local = threading.local()

    @property
    def shutting_down(self):
        return self._shutdown.is_set()

    def shutdown(self):
        """Reject new work and wake waiters; active bounded work may finish."""
        self._shutdown.set()
        with self._registry_lock:
            gates = tuple(self._chat_gates.values())
        for gate in gates:
            with gate.condition:
                gate.condition.notify_all()

    def _record_wait(self, resource, wait_ms):
        with self._metrics_lock:
            {
                "llm": self._llm_waits,
                "media": self._media_waits,
                "chat": self._chat_waits,
            }[resource].append(float(wait_ms))

    def _record_timeout(self, resource, background):
        with self._metrics_lock:
            if resource == "llm":
                self._llm_timeouts += 1
                if background:
                    self._autonomous_skipped_llm_busy += 1
            else:
                self._media_timeouts += 1
                if background:
                    self._autonomous_skipped_media_busy += 1

    @staticmethod
    def _acquire_until(semaphore, deadline, shutdown):
        while not shutdown.is_set():
            remaining = deadline - _monotonic()
            if remaining <= 0:
                return False
            if semaphore.acquire(timeout=min(remaining, 0.05)):
                return True
        return False

    @contextmanager
    def _resource_slot(
        self, resource, event_id, chat_id, *, background=False, timeout=None
    ):
        semaphore = self._llm if resource == "llm" else self._media
        default_timeout = (
            self.llm_timeout_seconds
            if resource == "llm" else self.media_timeout_seconds
        )
        # A manual image command acquires before download and then enters the
        # renderer. Re-entrancy avoids acquiring the same MEDIA_MAX=1 slot twice.
        depth_name = f"{resource}_depth"
        depth = int(getattr(self._local, depth_name, 0))
        if resource == "media" and depth:
            setattr(self._local, depth_name, depth + 1)
            try:
                yield Admission(True, resource, "reentrant", 0.0)
            finally:
                setattr(self._local, depth_name, depth)
            return

        started = _monotonic()
        acquired = False
        outcome = "shutdown" if self.shutting_down else "resource_busy"
        if not self.shutting_down:
            if background:
                acquired = semaphore.acquire(blocking=False)
            else:
                wait_for = default_timeout if timeout is None else max(0.0, float(timeout))
                acquired = self._acquire_until(
                    semaphore, _monotonic() + wait_for, self._shutdown
                )
            if acquired:
                outcome = "acquired"
            elif not self.shutting_down:
                outcome = "resource_busy" if background else "timeout"
        wait_ms = (_monotonic() - started) * 1000
        self._record_wait(resource, wait_ms)
        if not acquired:
            self._record_timeout(resource, background)
        log.info(
            "CONCURRENCY_ADMISSION event_id=%s chat_id=%s resource=%s "
            "outcome=%s wait_ms=%.3f background=%s",
            event_id, chat_id, resource, outcome, wait_ms, background,
        )
        if not acquired:
            yield Admission(False, resource, outcome, wait_ms)
            return

        setattr(self._local, depth_name, 1)
        with self._metrics_lock:
            if resource == "llm":
                self._active_llm += 1
                self._peak_llm = max(self._peak_llm, self._active_llm)
            else:
                self._active_media += 1
                self._peak_media = max(self._peak_media, self._active_media)
        gauge = (
            self._runtime.llm_call()
            if resource == "llm" and self._runtime is not None
            else self._runtime.media_job()
            if resource == "media" and self._runtime is not None
            else None
        )
        try:
            if gauge is None:
                yield Admission(True, resource, outcome, wait_ms)
            else:
                with gauge:
                    yield Admission(True, resource, outcome, wait_ms)
        finally:
            setattr(self._local, depth_name, 0)
            with self._metrics_lock:
                if resource == "llm":
                    self._active_llm -= 1
                else:
                    self._active_media -= 1
            semaphore.release()

    def llm_slot(self, event_id, chat_id, *, background=False, timeout=None):
        return self._resource_slot(
            "llm", event_id, chat_id, background=background, timeout=timeout
        )

    def media_slot(self, event_id, chat_id, *, background=False, timeout=None):
        return self._resource_slot(
            "media", event_id, chat_id, background=background, timeout=timeout
        )

    def _retain_gate(self, chat_id):
        key = int(chat_id)
        with self._registry_lock:
            gate = self._chat_gates.get(key)
            if gate is None:
                gate = _ChatGate()
                self._chat_gates[key] = gate
            gate.references += 1
            return key, gate

    def _release_gate_reference(self, key, gate):
        with self._registry_lock:
            gate.references -= 1
            if gate.references == 0 and not gate.active:
                if self._chat_gates.get(key) is gate:
                    self._chat_gates.pop(key, None)

    @staticmethod
    def _advance_cancelled(gate):
        while gate.serving in gate.cancelled:
            gate.cancelled.remove(gate.serving)
            gate.serving += 1

    @contextmanager
    def chat_event_slot(self, chat_id, event_id, *, background=False):
        """Serialize a complete decision/delivery/commit lifecycle per chat.

        Foreground tickets are FIFO in handler-arrival order. Optional
        autonomous work never queues behind an active or waiting user event.
        """
        key, gate = self._retain_gate(chat_id)
        started = _monotonic()
        ticket = None
        acquired = False
        outcome = "shutdown" if self.shutting_down else "chat_busy"
        try:
            with gate.condition:
                if background:
                    if not self.shutting_down and not gate.active and gate.next_ticket == gate.serving:
                        gate.active = True
                        acquired = True
                        outcome = "acquired"
                    else:
                        with self._metrics_lock:
                            self._autonomous_skipped_chat_busy += 1
                else:
                    ticket = gate.next_ticket
                    gate.next_ticket += 1
                    while not self.shutting_down:
                        if ticket == gate.serving and not gate.active:
                            gate.active = True
                            acquired = True
                            outcome = "acquired"
                            break
                        gate.condition.wait(timeout=0.05)
                    if not acquired:
                        gate.cancelled.add(ticket)
                        if not gate.active:
                            self._advance_cancelled(gate)
                        gate.condition.notify_all()

            wait_ms = (_monotonic() - started) * 1000
            self._record_wait("chat", wait_ms)
            log.info(
                "CONCURRENCY_ADMISSION event_id=%s chat_id=%s resource=chat "
                "outcome=%s wait_ms=%.3f background=%s",
                event_id, key, outcome, wait_ms, background,
            )
            if acquired:
                with self._metrics_lock:
                    self._chat_active += 1
                    self._peak_chat_active = max(
                        self._peak_chat_active, self._chat_active
                    )
            yield Admission(acquired, "chat", outcome, wait_ms)
        finally:
            if acquired:
                with gate.condition:
                    gate.active = False
                    with self._metrics_lock:
                        self._chat_active -= 1
                    if ticket is not None and gate.serving == ticket:
                        gate.serving += 1
                        self._advance_cancelled(gate)
                    gate.condition.notify_all()
            self._release_gate_reference(key, gate)

    @staticmethod
    def _percentile_50(values):
        if not values:
            return 0.0
        ordered = sorted(values)
        return float(ordered[(len(ordered) - 1) // 2])

    def snapshot(self):
        runtime = self._runtime.snapshot() if self._runtime is not None else {}
        with self._registry_lock:
            registry_size = len(self._chat_gates)
        with self._metrics_lock:
            values = {
                "llm_max_concurrency": self.llm_max_concurrency,
                "media_max_concurrency": self.media_max_concurrency,
                "llm_admission_wait_ms_p50": self._percentile_50(self._llm_waits),
                "llm_admission_wait_ms_max": max(self._llm_waits, default=0.0),
                "media_admission_wait_ms_p50": self._percentile_50(self._media_waits),
                "media_admission_wait_ms_max": max(self._media_waits, default=0.0),
                "chat_gate_wait_ms_p50": self._percentile_50(self._chat_waits),
                "chat_gate_wait_ms_max": max(self._chat_waits, default=0.0),
                "llm_admission_timeouts": self._llm_timeouts,
                "media_admission_timeouts": self._media_timeouts,
                "active_llm_calls": self._active_llm,
                "peak_active_llm_calls": self._peak_llm,
                "active_media_jobs": self._active_media,
                "peak_active_media_jobs": self._peak_media,
                "chat_gate_active": self._chat_active,
                "peak_chat_gate_active": self._peak_chat_active,
                "chat_gate_registry_size": registry_size,
                "autonomous_skipped_chat_busy": self._autonomous_skipped_chat_busy,
                "autonomous_skipped_llm_busy": self._autonomous_skipped_llm_busy,
                "autonomous_skipped_media_busy": self._autonomous_skipped_media_busy,
                "shutting_down": self.shutting_down,
            }
        values.update(runtime)
        return values


_process_controller = None
_process_controller_lock = threading.Lock()


def process_concurrency_controller(settings=None, runtime_telemetry=None):
    """Return the one controller shared by all production service instances."""
    global _process_controller
    with _process_controller_lock:
        if _process_controller is None:
            _process_controller = ConcurrencyController(
                llm_max_concurrency=getattr(settings, "llm_max_concurrency", 2),
                media_max_concurrency=getattr(settings, "media_max_concurrency", 1),
                llm_timeout_seconds=getattr(
                    settings, "llm_admission_timeout_seconds", 5.0
                ),
                media_timeout_seconds=getattr(
                    settings, "media_admission_timeout_seconds", 15.0
                ),
                runtime_telemetry=runtime_telemetry,
            )
        return _process_controller
