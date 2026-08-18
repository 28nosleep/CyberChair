"""Bounded process-local shutdown coordination for the synchronous runtime."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShutdownReport:
    reason: str
    grace_seconds: float
    elapsed_ms: int
    drained: bool
    unfinished: tuple[str, ...]
    errors: tuple[str, ...]
    resources: dict


class ShutdownCoordinator:
    """Stop admission immediately, then wait for active work on one deadline."""

    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    _WORK_KINDS = (
        "foreground", "scheduled", "memory", "autonomous", "startup"
    )

    def __init__(self, grace_seconds=30.0, *, clock=None, diagnostics=None):
        self.grace_seconds = max(0.0, float(grace_seconds))
        self._clock = clock or time.monotonic
        self._diagnostics = diagnostics or (lambda: {})
        self._condition = threading.Condition(threading.RLock())
        self._drain_lock = threading.Lock()
        self._state = self.RUNNING
        self._reason = None
        self._requested_at = None
        self._deadline = None
        self._components = []
        self._cleanup = []
        self._work = {kind: 0 for kind in self._WORK_KINDS}
        self._errors = []
        self._stopped_logged = set()
        self._report = None
        self.shutdown_event = threading.Event()

    @property
    def state(self):
        with self._condition:
            return self._state

    @property
    def is_draining(self):
        return self.state != self.RUNNING

    @property
    def is_running(self):
        return self.state == self.RUNNING

    @property
    def deadline(self):
        with self._condition:
            return self._deadline

    def register_component(self, name, *, stop, stopped=None):
        """Register a non-blocking stop request and optional stopped probe."""
        component = {
            "name": str(name),
            "stop": stop,
            "stopped": stopped or (lambda: True),
            "requested": False,
        }
        request_now = False
        with self._condition:
            self._components.append(component)
            request_now = self._state != self.RUNNING
        if request_now:
            self._request_component_stop(component)
        return component

    def register_cleanup(self, name, callback):
        with self._condition:
            self._cleanup.append((str(name), callback))

    @contextmanager
    def work(self, kind):
        """Admit a top-level lifecycle only while RUNNING and track it."""
        kind = str(kind)
        if kind not in self._work:
            raise ValueError(f"unknown shutdown work kind: {kind}")
        admitted = False
        with self._condition:
            if self._state == self.RUNNING:
                self._work[kind] += 1
                admitted = True
        try:
            yield admitted
        finally:
            if admitted:
                with self._condition:
                    self._work[kind] -= 1
                    self._condition.notify_all()

    def _request_component_stop(self, component):
        with self._condition:
            if component["requested"]:
                return
            component["requested"] = True
        try:
            component["stop"]()
        except Exception as error:
            with self._condition:
                self._errors.append(
                    f"{component['name']}:{type(error).__name__}"
                )

    def request_shutdown(self, reason="runtime_exit"):
        """Idempotently enter DRAINING and issue only non-blocking stop hooks."""
        with self._condition:
            if self._state != self.RUNNING:
                return False
            now = self._clock()
            self._state = self.DRAINING
            self._reason = str(reason)
            self._requested_at = now
            self._deadline = now + self.grace_seconds
            components = tuple(self._components)
            self.shutdown_event.set()
            self._condition.notify_all()
        for component in components:
            self._request_component_stop(component)
        return True

    def remaining_seconds(self):
        deadline = self.deadline
        if deadline is None:
            return self.grace_seconds
        return max(0.0, deadline - self._clock())

    def _resource_snapshot(self):
        with self._condition:
            work = dict(self._work)
        try:
            external = dict(self._diagnostics() or {})
        except Exception as error:
            with self._condition:
                self._errors.append(f"diagnostics:{type(error).__name__}")
            external = {}
        return {
            "active_foreground": work["foreground"],
            "active_scheduled": work["scheduled"],
            "active_memory": work["memory"],
            "active_autonomous": work["autonomous"],
            "active_startup": work["startup"],
            "active_llm": int(external.get("active_llm_calls", 0) or 0),
            "active_media": int(external.get("active_media_jobs", 0) or 0),
            "chat_gate_registry_size": int(
                external.get("chat_gate_registry_size", 0) or 0
            ),
            "typing_refreshers": int(external.get("typing_refreshers", 0) or 0),
        }

    def _component_status(self):
        unfinished = []
        for component in tuple(self._components):
            try:
                stopped = bool(component["stopped"]())
            except Exception as error:
                stopped = False
                marker = f"{component['name']}_probe:{type(error).__name__}"
                with self._condition:
                    if marker not in self._errors:
                        self._errors.append(marker)
            if stopped:
                if component["name"] not in self._stopped_logged:
                    self._stopped_logged.add(component["name"])
                    log.info(
                        "shutdown_component_stopped component=%s",
                        component["name"],
                    )
            else:
                unfinished.append(component["name"])
        return unfinished

    @staticmethod
    def _active_names(resources):
        return [name for name, value in resources.items() if int(value or 0) > 0]

    def _snapshot_drain_state(self):
        resources = self._resource_snapshot()
        unfinished = self._component_status() + self._active_names(resources)
        return resources, tuple(sorted(set(unfinished)))

    def drain(self, reason="runtime_exit"):
        """Wait once against the common deadline, then always return boundedly."""
        self.request_shutdown(reason)
        with self._drain_lock:
            if self._report is not None:
                return self._report
            log.info(
                "shutdown_requested reason=%s grace_seconds=%.3f",
                self._reason, self.grace_seconds,
            )
            log.info("shutdown_draining reason=%s", self._reason)
            resources, unfinished = self._snapshot_drain_state()
            while unfinished:
                remaining = self.remaining_seconds()
                if remaining <= 0:
                    break
                with self._condition:
                    self._condition.wait(timeout=min(0.05, remaining))
                resources, unfinished = self._snapshot_drain_state()

            drained = not unfinished
            if drained:
                log.info("shutdown_drain_complete reason=%s", self._reason)
            else:
                log.warning(
                    "shutdown_grace_expired reason=%s unfinished=%s",
                    self._reason, ",".join(unfinished),
                )

            deadline = self.deadline
            for name, callback in tuple(self._cleanup):
                try:
                    callback(deadline)
                except Exception as error:
                    with self._condition:
                        self._errors.append(f"{name}:{type(error).__name__}")

            started = (
                self._requested_at
                if self._requested_at is not None else self._clock()
            )
            elapsed_ms = int(max(0.0, self._clock() - started) * 1000)
            with self._condition:
                self._state = self.STOPPED
                errors = tuple(self._errors)
                self._condition.notify_all()
            self._report = ShutdownReport(
                reason=self._reason or str(reason),
                grace_seconds=self.grace_seconds,
                elapsed_ms=elapsed_ms,
                drained=drained,
                unfinished=unfinished,
                errors=errors,
                resources=resources,
            )
            log.info(
                "shutdown_completed reason=%s elapsed_ms=%s drained=%s "
                "active_foreground=%s active_llm=%s active_media=%s "
                "active_scheduled=%s active_memory=%s chat_gate_registry_size=%s",
                self._report.reason,
                self._report.elapsed_ms,
                str(self._report.drained).lower(),
                resources["active_foreground"],
                resources["active_llm"],
                resources["active_media"],
                resources["active_scheduled"],
                resources["active_memory"],
                resources["chat_gate_registry_size"],
            )
            return self._report
