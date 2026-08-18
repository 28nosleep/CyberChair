"""Per-event correlation, one-call LLM permits and runtime telemetry."""

import hashlib
import logging
import resource
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count


log = logging.getLogger(__name__)


def _compact_id(prefix, identity):
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def telegram_event_id(message):
    """Stable correlation id containing only Telegram technical identifiers."""
    chat_id = getattr(getattr(message, "chat", None), "id", 0)
    message_id = getattr(message, "message_id", getattr(message, "id", 0))
    return _compact_id("tg", f"telegram:user:{chat_id}:{message_id}")


def callback_event_id(chat_id, message_id, callback_id):
    return _compact_id(
        "cb", f"telegram:callback:{chat_id}:{message_id}:{callback_id}"
    )


def autonomous_event_id(chat_id, current):
    stamp = current.isoformat() if hasattr(current, "isoformat") else str(current)
    return _compact_id("auto", f"telegram:autonomous:{chat_id}:{stamp}")


def scheduled_event_id(chat_id, event_kind, event_key):
    """Stable content-free identity for one logical scheduled notification."""
    return _compact_id(
        "sched",
        f"telegram:scheduled:{int(chat_id)}:{event_kind}:{event_key}",
    )


def memory_event_id(chat_id, logical_day, start_cursor, end_row_id, sequence):
    """Durable summary-attempt identity containing no message/summary content."""
    return _compact_id(
        "mem",
        "memory:summary:"
        f"{int(chat_id)}:{logical_day}:{int(start_cursor)}:"
        f"{int(end_row_id)}:{int(sequence)}",
    )


_implicit_ids = count(1)


def implicit_event_id(event_type, chat_id):
    # Implicit ids are only for service/maintenance calls without a Telegram
    # update. Telegram user ids always use the deterministic function above.
    return _compact_id(
        "op", f"process:{event_type}:{chat_id}:{next(_implicit_ids)}"
    )


class EventLLMPermit:
    """Thread-safe one-way permit: failures never release the LLM budget."""

    def __init__(self, event_id):
        self.event_id = str(event_id)
        self._lock = threading.Lock()
        self._call = None
        self._denied = 0

    def try_acquire(self, call_type, provider_key):
        with self._lock:
            if self._call is not None:
                self._denied += 1
                return False
            self._call = {
                "call_type": str(call_type),
                "provider_key": str(provider_key),
            }
            return True

    @property
    def call_count(self):
        with self._lock:
            return 1 if self._call is not None else 0

    @property
    def denied_count(self):
        with self._lock:
            return self._denied

    @property
    def call(self):
        with self._lock:
            return dict(self._call) if self._call is not None else None


@dataclass
class EventContext:
    event_id: str
    event_type: str
    chat_id: int
    permit: EventLLMPermit = field(init=False)
    summary_requested: bool = False

    def __post_init__(self):
        self.permit = EventLLMPermit(self.event_id)


_current_event = ContextVar("cyberchair_event_context", default=None)


def current_event():
    return _current_event.get()


def current_event_id():
    event = current_event()
    return event.event_id if event is not None else None


@contextmanager
def bind_event(event):
    token = _current_event.set(event)
    try:
        yield event
    finally:
        _current_event.reset(token)


class RuntimeConcurrencyTelemetry:
    """Process-local gauges; R4 admission updates them around admitted work."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active_llm = 0
        self._peak_llm = 0
        self._active_media = 0
        self._peak_media = 0

    @contextmanager
    def llm_call(self):
        with self._lock:
            self._active_llm += 1
            self._peak_llm = max(self._peak_llm, self._active_llm)
        try:
            yield
        finally:
            with self._lock:
                self._active_llm -= 1

    @contextmanager
    def media_job(self):
        with self._lock:
            self._active_media += 1
            self._peak_media = max(self._peak_media, self._active_media)
        try:
            yield
        finally:
            with self._lock:
                self._active_media -= 1

    def snapshot(self):
        with self._lock:
            values = {
                "active_llm_calls": self._active_llm,
                "peak_active_llm_calls": self._peak_llm,
                "active_media_jobs": self._active_media,
                "peak_active_media_jobs": self._peak_media,
            }
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        values["peak_observed_rss_bytes"] = int(
            rss if sys.platform == "darwin" else rss * 1024
        )
        return values

    def reset_peaks_for_test(self):
        with self._lock:
            self._peak_llm = self._active_llm
            self._peak_media = self._active_media


runtime_concurrency = RuntimeConcurrencyTelemetry()


@contextmanager
def llm_network_call(repository, provider_key, call_type, controller=None):
    """Admit globally, then spend the event permit at network-attempt time."""
    event = current_event()
    implicit = event is None
    token = None
    if implicit:
        event = EventContext(
            implicit_event_id(call_type, repository.chat_id),
            "summary" if call_type == "summary" else "standalone",
            repository.chat_id,
        )
        token = _current_event.set(event)
    try:
        if controller is None:
            # Local import avoids making event identity depend on the admission
            # module. Production services inject the configured process singleton.
            from .concurrency import process_concurrency_controller
            controller = process_concurrency_controller(
                runtime_telemetry=runtime_concurrency
            )
        background = event.event_type in {"autonomous", "memory"}
        with controller.llm_slot(
            event.event_id, event.chat_id, background=background
        ) as admission:
            if not admission:
                repository.record_routing_event(
                    "llm_admission_busy" if background else "llm_admission_timeout",
                    event_id=event.event_id,
                    provider_key=provider_key,
                    call_type=call_type,
                )
                log.warning(
                    "LLM_ADMISSION_DENIED event_id=%s event_type=%s "
                    "provider_key=%s call_type=%s outcome=%s",
                    event.event_id, event.event_type, provider_key, call_type,
                    admission.outcome,
                )
                yield False
                return
            # Waiting for global capacity does not spend the R0 event budget.
            allowed = event.permit.try_acquire(call_type, provider_key)
            if not allowed:
                repository.record_routing_event(
                    "llm_permit_denied",
                    event_id=event.event_id,
                    provider_key=provider_key,
                    call_type=call_type,
                )
                log.error(
                    "LLM_EVENT_INVARIANT_DENIED event_id=%s event_type=%s "
                    "provider_key=%s call_type=%s llm_calls=%s",
                    event.event_id,
                    event.event_type,
                    provider_key,
                    call_type,
                    event.permit.call_count,
                )
                yield False
                return
            repository.record_routing_event(
                "llm_call_attempt",
                event_id=event.event_id,
                provider_key=provider_key,
                call_type=call_type,
            )
            log.info(
                "LLM_CALL_ATTEMPT event_id=%s event_type=%s provider_key=%s call_type=%s",
                event.event_id,
                event.event_type,
                provider_key,
                call_type,
            )
            # Failure/refusal/invalid output releases only the global slot. The
            # one-way EventLLMPermit stays spent exactly as in R0.
            yield True
    finally:
        if token is not None:
            _current_event.reset(token)
