"""Immutable R5 summary jobs and the single-process maintenance runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .event_context import (
    EventContext,
    bind_event,
    implicit_event_id,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummaryMessage:
    row_id: int
    message_id: int | None
    user_id: int | None
    username: str | None
    text: str
    created_at: str


@dataclass(frozen=True)
class SummaryJob:
    event_id: str
    chat_id: int
    logical_day: str
    start_cursor: int
    end_message_row_id: int
    prior_summary_json: str | None
    messages: tuple[SummaryMessage, ...]
    created_at: str
    claim_expires_at: str
    attempt_sequence: int


@dataclass(frozen=True)
class SummaryFinalizeResult:
    status: str
    cursor_before: int
    cursor_after: int
    remaining_messages: int = 0
    candidates_promoted: int = 0
    candidates_promoted_pruned: int = 0
    candidates_stale_pruned: int = 0

    @property
    def committed(self):
        return self.status == "committed"


@dataclass(frozen=True)
class MemoryMaintenanceResult:
    status: str
    event_id: str | None = None
    llm_calls: int = 0
    finalize: SummaryFinalizeResult | None = None


class MemoryMaintenanceRunner:
    """Run at most one background summary attempt for one chat/tick."""

    def __init__(
        self, memory_service, concurrency_controller, provider_resolver,
        provider_allowed=None,
    ):
        self.memory = memory_service
        self.concurrency = concurrency_controller
        self.provider_resolver = provider_resolver
        self.provider_allowed = provider_allowed or (lambda chat_id: True)

    def run_once(self, repository, chat_id, current=None):
        chat_id = int(chat_id)
        now = self.memory._utc(current or self.memory._now())
        if not self.provider_allowed(chat_id):
            return MemoryMaintenanceResult("provider_not_allowed")
        provider = self.provider_resolver(chat_id)
        if provider is None or not getattr(provider, "available", False):
            repository.record_routing_event(
                "summary_provider_unavailable", call_type="summary"
            )
            return MemoryMaintenanceResult("provider_unavailable")

        # This optional gate protects only the short read/claim phase. It is
        # intentionally released before the provider call so a user event of
        # the same chat never waits for background summary network I/O.
        probe_id = implicit_event_id("memory_probe", chat_id)
        with self.concurrency.chat_event_slot(
            chat_id, probe_id, background=True
        ) as admission:
            if not admission:
                repository.record_routing_event(
                    "summary_deferred_chat_busy", event_id=probe_id,
                    call_type="summary",
                )
                return MemoryMaintenanceResult("chat_busy", probe_id)
            try:
                job, status = self.memory.prepare_summary_job(
                    repository, chat_id, now
                )
            except Exception:
                log.exception(
                    "SUMMARY_PREPARATION_FAILED event_id=%s chat_id=%s",
                    probe_id, chat_id,
                )
                return MemoryMaintenanceResult("preparation_failure", probe_id)
        if job is None:
            return MemoryMaintenanceResult(status)

        event = EventContext(job.event_id, "memory", chat_id)
        with bind_event(event):
            try:
                result = self.memory.execute_summary_job(
                    repository, job, provider, now
                )
            except Exception:
                # A post-provider SQLite failure is intentionally not followed
                # by another provider call.  The durable lease expires and the
                # exact range becomes claimable by a fresh maintenance event.
                log.exception(
                    "SUMMARY_FINALIZE_FAILED event_id=%s chat_id=%s",
                    job.event_id, chat_id,
                )
                return MemoryMaintenanceResult(
                    "storage_failure", job.event_id, event.permit.call_count
                )
        log.info(
            "MEMORY_EVENT event_id=%s chat_id=%s status=%s llm_calls=%s",
            job.event_id, chat_id, result.status, event.permit.call_count,
        )
        return MemoryMaintenanceResult(
            result.status, job.event_id, event.permit.call_count, result
        )
