"""Foreground memory ingestion and R5 maintenance facade.

This boundary exposes cohesive memory entry points while leaving snapshot
construction, repository storage and background SummaryJob execution in their
existing specialized components.
"""

import logging
from datetime import datetime, timezone

from .preprocessing import normalize_spaces, rejection_reason


log = logging.getLogger("learning.service")


class MemoryFacade:
    """Own ingestion/scheduling glue; never performs foreground summary LLM."""

    def __init__(
        self, *, settings, repository, normalize_event, enabled, triggers,
        invalidate_generation, memory_maintenance, troll_mode,
        provider_name, provider_available, activity_percent,
        autonomous_enabled, media_enabled, relationship_model,
        moment_detector, evidence_engine,
    ):
        self.settings=settings
        self.repository=repository
        self._normalized_event=normalize_event
        self._enabled=enabled
        self.triggers=triggers
        self.invalidate_generation=invalidate_generation
        self.memory_maintenance=memory_maintenance
        self.troll_mode=troll_mode
        self.llm_provider_name=provider_name
        self.provider_available=provider_available
        self.activity_percent=activity_percent
        self.autonomous_enabled=autonomous_enabled
        self.media_enabled=media_enabled
        self.relationship_model=relationship_model
        self.moment_detector=moment_detector
        self.evidence_engine=evidence_engine

    def ingest(self, message, refresh_memory=True):
        event = self._normalized_event(message)
        chat_id = event.chat_id
        if not self._enabled(chat_id, "learning"):
            return False, "disabled"
        if event.user_id is None or event.user_is_bot:
            return False, "bot"
        reason = rejection_reason(
            event.effective_text, self.settings.max_stored_text_length
        )
        if reason:
            log.debug("Learning message rejected chat=%s reason=%s", chat_id, reason)
            return False, reason
        repository = self.repository(chat_id)
        inserted, insert_reason = repository.add_message(
            event.message_id,
            event.user_id,
            event.username,
            normalize_spaces(event.effective_text),
            event.timestamp,
            event.reply_to_message_id,
            event.reply_to_message_id is not None,
            return_reason=True,
        )
        if inserted:
            recent_rows = repository.recent_messages(
                self.moment_detector.max_messages
            )
            moments = self.moment_detector.detect(
                recent_rows, event.message_id
            )
            self.relationship_model.observe(repository, event, recent_rows)
            self.evidence_engine.capture_message(repository, event, moments)
            self.triggers.note_message(chat_id)
            self.invalidate_generation(chat_id)
            count = repository.count()
            log.info("Learning message accepted chat=%s count=%s", chat_id, count)
            if count == self.settings.min_training_messages:
                log.info("Minimum training volume reached chat=%s", chat_id)
            if refresh_memory:
                pending_stamp = (
                    event.timestamp or datetime.now(timezone.utc)
                ).isoformat()
                if repository.mark_summary_pending(pending_stamp):
                    repository.record_routing_event(
                        "summary_due", event_id=event.event_id,
                        call_type="summary",
                    )
        return inserted, None if inserted else insert_reason or "duplicate"

    def run_memory_maintenance(self, chat_id, current=None):
        """Run at most one separately correlated background summary event."""
        repository = self.repository(chat_id)
        repository.run_persistence_maintenance(
            current,
            llm_retention_days=self.settings.llm_call_retention_days,
            routing_retention_days=self.settings.routing_event_retention_days,
            scheduled_retention_days=self.settings.scheduled_event_retention_days,
            evidence_retention_days=self.settings.evidence_retention_days,
            interval_seconds=(
                self.settings.persistence_maintenance_interval_seconds
            ),
        )
        return self.memory_maintenance.run_once(
            repository, chat_id, current=current
        )

    def persistence_diagnostics(self, chat_id):
        return self.repository(chat_id).persistence_diagnostics()

    def format_persistence_diagnostics(self, chat_id):
        report = self.persistence_diagnostics(chat_id)
        rows = report["rows_by_table"]
        return "\n".join((
            "DB PERSISTENCE",
            f"schema_version: {report['schema_version']}",
            f"db_size_bytes: {report['db_size_bytes']}",
            f"page_count: {report['page_count']}",
            f"freelist_count: {report['freelist_count']}",
            "rows_by_table: " + ", ".join(
                f"{name}={rows[name]}" for name in sorted(rows)
            ),
            f"llm_calls_oldest: {report['oldest']['llm_calls']}",
            f"routing_events_oldest: {report['oldest']['routing_events']}",
            f"daily_summaries_oldest: {report['oldest']['daily_summaries']}",
            f"candidate_oldest: {report['oldest']['memory_candidates']}",
            f"migration_current: {str(report['migration_current']).lower()}",
        ))

    def status(self, chat_id):
        repository = self.repository(chat_id)
        count = repository.count()
        total = repository.statistics()["total_messages"]
        return {
            "count": total,
            "short_memory_count": count,
            "ready": count >= self.settings.min_training_messages,
            "learning": self._enabled(chat_id, "learning"),
            "talk": self._enabled(chat_id, "talk"),
            "troll_mode": self.troll_mode(chat_id),
            "provider": self.llm_provider_name(chat_id),
            "provider_available": self.provider_available(chat_id),
            "llm": self.provider_available(chat_id),
            "activity_percent": self.activity_percent(chat_id),
            "autonomous_enabled": self.autonomous_enabled(chat_id),
            "media_enabled": self.media_enabled(chat_id),
        }

    def memory_lifecycle_diagnostics(self, chat_id, current=None):
        return self.repository(chat_id).memory_lifecycle_report(current)

    def format_memory_lifecycle_diagnostics(self, chat_id, current=None):
        report = self.memory_lifecycle_diagnostics(chat_id, current)
        return "\n".join((
            "MEMORY LIFECYCLE",
            f"chats_with_backlog: {int(report['backlog_messages'] > 0)}",
            f"total_unsummarized_messages: {report['backlog_messages']}",
            f"max_backlog_messages: {report['backlog_messages']}",
            "oldest_unsummarized_age: "
            f"{report['oldest_unsummarized_age_seconds']}",
            f"summary_attempts: {report['summary_attempts']}",
            f"summary_success: {report['summary_success']}",
            f"summary_failures: {report['summary_failures']}",
            "summary_resource_deferrals: "
            f"{report['summary_resource_deferrals']}",
            "max_summary_calls_per_memory_event: "
            f"{report['max_summary_calls_per_memory_event']}",
            f"foreground_summary_calls: {report['foreground_summary_calls']}",
            f"active_candidates: {report['active_candidates']}",
            f"promoted_pruned: {report['promoted_pruned']}",
            f"stale_pruned: {report['stale_pruned']}",
        ))
