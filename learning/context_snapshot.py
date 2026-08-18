"""Immutable per-event context view loaded through one bounded read session."""

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Mapping

from .memory_service import normalize_summary


log = logging.getLogger(__name__)


def format_context_snapshot_read_diagnostic(before_profiles, after_profiles):
    """Format synthetic BEFORE/AFTER query profiles without repository data."""
    before = tuple(before_profiles)
    after = tuple(after_profiles)

    def average(values, key):
        return (
            sum(float(item.get(key, 0)) for item in values) / len(values)
            if values else 0.0
        )

    lines = [
        "CONTEXT SNAPSHOT",
        f"events: {len(after)}",
        f"avg_db_connections_before: {average(before, 'connections'):.2f}",
        f"avg_db_connections_after: {average(after, 'connections'):.2f}",
    ]
    for key, label in (
        ("summary_reads", "summary_reads_per_event"),
        ("dialogue_reads", "dialogue_reads_per_event"),
        ("stable_memory_reads", "stable_memory_reads_per_event"),
        ("generated_history_reads", "recent_generated_reads_per_event"),
    ):
        lines.append(
            f"{label}: before={average(before, key):.2f} "
            f"after={average(after, key):.2f}"
        )
    return "\n".join(lines)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _frozen_row(value):
    return _freeze(dict(value or {}))


@dataclass(frozen=True)
class SnapshotMetrics:
    build_ms: float
    db_connections: int
    queries: int
    dialogue_rows: int
    memory_rows: int


@dataclass(frozen=True)
class SnapshotIdentity:
    event_id: str
    chat_id: int
    message_id: int | None = None
    user_id: int | None = None


@dataclass(frozen=True)
class MediaContext:
    recent_usage: tuple[Mapping, ...] = ()
    tagged_gifs: tuple[Mapping, ...] = ()
    tagged_stickers: tuple[Mapping, ...] = ()


@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable, event-local view of bounded repository state."""

    event_id: str
    chat_id: int
    message_id: int | None
    logical_day: str
    built_at: datetime
    recent_dialogue: tuple[Mapping, ...]
    latest_message: Mapping | None
    message_count: int
    current_summary: Mapping
    stable_memories: tuple[str, ...]
    recent_generated: tuple[Mapping, ...]
    resolved_settings: Mapping
    pending: Mapping | None
    sections_loaded: tuple[str, ...]
    metrics: SnapshotMetrics
    media: MediaContext | None = None

    def setting(self, key, default=None):
        return self.resolved_settings.get(key, default)

    @property
    def recent_generated_texts(self):
        return tuple(row.get("text", "") for row in self.recent_generated)

    def with_media(self, media, *, queries, build_ms):
        sections = tuple(dict.fromkeys((*self.sections_loaded, "media")))
        return replace(
            self,
            media=media,
            sections_loaded=sections,
            metrics=replace(
                self.metrics,
                build_ms=self.metrics.build_ms + build_ms,
                db_connections=self.metrics.db_connections + 1,
                queries=self.metrics.queries + queries,
            ),
        )


class ContextSnapshotBuilder:
    """Loads shared event reads; it contains no routing and never calls an LLM."""

    def __init__(self, settings, memory_service):
        self.settings = settings
        self.memory = memory_service

    @staticmethod
    def _utc(value):
        value = value or datetime.now(timezone.utc)
        return (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None else value.astimezone(timezone.utc)
        )

    def build(self, event, repository, current=None):
        started = time.perf_counter()
        now = self._utc(current or self.memory._now())
        logical_day = self.memory.logical_day(now)
        cutoff = (
            now - timedelta(minutes=self.settings.short_memory_minutes)
        ).isoformat()
        values = repository.load_context_snapshot_inputs(
            since_iso=cutoff,
            dialogue_limit=min(50, self.settings.max_messages_per_chat),
            logical_day=logical_day,
            generated_limit=max(0, self.settings.max_messages_per_chat),
            stable_limit=20,
            pending_user_id=getattr(event, "user_id", None),
        )
        query_count = 8 if getattr(event, "user_id", None) is not None else 7
        snapshot = ContextSnapshot(
            event_id=event.event_id,
            chat_id=int(event.chat_id),
            message_id=getattr(event, "message_id", None),
            logical_day=logical_day,
            built_at=now,
            recent_dialogue=tuple(
                _frozen_row(row) for row in values["recent_dialogue"]
            ),
            latest_message=(
                _frozen_row(values["latest_message"])
                if values["latest_message"] else None
            ),
            message_count=values["message_count"],
            current_summary=_frozen_row(
                normalize_summary(values["current_summary"] or {})
            ),
            stable_memories=tuple(values["stable_memories"]),
            recent_generated=tuple(
                _frozen_row(row) for row in values["recent_generated"]
            ),
            resolved_settings=_frozen_row(values["settings"]),
            pending=(
                _frozen_row(values["pending"])
                if values["pending"] else None
            ),
            sections_loaded=(
                "dialogue", "summary", "stable_memory", "recent_generated",
                "settings",
                *(("pending",) if getattr(event, "user_id", None) is not None else ()),
            ),
            metrics=SnapshotMetrics(
                build_ms=(time.perf_counter() - started) * 1000,
                db_connections=1,
                queries=query_count,
                dialogue_rows=len(values["recent_dialogue"]),
                memory_rows=len(values["stable_memories"]),
            ),
        )
        log.info(
            "CONTEXT_SNAPSHOT event_id=%s build_ms=%.3f db_connections=%s "
            "queries=%s sections=%s dialogue_rows=%s memory_rows=%s",
            snapshot.event_id, snapshot.metrics.build_ms,
            snapshot.metrics.db_connections, snapshot.metrics.queries,
            ",".join(snapshot.sections_loaded),
            snapshot.metrics.dialogue_rows, snapshot.metrics.memory_rows,
        )
        return snapshot

    def enrich_media(self, snapshot, repository):
        if snapshot.media is not None:
            return snapshot
        started = time.perf_counter()
        values = repository.load_media_context_inputs()
        media = MediaContext(
            recent_usage=tuple(_frozen_row(row) for row in values["recent_usage"]),
            tagged_gifs=tuple(_frozen_row(row) for row in values["tagged_gifs"]),
            tagged_stickers=tuple(
                _frozen_row(row) for row in values["tagged_stickers"]
            ),
        )
        enriched = snapshot.with_media(
            media, queries=3,
            build_ms=(time.perf_counter() - started) * 1000,
        )
        log.info(
            "CONTEXT_SNAPSHOT_MEDIA event_id=%s usage_rows=%s tagged_rows=%s",
            snapshot.event_id, len(media.recent_usage),
            len(media.tagged_gifs) + len(media.tagged_stickers),
        )
        return enriched
