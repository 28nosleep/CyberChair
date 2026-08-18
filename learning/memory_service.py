import json
import logging
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .llm_prompts import build_summarize_request
from .event_context import current_event_id, llm_network_call
from .memory_maintenance import (
    MemoryMaintenanceRunner,
    SummaryFinalizeResult,
    SummaryJob,
    SummaryMessage,
)
from .preprocessing import lexical_stem, normalize_spaces, significant_words
from .repository import memories_are_similar, normalize_memory


log = logging.getLogger(__name__)


def _compact_list(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    result = []
    for value in values:
        clean = normalize_spaces(str(value))[:200]
        if clean and not any(memories_are_similar(clean, old) for old in result):
            result.append(clean)
        if len(result) >= 6:
            break
    return result


def normalize_summary(summary):
    """Convert old and new provider payloads to one compact canonical shape."""
    source = summary if isinstance(summary, Mapping) else {}
    mood = source.get("current_mood", source.get("mood", ""))
    if isinstance(mood, (list, tuple)):
        mood = ", ".join(str(item) for item in mood if str(item).strip())
    if mood is None:
        mood = ""
    candidates = source.get(
        "memory_candidates", source.get("stable_memory_candidates", [])
    )
    return {
        "main_topics": _compact_list(source.get("main_topics", source.get("topics", []))),
        "current_mood": normalize_spaces(str(mood))[:200],
        "active_conflicts": _compact_list(source.get("active_conflicts", [])),
        "inside_jokes": _compact_list(
            source.get("inside_jokes", source.get("local_memes", []))
        ),
        "frequently_mentioned_people": _compact_list(
            source.get("frequently_mentioned_people", source.get("people", []))
        ),
        "notable_events": _compact_list(
            source.get("notable_events", source.get("events", []))
        ),
        "repeated_phrases": _compact_list(source.get("repeated_phrases", [])),
        "callback_jokes": _compact_list(source.get("callback_jokes", [])),
        "memory_candidates": _compact_list(candidates),
    }


class MemoryService:
    def __init__(self, settings, llm_provider, speaker_name, clock=None,
                 provider_resolver=None, concurrency_controller=None):
        self.settings = settings
        self.llm_provider = llm_provider
        self.provider_resolver = provider_resolver or (lambda chat_id: llm_provider)
        self.concurrency = concurrency_controller
        self._speaker_name = speaker_name
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timezone = ZoneInfo(settings.timezone_name)

    def _now(self):
        current = self._clock()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current

    def logical_day(self, current=None):
        return (current or self._now()).astimezone(self._timezone).date().isoformat()

    @staticmethod
    def _utc(value):
        value = value or datetime.now(timezone.utc)
        return (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None else value.astimezone(timezone.utc)
        )

    @staticmethod
    def _relevant_values(values, terms, limit=2):
        result = []
        for index, value in enumerate(values or ()):
            clean = normalize_spaces(str(value))[:200]
            value_terms = set(significant_words(clean))
            stems = {lexical_stem(term) for term in terms if len(term) >= 3}
            value_stems = {lexical_stem(term) for term in value_terms if len(term) >= 3}
            exact = len(value_terms & terms)
            stem = len(stems & value_stems)
            if clean and terms and (exact or stem):
                result.append((exact * 6 + stem * 2, -index, clean))
        return [item[2] for item in sorted(result, reverse=True)[:limit]]

    def relevant_memory(self, repository, context=None, dominant_topic=None,
                        conversation_type=None):
        """Select facts that can affect this turn instead of serialising all memory."""
        return self.relevant_memory_from_values(
            normalize_summary(
                repository.summary_for_day(self.logical_day()) or {}
            ),
            repository.stable_memories(20),
            context, dominant_topic, conversation_type,
        )

    def relevant_memory_from_values(
        self, summary, stable_memories, context=None, dominant_topic=None,
        conversation_type=None,
    ):
        """Pure relevance selection over already-loaded snapshot values."""
        del conversation_type  # Retained for compatibility with existing scoring API.
        seed = f"{context or ''} {dominant_topic or ''}"
        terms = set(significant_words(seed))
        summary = normalize_summary(summary or {})
        relevant_summary = {}
        for key in (
            "main_topics", "active_conflicts", "inside_jokes",
            "frequently_mentioned_people", "notable_events", "repeated_phrases",
            "callback_jokes",
        ):
            selected = self._relevant_values(summary.get(key), terms)
            if selected:
                relevant_summary[key] = selected
        # Mood only changes a reply when this is clearly a mood/topic match.
        mood = summary.get("current_mood", "")
        if mood and terms & set(re.findall(r"[\wёЁ-]{4,}", mood.casefold())):
            relevant_summary["current_mood"] = mood
        return {
            "day_summary": relevant_summary or None,
            "stable_chat_memory": self._relevant_values(
                stable_memories, terms
            ),
        }

    def short_term_context(self, repository, context=None, max_chars=5000,
                           max_messages=None, dominant_topic=None,
                           conversation_type=None):
        rows = self.short_term_rows(repository)
        terms = set(re.findall(r"[\wёЁ-]{4,}", (context or "").casefold()))
        scored = []
        for position, row in enumerate(rows):
            row_terms = set(re.findall(r"[\wёЁ-]{4,}", row["text"].casefold()))
            score = len(terms & row_terms) * 10 + position / max(1, len(rows))
            scored.append((score, position, row))
        selected = sorted(
            sorted(scored, reverse=True)[: min(
                max_messages or self.settings.context_message_limit,
                self.settings.context_message_limit,
                20,
            )],
            key=lambda item: item[1],
        )
        lines = []
        length = 0
        for _, _, row in selected:
            text = normalize_spaces(row["text"])
            if not text:
                continue
            line = f"{self._speaker_name(row)}: {text[:350]}"
            if length + len(line) > max_chars:
                continue
            lines.append(line)
            length += len(line)
        memory = self.relevant_memory(
            repository, context, dominant_topic, conversation_type
        )
        header = "Сжатая память чата: " + json.dumps(
            memory, ensure_ascii=False, separators=(",", ":")
        )[:2200]
        return header + (
            "\nПоследние релевантные реплики:\n" + "\n".join(lines) if lines else ""
        )

    def short_term_context_from_snapshot(
        self, snapshot, context=None, max_chars=5000, max_messages=None,
        dominant_topic=None, conversation_type=None, relevant_memory=None,
    ):
        return self._short_term_context_from_values(
            snapshot.recent_dialogue,
            snapshot.current_summary,
            snapshot.stable_memories,
            context, max_chars, max_messages, dominant_topic, conversation_type,
            relevant_memory,
        )

    def _short_term_context_from_values(
        self, rows, summary, stable_memories, context=None, max_chars=5000,
        max_messages=None, dominant_topic=None, conversation_type=None,
        relevant_memory=None,
    ):
        terms = set(re.findall(r"[\wёЁ-]{4,}", (context or "").casefold()))
        scored = []
        for position, row in enumerate(rows):
            row_terms = set(re.findall(r"[\wёЁ-]{4,}", row["text"].casefold()))
            score = len(terms & row_terms) * 10 + position / max(1, len(rows))
            scored.append((score, position, row))
        selected = sorted(
            sorted(scored, reverse=True)[: min(
                max_messages or self.settings.context_message_limit,
                self.settings.context_message_limit,
                20,
            )],
            key=lambda item: item[1],
        )
        lines = []
        length = 0
        for _, _, row in selected:
            text = normalize_spaces(row["text"])
            if not text:
                continue
            line = f"{self._speaker_name(row)}: {text[:350]}"
            if length + len(line) > max_chars:
                continue
            lines.append(line)
            length += len(line)
        memory = relevant_memory
        if memory is None:
            memory = self.relevant_memory_from_values(
                summary, stable_memories, context, dominant_topic,
                conversation_type,
            )
        header = "Сжатая память чата: " + json.dumps(
            memory, ensure_ascii=False, separators=(",", ":")
        )[:2200]
        return header + (
            "\nПоследние релевантные реплики:\n" + "\n".join(lines) if lines else ""
        )

    def short_term_rows(self, repository):
        cutoff = (
            self._now() - timedelta(minutes=self.settings.short_memory_minutes)
        ).isoformat()
        return repository.short_term_dialogue(
            cutoff,
            min(50, self.settings.max_messages_per_chat),
        )

    def _dialogue_fragment(self, rows):
        return "\n".join(
            f"{self._speaker_name({**row, 'speaker': 'user'})}: "
            f"{normalize_spaces(row['text'])[:350]}"
            for row in rows
            if normalize_spaces(row["text"])
        )

    def _candidate_observations(self, summary, rows, legacy_evidence=False):
        observations = []
        normalized_messages = [normalize_memory(row["text"]) for row in rows]
        for candidate in summary.get("memory_candidates", []):
            if any(memories_are_similar(candidate, old[0]) for old in observations):
                continue
            core = re.split(r"\s+[—–-]\s+", candidate, maxsplit=1)[0]
            normalized_core = normalize_memory(core)
            core_words = set(normalized_core.split())
            evidence = sum(
                bool(
                    normalized_core
                    and (
                        normalized_core in message
                        or (
                            core_words
                            and core_words.issubset(set(message.split()))
                        )
                    )
                )
                for message in normalized_messages
            )
            observations.append(
                (candidate, min(2, max(1, evidence)) if legacy_evidence else 1)
            )
        return observations

    def _row_logical_day(self, row):
        try:
            created = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            created = self._now()
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return created.astimezone(self._timezone).date().isoformat()

    def prepare_summary_job(self, repository, chat_id, current=None):
        """Build and lease one immutable, day-local, cursor-bounded job."""
        now = self._utc(current or self._now())
        backlog = repository.summary_backlog_state(now)
        if backlog["backlog_messages"] <= 0:
            return None, "idle"
        if backlog["pending_since"] is None:
            repository.mark_summary_pending(now.isoformat())
            backlog = repository.summary_backlog_state(now)
        pending_since = datetime.fromisoformat(backlog["pending_since"])
        if pending_since.tzinfo is None:
            pending_since = pending_since.replace(tzinfo=timezone.utc)
        rows = repository.messages_after(
            backlog["last_message_row_id"], self.settings.summary_batch_messages
        )
        if not rows:
            return None, "idle"
        first_day = self._row_logical_day(rows[0])
        due = (
            backlog["backlog_messages"] >= self.settings.summary_message_interval
            or (now - pending_since).total_seconds()
            >= self.settings.summary_time_interval
            or first_day < self.logical_day(now)
        )
        if not due:
            return None, "not_due"
        repository.record_routing_event("summary_due", call_type="summary")

        selected = []
        used_chars = 0
        for row in rows:
            if self._row_logical_day(row) != first_day:
                break
            rendered_chars = min(350, len(normalize_spaces(row["text"]))) + 32
            if selected and used_chars + rendered_chars > self.settings.summary_batch_chars:
                break
            selected.append(row)
            used_chars += rendered_chars
        if not selected:
            return None, "empty"
        claim, status = repository.claim_summary_range(
            backlog["last_message_row_id"], selected[-1]["id"], first_day,
            now, self.settings.summary_claim_lease_seconds,
        )
        if claim is None:
            if status == "backoff":
                repository.record_routing_event(
                    "summary_deferred_backoff", call_type="summary"
                )
            return None, status
        previous = repository.summary_for_day(first_day)
        previous_json = (
            json.dumps(
                normalize_summary(previous), ensure_ascii=False,
                separators=(",", ":"),
            )
            if previous else None
        )
        job = SummaryJob(
            event_id=claim["event_id"],
            chat_id=int(chat_id),
            logical_day=first_day,
            start_cursor=int(backlog["last_message_row_id"]),
            end_message_row_id=int(selected[-1]["id"]),
            prior_summary_json=previous_json,
            messages=tuple(
                SummaryMessage(
                    row_id=int(row["id"]),
                    message_id=row.get("message_id"),
                    user_id=row.get("user_id"),
                    username=row.get("username"),
                    text=str(row.get("text") or ""),
                    created_at=str(row.get("created_at") or ""),
                )
                for row in selected
            ),
            created_at=claim["created_at"],
            claim_expires_at=claim["claim_expires_at"],
            attempt_sequence=claim["attempt_sequence"],
        )
        repository.record_routing_event(
            "summary_job_created", event_id=job.event_id, call_type="summary"
        )
        repository.record_routing_event(
            "summary_claimed", event_id=job.event_id, call_type="summary"
        )
        return job, "claimed"

    def execute_summary_job(self, repository, job, provider, current=None):
        """Call the provider once, then conditionally finalize the fixed range."""
        now = self._utc(current or self._now())
        rows = [
            {
                "id": item.row_id,
                "message_id": item.message_id,
                "user_id": item.user_id,
                "username": item.username,
                "text": item.text,
                "created_at": item.created_at,
            }
            for item in job.messages
        ]
        dialogue = self._dialogue_fragment(rows)
        previous = (
            json.loads(job.prior_summary_json)
            if job.prior_summary_json else None
        )
        safety_identifier = __import__("hashlib").sha256(
            f"cyberchair-memory:{job.chat_id}".encode("utf-8")
        ).hexdigest()[:32]
        request = build_summarize_request(
            dialogue, previous, safety_identifier,
            max_output_tokens=self.settings.summary_max_output_tokens,
        )
        provider_key = str(getattr(
            provider, "provider_key", getattr(provider, "provider_label", "injected")
        )).casefold()
        try:
            with llm_network_call(
                repository, provider_key, "summary", self.concurrency
            ) as allowed:
                if not allowed:
                    repository.release_summary_claim(job.event_id)
                    repository.record_routing_event(
                        "summary_deferred_resource_busy", event_id=job.event_id,
                        provider_key=provider_key, call_type="summary",
                    )
                    return SummaryFinalizeResult(
                        "resource_deferred", job.start_cursor, job.start_cursor
                    )
                repository.record_routing_event(
                    "summary_attempt", event_id=job.event_id,
                    provider_key=provider_key, call_type="summary",
                )
                summary = provider.summarize(replace(request, metadata={
                    "chat_id": job.chat_id,
                    "call_type": "summary",
                    "purpose": "summary",
                    "event_id": current_event_id(),
                }))
        except Exception:
            log.exception(
                "SUMMARY_PROVIDER_FAILED event_id=%s provider=%s",
                job.event_id, provider_key,
            )
            summary = None
        if not isinstance(summary, dict) or not any(
            key in summary for key in (
                "main_topics", "topics", "current_mood", "mood",
                "memory_candidates", "stable_memory_candidates",
            )
        ):
            repository.fail_summary_claim(
                job.event_id, now,
                self.settings.summary_failure_backoff_base_seconds,
                self.settings.summary_failure_backoff_cap_seconds,
            )
            repository.record_routing_event(
                "summary_failure", event_id=job.event_id,
                provider_key=provider_key, call_type="summary",
            )
            return SummaryFinalizeResult(
                "provider_failure", job.start_cursor, job.start_cursor
            )
        legacy_candidates = (
            "stable_memory_candidates" in summary
            and "memory_candidates" not in summary
        )
        compact_summary = normalize_summary(summary)
        result = repository.finalize_summary_job(
            job,
            compact_summary,
            self._candidate_observations(
                compact_summary, rows, legacy_evidence=legacy_candidates
            ),
            now,
            stable_limit=self.settings.max_long_memories,
            recent_limit=self.settings.max_messages_per_chat,
            max_candidates=self.settings.max_memory_candidates,
            candidate_stale_days=self.settings.memory_candidate_stale_days,
            promoted_retention_days=(
                self.settings.memory_candidate_promoted_retention_days
            ),
            daily_summary_retention_days=self.settings.daily_summary_retention_days,
        )
        repository.record_routing_event(
            "summary_success" if result.committed else "summary_stale_finalize",
            event_id=job.event_id, provider_key=provider_key,
            call_type="summary",
        )
        if result.candidates_promoted_pruned:
            repository.record_routing_event(
                "summary_candidates_promoted_pruned", event_id=job.event_id,
                call_type="summary",
            )
        if result.candidates_stale_pruned:
            repository.record_routing_event(
                "summary_candidates_stale_pruned", event_id=job.event_id,
                call_type="summary",
            )
        return result

    def maybe_refresh(self, repository, chat_id):
        """Compatibility facade; every call is still a separate memory event."""
        runner = MemoryMaintenanceRunner(
            self, self.concurrency, self.provider_resolver
        )
        return runner.run_once(
            repository, chat_id, current=self._now()
        ).status == "committed"
