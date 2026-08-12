import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .llm_prompts import build_summarize_request
from .preprocessing import normalize_spaces
from .repository import memories_are_similar, normalize_memory


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
    source = summary if isinstance(summary, dict) else {}
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
                 provider_resolver=None):
        self.settings = settings
        self.llm_provider = llm_provider
        self.provider_resolver = provider_resolver or (lambda chat_id: llm_provider)
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
    def _relevant_values(values, terms, limit=2):
        result = []
        for value in values or ():
            clean = normalize_spaces(str(value))[:200]
            value_terms = set(re.findall(r"[\wёЁ-]{4,}", clean.casefold()))
            if clean and terms and value_terms & terms:
                result.append(clean)
            if len(result) >= limit:
                break
        return result

    def relevant_memory(self, repository, context=None, dominant_topic=None,
                        conversation_type=None):
        """Select facts that can affect this turn instead of serialising all memory."""
        seed = f"{context or ''} {dominant_topic or ''}"
        terms = set(re.findall(r"[\wёЁ-]{4,}", seed.casefold()))
        summary = normalize_summary(repository.summary_for_day(self.logical_day()) or {})
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
                repository.stable_memories(20), terms
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

    def maybe_refresh(self, repository, chat_id):
        state = repository.summary_state()
        rows = repository.messages_after(state["last_message_row_id"])
        provider = self.provider_resolver(chat_id)
        if not rows or not provider.available:
            return False
        now = self._now()
        repository.mark_summary_pending(now.isoformat())
        state = repository.summary_state()
        pending_since = datetime.fromisoformat(state["pending_since"])
        due = (
            len(rows) >= self.settings.summary_message_interval
            or (now - pending_since).total_seconds() >= self.settings.summary_time_interval
        )
        if not due:
            return False
        day = self.logical_day(now)
        previous = repository.summary_for_day(day)
        previous = normalize_summary(previous) if previous else None
        fragment_rows = rows[:50]
        dialogue = self._dialogue_fragment(fragment_rows)
        safety_identifier = __import__("hashlib").sha256(
            f"cyberchair-memory:{chat_id}".encode("utf-8")
        ).hexdigest()[:32]
        if not dialogue.strip():
            return False
        request = build_summarize_request(
            dialogue, previous, safety_identifier,
            max_output_tokens=self.settings.summary_max_output_tokens,
        )
        summary = provider.summarize(replace(request, metadata={
            "chat_id": chat_id, "call_type": "summary", "purpose": "summary",
        }))
        if not summary:
            return False
        legacy_candidates = (
            isinstance(summary, dict)
            and "stable_memory_candidates" in summary
            and "memory_candidates" not in summary
        )
        compact_summary = normalize_summary(summary)
        repository.finalize_summary(
            day,
            compact_summary,
            fragment_rows[-1]["id"],
            self._candidate_observations(
                compact_summary,
                fragment_rows,
                legacy_evidence=legacy_candidates,
            ),
            self.settings.max_long_memories,
        )
        return True
