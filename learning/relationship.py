"""Slow, local per-participant relationship state."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re

from .preprocessing import normalize_spaces


POSITIVE_RE = re.compile(
    r"(?:\bспасибо\b|\bспс\b|\bкрасав\w*\b|\bхорош\w*\b|\bсоглас\w*\b|❤️|❤|👍)",
    re.I,
)
NEGATIVE_RE = re.compile(
    r"(?:\bзаткнись\b|\bиди\s+нахуй\b|\bмудак\w*\b|\bдолбо[её]б\w*\b|👎)",
    re.I,
)
PLAYFUL_RE = re.compile(r"(?:ахах|хаха|лол|кек|рофл|💀|😂)", re.I)
CONFLICT_RE = re.compile(r"(?:\bнет\b|\bнеправда\b|\bчушь\b|\bбред\b|\bвр[её]шь\b)", re.I)


@dataclass(frozen=True)
class Relationship:
    chat_id: int
    user_id: int
    affinity: float = 0.5
    irritation: float = 0.0
    respect: float = 0.5
    interest: float = 0.25
    troll_tendency: float = 0.25
    familiarity: float = 0.0
    interaction_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    last_interaction_at: str | None = None

    def debug(self):
        return asdict(self)


class RelationshipModel:
    """Update bounded state from observable behavior, never from an LLM."""

    @staticmethod
    def _clip(value):
        return round(max(0.0, min(1.0, float(value))), 4)

    def current(self, repository, user_id):
        row = repository.relationship(user_id)
        if row:
            return Relationship(**row)
        return Relationship(repository.chat_id, int(user_id))

    def observe(self, repository, event, recent_rows=()):
        if event.user_id is None:
            return None
        current = self.current(repository, event.user_id)
        text = normalize_spaces(event.effective_text)
        positive = bool(POSITIVE_RE.search(text))
        negative = bool(NEGATIVE_RE.search(text))
        playful = bool(PLAYFUL_RE.search(text))
        conflict = bool(CONFLICT_RE.search(text))
        is_reply = event.reply_to_message_id is not None
        prior_by_user = [
            row for row in recent_rows
            if row.get("user_id") == event.user_id
            and row.get("message_id") != event.message_id
        ]
        normalized = text.casefold().replace("ё", "е")
        repeated = sum(
            normalize_spaces(row.get("text", "")).casefold().replace("ё", "е")
            == normalized
            for row in prior_by_user[-8:]
        )

        # Every delta is deliberately small. One extreme message cannot rewrite
        # the relationship; repeated behavior accumulates over many events.
        affinity_delta = 0.012 if positive else -0.012 if negative else 0.0
        irritation_delta = 0.014 if negative else 0.007 if repeated else -0.002
        respect_delta = 0.008 if positive or (len(text) >= 80 and "?" not in text) else 0.0
        if conflict:
            respect_delta -= 0.004
        interest_delta = 0.007 if len(text.split()) >= 8 or is_reply else -0.001
        troll_delta = 0.010 if playful else 0.006 if conflict else -0.001
        familiarity_delta = min(0.012, 0.004 + (0.003 if is_reply else 0.0))

        updated = Relationship(
            chat_id=repository.chat_id,
            user_id=int(event.user_id),
            affinity=self._clip(current.affinity + affinity_delta),
            irritation=self._clip(current.irritation + irritation_delta),
            respect=self._clip(current.respect + respect_delta),
            interest=self._clip(current.interest + interest_delta),
            troll_tendency=self._clip(current.troll_tendency + troll_delta),
            familiarity=self._clip(current.familiarity + familiarity_delta),
            interaction_count=current.interaction_count + 1,
            positive_count=current.positive_count + int(positive),
            negative_count=current.negative_count + int(negative or conflict),
            last_interaction_at=(
                event.timestamp.isoformat()
                if getattr(event.timestamp, "isoformat", None)
                else str(event.timestamp)
            ),
        )
        repository.save_relationship(updated)
        return updated

    def observe_callback(self, repository, event):
        if event.user_id is None:
            return None
        current = self.current(repository, event.user_id)
        updated = Relationship(
            chat_id=repository.chat_id,
            user_id=int(event.user_id),
            affinity=self._clip(current.affinity + .002),
            irritation=current.irritation,
            respect=current.respect,
            interest=self._clip(current.interest + .003),
            troll_tendency=current.troll_tendency,
            familiarity=self._clip(current.familiarity + .006),
            interaction_count=current.interaction_count + 1,
            positive_count=current.positive_count,
            negative_count=current.negative_count,
            last_interaction_at=datetime.now(timezone.utc).isoformat(),
        )
        repository.save_relationship(updated)
        return updated
