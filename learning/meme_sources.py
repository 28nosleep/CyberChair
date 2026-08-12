"""Source selection for the explicit ``с м стул`` command.

The selector is deliberately local: an unavailable LLM must never turn a meme
command into a cooldown notice.  It also keeps the command's anti-repeat state
separate from normal conversational-media scoring.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import re

from .preprocessing import normalize_spaces, significant_words
from .repository import normalize_memory


@dataclass(frozen=True)
class MemeSource:
    kind: str
    text: str
    message_id: int | None = None
    user_id: int | None = None
    score: float = 0.0


class MemeSourceSelector:
    """Choose varied meme material, preferring actual chat quotes."""

    WEIGHTS = (("fresh", 45), ("old", 30), ("callback", 20), ("absurd", 5))
    FALLBACK_ORDER = ("old", "fresh", "callback", "markov", "phrase")

    def __init__(self, rng, recent_limit=8):
        self.rng = rng
        self._source_ids = defaultdict(lambda: deque(maxlen=recent_limit))
        self._users = defaultdict(lambda: deque(maxlen=3))
        self._templates = defaultdict(lambda: deque(maxlen=2))
        self._callbacks = defaultdict(lambda: deque(maxlen=recent_limit))
        self._kinds = defaultdict(lambda: deque(maxlen=2))

    @staticmethod
    def _clean(text):
        return normalize_spaces(str(text or ""))[:180]

    def rank_old_quotes(self, rows, current_text="", topic=""):
        """Rank old rows; never sample an old quote uniformly at random."""
        current_terms = set(significant_words(f"{current_text} {topic}"))
        current_normalized = normalize_memory(current_text)
        ranked = []
        for row in rows:
            text = self._clean(row.get("text"))
            if not text or len(text) < 8 or len(text) > 240:
                continue
            message_id, user_id = row.get("message_id"), row.get("user_id")
            if message_id in self._source_ids[row.get("chat_id", 0)]:
                continue
            words = significant_words(text)
            score = 0.0
            # Short quotable messages age particularly well in a meme.
            score += max(0, 4.0 - abs(len(text) - 72) / 24)
            score += min(2.5, text.count("!") + text.count("?") + text.count("…"))
            score += min(3.0, int(row.get("reply_count") or 0) * .75)
            score += 2.5 if row.get("is_reply") else 0.0
            score += 2.5 if row.get("last_used_at") is None else -2.0
            overlap = len(set(words) & current_terms)
            score += min(4.0, overlap * 1.25)
            # A confident old statement that clashes with the current wording
            # is a good compromising callback.
            if current_normalized and any(token in text.casefold() for token in ("никогда", "всегда", "точно", "легко", "не буду")):
                score += 1.5
            if re.search(r"\b(я|мы)\b.{0,50}\b(никогда|всегда|легко|точно|завтра)\b", text, re.I):
                score += 2.0
            if user_id in self._users[row.get("chat_id", 0)]:
                score -= 4.0
            ranked.append(MemeSource("old", text, message_id, user_id, score))
        return sorted(ranked, key=lambda item: (item.score, item.message_id or 0), reverse=True)

    def _fresh(self, rows, chat_id):
        candidates = []
        for row in reversed(rows[-12:]):
            text = self._clean(row.get("text"))
            if not text or row.get("message_id") in self._source_ids[chat_id]:
                continue
            if row.get("user_id") in self._users[chat_id]:
                continue
            candidates.append(MemeSource("fresh", text, row.get("message_id"), row.get("user_id")))
        return candidates

    def choose(self, chat_id, rows, callbacks=(), current_text="", topic="", fallback=False):
        rows = [dict(row, chat_id=chat_id) for row in rows if row.get("text")]
        fresh = self._fresh(rows, chat_id)
        old = self.rank_old_quotes(rows[:-8], current_text, topic)
        callbacks = [self._clean(value) for value in callbacks]
        callbacks = [value for value in callbacks if value and value not in self._callbacks[chat_id]]
        pools = {"fresh": fresh, "old": old, "callback": [MemeSource("callback", value) for value in callbacks]}
        order = self.FALLBACK_ORDER if fallback else self._weighted_order()
        for kind in order:
            if kind in pools and pools[kind]:
                # Ranking decides which old quote is resurrected.  Fresh/callback
                # selection remains light-weight but anti-repeat guarded.
                return pools[kind][0] if kind == "old" else self.rng.choice(pools[kind])
        return MemeSource("phrase", "")

    def _weighted_order(self):
        bag = [kind for kind, weight in self.WEIGHTS for _ in range(weight)]
        first = self.rng.choice(bag)
        return (first,) + tuple(kind for kind, _ in self.WEIGHTS if kind != first)

    def record(self, chat_id, source, template_id=None):
        if source.message_id is not None:
            self._source_ids[chat_id].append(source.message_id)
        if source.user_id is not None:
            self._users[chat_id].append(source.user_id)
        if source.kind == "callback" and source.text:
            self._callbacks[chat_id].append(source.text)
        if source.kind:
            self._kinds[chat_id].append(source.kind)
        if template_id:
            self._templates[chat_id].append(template_id)

    def recent_templates(self, chat_id):
        return set(self._templates[chat_id])

    def markov_allowed(self, chat_id):
        return "markov" not in self._kinds[chat_id]

    def clear_chat(self, chat_id):
        for values in (self._source_ids, self._users, self._templates, self._callbacks, self._kinds):
            values.pop(chat_id, None)
