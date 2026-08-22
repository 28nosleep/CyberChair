import random
import time
import math
import threading
from collections import defaultdict, deque


class TriggerEngine:
    def __init__(self, settings, rng=None, clock=None):
        self.settings = settings
        self.rng = rng or random
        self.clock = clock or time.monotonic
        self._last = defaultdict(dict)
        self._history = defaultdict(deque)
        self._chair_history = defaultdict(deque)
        self._cooldown_notices = defaultdict(dict)
        self._lock = threading.RLock()

    def note_message(self, chat_id):
        now = self.clock()
        history = self._history[chat_id]
        history.append(now)
        while history and now - history[0] > 300:
            history.popleft()

    def observed_message_count(self, chat_id):
        """Number of runtime-ingested messages still known to this process."""
        with self._lock:
            return len(self._history[chat_id])

    def note_chair(self, chat_id):
        now = self.clock()
        history = self._chair_history[chat_id]
        history.append(now)
        while history and now - history[0] > 60:
            history.popleft()
        return len(history)

    def allowed(self, chat_id, kind, addressed=False):
        now = self.clock()
        cooldown = self.settings.addressed_cooldown if addressed else self.settings.generated_cooldown
        last = self._last[chat_id].get(kind, float("-inf"))
        if now - last < cooldown:
            return False
        hour_events = self._last[chat_id].get("hour_events", [])
        hour_events[:] = [stamp for stamp in hour_events if now - stamp < 3600]
        return len(hour_events) < self.settings.max_generated_per_hour

    def cooldown_remaining(self, chat_id, kind, addressed=False):
        cooldown = (
            self.settings.addressed_cooldown
            if addressed
            else self.settings.generated_cooldown
        )
        last = self._last[chat_id].get(kind)
        if last is None:
            return 0
        return max(0, math.ceil(cooldown - (self.clock() - last)))

    def consume_cooldown_notice(self, chat_id, kind, addressed=False):
        """Return remaining time at most once a minute during a cooldown."""
        with self._lock:
            remaining = self.cooldown_remaining(chat_id, kind, addressed)
            if remaining <= 0:
                return 0
            now = self.clock()
            last_notice = self._cooldown_notices[chat_id].get(kind)
            if last_notice is not None and now - last_notice < 60:
                return 0
            self._cooldown_notices[chat_id][kind] = now
            return remaining

    def commit(self, chat_id, kind):
        now = self.clock()
        self._last[chat_id][kind] = now
        self._last[chat_id].setdefault("hour_events", []).append(now)

    def decide_user_reply(self, chat_id, replies_to_bot=False, mentioned=False, special=False):
        """Compatibility wrapper; LearningService now uses ConversationPolicy."""
        addressed = replies_to_bot or mentioned or special
        if not self.allowed(chat_id, "addressed" if addressed else "random", addressed):
            return None
        if addressed:
            chance = self.settings.reply_to_stul_chance
        else:
            active_bonus = self.settings.active_chat_reply_chance if len(self._history[chat_id]) >= 8 else 0
            chance = self.settings.random_reply_chance + active_bonus
        return ("addressed" if addressed else "random") if self.rng.random() < chance else None
