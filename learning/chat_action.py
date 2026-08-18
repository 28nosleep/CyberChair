import logging
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass
class _ActiveAction:
    action: str
    stop: threading.Event
    thread: threading.Thread
    references: int = 1


class ChatActionSession:
    def __init__(self, manager, chat_id, started_at):
        self.manager = manager
        self.chat_id = chat_id
        self.started_at = started_at

    def ensure_visible(self, producer, text):
        self.manager.ensure_visible(self.started_at, producer, text)


class ChatActionManager:
    """Best-effort Telegram actions with bounded refresh and human jitter."""

    def __init__(self, bot, refresh_interval=4.0, rng=None, clock=None, sleeper=None):
        self.bot = bot
        self.refresh_interval = max(0.05, float(refresh_interval))
        self.rng = rng or random
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._active = {}
        self._lock = threading.RLock()
        self._shutdown = threading.Event()

    def _send(self, chat_id, action):
        try:
            self.bot.send_chat_action(chat_id, action)
            return True
        except Exception as error:
            log.warning(
                "Telegram chat action failed chat=%s action=%s error=%s",
                chat_id, action, type(error).__name__,
            )
            return False

    def _refresh(self, chat_id, stop):
        while not self._shutdown.is_set() and not stop.wait(self.refresh_interval):
            with self._lock:
                active = self._active.get(chat_id)
                if active is None or active.stop is not stop:
                    return
                action = active.action
            self._send(chat_id, action)

    @contextmanager
    def activity(self, chat_id, action="typing", producer=None):
        """Start immediately and reuse an existing per-chat loop if necessary."""
        started_at = self.clock()
        owner = False
        with self._lock:
            if self._shutdown.is_set():
                yield_disabled = True
                active = None
            else:
                yield_disabled = False
                active = self._active.get(chat_id)
            if not yield_disabled and active is None:
                stop = threading.Event()
                thread = threading.Thread(
                    target=self._refresh,
                    args=(chat_id, stop),
                    name=f"chat-action-{chat_id}",
                    daemon=True,
                )
                active = _ActiveAction(action, stop, thread)
                self._active[chat_id] = active
                owner = True
            elif not yield_disabled:
                active.references += 1
                active.action = action
        if yield_disabled:
            yield ChatActionSession(self, chat_id, started_at)
            return
        if not self._shutdown.is_set():
            self._send(chat_id, action)
        if owner:
            active.thread.start()
        try:
            yield ChatActionSession(self, chat_id, started_at)
        finally:
            thread = None
            with self._lock:
                current = self._active.get(chat_id)
                if current is active:
                    current.references -= 1
                    if current.references <= 0:
                        self._active.pop(chat_id, None)
                        current.stop.set()
                        thread = current.thread
            if thread and thread is not threading.current_thread():
                thread.join(timeout=min(1.0, self.refresh_interval + 0.1))

    def ensure_visible(self, started_at, producer, text):
        """Add only the small local delay still missing after preparation."""
        if producer == "llm" or self._shutdown.is_set():
            return
        words = len(str(text or "").split())
        if producer == "markov":
            low, high = 0.7, 1.5
        elif words <= 3:
            low, high = 0.4, 1.2
        else:
            low, high = 0.8, 1.8
        length_factor = min(1.0, words / 18)
        target = low + (high - low) * (
            0.35 * length_factor + 0.65 * self.rng.random()
        )
        remaining = target - max(0.0, self.clock() - started_at)
        if remaining > 0:
            self.sleeper(min(high, remaining))

    def active_count(self, chat_id=None):
        with self._lock:
            if chat_id is None:
                return len(self._active)
            return int(chat_id in self._active)

    def shutdown(self):
        """Stop refresh API calls without waiting for active foreground work."""
        self._shutdown.set()
        with self._lock:
            active = tuple(self._active.values())
        for item in active:
            item.stop.set()

    def worker_count(self):
        with self._lock:
            return sum(int(item.thread.is_alive()) for item in self._active.values())


MEDIA_CHAT_ACTIONS = {
    "gif": "upload_video",
    "sticker": "choose_sticker",
    "meme": "upload_photo",
}
