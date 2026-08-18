import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from .memory_service import normalize_summary
from .normalized_event import NormalizedEvent, normalize_telegram_event
from .repository import normalize_memory


WORD_RE = re.compile(r"[a-zа-яё][a-zа-яё-]{2,}", re.IGNORECASE)
HUMOR_RE = re.compile(
    r"(?:а?ха){2,}|\b(?:лол|кек|ору|орнул|ржу|мем|прикол|шутк\w*|угар)\b|[😂🤣😁😹🤡]",
    re.IGNORECASE,
)
ARGUMENT_RE = re.compile(
    r"\b(?:нет|неправда|не соглас\w*|ты не понял\w*|какой бред|это бред|чушь|"
    r"ерунда|ошибаешься|возражаю|докажи|аргумент\w*|перечитай)\b",
    re.IGNORECASE,
)
SERIOUS_RE = re.compile(
    r"\b(?:проблем\w*|решени\w*|причин\w*|следстви\w*|вариант\w*|"
    r"предлагаю|необходимо|важно|риск\w*|анализ\w*|потому что|следовательно)\b",
    re.IGNORECASE,
)
WORK_RE = re.compile(
    r"\b(?:задач\w*|срок\w*|дедлайн\w*|релиз\w*|баг\w*|ошибк\w*|созвон\w*|"
    r"сервер\w*|проект\w*|отч[её]т\w*|клиент\w*|депло\w*|выкат\w*|прод(?:акшн\w*)?|"
    r"тест\w*|тикет\w*|спринт\w*|ревью\w*|коммит\w*|пулл.?реквест\w*|"
    r"техническ\w*|требовани\w*|приоритет\w*|план\w*)\b",
    re.IGNORECASE,
)
EMOJI_ONLY_RE = re.compile(r"^[\W_\d]+$", re.UNICODE)

STOP_WORDS = {
    "это", "как", "что", "все", "для", "или", "еще", "уже", "только", "просто",
    "очень", "тебя", "меня", "тебе", "мне", "они", "она", "оно", "его", "такой",
    "такая", "такое", "когда", "если", "потом", "тут", "там", "здесь", "сейчас",
    "будет", "было", "быть", "есть", "нет", "надо", "можно", "нельзя", "который",
    "почему", "потому", "тоже", "этот", "эта", "эти", "того", "вообще", "снова",
}


@dataclass(frozen=True)
class ChatState:
    activity_level: str
    silence_seconds: int | None
    conversation_type: str
    dominant_topic: str | None
    topic_strength: float
    humor_score: float
    argument_score: float
    serious_score: float
    work_score: float
    reply_density: float
    participant_count: int
    target_message_id: int | None
    target_user_id: int | None
    confidence: float

    def debug(self):
        return asdict(self)


def _stamp(value):
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


class ChatStateAnalyzer:
    def __init__(self, settings, memory_service, clock=None):
        self.settings = settings
        self.memory_service = memory_service
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self, supplied=None):
        current = supplied or self._clock()
        return current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current

    def _incoming_row(self, message, now):
        if message is None:
            return None
        event = (
            message if isinstance(message, NormalizedEvent)
            else normalize_telegram_event(message)
        )
        created_at = event.timestamp or now
        return {
            "row_id": None,
            "message_id": event.message_id,
            "user_id": event.user_id,
            "username": event.username,
            "text": event.effective_text,
            "created_at": created_at.isoformat(),
            "speaker": "cyberchair" if event.user_is_bot else "user",
            "reply_to_message_id": event.reply_to_message_id,
            "is_reply": int(event.reply_to_message_id is not None),
        }

    def _rows(self, repository, incoming_message, now):
        rows = list(self.memory_service.short_term_rows(repository))
        incoming = self._incoming_row(incoming_message, now)
        if incoming and not any(
            row.get("message_id") == incoming.get("message_id")
            and row.get("speaker") == incoming.get("speaker")
            for row in rows
        ):
            rows.append(incoming)
        return sorted(rows, key=lambda row: _stamp(row.get("created_at")) or now)

    def _activity(self, rows, silence_seconds, now):
        user_rows = [row for row in rows if row.get("speaker") != "cyberchair"]
        in_one_minute = sum(
            bool((stamp := _stamp(row.get("created_at"))) and now - stamp <= timedelta(minutes=1))
            for row in user_rows
        )
        in_five_minutes = sum(
            bool((stamp := _stamp(row.get("created_at"))) and now - stamp <= timedelta(minutes=5))
            for row in user_rows
        )
        recent_participants = {
            row.get("user_id")
            for row in user_rows
            if row.get("user_id") is not None
            and (stamp := _stamp(row.get("created_at")))
            and now - stamp <= timedelta(minutes=1)
        }
        if (
            in_one_minute >= self.settings.state_burst_messages_1m
            and len(recent_participants) >= self.settings.state_burst_participants
        ):
            return "burst"
        if (
            in_five_minutes >= self.settings.state_high_messages_5m
            or in_one_minute >= self.settings.state_high_messages_1m
        ):
            return "high"
        if (
            len(user_rows) <= self.settings.state_low_message_count
            or silence_seconds is None
            or silence_seconds >= self.settings.state_low_silence_seconds
        ):
            return "low"
        return "normal"

    def _dominant_topic(self, rows, summary):
        user_rows = [row for row in rows if row.get("speaker") != "cyberchair"]
        counts = Counter()
        coverage = defaultdict(set)
        for position, row in enumerate(user_rows):
            words = {
                normalize_memory(word)
                for word in WORD_RE.findall(row.get("text", ""))
                if normalize_memory(word) not in STOP_WORDS
            }
            for word in words:
                counts[word] += 1
                coverage[word].add(position)
        live_candidates = [
            word
            for word, count in counts.items()
            if count >= self.settings.state_topic_min_occurrences
            and len(coverage[word]) >= self.settings.state_topic_min_messages
        ]
        if not live_candidates:
            return None, 0.0
        summary_words = set()
        for topic in summary.get("main_topics", []):
            summary_words.update(normalize_memory(topic).split())
        topic = max(
            live_candidates,
            key=lambda word: (counts[word] + (0.5 if word in summary_words else 0), len(word)),
        )
        strength = min(
            1.0,
            (counts[topic] / max(1, len(user_rows))) * 0.7
            + (len(coverage[topic]) / max(1, len(user_rows))) * 0.3,
        )
        return topic, round(strength, 3)

    def _scores(self, rows):
        user_rows = [row for row in rows if row.get("speaker") != "cyberchair"]
        count = len(user_rows)
        if not count:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        texts = [row.get("text", "") for row in user_rows]
        word_counts = [len(WORD_RE.findall(text)) for text in texts]
        humor_ratio = sum(bool(HUMOR_RE.search(text)) for text in texts) / count
        short_ratio = sum(words <= 5 for words in word_counts) / count if count >= 4 else 0
        reaction_ratio = sum(
            bool(EMOJI_ONLY_RE.fullmatch(text.strip())) for text in texts if text.strip()
        ) / count
        humor = min(1.0, humor_ratio * 0.65 + short_ratio * 0.2 + reaction_ratio * 0.25)

        id_to_user = {
            row.get("message_id"): row.get("user_id")
            for row in user_rows
            if row.get("message_id") is not None
        }
        reply_rows = [row for row in user_rows if row.get("reply_to_message_id") is not None]
        reply_density = len(reply_rows) / count
        directions = {
            (row.get("user_id"), id_to_user.get(row.get("reply_to_message_id")))
            for row in reply_rows
            if id_to_user.get(row.get("reply_to_message_id")) is not None
            and row.get("user_id") != id_to_user.get(row.get("reply_to_message_id"))
        }
        reciprocal = any((target, source) in directions for source, target in directions)
        conflict_ratio = sum(bool(ARGUMENT_RE.search(text)) for text in texts) / count
        argument = min(
            1.0,
            conflict_ratio * 0.55 + reply_density * 0.3 + (0.25 if reciprocal else 0),
        )

        long_ratio = sum(words >= 14 for words in word_counts) / count
        structured_ratio = sum(bool(SERIOUS_RE.search(text)) for text in texts) / count
        serious = min(
            1.0,
            max(0.0, long_ratio * 0.55 + structured_ratio * 0.35 + reply_density * 0.1 - humor * 0.3),
        )
        work_ratio = sum(bool(WORK_RE.search(text)) for text in texts) / count
        work = min(1.0, work_ratio * 0.85 + structured_ratio * 0.15)
        return tuple(round(value, 3) for value in (humor, argument, serious, work, reply_density))

    def _conversation_type(self, humor, argument, serious, work):
        scores = {
            "humor": humor,
            "argument": argument,
            "serious": serious,
            "work": work,
        }
        strong = [name for name, score in scores.items() if score >= 0.5]
        if len(strong) >= 2:
            return "mixed"
        best, score = max(scores.items(), key=lambda item: item[1])
        return best if score >= 0.42 else "casual"

    def _target(self, rows, last_target_user_id, answered_message_ids):
        candidates = [
            row
            for row in rows
            if row.get("speaker") != "cyberchair" and row.get("message_id") is not None
        ]
        if not candidates:
            return None, None
        unanswered = [
            row for row in candidates if row.get("message_id") not in answered_message_ids
        ]
        if unanswered:
            candidates = unanswered
        alternatives = [
            row for row in candidates if row.get("user_id") != last_target_user_id
        ]
        if last_target_user_id is not None and alternatives:
            candidates = alternatives
        elif last_target_user_id is not None:
            return None, None
        reply_counts = Counter(
            row.get("reply_to_message_id")
            for row in rows
            if row.get("reply_to_message_id") is not None
        )
        scored = []
        for position, row in enumerate(candidates):
            text = row.get("text", "")
            signal = bool(HUMOR_RE.search(text) or ARGUMENT_RE.search(text))
            unusual = len(WORD_RE.findall(text)) >= 12 or "?" in text
            freshness = (position + 1) / max(1, len(candidates))
            score = reply_counts[row.get("message_id")] * 2 + signal * 1.4 + unusual * 0.5 + freshness
            scored.append((score, row))
        score, target = max(scored, key=lambda item: item[0])
        if score < 1.2:
            return None, None
        return target.get("message_id"), target.get("user_id")

    def analyze(
        self,
        repository,
        incoming_message=None,
        bot_id=None,
        last_target_user_id=None,
        answered_message_ids=(),
        now=None,
        snapshot=None,
    ):
        del bot_id  # Bot-authored rows are identified by their stored speaker.
        current = self._now(now)
        if snapshot is None:
            rows = self._rows(repository, incoming_message, current)
            latest_rows = repository.recent_messages(1)
            latest_stamp = (
                _stamp(latest_rows[-1]["created_at"]) if latest_rows else None
            )
            summary = normalize_summary(
                repository.summary_for_day(
                    self.memory_service.logical_day(current)
                ) or {}
            )
        else:
            rows = list(snapshot.recent_dialogue)
            incoming = self._incoming_row(incoming_message, current)
            if incoming and not any(
                row.get("message_id") == incoming.get("message_id")
                and row.get("speaker") == incoming.get("speaker")
                for row in rows
            ):
                rows.append(incoming)
            rows = sorted(
                rows, key=lambda row: _stamp(row.get("created_at")) or current
            )
            latest_stamp = _stamp(
                snapshot.latest_message.get("created_at")
            ) if snapshot.latest_message else None
            summary = normalize_summary(snapshot.current_summary)
        if incoming_message is not None:
            incoming_stamp = _stamp(self._incoming_row(incoming_message, current)["created_at"])
            if incoming_stamp and (latest_stamp is None or incoming_stamp > latest_stamp):
                latest_stamp = incoming_stamp
        silence = max(0, int((current - latest_stamp).total_seconds())) if latest_stamp else None
        activity = self._activity(rows, silence, current)
        dominant_topic, topic_strength = self._dominant_topic(rows, summary)
        humor, argument, serious, work, reply_density = self._scores(rows)
        conversation_type = self._conversation_type(humor, argument, serious, work)
        target_message_id, target_user_id = self._target(
            rows, last_target_user_id, set(answered_message_ids)
        )
        participants = {
            row.get("user_id")
            for row in rows
            if row.get("speaker") != "cyberchair" and row.get("user_id") is not None
        }
        confidence = min(
            1.0,
            0.25
            + max(humor, argument, serious, work, topic_strength) * 0.55
            + min(len(rows), 10) * 0.02,
        )
        return ChatState(
            activity_level=activity,
            silence_seconds=silence,
            conversation_type=conversation_type,
            dominant_topic=dominant_topic,
            topic_strength=topic_strength,
            humor_score=humor,
            argument_score=argument,
            serious_score=serious,
            work_score=work,
            reply_density=reply_density,
            participant_count=len(participants),
            target_message_id=target_message_id,
            target_user_id=target_user_id,
            confidence=round(confidence, 3),
        )
