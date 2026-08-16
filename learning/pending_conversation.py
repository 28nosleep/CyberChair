import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .preprocessing import normalize_spaces


@dataclass(frozen=True)
class PendingConversation:
    chat_id: int
    user_id: int
    bot_message_id: int | None
    original_message_id: int | None
    original_question: str
    clarification_question: str
    intent: str
    context: str
    expected_type: str
    mode: str
    created_at: datetime


_GENERIC_CLARIFICATION_RE = re.compile(
    r"(?:уточни\s+(?:контекст|желаем)|дай\s+больше\s+детал|"
    r"что\s+именно\s+происходит|что\s+уже\s+пробовал|"
    r"предоставь\s+дополнительную\s+информацию)",
    re.I,
)
_QUESTION_REQUEST_RE = re.compile(
    r"(?:\?|\b(?:кидай|скажи|напиши|назови|укажи|между\s+чем|"
    r"какой|какая|какие|сколько|винда\s+или\s+мак)\b)"
    r"[^.!?\n]{0,100}$",
    re.I,
)


def expected_answer_type(question):
    value = normalize_spaces(question or "").casefold()
    if re.search(r"(?:рост|вес|кг|см)", value):
        return "measurements"
    if re.search(r"(?:бюджет|рубл|тысяч|сколько\s+готов)", value):
        return "budget"
    if re.search(r"(?:между\s+чем|вариант|что\s+выбира)", value):
        return "choices"
    if re.search(r"(?:винда|windows|мак|mac|android|ios|платформ)", value):
        return "platform"
    return "short_answer"


def extract_clarification(response):
    """Return one concrete final follow-up request, never a generic fallback."""
    text = normalize_spaces(response or "")
    if not text or _GENERIC_CLARIFICATION_RE.search(text):
        return None
    tail = re.split(r"(?<=[.!?])\s+|\n+", text)[-1].strip()
    if len(tail) > 180 or not _QUESTION_REQUEST_RE.search(tail):
        return None
    return tail


def pending_mode(response, clarification):
    """A follow-up after a completed answer is advisory, not a dialogue lock."""
    text = normalize_spaces(response or "")
    prefix = text[:-len(clarification)].strip(" \n.,;:—-") if clarification else ""
    # A real answer has enough substance before the optional request. Short
    # lead-ins such as "ну" or "тогда" are not an answer to the original turn.
    return "soft" if len(prefix.split()) >= 4 else "hard"


def looks_like_continuation(text, expected_type, mode="hard"):
    value = normalize_spaces(text or "").casefold()
    if not value or len(value) > 500:
        return False
    if re.search(
        r"(?:^|\s)(?:стул\w*|стуль\w*)\b|^(?:как|почему|зачем|кто|что\s+(?:делать|купить|выбрать))\b",
        value,
    ):
        return False
    if expected_type == "measurements":
        return bool(re.search(
            r"\b\d{2,3}\s*(?:[/xх]|и|,)\s*\d{2,3}\b|"
            r"\b(?:рост|вес)\s*\d{2,3}\b", value,
        ))
    if expected_type == "budget":
        return len(value.split()) <= 10 and bool(
            re.search(r"\d|тысяч|к|руб|без\s+лимита", value)
        )
    if expected_type == "choices":
        return len(value.split()) <= 16 and bool(
            re.search(r"\b(?:и|или|vs|либо|между)\b", value)
        )
    if expected_type == "platform":
        return bool(re.search(r"\b(?:windows|винд|mac|мак|linux|линукс|ios|android)\b", value))
    # A fresh explicit question is a new turn, not an accidental answer to an
    # old pending request. Short facts and fragments are natural continuations.
    return mode == "hard" and "?" not in value and len(value.split()) <= 20


def utc_now():
    return datetime.now(timezone.utc)
