import re
from dataclasses import asdict, dataclass

from .preprocessing import normalize_spaces


TRIVIAL = "trivial"
SOCIAL = "social"
SUBSTANTIVE = "substantive"


@dataclass(frozen=True)
class ResponseDecision:
    required: bool
    intent: str
    priority: str
    producer: str
    reason: str
    direct_reply: bool = False

    def debug(self):
        return asdict(self)


class LocalIntentClassifier:
    """Deterministic direct-message classifier. It never calls a provider."""

    _substantive = re.compile(
        r"(?:^|\s)(?:как|почему|зачем|сколько|ка(?:кой|кая|кие|кое)|кто|"
        r"что\s+(?:делать|думаешь|лучше|выбрать|купить|решили|случилось|происходит|нужно|будет)|посоветуй|подскажи|помоги|"
        r"объясни|расскажи|можно\s+ли|стоит\s+ли|куда|когда|где|откуда|"
        r"настрой|почини|сравни|выбери|разберись)(?:\s|$)",
        re.I,
    )
    _dependent_reply = re.compile(
        r"^(?:а\s+)?(?:почему|зачем|как|кто|что|где|когда|а если|если наоборот|"
        r"и что|что дальше|в смысле|это как|можно подробнее)\b",
        re.I,
    )
    _trivial = {
        "", "ау", "ага", "да", "нет", "неа", "ахах", "ахаха", "хаха",
        "лол", "кек", "чё", "че", "что", "ну", "ок", "окей", "пиздец",
        "ты тут", "живой", "алло", "эй", "💀", "🪑", "+", "-",
    }
    _social = re.compile(
        r"\b(?:долбо[её]б|лох|мудак|иди\s+нахуй|сам\s+ты|нес[её]шь|охуел|"
        r"виноват|соя|сойджак|кукд|кринж|база|бейсд|нпс|красава)\b",
        re.I,
    )

    def classify(self, text, *, direct_reply=False):
        normalized = normalize_spaces(text or "").casefold().strip(".,!?…:;—- ")
        if direct_reply and self._dependent_reply.search(normalized):
            return SUBSTANTIVE
        if re.fullmatch(r"кто\s+тут\s+(?:соя|сойджак|лох|нпс)", normalized):
            return SOCIAL
        if self._substantive.search(normalized):
            return SUBSTANTIVE
        if normalized in self._trivial or len(normalized.split()) <= 2 and not self._social.search(normalized):
            return TRIVIAL
        if self._social.search(normalized):
            return SOCIAL
        # A question with actual subject matter is substantive, but punctuation
        # is deliberately only one signal rather than the classifier itself.
        if "?" in (text or "") and len(normalized.split()) >= 3:
            return SUBSTANTIVE
        return SOCIAL


class DirectAddressRouter:
    def __init__(self, classifier=None):
        self.classifier = classifier or LocalIntentClassifier()

    def decide(self, text, *, direct_reply=False, ai_available=True,
               budget_exceeded=False, social_ai_useful=False):
        intent = self.classifier.classify(text, direct_reply=direct_reply)
        if intent == TRIVIAL:
            return ResponseDecision(True, intent, "P1", "local", "trivial_local", direct_reply)
        if intent == SOCIAL:
            producer = "grok" if social_ai_useful and ai_available and not budget_exceeded else "local"
            return ResponseDecision(True, intent, "P2", producer, "social_contextual", direct_reply)
        # The soft budget protects low-value calls. P3 remains AI-preferred so a
        # useful question is never silently dropped merely because spend is high.
        producer = "grok" if ai_available else "local"
        reason = "substantive_ai" if ai_available else "substantive_local_fallback"
        return ResponseDecision(True, intent, "P3", producer, reason, direct_reply)
