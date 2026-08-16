import re
from datetime import datetime, timedelta, timezone

from .direct_address import SOCIAL, SUBSTANTIVE
from .preprocessing import normalize_spaces


RESPONSES = {
    "summon": (
        "чё", "ну", "🪑", "chairOS на связи", "хули", "цель обнаружена",
        "да тут я", "говори давай", "стул загружен, вещай",
    ),
    "acknowledgement": (
        "ага", "принято", "ну допустим", "слышу", "зафиксировал эту хуйню",
        "chairOS кивнул", "ладно, живи",
    ),
    "laugh": (
        "ахах, блять", "💀", "это уже канон", "ору", "пик диалога",
        "брейнрот принят", "смешно вышло, не спорю",
    ),
    "insult_to_chair": (
        "сам такой, биологический объект", "сильный аргумент, лил бро",
        "ругайся точнее, я записываю", "chairOS пережил и не такое",
        "цель обнаружена. интеллект пока нет", "ну поплачь об обивку",
    ),
    "confusion": (
        "в смысле", "раскрой эту шизотеорию", "чё конкретно", "говори словами",
        "контекст потерян где-то между соей и крашаутом",
    ),
    "argument": (
        "звучит как коуп", "сильный ларп, фактов пока ноль",
        "ну и кто тут сейчас нпс", "аргумент мид, подача гигачад",
        "chairOS фиксирует крашаут", "бро живёт у тебя рент фри",
    ),
    "dismissal": (
        "иди сам, у меня колёсики", "неубедительно", "отклонено по причине skill issue",
        "ну давай, выдохни", "протокол обиды не найден",
    ),
    "fallback": (
        "говори давай", "я слушаю", "ну и", "продолжай эту хуйню",
        "chairOS ждёт конкретики", "принял, что дальше",
    ),
    "substantive_fallback": (
        "давай детали: что именно происходит и что уже пробовал",
        "накидай исходные данные и ограничения, разберём без гадания",
        "уточни контекст и желаемый результат — иначе это ларп диагностики",
    ),
}

NEUTRAL_RESPONSES = {
    "trivial": ("слушаю", "да, я здесь", "говорите", "на связи"),
    "social": ("принято", "понял", "давайте по существу", "аргумент услышал"),
    "substantive": RESPONSES["substantive_fallback"],
}


class LocalResponder:
    """Data-driven CyberChair fallback with local anti-repeat and meme cooldowns."""

    def __init__(self, lexicon, rng):
        self.lexicon = lexicon
        self.rng = rng

    @staticmethod
    def _category(text, intent):
        normalized = normalize_spaces(text or "").casefold()
        if intent == SUBSTANTIVE:
            return "substantive_fallback"
        if re.search(r"(?:ахах|хаха|лол|кек|💀)", normalized):
            return "laugh"
        if re.search(r"(?:долбо[её]б|лох|мудак|иди\s+нахуй|сам\s+ты)", normalized):
            return "insult_to_chair" if re.search(r"(?:ты|стул|стульчик)", normalized) else "dismissal"
        if re.search(r"(?:ч[её]|в смысле|что)", normalized):
            return "confusion"
        if intent == SOCIAL:
            return "argument"
        if re.sub(r"\b(?:ау\s+)?стул\w*|\bстуль\w*", "", normalized).strip(" ,.!?…") == "":
            return "summon"
        return "acknowledgement"

    def respond(self, chat_id, text, intent, repository, excluded_meme_ids=(),
                excluded_meme_groups=(), troll_mode=True):
        category = self._category(text, intent)
        variants = list(
            RESPONSES[category] if troll_mode else NEUTRAL_RESPONSES[intent]
        )
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        recent = [row["text"] for row in repository.generated_since(since)][-12:]
        fresh = [item for item in variants if item not in recent]
        candidates = fresh or variants
        index = min(len(candidates) - 1, int(self.rng.random() * len(candidates)))
        result = candidates[index]

        # At most one lexicon concept, and only when it naturally fits a social
        # response. Existing lexicon cooldown groups are shared with AI persona.
        used = ()
        if troll_mode and intent == SOCIAL and self.rng.random() < .28:
            selected = self.lexicon.select(
                text, {"humor", "mocking", "argument"}, .65,
                excluded_meme_ids, excluded_meme_groups, limit=1,
            )
            if selected and selected[0].output not in result:
                result = f"{result}, {selected[0].output}"
                used = (selected[0],)
        return result, used
