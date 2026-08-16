import re
from datetime import datetime, timedelta, timezone

from .direct_address import SOCIAL, SUBSTANTIVE
from .preprocessing import normalize_spaces
from .pending_conversation import is_ambiguous_choice_request, question_intent, HOW_TO


PROFANITY_RE = re.compile(
    r"\b(?:бля(?:ть|дь)?|ебать|ебан\w*|нахуй|пиздец|хуйн\w*|охуел\w*|"
    r"заеб\w*|долбо[её]б\w*|проеб\w*|хуяр\w*|дохуя|нихуя|хули)\b",
    re.I,
)


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
        "ебать, chairOS уничтожен словом из трёх букв", "иди нахуй точнее, я протоколирую",
    ),
    "confusion": (
        "в смысле", "раскрой эту шизотеорию", "чё конкретно", "говори словами",
        "контекст потерян где-то между соей и крашаутом",
    ),
    "argument": (
        "звучит как коуп", "сильный ларп, фактов пока ноль",
        "ну и кто тут сейчас нпс", "аргумент мид, подача гигачад",
        "chairOS фиксирует крашаут", "бро живёт у тебя рент фри",
        "нихуя себе уверенность при нулевой доказательной базе",
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
        "chairOS сейчас без внешнего мозга, но вопрос понял; повтори через минуту",
        "внешний мозг отвалился, не буду выдумывать ответ из обивки; попробуй ещё раз через минуту",
        "вопрос принят, но сейчас лучше повторить его через минуту, чем получить фанфик от колёсиков",
    ),
    "how_to_unavailable": (
        "chairOS сейчас без больших мозгов, но вопрос понял; повтори через минуту",
        "вопрос понятен, просто внешний мозг отвалился; попробуй ещё раз через минуту",
    ),
}

NEUTRAL_RESPONSES = {
    "trivial": ("слушаю", "да, я здесь", "говорите", "на связи"),
    "social": ("принято", "понял", "давайте по существу", "аргумент услышал"),
    "substantive": (
        "уточните предмет и цель, чтобы ответ был полезным",
    ),
}


class LocalResponder:
    """Data-driven CyberChair fallback with local anti-repeat and meme cooldowns."""

    def __init__(self, lexicon, rng):
        self.lexicon = lexicon
        self.rng = rng

    def _troll_user(self, text, repository, excluded_meme_ids, excluded_meme_groups):
        """A real roast fallback: never turn provider failure into advice."""
        normalized = normalize_spaces(text or "").casefold()
        topic = next((label for pattern, label in (
            (r"docker|nginx|dns|redis|vpn|wi.?fi|тест", "инфраструктуру"),
            (r"деньг|накоп|кредит|банк|квартир", "кошелёк"),
            (r"голов|температур|бессон|спин|врач|вес|бег", "биологический квест"),
            (r"айфон|пиксел|android|ios|телефон|ssd|ноутбук|монитор|роутер|клавиатур|пк", "техношопинг"),
            (r"паст|рис|кофе|готов", "кухню"),
            (r"рэп|альбом|стрим|гитар|рисова", "творческую карьеру"),
            (r"игр|elden", "твой квест"),
            (r"работ|резюме|переговор|команд", "карьерную арку"),
            (r"девушк|расстав", "романтический сериал"),
            (r"кот|дом|переезд", "бытовой survival"),
        ) if re.search(pattern, normalized)), "этот план")
        callbacks = []
        terms = set(re.findall(r"[а-яёa-z0-9]{4,}", normalized))
        for item in repository.stable_memories(20):
            value = normalize_spaces(str(item))
            if terms & set(re.findall(r"[а-яёa-z0-9]{4,}", value.casefold())):
                callbacks.append(value[:100])
        if callbacks:
            variants = [
                f"вчерашний лор про «{callbacks[0]}» уже был намёком, а ты всё равно пришёл прокачивать {topic} как побочный квест",
                f"chairOS сверил память: «{callbacks[0]}». похоже, {topic} у тебя не вопрос, а сезонная арка без финала",
            ]
        else:
            variants = [
                f"бро смотрит на {topic} так, будто финальный босс сам выдаст ему гайд после этой формулировки",
                f"достижение разблокировано: спросить про {topic} с уверенностью человека, который уже проиграл туториал",
                f"chairOS фиксирует: {topic} снова пытаются закрыть одним сообщением, как будто реальность подписана на премиум",
                f"по постановке видно: {topic} у тебя уже не проблема, а франшиза с нулевым бюджетом",
                f"это не вопрос про {topic}, это заявка на то, чтобы взрослый интернет сделал за тебя домашку",
                f"у тебя с {topic} такой контакт, будто ты открыл меню настроек и сразу начал переговоры с богом",
            ]
        result = variants[min(len(variants) - 1, int(self.rng.random() * len(variants)))]
        selected = self.lexicon.select(
            text, {"mocking", "humor"}, .7, excluded_meme_ids,
            excluded_meme_groups, limit=1, recent_concepts=excluded_meme_groups,
        )
        if selected and selected[0].output not in result and self.rng.random() < .35:
            result = f"{result}, {selected[0].output}"
            return result, (selected[0],)
        return result, ()

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
                excluded_meme_groups=(), troll_mode=True, troll_intensity=.6,
                behavior_mode="useful_answer"):
        if intent == SUBSTANTIVE and behavior_mode == "troll_user":
            return self._troll_user(text, repository, excluded_meme_ids, excluded_meme_groups)
        category = self._category(text, intent)
        normalized = normalize_spaces(text or "").casefold()
        if intent == SUBSTANTIVE:
            if re.search(r"\bкак\s+набрать\s+вес\b", normalized):
                return ((
                    "начни с профицита 250–350 ккал, белка 1.6–2 г/кг и силовых с прогрессией; "
                    "две недели вес стоит — добавь ещё 150–200 ккал, без массонабора в пельмень"
                    if troll_mode else
                    "начните с профицита 250–350 ккал, белка 1.6–2 г/кг и силовых с прогрессией; "
                    "если две недели вес стоит, добавьте ещё 150–200 ккал"
                ), ())
            if is_ambiguous_choice_request(normalized):
                return (
                    "между чем выбираешь, лил бро" if troll_mode else "между чем выбираете?",
                    (),
                )
            # This branch is only used after the preferred provider failed or
            # was unavailable. Never pretend an already clear how-to lacks a
            # subject or goal.
            if question_intent(normalized) == HOW_TO:
                category = "how_to_unavailable"
        variants = list(
            RESPONSES[category] if troll_mode else NEUTRAL_RESPONSES[intent]
        )
        if troll_mode and troll_intensity <= .4:
            # Low intensity stays the same character but does not randomly jump
            # to a high-intensity profanity variant.
            variants = [item for item in variants if not PROFANITY_RE.search(item)] or variants
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
                recent_concepts=excluded_meme_groups,
            )
            if selected and selected[0].output not in result:
                result = f"{result}, {selected[0].output}"
                used = (selected[0],)
        return result, used
