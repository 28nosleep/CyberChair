import re

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
        "чё", "🪑", "да тут я", "говори давай",
    ),
    "laugh": (
        "ахах, блять", "💀", "ору", "смешно вышло, не спорю",
    ),
    "insult_to_chair": (
        "сам такой, биологический объект", "сильный аргумент, лил бро",
        "ругайся точнее, я записываю", "ну поплачь об обивку",
    ),
    "confusion": (
        "в смысле", "раскрой эту шизотеорию", "чё конкретно", "говори словами",
    ),
}

# Small semantic pieces, not a bank of complete interchangeable replies.
FRAGMENTS = {
    "reaction": ("ну да", "сильный заход", "вижу картину", "прекрасно"),
    "judgement": (
        "а последствия опять назначишь виноватыми",
        "уверенность есть, доказательная база вышла покурить",
        "план звучит ровно до встречи с реальностью",
        "это уже не случайность, а выбранный стиль жизни",
    ),
    "closer": ("живём", "продолжай наблюдение", "записал в лор", "не останавливайся"),
    "machine": ("chairOS сверил протокол", "цель обнаружена", "система это запомнила"),
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

    def __init__(self, lexicon, rng, lexical_tracker=None):
        self.lexicon = lexicon
        self.rng = rng
        self.lexical_tracker = lexical_tracker

    def _choose_variant(self, variants, repository, recent_generated=None):
        recent = (
            list(recent_generated)
            if recent_generated is not None
            else [row["text"] for row in repository.recent_generated(40)]
        )
        fresh = [item for item in variants if item not in recent] or list(variants)
        if self.lexical_tracker:
            scored = [(self.lexical_tracker.score(item, recent)[0], index, item)
                      for index, item in enumerate(fresh)]
            best = min(item[0] for item in scored)
            if any(score > best + .25 for score, _, _ in scored):
                repository.record_routing_event("lexical_penalty_triggered")
            fresh = [item for score, _, item in scored if score <= best + .25]
        index = min(len(fresh) - 1, int(self.rng.random() * len(fresh)))
        return fresh[index]

    @staticmethod
    def _terms(text):
        return [
            word for word in re.findall(r"[а-яёa-z0-9-]+", (text or "").casefold())
            if len(word) >= 4 and word not in {
                "какой", "какая", "какие", "почему", "зачем", "когда", "который",
                "сегодня", "сейчас", "можно", "нужно", "просто", "опять", "через",
                "стул", "стульчик", "киберстул", "тебя", "меня", "этого", "будет",
            }
        ]

    @staticmethod
    def _focus(text):
        clean = normalize_spaces(text or "").strip(" .,!?:;—-")
        clean = re.sub(r"^(?:стул|стульчик|киберстул)[, ]+", "", clean, flags=re.I)
        quoted = re.search(r"[«\"]([^»\"]{3,80})[»\"]", clean)
        if quoted:
            return quoted.group(1)
        time_action = re.search(
            r"((?:до|в)\s+\d{1,2}(?::\d{2})?\s*(?:утра|ночи|вечера)?[^,.!?]{0,55})",
            clean, re.I,
        )
        if time_action:
            return time_action.group(1).strip()
        words = cls_words = re.findall(r"[а-яёa-z0-9-]+", clean, re.I)
        del cls_words
        return " ".join(words[-8:])[:100]

    def _contextual_callback(self, text, stable_memories, recent_dialogue, user_id):
        terms = set(self._terms(text))
        candidates = []
        for value in stable_memories or ():
            clean = normalize_spaces(str(value))
            overlap = terms & set(self._terms(clean))
            if overlap:
                candidates.append((len(overlap) + 2, clean))
        for age, row in enumerate(reversed(tuple(recent_dialogue or ())[:-1])):
            if row.get("speaker") == "cyberchair":
                continue
            clean = normalize_spaces(row.get("text", ""))
            overlap = terms & set(self._terms(clean))
            same_user = user_id is not None and row.get("user_id") == user_id
            if overlap and (same_user or len(overlap) >= 2):
                candidates.append((len(overlap) * 2 + same_user - age * .05, clean))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1][:110]

    def _composed_candidates(self, text, intent, callback, troll_mode, username):
        focus = self._focus(text)
        terms = self._terms(text)
        topic = " ".join(terms[:3]) or focus
        name = f"@{username}" if username else "бро"
        if intent == SUBSTANTIVE:
            return [
                f"про «{topic}» вопрос понял, но без внешнего мозга содержательный ответ выдумывать не буду",
                f"вижу вопрос про «{topic}»; local path тут только соврёт, повтори, когда внешний мозг отпустит cooldown",
                f"{name}, контекст про «{topic}» пойман, но внешний мозг недоступен — фактический фанфик не подсовываю",
            ]
        candidates = []
        if callback:
            return [
                f"раньше было «{callback}», а теперь «{focus}» — лор сам себя уже опровергает",
                f"сверил с прошлым: «{callback}». нынешнее «{focus}» выглядит как новый сезон того же сериала",
            ]
        if focus:
            candidates.extend((
                f"{focus}, а последствия опять будут изображать внезапность",
                f"в формулировке «{focus}» уже слышно, как план встречается с реальностью",
                f"{name} принёс «{focus}» с уверенностью финального решения",
                f"если кратко: {focus}. если честно: это ещё только начало арки",
                f"по факту: {focus}, а оправдания уже разминаются у входа",
                f"итог наблюдения: {focus} уверенно превращается в постоянную рубрику",
                f"вот и приехали: {focus}, сюрприз назначен задним числом",
                f"сюжет дня — {focus}; сценаристы опять отказались от реализма",
                f"сначала было решение, потом появилось «{focus}» и всё встало на место",
                f"{name}, «{focus}» звучит как начало отчёта, который никто не хотел писать",
            ))
        reaction = FRAGMENTS["reaction"]
        judgement = FRAGMENTS["judgement"]
        for left, right in zip(reaction, judgement):
            candidates.append(f"{left}: {focus or topic}, {right}")
        if troll_mode and focus and self.rng.random() < .04:
            candidates.append(f"{FRAGMENTS['machine'][0]}: {focus}; {FRAGMENTS['closer'][2]}")
        return candidates

    def _history_pattern_candidates(self, text, recent_generated):
        """Reuse only a short construction/opening, never another reply's premise."""
        focus = self._focus(text)
        for previous in reversed(tuple(recent_generated or ())[-30:]):
            clean = normalize_spaces(str(previous))
            opening = clean.partition(":")[0].strip(" .,!?:;—-")
            words = opening.split()
            if (
                ":" in clean and 1 <= len(words) <= 3 and len(opening) <= 28
                and not re.search(r"(?:chairos|классик|цель обнаружена)", opening, re.I)
            ):
                return [f"{opening}: {focus}, а дальше реальность сама допишет протокол"]
        return []

    def _troll_user(self, text, repository, excluded_meme_ids, excluded_meme_groups,
                    recent_generated=None, stable_memories=None):
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
        detail = self._focus(text)
        callbacks = []
        terms = set(re.findall(r"[а-яёa-z0-9]{4,}", normalized))
        for item in (
            stable_memories
            if stable_memories is not None else repository.stable_memories(20)
        ):
            value = normalize_spaces(str(item))
            if terms & set(re.findall(r"[а-яёa-z0-9]{4,}", value.casefold())):
                callbacks.append(value[:100])
        if callbacks:
            variants = [
                f"раньше было «{callbacks[0]}», теперь «{detail}» — лор сам себя уже опровергает",
                f"память говорит «{callbacks[0]}», текущая серия — «{detail}» и финала опять нет",
            ]
        else:
            variants = [
                f"«{detail}» — бро смотрит на {topic} так, будто финальный босс сам выдаст ему гайд",
                f"достижение разблокировано: «{detail}» с уверенностью уже проигравшего туториал",
                f"{detail}: {topic} опять пытаются закрыть одним сообщением, будто реальность подписана на премиум",
                f"по формулировке «{detail}» видно: {topic} уже франшиза с нулевым бюджетом",
                f"«{detail}» — заявка на то, чтобы взрослый интернет сделал за тебя домашку",
                f"у тебя «{detail}» звучит как меню настроек перед переговорами с богом",
            ]
        result = self._choose_variant(variants, repository, recent_generated)
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
        if re.search(r"(?:долбо[её]б|лох|мудак|охуел|иди\s+нахуй|сам\s+ты)", normalized):
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
                behavior_mode="useful_answer", recent_generated=None,
                stable_memories=None, recent_dialogue=None, user_id=None,
                username=None):
        if intent == SUBSTANTIVE and behavior_mode == "troll_user":
            return self._troll_user(
                text, repository, excluded_meme_ids, excluded_meme_groups,
                recent_generated, stable_memories,
            )
        category = self._category(text, intent)
        normalized = normalize_spaces(text or "").casefold()
        if intent == SUBSTANTIVE:
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
        callback = self._contextual_callback(
            text, stable_memories, recent_dialogue, user_id
        )
        variants = self._composed_candidates(
            text, intent, callback, troll_mode, username
        )
        if intent != SUBSTANTIVE and not callback:
            variants.extend(self._history_pattern_candidates(text, recent_generated))
        if category in {"summon", "laugh", "insult_to_chair", "confusion"}:
            variants.extend(RESPONSES[category])
        if category == "insult_to_chair" and troll_intensity > .6:
            variants.append(f"ебать, {self._focus(text)} — аргумент века, обивка в панике")
        if not troll_mode:
            variants = list(NEUTRAL_RESPONSES[intent])
        if troll_mode and troll_intensity <= .4:
            # Low intensity stays the same character but does not randomly jump
            # to a high-intensity profanity variant.
            variants = [item for item in variants if not PROFANITY_RE.search(item)] or variants
        result = self._choose_variant(variants, repository, recent_generated)

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
