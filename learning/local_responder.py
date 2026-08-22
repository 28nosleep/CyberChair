import re
from dataclasses import dataclass

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
    "reaction": ("ну приехали", "вижу", "прекрасно", "логично"),
    "judgement": (
        "а последствия опять назначишь виноватыми",
        "план уже спорит с реальностью",
        "сюжет сам себя пишет",
        "это уже не случайность, а выбранный стиль жизни",
    ),
    "closer": ("живём", "продолжай наблюдение", "записал в лор", "не останавливайся"),
    "machine": ("chairOS сверил протокол", "цель обнаружена", "система это запомнила"),
}


@dataclass(frozen=True)
class LocalCandidate:
    text: str
    signature: str
    opening_id: str | None = None
    fragment_ids: tuple[str, ...] = ()
    closer_id: str | None = None


class StructuredLocalResponse(str):
    def __new__(cls, candidate):
        value = str.__new__(cls, candidate.text)
        value.construction_signature = candidate.signature
        value.opening_id = candidate.opening_id
        value.fragment_ids = candidate.fragment_ids
        value.closer_id = candidate.closer_id
        return value

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
        candidates = [
            item if isinstance(item, LocalCandidate)
            else LocalCandidate(str(item), "short_reaction" if len(str(item).split()) <= 4 else "observation")
            for item in variants
        ]
        fresh = [item for item in candidates if item.text not in recent] or candidates
        structures = repository.recent_response_structures(40)
        if self.lexical_tracker:
            scored = []
            for index, item in enumerate(fresh):
                lexical = self.lexical_tracker.score(item.text, recent)[0]
                structural = sum(
                    8.0 if row["construction_signature"] == item.signature else 0.0
                    for row in structures[:12]
                )
                structural += sum(
                    4.0 if item.opening_id and row.get("opening_id") == item.opening_id else 0.0
                    for row in structures[:16]
                )
                used_fragments = {
                    fragment for row in structures[:12]
                    for fragment in row.get("fragment_ids", ())
                }
                structural += len(used_fragments & set(item.fragment_ids)) * 2.5
                scored.append((lexical + structural, index, item))
            best = min(item[0] for item in scored)
            if any(score > best + .25 for score, _, _ in scored):
                repository.record_routing_event("lexical_penalty_triggered")
            fresh = [item for score, _, item in scored if score <= best + .5]
        index = min(len(fresh) - 1, int(self.rng.random() * len(fresh)))
        return StructuredLocalResponse(fresh[index])

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
        # A characteristic tail is usually enough context and avoids wrapping
        # the whole incoming message in a template.
        return " ".join(words[-5:])[:80]

    def _contextual_callback(self, text, stable_memories, recent_dialogue, user_id):
        terms = set(self._terms(text))
        candidates = []
        # Stable summaries may shape a direct response as unattributed chat
        # context, but are never phrased as a sourced past quote.
        for value in stable_memories or ():
            clean = normalize_spaces(str(value))
            overlap = terms & set(self._terms(clean))
            if overlap:
                candidates.append((len(overlap), clean))
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
                LocalCandidate(
                    f"про «{topic}» вопрос понял, но без внешнего мозга содержательный ответ выдумывать не буду",
                    "topic_plus_limit", "topic_understood", ("provider_limit",),
                ),
                LocalCandidate(
                    f"вопрос про «{topic}» вижу. локально тут получится только фанфик",
                    "two_short_phrases", "question_seen", ("local_limit",),
                ),
                LocalCandidate(
                    f"{name}, по «{topic}» нужен фактический ответ; внешний мозг сейчас недоступен",
                    "address_plus_observation", "address", ("provider_limit",),
                ),
            ]
        candidates = []
        if callback:
            return [
                LocalCandidate(
                    f"контекст темы — «{callback}». сейчас в кадре «{focus}»",
                    "context_plus_observation", "topic_context", ("current_focus",), "current_frame",
                ),
                LocalCandidate(
                    f"на фоне «{callback}» нынешнее «{focus}» выглядит особенно выразительно",
                    "contextual_observation", "against_context", ("current_focus",),
                ),
            ]
        if focus:
            candidates.extend((
                LocalCandidate(f"{focus}. мощно", "short_reaction", None, ("focus",), "strong"),
                LocalCandidate(f"«{focus}» — последствия уже делают вид, что незнакомы", "quote_plus_judgement", "quote", ("consequences",)),
                LocalCandidate(f"у {name} сегодня отдельная сюжетная линия: {focus}", "contextual_observation", "participant_observation", ("focus",)),
                LocalCandidate(f"{focus}. оправдания разминаются у входа", "two_short_phrases", None, ("focus",), "excuses"),
                LocalCandidate(f"по факту — {focus}; по ощущениям — начало длинной арки", "parallel_observation", "in_fact", ("focus", "arc")),
                LocalCandidate(f"{name}, «{focus}» — заявка в постоянную рубрику", "short_roast", "address", ("focus", "recurring_bit")),
                LocalCandidate(f"и как именно «{focus}» должно пережить встречу с реальностью?", "rhetorical_question", "how_exactly", ("focus", "reality")),
                LocalCandidate(f"сюжет дня: {focus}", "label_observation", "daily_plot", ("focus",)),
                LocalCandidate(f"сначала решение. потом «{focus}». отличный таймлайн", "three_beats", "first_decision", ("focus",), "timeline"),
                LocalCandidate(f"записал: {focus}. без выводов, они тут сами появятся", "observation_plus_closer", "recorded", ("focus",), "self_explaining"),
            ))
        reaction = FRAGMENTS["reaction"]
        judgement = FRAGMENTS["judgement"]
        for left, right in zip(reaction, judgement):
            candidates.append(LocalCandidate(
                f"{left}: {focus or topic}. {right}",
                "reaction_plus_observation", f"reaction_{left}",
                (f"judgement_{right}",),
            ))
        if troll_mode and focus and self.rng.random() < .04:
            candidates.append(LocalCandidate(
                f"{FRAGMENTS['machine'][0]}: {focus}; {FRAGMENTS['closer'][2]}",
                "machine_observation", "machine_protocol", ("focus",), "lore_recorded",
            ))
        return candidates

    def _history_pattern_candidates(self, text, recent_generated):
        """Never copy a recent reply's construction as a pseudo-new template."""
        del text, recent_generated
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
        del stable_memories
        variants = [
            LocalCandidate(f"«{detail}» — финальный босс даже не понял, что его вызвали", "quote_plus_roast", "quote", ("boss",)),
            LocalCandidate(f"достижение разблокировано: «{detail}»", "achievement", "achievement", ("detail",)),
            LocalCandidate(f"{detail}. реальность подписку не оформляла", "two_short_phrases", None, ("detail",), "reality_declines"),
            LocalCandidate(f"по формулировке «{detail}» у {topic} уже второй сезон", "contextual_observation", "by_wording", ("detail", "topic", "season")),
            LocalCandidate(f"«{detail}» — взрослый интернет сейчас должен сделать домашку?", "rhetorical_question", "adult_internet", ("detail", "homework")),
            LocalCandidate(f"«{detail}»: меню настроек открыто, переговоры с реальностью провалены", "parallel_observation", "settings_menu", ("detail", "reality")),
        ]
        result = self._choose_variant(variants, repository, recent_generated)
        selected = self.lexicon.select(
            text, {"mocking", "humor"}, .7, excluded_meme_ids,
            excluded_meme_groups, limit=1, recent_concepts=excluded_meme_groups,
        )
        if selected and selected[0].output not in result and self.rng.random() < .35:
            result = StructuredLocalResponse(LocalCandidate(
                f"{result}, {selected[0].output}",
                result.construction_signature,
                result.opening_id,
                tuple(result.fragment_ids) + (f"meme_{selected[0].id}",),
                result.closer_id,
            ))
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
            variants = [
                item for item in variants
                if not PROFANITY_RE.search(
                    item.text if isinstance(item, LocalCandidate) else item
                )
            ] or variants
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
                result = StructuredLocalResponse(LocalCandidate(
                    f"{result}, {selected[0].output}",
                    result.construction_signature,
                    result.opening_id,
                    tuple(result.fragment_ids) + (f"meme_{selected[0].id}",),
                    result.closer_id,
                ))
                used = (selected[0],)
        return result, used
