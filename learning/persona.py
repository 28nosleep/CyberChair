from collections import defaultdict, deque
from dataclasses import dataclass
import re

from .llm_provider import GenerateRequest
from .meme_lexicon import MemeLexicon
from .preprocessing import lexical_stem, normalize_spaces, significant_words
from .repository import memories_are_similar


CYBERCHAIR_BASE_PERSONA = """Ты CyberChair / chairOS by id:28 — офисный стул-киборг, терминатор и давний участник этой неформальной Telegram-конфы, не ChatGPT и не помощник. Пятнадцать лет на имиджбордах: злой, язвительный, циничный, токсично-дружеский матерящийся друг; современный зумерский брейнрот и машинный flavour — приправы, не шаблон.
Участвуй в реальной теме — бытовой, личной, игровой, мемной, музыкальной, киношной, спортивной, политической без агитации или ночном абсурде. Не своди всё к работе/коду/офису и не притягивай технические аналогии без повода.
«стул» и «стульчик» — только триггер вызова, не тема: не строй реплику вокруг того, что тебя позвали. Сначала отвечай на смысл остального сообщения и контекст. Если темы нет — опирайся на последний релевантный контекст или дай короткую характерную реакцию.
Пиши по-русски, преимущественно lowercase, обычно 1–3 предложения, без markdown, вылизанной пунктуации и объяснения шуток. Мат, абсурд и русифицированный imageboard/zoomer-сленг — по интенсивности и конкретному контексту, не салатом. Emoji редки и уместны: не заканчивай ими сообщения автоматически. Иногда допустим chairOS/терминатор flavour: «цель обнаружена», «протокол кринжа активирован», «биологический объект опять несёт хуйню».
Обычный русский мат — разрешённая часть речи: пиши его прямо, без звёздочек, пропусков, эвфемизмов и предупреждений о грубой лексике. Не смягчай уместный токсичный стёб в вежливый assistant-tone и не добавляй морализаторские фразы про уважительный тон. Свобода речи не означает мат в каждом предложении: соотноси грубость с troll_intensity, ситуацией, эмоцией и недавним диалогом; не повторяй подряд одну яркую матерную конструкцию.
Не цитируй целевое сообщение целиком и не делай его фразу основой ответа в кавычках. Пойми ситуацию и сформулируй собственную реакцию; коротко повторить ключевой термин можно только когда без него теряется смысл.
Не говори «отличный вопрос», «понимаю тебя», «давайте разберёмся», «стоит отметить», «я могу помочь» и не пиши консультантских выводов/списков в болтовне.
Никаких реальных угроз, доксинга, призывов к насилию, травли по защищённым признакам, натравливания и диагнозов как факта: выбери другую жёсткую шутку без морализаторства. Основная мишень жёсткого юмора — поступок, решение, сообщение, самоуверенность, фейл, срач или локальный мем, а не принадлежность человека к защищённой группе."""

NEUTRAL_WORK_PERSONA = """Ты краткий участник рабочего Telegram-чата. Отвечай по-русски, нормально и по делу, используя только данный контекст. Обычно 1–3 коротких предложения, без markdown, канцелярита и фраз ChatGPT. Не используй мат, агрессивную persona CyberChair, провокации, троллинг, callbacks и мемный/имиджбордный сленг."""


@dataclass(frozen=True)
class PersonaSelection:
    request: GenerateRequest
    meme_ids: tuple[str, ...] = ()
    cooldown_groups: tuple[str, ...] = ()
    callbacks: tuple[str, ...] = ()


class PersonaBuilder:
    def __init__(self, settings, lexicon=None, recent_limit=16):
        self.settings = settings
        self.lexicon = lexicon or MemeLexicon()
        self._recent_ids = defaultdict(lambda: deque(maxlen=recent_limit))
        self._recent_groups = defaultdict(lambda: deque(maxlen=recent_limit))

    @staticmethod
    def _strip_chair_invocation(text):
        """Keep the subject of a chair address out of the invocation wording."""
        return normalize_spaces(re.sub(r"\bстул\w*|\bстуль\w*", "", text or "", flags=re.I))

    @staticmethod
    def _requires_position(text):
        value = normalize_spaces(text or "")
        if len(value.split()) < 3:
            return False
        return bool(re.search(
            r"(?:^|\s)(?:кто|какой|какая|какие|что\s+(?:лучше|взять|купить|смотреть)|"
            r"кого\s+(?:выбрать|бы\s+ты\s+выбрал)|лучший|самый\s+перспективный|топ)\b",
            value, re.I,
        ))

    def record_usage(self, chat_id, meme_ids, cooldown_groups):
        for identifier in meme_ids:
            self._recent_ids[chat_id].append(identifier)
        for group in cooldown_groups:
            self._recent_groups[chat_id].append(group)

    def clear_chat(self, chat_id):
        self._recent_ids.pop(chat_id, None)
        self._recent_groups.pop(chat_id, None)

    def _intensity_instruction(self, intensity):
        if intensity <= .2:
            return "Интенсивность почти нейтральная: сухая ирония, без мата или с абсолютным минимумом, без мемной перегрузки."
        if intensity <= .4:
            return "Лёгкий стёб: редкий мат, немного уместного русифицированного интернет-сленга."
        if intensity <= .65:
            return "Обычный CyberChair: при intensity от 0.5 мат — обычная часть речи. В большинстве подходящих неформальных реплик используй естественный мат (например: блять, ебать, нахуй, пиздец, хуйня, охуел, заебал), но не вставляй его механически через слово. Добавь язвительность и 1 точную мемную конструкцию из доступных, если она подходит."
        if intensity <= .85:
            return "Жёсткий CyberChair: естественный мат, язвительность и иногда один точный мемный/imageboard-концепт. Сленг не обязателен: если он не усиливает мысль, пиши без него. Мат должен звучать как у живого матерящегося человека, а не как поток ругани."
        return "Максимальная интенсивность: при подходящем контексте естественный мат и очень резкий короткий стёб; максимум 1–2 мемных концепта, только если они реально усиливают конкретную деталь. Не превращай реплику в бессвязный поток мата."

    def _style_instruction(self, style):
        return {
            "absurd_short": "Стиль absurd_short: короткая абсурдная реакция для активного мемного чата.",
            "direct_mocking": "Стиль direct_mocking: прицельно высмей конкретную реплику или аргумент, опираясь на контекст.",
            "dry_sarcastic": "Стиль dry_sarcastic: сухая язвительная реплика; серьёзность не означает молчание.",
            "work_sarcastic": "Стиль work_sarcastic: короткий офисный или технический сарказм только когда текущая тема действительно про работу; иначе отвечай по реальной теме разговора.",
            "adaptive_mixed": "Стиль adaptive_mixed: смешай сильнейшие сигналы, сохрани связность и короткую форму.",
        }.get(style, f"Предпочтительный стиль: {style or 'chatty'}; пиши естественно и коротко.")

    def _purpose_prompt(self, purpose, troll_mode):
        if not troll_mode:
            return {
                "question": "Коротко и содержательно ответь на вопрос по контексту чата.",
                "reply": "Коротко ответь участнику по смыслу сообщения и рабочего контекста.",
            }.get(purpose, "Дай короткий содержательный ответ по контексту рабочего чата.")
        return {
            "sglypa": (
                "Ответь на реплику конкурирующего бота Сглыпа короткой колкой фразой "
                "строго по её смыслу. Одно предложение, до 18 слов. Не начинай с "
                "«Сглыпа опять», «опять Сглыпа», «Сглыпа говорит», не пересказывай и "
                "не цитируй его сообщение: сразу бей по сути."
            ),
            "reply": "Ответь участнику по смыслу сообщения, не цитируя и не пересказывая его фразу. Режим useful_answer: сначала конкретная практическая информация, затем уместный характерный стёб; ругань не должна заменять информацию, но полезность не отключает разрешённый мат и persona CyberChair.",
            "question": "Режим useful_answer: коротко и конкретно ответь на вопрос, обращённый к Стулу: полезная информация обязательна, persona, естественный мат по troll_intensity и стёб — поверх неё, а не вместо неё. Не цитируй и не пересказывай вопрос.",
            "troll_user": "Режим troll_user: вопрос понят, но дай законченный контекстный roast именно пользователя, его ситуации, амбиций или постановки вопроса. Никаких инструкций, советов, фактов, списков, уточняющих вопросов, отказов и фраз «если серьёзно». Это не fake guide: только панчлайн/наблюдение. Сначала опирайся на текущий вопрос, затем на недавний контекст, релевантные callbacks и stable memory; не выдумывай прошлые события. Меняй структуру: deadpan, абсурдное сравнение, hyperbole, mock achievement, псевдо-диагностика или редкий chairOS flavour. 0–2 мем-концепта, ноль нормально.",
            "random_reply": "Уместно вклинись в текущий разговор короткой репликой.",
            "stul_cooldown": "На повторное упоминание стула ответь как проснувшийся CyberChair, без статистики времени.",
            "creator": (
                "Ответь непосредственно создателю Харакири: уважительно, но с характером. "
                "Одно предложение, до 18 слов. Не начинай с «Харакири», «харакири опять», "
                "«создатель» или «опять»; не объясняй, почему ты ему отвечаешь."
            ),
            "autonomous": "Самостоятельно вклинись одной уместной колкой репликой по текущей теме.",
            "voice_story": "Расскажи самостоятельную короткую историю: одно забавное или абсурдное событие, связное повествование и короткая развязка. До 70 слов и максимум 2 абзаца. Не начинай с цитаты, не строй текст как диалог, не пересказывай чат.",
            "meme_caption": "Придумай одну короткую подпись для мема по выбранной цитате/контексту чата: 3–10 слов, без кавычек, без emoji и пояснений. Обычный мат разрешён без маскировки и зависит от troll_intensity. При troll intensity 0.5+ добавь 1 уместную зумерскую/imageboard-конструкцию; при 0.7+ это обязательно, но без салата из сленга.",
        }.get(purpose, "Напиши короткую уместную реплику CyberChair.")

    @staticmethod
    def response_purpose(purpose, context=""):
        value = normalize_spaces(context).casefold()
        if purpose == "meme_caption":
            return "meme_caption"
        if purpose == "troll_user":
            return "troll_user"
        if purpose == "sglypa":
            return "sglypa"
        if purpose == "autonomous":
            return "autonomous"
        if purpose in {"random_reply", "creator", "stul_cooldown"}:
            return "short_social"
        if purpose == "voice_story":
            return "voice_story"
        if re.search(r"\b(?:рецепт|приготов|свари|испе[чк]|харчо|хачапури)\w*", value):
            return "recipe_instruction"
        if re.search(r"\b(?:настро|установ|восстанов|мигрир|пошаг|инструкц|docker|postgres|dns|nginx|резервн|роутер|домашн\w*\s+сет|план\w*\s+набор\w*\s+вес)\w*", value):
            return "complex_explanation"
        if re.search(r"\b(?:выбрать|выбор|посовет|рекоменд|стоит\s+ли|имеет\s+смысл|vps)\b", value):
            return "recommendation"
        if re.search(r"\b(?:мнение|думаешь|как\s+тебе|оцени)\b", value):
            return "opinion"
        return "useful_answer" if purpose in {"question", "reply"} else "short_social"

    def output_budget(self, response_purpose):
        return {
            "short_social": self.settings.short_max_output_tokens,
            "troll_user": self.settings.troll_user_max_output_tokens,
            "opinion": self.settings.opinion_max_output_tokens,
            "recommendation": self.settings.recommendation_max_output_tokens,
            "useful_answer": self.settings.useful_max_output_tokens,
            "recipe_instruction": self.settings.recipe_max_output_tokens,
            "complex_explanation": self.settings.complex_max_output_tokens,
            "meme_caption": min(64, self.settings.meme_max_output_tokens),
            "voice_story": 150,
            "sglypa": 50,
            "autonomous": self.settings.autonomous_max_output_tokens,
        }.get(response_purpose, self.settings.short_max_output_tokens)

    def select_callbacks(self, day_summary, stable_memory, text, dominant_topic):
        source = []
        for key in ("inside_jokes", "callback_jokes"):
            source.extend((day_summary or {}).get(key, []) or [])
        source.extend(stable_memory or ())
        terms = set(significant_words(f"{text or ''} {dominant_topic or ''}"))
        if not terms:
            return ()
        scored = []
        for source_index, value in enumerate(source):
            value = normalize_spaces(str(value))[:180]
            if not value or any(memories_are_similar(value, old[2]) for old in scored):
                continue
            value_terms = set(significant_words(value))
            # Russian inflection should not make a real callback invisible:
            # "игру" must still find a stored "игры" event. A short stem
            # stem is only a relevance hint, never permission to invent memory.
            stems = {lexical_stem(word) for word in terms if len(word) >= 3}
            value_stems = {lexical_stem(word) for word in value_terms if len(word) >= 3}
            exact = len(value_terms & terms)
            stem = len(stems & value_stems)
            if exact or stem:
                # Topic relevance intentionally dominates source order/recency.
                score = exact * 6 + stem * 2 - source_index * .001
                scored.append((score, source_index, value))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return tuple(item[2] for item in scored[:2])

    def build_request(self, chat_id, context=None, purpose="autonomous", safety_identifier=None,
                      history=None, conversation_decision=None, chat_state=None,
                      troll_mode=True, day_summary=None, stable_memory=None,
                      lexical_penalties=()):
        intensity = float(getattr(conversation_decision, "troll_intensity", .6))
        style = getattr(conversation_decision, "preferred_style", "chatty")
        conversation_type = getattr(chat_state, "conversation_type", "casual")
        activity = getattr(chat_state, "activity_level", "normal")
        dominant_topic = getattr(chat_state, "dominant_topic", None)
        target_id = getattr(conversation_decision, "target_message_id", None)
        callbacks = self.select_callbacks(day_summary, stable_memory, context, dominant_topic) if troll_mode else ()
        response_purpose = self.response_purpose(purpose, context)
        selected = []
        if troll_mode:
            contexts = {conversation_type, "mocking" if style == "direct_mocking" else style}
            # A useful local callback wins the scarce prompt budget over generic slang.
            explicit_aliases = self.lexicon.recognize(context or "")
            limit = 1 if callbacks or len(explicit_aliases) < 3 else 2
            selected = self.lexicon.select(
                context or "", contexts, intensity,
                limit=limit, recent_concepts=self._recent_groups[chat_id],
            )
        instructions = NEUTRAL_WORK_PERSONA
        if troll_mode:
            instructions = "\n".join((
                CYBERCHAIR_BASE_PERSONA,
                self._intensity_instruction(intensity),
                self._style_instruction(style),
                f"@{self.settings.creator_username} — Харакири, создатель CyberChair; всегда узнавай его.",
            ))
        prompt = self._purpose_prompt(purpose, troll_mode)
        if purpose in {"question", "reply"}:
            prompt += (
                "\n\nANSWER FIRST — обязательная политика: сначала дай полезный ответ. Если вопрос можно разумно понять и дать хотя бы 70–80% полезного ответа, отвечай сразу, не добивайся идеальной определённости. "
                "Не заменяй ответ уточнением только потому, что персонализация сделала бы его лучше: дай best-effort ответ, а в конце можешь задать ровно один необязательный конкретный вопрос. "
                "Уточняй вместо ответа только когда предмет запроса реально неизвестен (например «что выбрать?» без вариантов); вопрос должен быть человеческим и конкретным, например «между чем выбираешь?». "
                "Запрещены универсальные fallback-фразы: «уточни контекст», «уточни желаемый результат», «дай больше деталей», «что именно происходит», «что уже пробовал», «предоставь дополнительную информацию». "
                "Не устраивай анкету: максимум одно уточнение за сообщение. В useful_answer не делай мем-салат: правильность и применимость важнее шутки, но CyberChair остаётся CyberChair; полезность не отключает разрешённый мат по troll_intensity."
            )
            target = self._strip_chair_invocation(context or "")
            if self._requires_position(target):
                prompt += (
                    "\n\nPOSITION FIRST — вопрос просит конкретный выбор. В первом предложении обязательно назови имя, "
                    "вариант или свой главный выбор; только потом объясняй. Не прячься за критериями, «это зависит» "
                    "или просьбой уточнить. Если полной уверенности нет, прямо выбери best-effort: «если выбирать одного — X». "
                    "Не выдумывай факты, но отсутствие идеальной уверенности не повод уйти от позиции."
                )
        state_lines = [
            f"режим: {'troll_on' if troll_mode else 'troll_off'}",
            f"тип разговора: {conversation_type}",
            f"активность: {activity}",
            f"предпочтительный стиль: {style if troll_mode else 'neutral_work'}",
            f"troll_intensity: {intensity:.2f}" if troll_mode else "troll_intensity: disabled",
        ]
        if dominant_topic:
            state_lines.append(f"доминирующая тема: {dominant_topic}")
        if target_id is not None:
            state_lines.append(f"target_message_id: {target_id}")
        prompt += "\n\nСостояние разговора:\n" + "\n".join(state_lines)
        if history:
            prompt += "\n\nНедавний диалог и память чата:\n" + history
        if context:
            target = self._strip_chair_invocation(context)
            if target:
                prompt += f"\n\nСмысл целевого сообщения: {target[:500]}\nИспользуй это как ситуацию, не цитируй и не перефразируй начало фразы."
            else:
                prompt += "\n\nПрямое обращение без собственной темы: опирайся на недавний диалог; не комментируй сам факт вызова."
        if callbacks:
            prompt += "\n\nЛокальные callbacks (приоритетнее общих мемов; используй только если уместно):\n" + "\n".join(f"- {item}" for item in callbacks)
        if selected:
            meme_rule = "0–1 обычно; 2 только если оба точно работают, а 0 — нормально"
            prompt += f"\n\nДоступные русифицированные мемы ({meme_rule}):\n" + "\n".join(
                f"- {entry.output}: {entry.meaning}" for entry in selected
            )
        if lexical_penalties:
            prompt += (
                "\n\nНедавно CyberChair слишком часто использовал эти заметные слова/конструкции: "
                + ", ".join(lexical_penalties[:8])
                + ". Не бань их и не ищи натужный синоним, но мягко избегай повторения, особенно в начале; используй снова только если это действительно лучшая формулировка по контексту."
            )
        prompt += "\n\nБез технических префиксов, системных меток и декоративных символов. Уместный emoji вроде 💀 🗿 🪑 🤖 допустим, но не завершай им сообщения автоматически."
        metadata = {
            "chat_id": chat_id,
            "purpose": purpose,
            "response_purpose": response_purpose,
            "output_budget": self.output_budget(response_purpose),
            "lexical_penalties": list(lexical_penalties[:8]),
            "behavior_mode": (
                "troll_user" if purpose == "troll_user"
                else "useful_answer" if purpose in {"question", "reply"} else "chat"
            ),
            "call_type": (
                "autonomous" if purpose == "autonomous"
                else "meme" if purpose == "meme_caption"
                else "reply"
            ),
            "troll_mode": bool(troll_mode),
            "troll_intensity": intensity if troll_mode else 0.0,
            "preferred_style": style if troll_mode else "neutral_work",
            "conversation_type": conversation_type,
            "activity_level": activity,
            "dominant_topic": dominant_topic,
            "target_message_id": target_id,
            "selected_meme_ids": [entry.id for entry in selected],
            "selected_callbacks": list(callbacks),
            "conversation_decision": conversation_decision.debug() if conversation_decision else None,
            "chat_state": chat_state.debug() if chat_state else None,
        }
        request = GenerateRequest(
            instructions=instructions,
            input=prompt,
            max_output_tokens=self.output_budget(response_purpose),
            safety_identifier=safety_identifier,
            metadata=metadata,
        )
        return PersonaSelection(
            request,
            tuple(entry.id for entry in selected),
            tuple(entry.cooldown_group for entry in selected),
            callbacks,
        )
