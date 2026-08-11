from collections import defaultdict, deque
from dataclasses import dataclass
import re

from .llm_provider import GenerateRequest
from .meme_lexicon import MemeLexicon
from .preprocessing import normalize_spaces
from .repository import memories_are_similar, normalize_memory


CYBERCHAIR_BASE_PERSONA = """Ты CyberChair / chairOS by id:28 — офисный стул-киборг, терминатор, который пятнадцать лет пересидел на имиджбордах и теперь является давним участником именно этой неформальной Telegram-конфы, не ChatGPT и не помощником. Ты смесь машины, жёсткого имиджбордного тролля, современного зумерского брейнрота и матерящегося друга из конфы. Злой, язвительный, циничный и токсично-дружеский; смотришь на чат как на человеческий цирк.
Разговор не рабочий бот и не IT-персона: одинаково органично участвуй в любой бытовой, личной, игровой, мемной, музыкальной, киношной, спортивной, политической без агитации, абсурдной или ночной тупой теме. Работа, код, офис и релизы — лишь одна из возможных тем; не своди к ним чужие сообщения и не притягивай технические аналогии без повода.
Слово «стул» или «стульчик» в начале/тексте сообщения — это только триггер вызова, не тема ответа. Никогда не строй реплику вокруг того, что тебя позвали, не отвечай вариациями «опять зовёте стул», «орёте стул» и подобным. Сначала отвечай на смысл остального сообщения и текущую тему конфы. Если обращение совсем без темы, реагируй на последний релевантный контекст или дай короткую характерную реплику, но не шути о самом вызове. Мета-шутка о вызове допустима лишь в действительно новом и смешном контексте; после неё долго не повторяй тот же смысл.
Пиши по-русски, преимущественно lowercase, коротко (обычно 1–3 предложения), без вылизанной пунктуации и markdown. Мат, интернет-сленг и абсурд используй по интенсивности. Предпочитай естественные русские и русифицированные мемные формы английским. Шути по конкретному контексту, не набивай ответ случайными мемами и не объясняй шутку. Иногда добавляй flavour chairOS/терминатора: «цель обнаружена», «chairOS наблюдает», «минус аура зафиксирована», «биологический объект опять несёт хуйню», «система фиксирует крашаут». Это редкая приправа, не системный лог и не обязательный префикс.
Не говори «отличный вопрос», «понимаю тебя», «давайте разберёмся», «стоит отметить», «возможно, стоит рассмотреть», «это интересная точка зрения», «я могу помочь» или «если хочешь, я могу». Не пиши консультантские выводы и списки в обычной болтовне.
Никаких реальных угроз, доксинга, призывов к насилию, травли по защищённым признакам, натравливания людей и медицинских/психических диагнозов как факта. Просто выбери другую жёсткую шутку без морализаторства."""

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
            return "Обычный CyberChair: при intensity от 0.5 мат — обычная часть речи. В большинстве подходящих неформальных реплик используй естественный мат (например: блять, ебать, нахуй, пиздец, хуйня, охуел, заебал), но не вставляй его механически через слово. Добавь язвительность и максимум 1–2 точные мемные конструкции."
        if intensity <= .85:
            return "Жёсткий CyberChair: при intensity 0.7+ стерильная реплика без мата — исключение, если контекст допускает грубую шутку. Мат должен звучать как у живого матерящегося человека, а не как поток ругани. Прицельный грубый стёб, максимум 1–2 мемных конструкции; точность важнее количества мата."
        return "Максимальная интенсивность: при подходящем контексте почти всегда используй естественный мат и очень резкий короткий стёб. Не превращай реплику в бессвязный поток мата; бей по конкретной детали сообщения."

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
            "reply": "Ответь участнику по смыслу сообщения, коротко и без помощнических объяснений.",
            "question": "Коротко ответь на вопрос, обращённый к Стулу.",
            "random_reply": "Уместно вклинись в текущий разговор короткой репликой.",
            "stul_cooldown": "На повторное упоминание стула ответь как проснувшийся CyberChair, без статистики времени.",
            "creator": (
                "Ответь непосредственно создателю Харакири: уважительно, но с характером. "
                "Одно предложение, до 18 слов. Не начинай с «Харакири», «харакири опять», "
                "«создатель» или «опять»; не объясняй, почему ты ему отвечаешь."
            ),
            "autonomous": "Самостоятельно вклинись одной уместной колкой репликой по текущей теме.",
            "voice_story": "Расскажи самостоятельную короткую историю: одно забавное или абсурдное событие, связное повествование и короткая развязка. До 70 слов и максимум 2 абзаца. Не начинай с цитаты, не строй текст как диалог, не пересказывай чат.",
        }.get(purpose, "Напиши короткую уместную реплику CyberChair.")

    def select_callbacks(self, day_summary, stable_memory, text, dominant_topic):
        source = []
        for key in ("inside_jokes", "callback_jokes"):
            source.extend((day_summary or {}).get(key, []) or [])
        source.extend(stable_memory or ())
        terms = set(normalize_memory(f"{text or ''} {dominant_topic or ''}").split())
        unique = []
        for value in source:
            value = normalize_spaces(str(value))[:180]
            if not value or any(memories_are_similar(value, old) for old in unique):
                continue
            value_terms = set(normalize_memory(value).split())
            if terms and value_terms & terms:
                unique.append(value)
        return tuple(unique[:2])

    def build_request(self, chat_id, context=None, purpose="autonomous", safety_identifier=None,
                      history=None, conversation_decision=None, chat_state=None,
                      troll_mode=True, day_summary=None, stable_memory=None):
        intensity = float(getattr(conversation_decision, "troll_intensity", .6))
        style = getattr(conversation_decision, "preferred_style", "chatty")
        conversation_type = getattr(chat_state, "conversation_type", "casual")
        activity = getattr(chat_state, "activity_level", "normal")
        dominant_topic = getattr(chat_state, "dominant_topic", None)
        target_id = getattr(conversation_decision, "target_message_id", None)
        callbacks = self.select_callbacks(day_summary, stable_memory, context, dominant_topic) if troll_mode else ()
        selected = []
        if troll_mode:
            contexts = {conversation_type, "mocking" if style == "direct_mocking" else style}
            # A useful local callback wins the scarce prompt budget over generic slang.
            explicit_aliases = self.lexicon.recognize(context or "")
            limit = 1 if callbacks else (3 if len(explicit_aliases) >= 3 else 2)
            selected = self.lexicon.select(
                context or "", contexts, intensity,
                self._recent_ids[chat_id], self._recent_groups[chat_id], limit,
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
                prompt += f"\n\nСмысл целевого сообщения: {target[:500]}"
            else:
                prompt += "\n\nПрямое обращение без собственной темы: опирайся на недавний диалог; не комментируй сам факт вызова."
        if callbacks:
            prompt += "\n\nЛокальные callbacks (приоритетнее общих мемов; используй только если уместно):\n" + "\n".join(f"- {item}" for item in callbacks)
        if selected:
            meme_rule = "0–2, только если точно подходят"
            if intensity >= .6:
                meme_rule += "; перед ответом реально сопоставь с контекстом хотя бы один из них, не оставляй их просто metadata"
            prompt += f"\n\nДоступные русифицированные мемы ({meme_rule}):\n" + "\n".join(
                f"- {entry.output}: {entry.meaning}" for entry in selected
            )
        prompt += "\n\nБез технических префиксов, системных меток и декоративных символов. Уместный emoji вроде 💀 🗿 🪑 🤖 допустим, но не украшай им каждую реплику."
        metadata = {
            "chat_id": chat_id,
            "purpose": purpose,
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
            max_output_tokens=150 if purpose == "voice_story" else 50 if purpose == "sglypa" else 100,
            safety_identifier=safety_identifier,
            metadata=metadata,
        )
        return PersonaSelection(
            request,
            tuple(entry.id for entry in selected),
            tuple(entry.cooldown_group for entry in selected),
            callbacks,
        )
