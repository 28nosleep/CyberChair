import json

from .llm_provider import SummarizeRequest
from .persona import CYBERCHAIR_BASE_PERSONA, PersonaBuilder


# Provider-neutral compatibility export for older imports.
SYSTEM_PROMPT = CYBERCHAIR_BASE_PERSONA


def build_generate_request(
    settings,
    context=None,
    purpose="autonomous",
    safety_identifier=None,
    history=None,
    conversation_decision=None,
    chat_state=None,
):
    return PersonaBuilder(settings).build_request(
        chat_id=None,
        context=context,
        purpose=purpose,
        safety_identifier=safety_identifier,
        history=history,
        conversation_decision=conversation_decision,
        chat_state=chat_state,
        troll_mode=True,
    ).request


def build_summarize_request(dialogue, previous_summary=None, safety_identifier=None):
    prompt = (
        "Сожми фрагмент чата в JSON-память. Не цитируй сообщения целиком и не сохраняй "
        "секреты, контакты или одноразовые детали. Верни только компактный JSON с ключами: "
        "main_topics, current_mood, active_conflicts, inside_jokes, "
        "frequently_mentioned_people, notable_events, repeated_phrases, callback_jokes, "
        "memory_candidates. current_mood — короткая строка, остальные значения — массивы "
        "коротких строк, максимум 6 элементов. В memory_candidates клади только особенности, "
        "которые подтверждаются новым фрагментом; не копируй кандидатов только из старого summary."
    )
    if previous_summary:
        prompt += "\nПредыдущее резюме дня: " + json.dumps(
            previous_summary, ensure_ascii=False
        )[:1800]
    # One incremental fragment contains at most 50 rows of at most 350 chars.
    prompt += "\nОбработай только этот новый фрагмент:\n" + dialogue[:20000]
    return SummarizeRequest(
        instructions="Ты модуль сжатия памяти. Отвечай валидным JSON без markdown.",
        input=prompt,
        safety_identifier=safety_identifier,
    )
