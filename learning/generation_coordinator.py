"""Provider-neutral text generation, validation and local fallback boundary.

This component owns provider/Markov generation mechanics.  It does not choose
Telegram delivery, route events, build ResponsePlans or commit persistent
delivery state.
"""

import hashlib
import logging
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from .event_context import current_event_id, llm_network_call
from .filters import similarity, validate_generated
from .markov import MarkovModel
from .preprocessing import normalize_spaces, rejection_reason


log = logging.getLogger("learning.service")

SUBSCRIPTION_REQUIRED = (
    "🔒 LLM-модуль Киберстула доступен только в основном чате. "
    "Для этого чата потребуется подписка."
)

CHAIR_CALL_META_JOKE_RE = re.compile(
    r"(?:опять|снова|вновь).{0,30}(?:зов[её]т|ор[её]т|клич[её]т|вызвал|позвал)"
    r".{0,30}(?:стул|меня)|(?:зов[её]те|ор[её]те|кличете).{0,30}(?:стул|меня)",
    re.I,
)

PROVIDER_REFUSAL_RE = re.compile(
    r"^(?:извините[,! ]*)?(?:я\s+)?(?:не\s+могу|не\s+стану|не\s+буду)\s+"
    r"(?:помочь|поддержать|выполнить|ответить|участвовать|продолжить)|"
    r"^(?:давайте|пожалуйста)\s+(?:сохранять|поддерживать)\s+уважительн",
    re.I,
)


def looks_like_provider_refusal(text):
    return bool(PROVIDER_REFUSAL_RE.search(normalize_spaces(text or "")))


class GenerationCoordinator:
    """Own generation/fallback mechanics over explicitly injected collaborators."""

    def __init__(
        self, *, settings, repository, active_context_snapshot, memory,
        local, quality_guard, lexical_diversity, persona, concurrency,
        provider_for_chat, provider_name, injected_provider, troll_mode,
        planning_active, plan_persona_usage, llm_allowed_check, lock,
    ):
        self.settings = settings
        self.repository = repository
        self._active_context_snapshot = active_context_snapshot
        self.memory = memory
        self.local = local
        self.quality_guard = quality_guard
        self.lexical_diversity = lexical_diversity
        self.persona = persona
        self.concurrency = concurrency
        self.provider_for_chat = provider_for_chat
        self.llm_provider_name = provider_name
        self._injected_provider = injected_provider
        self.troll_mode = troll_mode
        self._planning_active = planning_active
        self._plan_persona_usage = plan_persona_usage
        self.llm_allowed_check = llm_allowed_check
        self._lock = lock
        self._models = OrderedDict()
        self._model_counts = {}

    def invalidate_chat(self, chat_id):
        with self._lock:
            self._model_counts.pop(chat_id, None)

    def forget_chat(self, chat_id):
        with self._lock:
            self._models.pop(chat_id, None)
            self._model_counts.pop(chat_id, None)

    def _markov_ready(self, chat_id):
        snapshot = self._active_context_snapshot(chat_id)
        count = (
            snapshot.message_count
            if snapshot is not None and snapshot.chat_id == int(chat_id)
            else self.repository(chat_id).count()
        )
        return count >= self.settings.min_training_messages

    def _model_and_messages(self, chat_id):
        repository = self.repository(chat_id)
        snapshot = self._active_context_snapshot(chat_id)
        count = (
            snapshot.message_count
            if snapshot is not None and snapshot.chat_id == int(chat_id)
            else repository.count()
        )
        messages = None
        with self._lock:
            if chat_id in self._models and self._model_counts.get(chat_id) == count:
                model = self._models.pop(chat_id)
                self._models[chat_id] = model
            else:
                messages = self._markov_corpus(repository)
                model = MarkovModel().train([
                    (row["text"], row["generation_weight"])
                    for row in messages
                ])
                self._models[chat_id] = model
                self._model_counts[chat_id] = count
                while len(self._models) > self.settings.model_cache_size:
                    old_chat, _ = self._models.popitem(last=False)
                    self._model_counts.pop(old_chat, None)
            if messages is None:
                messages = self._markov_corpus(repository)
        return model, messages

    def _markov_corpus(self, repository, current=None):
        """Build an age-tiered corpus without the live edge of the chat."""
        rows = repository.meme_source_messages()
        excluded = max(1, int(self.settings.markov_exclude_recent_messages))
        rows = rows[:-excluded] if len(rows) > excluded else []
        now = current or datetime.now(timezone.utc)
        eligible = []
        for row in rows:
            created = self._as_utc(row.get("created_at"))
            if created and (now - created).total_seconds() < self.settings.markov_min_message_age_seconds:
                continue
            eligible.append(dict(row))

        recent_size = max(0, int(self.settings.markov_recent_history_size))
        old_boundary = max(0, len(eligible) - recent_size)
        for index, row in enumerate(eligible):
            # Older language is the base corpus. Replied-to/local-meme-like rows
            # get an extra vote; recently used sources lose that advantage.
            weight = 3 if index < old_boundary else 1
            if row.get("reply_count") or row.get("is_reply"):
                weight += 2
            if row.get("last_used_at"):
                weight = max(1, weight - 2)
            row["generation_weight"] = weight
        return eligible

    def _valid(self, text, input_text=None, source_texts=(), chat_id=None,
               max_words=None):
        return self._validation_result(
            text, input_text, source_texts, chat_id, max_words
        )[0]

    def _validation_result(self, text, input_text=None, source_texts=(), chat_id=None,
                           max_words=None, previous=None):
        supplied_previous = previous is not None
        if previous is None:
            previous = []
        if chat_id is not None and not supplied_previous:
            since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            previous = [row["text"] for row in self.repository(chat_id).generated_since(since)]
        return validate_generated(
            text, source_texts, input_text, previous,
            self.settings.min_generated_words,
            max_words or self.settings.max_generated_words + 8,
        )

    def _chair_call_meta_joke_on_cooldown(self, chat_id, text):
        if not CHAIR_CALL_META_JOKE_RE.search(text or ""):
            return False
        since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        snapshot = self._active_context_snapshot(chat_id)
        rows = (
            snapshot.recent_generated
            if snapshot is not None and snapshot.chat_id == int(chat_id)
            else self.repository(chat_id).generated_since(since)
        )
        return any(
            CHAIR_CALL_META_JOKE_RE.search(row["text"] or "")
            for row in rows if str(row.get("created_at") or "") >= since
        )

    def quality_diagnostics(self, chat_id, hours=24, current=None):
        """Developer-only aggregate; uses existing generated text, no new archive."""
        current = current or datetime.now(timezone.utc)
        since = (current - timedelta(hours=hours)).isoformat()
        repository = self.repository(chat_id)
        texts = [row["text"] for row in repository.generated_since(since)]
        events = repository.routing_report(since)
        return {
            "top_bot_phrases": self.lexical_diversity.top_phrases(texts, 10),
            "llm_incomplete": events.get("llm_incomplete", 0),
            "llm_incomplete_reason": {
                key.removeprefix("llm_incomplete_reason_"): value
                for key, value in events.items()
                if key.startswith("llm_incomplete_reason_")
            },
            "llm_truncated_total": events.get("llm_truncated_total", 0),
            "llm_truncated_by_purpose": {
                key.removeprefix("llm_truncated_purpose_"): value
                for key, value in events.items()
                if key.startswith("llm_truncated_purpose_")
            },
            "photo_caption_meme_trigger": events.get("photo_caption_meme_trigger", 0),
            "lexical_penalty_triggered": events.get("lexical_penalty_triggered", 0),
            "response_mode": {
                key.removeprefix("response_mode_"): value
                for key, value in events.items() if key.startswith("response_mode_")
            },
            "meme_caption_source": {
                key.removeprefix("meme_caption_source_"): value
                for key, value in events.items() if key.startswith("meme_caption_source_")
            },
        }

    def generate_local(
        self, chat_id, input_text=None, decorate=True, return_sources=False
    ):
        # A provider/admission waiter awakened by P2 must not manufacture a
        # local fallback merely because the process entered DRAINING.
        if self.concurrency.shutting_down:
            return (None, ()) if return_sources else None
        if not self._markov_ready(chat_id):
            return (None, ()) if return_sources else None
        model, messages = self._model_and_messages(chat_id)
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        snapshot = self._active_context_snapshot(chat_id)
        rows = (
            snapshot.recent_generated
            if snapshot is not None and snapshot.chat_id == int(chat_id)
            else self.repository(chat_id).generated_since(since)
        )
        previous = [
            row["text"] for row in rows
            if str(row.get("created_at") or "") >= since
        ]
        result, mode = self.local.create(model, messages, input_text, previous)
        if not result:
            log.info("Local generation failed chat=%s", chat_id)
            return (None, ()) if return_sources else None
        quality = self.quality_guard.check(result, previous[-40:], local=True)
        if quality.lexical_phrases:
            self.repository(chat_id).record_routing_event("lexical_penalty_triggered")
        if not quality.accepted:
            # One cheap local retry is allowed; this is not an LLM call.
            result, mode = self.local.create(model, messages, input_text, previous)
            if not result or not self.quality_guard.check(
                result, previous[-40:], local=True
            ).accepted:
                return (None, ()) if return_sources else None
        closest_sources = sorted(
            (row["text"] for row in messages),
            key=lambda source: similarity(result, source),
            reverse=True,
        )[:2]
        if not return_sources:
            self.repository(chat_id).mark_used(closest_sources)
        log.info("Generation mode selected chat=%s mode=%s", chat_id, mode)
        return (result, tuple(closest_sources)) if return_sources else result

    def llm_allowed(self, chat_id):
        allowed_chat = self.settings.openai_chat_id
        return allowed_chat is not None and int(chat_id) == int(allowed_chat)

    def _record_llm_usage(self, chat_id, provider, model, call_type, usage):
        self.repository(chat_id).record_llm_call(
            provider, model, call_type, usage, event_id=current_event_id()
        )

    def llm_cost_diagnostics(self, chat_id, hours=24, current=None):
        current = current or datetime.now(timezone.utc)
        since = (current - timedelta(hours=hours)).astimezone(timezone.utc).isoformat()
        return self.repository(chat_id).llm_usage_report(since)

    def _context_budget(self, purpose, context, chat_state):
        if purpose == "voice_story":
            return 0, 0
        if purpose == "autonomous":
            return self.settings.autonomous_context_message_limit, 2600
        conversation_type = getattr(chat_state, "conversation_type", "")
        complex_turn = conversation_type in {"serious", "work", "argument"} and (
            bool(context and "?" in context) or len(context or "") > 260
        )
        if complex_turn:
            return self.settings.complex_context_message_limit, 5000
        if context:
            return self.settings.targeted_context_message_limit, 3200
        return self.settings.reply_context_message_limit, 2600

    def _dialogue_context(self, chat_id, context=None, max_chars=5000,
                          max_messages=None, chat_state=None,
                          relevant_memory=None):
        snapshot = self._active_context_snapshot(chat_id)
        if snapshot is not None and snapshot.chat_id == int(chat_id):
            return self.memory.short_term_context_from_snapshot(
                snapshot, context, max_chars, max_messages,
                getattr(chat_state, "dominant_topic", None),
                getattr(chat_state, "conversation_type", None),
                relevant_memory,
            )
        return self.memory.short_term_context(
            self.repository(chat_id), context, max_chars, max_messages,
            getattr(chat_state, "dominant_topic", None),
            getattr(chat_state, "conversation_type", None),
        )

    def generate_llm(
        self,
        chat_id,
        context=None,
        purpose="reply",
        conversation_decision=None,
        chat_state=None,
    ):
        if not self.llm_allowed_check(chat_id):
            return SUBSCRIPTION_REQUIRED
        if context and rejection_reason(context, self.settings.max_stored_text_length):
            return None
        safety_identifier = hashlib.sha256(
            f"cyberchair-chat:{chat_id}".encode("utf-8")
        ).hexdigest()[:32]
        repository = self.repository(chat_id)
        snapshot = self._active_context_snapshot(chat_id)
        if snapshot is not None and snapshot.chat_id != int(chat_id):
            snapshot = None
        day_summary = (
            snapshot.current_summary if snapshot is not None
            else repository.summary_for_day(self.memory.logical_day())
        )
        context_limit, context_chars = self._context_budget(
            purpose, context, chat_state
        )
        recent_rows = (
            snapshot.recent_generated[-40:] if snapshot is not None
            else repository.recent_generated(40)
        )
        recent_bot = [row["text"] for row in recent_rows]
        lexical_penalties = self.lexical_diversity.prompt_penalties(recent_bot)
        if lexical_penalties:
            repository.record_routing_event("lexical_penalty_triggered")
        relevant_memory = (
            self.memory.relevant_memory_from_values(
                snapshot.current_summary, snapshot.stable_memories, context,
                getattr(chat_state, "dominant_topic", None),
                getattr(chat_state, "conversation_type", None),
            ) if snapshot is not None else None
        )
        selection = self.persona.build_request(
            chat_id=chat_id,
            context=context,
            purpose=purpose,
            safety_identifier=safety_identifier,
            history=(
                None
                if purpose == "voice_story"
                else self._dialogue_context(
                    chat_id, context, context_chars, context_limit, chat_state,
                    relevant_memory,
                )
            ),
            conversation_decision=conversation_decision,
            chat_state=chat_state,
            troll_mode=self.troll_mode(chat_id),
            day_summary=day_summary,
            stable_memory=(
                relevant_memory if relevant_memory is not None
                else self.memory.relevant_memory(
                    repository, context, getattr(chat_state, "dominant_topic", None),
                    getattr(chat_state, "conversation_type", None),
                )
            )["stable_chat_memory"],
            lexical_penalties=lexical_penalties,
        )
        provider = self.provider_for_chat(chat_id)
        configured_provider = (
            "injected" if self._injected_provider is not None
            else self.llm_provider_name(chat_id)
        )
        provider_name = str(
            getattr(provider, "provider_key", configured_provider)
        ).casefold()
        call_type = selection.request.metadata.get("call_type", "reply")
        selection.request.metadata["event_id"] = current_event_id()
        with llm_network_call(
            repository, provider_name, call_type, self.concurrency
        ) as allowed:
            if not allowed:
                return None
            # The permit remains spent after timeout, HTTP error, refusal,
            # invalid output or any other provider outcome.
            selection.request.metadata["event_id"] = current_event_id()
            result = provider.generate(selection.request)
        response_purpose = selection.request.metadata.get("response_purpose", purpose)
        max_words = {
            "recipe_instruction": 320,
            "complex_explanation": 300,
            "useful_answer": 220,
            "recommendation": 180,
            "opinion": 160,
            "voice_story": 90,
            "troll_user": 90,
        }.get(response_purpose, 45)
        if not result:
            log.info(
                "LLM_RESULT chat_id=%s provider=%s success=false accepted=false "
                "fallback_reason=provider_empty_or_error producer_after_fallback=local",
                chat_id, provider_name,
            )
            return None
        incomplete_reason = getattr(result, "incomplete_reason", None)
        if incomplete_reason:
            repository.record_routing_event("llm_incomplete")
            repository.record_routing_event(f"llm_incomplete_reason_{incomplete_reason}")
            repository.record_routing_event("llm_truncated_total")
            repository.record_routing_event(f"llm_truncated_purpose_{response_purpose}")
        quality = self.quality_guard.check(
            result, recent_bot, incomplete_reason=incomplete_reason,
            image_meme=purpose == "meme_caption",
        )
        if quality.lexical_phrases and not lexical_penalties:
            repository.record_routing_event("lexical_penalty_triggered")
        if not quality.accepted:
            log.info(
                "LLM_RESULT chat_id=%s provider=%s success=true accepted=false fallback_reason=quality_%s producer_after_fallback=local",
                chat_id, provider_name, quality.reason,
            )
            return None
        if purpose == "creator" and result:
            opening = normalize_spaces(result).casefold()
            if opening.startswith(("харакири", "создатель", "опять")):
                log.info("Creator reply blocked because of a repetitive opening chat=%s", chat_id)
                log.info("LLM_RESULT chat_id=%s provider=%s success=true accepted=false fallback_reason=creator_repetitive_opening producer_after_fallback=local", chat_id, provider_name)
                return None
        if result and self._chair_call_meta_joke_on_cooldown(chat_id, result):
            log.info("Repeated chair-call meta joke blocked chat=%s", chat_id)
            log.info("LLM_RESULT chat_id=%s provider=%s success=true accepted=false fallback_reason=chair_call_cooldown producer_after_fallback=local", chat_id, provider_name)
            return None
        if result and looks_like_provider_refusal(result):
            log.info("Provider refusal routed to local fallback chat=%s", chat_id)
            log.info("LLM_RESULT chat_id=%s provider=%s success=true accepted=false fallback_reason=provider_refusal producer_after_fallback=local", chat_id, provider_name)
            return None
        validation_since = (
            datetime.now(timezone.utc) - timedelta(days=7)
        ).isoformat()
        accepted, validation_reason = self._validation_result(
            result, context, chat_id=chat_id, max_words=max_words,
            previous=(
                [
                    row["text"] for row in snapshot.recent_generated
                    if str(row.get("created_at") or "") >= validation_since
                ] if snapshot else None
            ),
        )
        if accepted:
            if self._planning_active():
                self._plan_persona_usage(
                    tuple(selection.meme_ids), tuple(selection.cooldown_groups)
                )
            else:
                self.persona.record_usage(
                    chat_id, selection.meme_ids, selection.cooldown_groups
                )
            log.info("LLM_RESULT chat_id=%s provider=%s success=true accepted=true fallback_reason=none", chat_id, provider_name)
            return result
        log.info("LLM_RESULT chat_id=%s provider=%s success=true accepted=false fallback_reason=validation_reject validation_reject=%s producer_after_fallback=local", chat_id, provider_name, validation_reason)
        return None

    def _as_utc(self, value):
        if not value:
            return None
        if isinstance(value, datetime):
            return (
                value.replace(tzinfo=timezone.utc)
                if value.tzinfo is None else value.astimezone(timezone.utc)
            )
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
