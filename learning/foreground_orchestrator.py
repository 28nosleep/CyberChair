"""Foreground domain orchestration for one normalized event.

The orchestrator owns existing route coordination and final-plan arbitration.
It does not parse Telegram DTOs, deliver Telegram payloads, implement provider,
media or memory algorithms, or commit a delivered ResponsePlan.
"""

import hashlib
import logging
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from .direct_address import SOCIAL, SUBSTANTIVE
from .media_service import MediaDecision
from .pending_conversation import (
    PendingConversation, choice_declined, extract_choice_alternatives,
    looks_like_continuation, question_intent,
)
from .preprocessing import normalize_spaces, significant_words
from .response_selector import ResponseKind
from .response_plan import (
    EvidenceUsageCommit, GeneratedCommit, MediaUsageCommit, PendingFinalize, PersonaUsageCommit,
    PolicyTargetCommit, Producer, RoutingCommit, SourceUsageCommit,
    StructureUsageCommit, TriggerCommit,
)


# Preserve the established diagnostic logger while moving code ownership.
log = logging.getLogger("learning.service")


class ForegroundOrchestrator:
    """Own route ordering for foreground user events."""

    def __init__(
        self, *, settings, repository, normalize_event, active_snapshot,
        as_utc, enabled, context_snapshot, media_context_snapshot,
        create_response_plan, pending_create_action, delivery_type_for_media,
        prepare_reaction_response,
        provider_for_chat, provider_available, generate_llm,
        llm_allowed, troll_mode, media_enabled,
        budget_exceeded, chair_meta_on_cooldown,
        message_context, response_planning, planning_requested,
        deterministic_media_roll, policy_quiet_hours,
        response_activity, ensure_action_visible, chat_state_analyzer,
        conversation_policy, direct_router, local_responder, date_time_utility,
        media, memory, persona, meme_lexicon, relationship_model,
        moment_detector, evidence_engine, response_selector, triggers,
        rng, lock, clock,
    ):
        self.settings = settings
        self.repository = repository
        self._normalized_event = normalize_event
        self._active_context_snapshot = active_snapshot
        self._as_utc = as_utc
        self._enabled = enabled
        self.context_snapshot = context_snapshot
        self.media_context_snapshot = media_context_snapshot
        self._create_response_plan = create_response_plan
        self._pending_create_action = pending_create_action
        self._delivery_type_for_media = delivery_type_for_media
        self.prepare_reaction_response = prepare_reaction_response
        self.provider_for_chat = provider_for_chat
        self.provider_available = provider_available
        self.generate_llm = generate_llm
        self.llm_allowed = llm_allowed
        self.troll_mode = troll_mode
        self.media_enabled = media_enabled
        self._budget_exceeded = budget_exceeded
        self._chair_call_meta_joke_on_cooldown = chair_meta_on_cooldown
        self._message_context = message_context
        self.response_planning = response_planning
        self._planning_requested = planning_requested
        self.deterministic_media_roll = deterministic_media_roll
        self.policy_quiet_hours = policy_quiet_hours
        self._response_activity = response_activity
        self._ensure_action_visible = ensure_action_visible
        self.chat_state_analyzer = chat_state_analyzer
        self.conversation_policy = conversation_policy
        self.direct_router = direct_router
        self.local_responder = local_responder
        self.date_time_utility = date_time_utility
        self.media = media
        self.memory = memory
        self.persona = persona
        self.meme_lexicon = meme_lexicon
        self.relationship_model = relationship_model
        self.moment_detector = moment_detector
        self.evidence_engine = evidence_engine
        self.response_selector = response_selector
        self.triggers = triggers
        self.rng = rng
        self._lock = lock
        self._clock = clock
        self._last_policy_target_user = {}
        self._policy_target_streak = {}
        self._policy_answered_messages = {}
        self._last_chat_state = {}
        self._last_conversation_decision = {}
        self._last_direct_decision = {}
        self._voice_cooldown_notices = {}

    def bind_runtime_ports(self, **ports):
        """Refresh injected facade ports without taking a service back-reference."""
        for name, value in ports.items():
            setattr(self, name, value)

    def forget_chat(self, chat_id):
        with self._lock:
            self._last_policy_target_user.pop(chat_id, None)
            self._policy_target_streak.pop(chat_id, None)
            self._policy_answered_messages.pop(chat_id, None)
            self._last_chat_state.pop(chat_id, None)
            self._last_conversation_decision.pop(chat_id, None)
            self._last_direct_decision.pop(chat_id, None)
            self._voice_cooldown_notices.pop(chat_id, None)

    def _policy_quiet_hours(self):
        hour = datetime.now(self.memory._timezone).hour
        if self.settings.quiet_start_hour > self.settings.quiet_end_hour:
            return hour >= self.settings.quiet_start_hour or hour < self.settings.quiet_end_hour
        return self.settings.quiet_start_hour <= hour < self.settings.quiet_end_hour

    def conversation_diagnostics(self, chat_id):
        state = self._last_chat_state.get(chat_id)
        decision = self._last_conversation_decision.get(chat_id)
        return {
            "state": state.debug() if state else None,
            "decision": decision.debug() if decision else None,
        }


    def _remember_policy_target(self, chat_id, message):
        event = self._normalized_event(message)
        self._remember_policy_identity(chat_id, event.user_id, event.message_id)

    def _remember_policy_identity(self, chat_id, actual_user, actual_message):
        if actual_user is not None:
            self._last_policy_target_user[chat_id] = actual_user
            previous_user, previous_count = self._policy_target_streak.get(
                chat_id, (None, 0)
            )
            self._policy_target_streak[chat_id] = (
                actual_user,
                previous_count + 1 if previous_user == actual_user else 1,
            )
        if actual_message is not None:
            answered = self._policy_answered_messages.setdefault(chat_id, [])
            answered.append(actual_message)
            del answered[:-20]

    def _deterministic_media_roll(self, chat_id, message_id, salt):
        digest = hashlib.sha256(
            f"{chat_id}:{message_id}:{salt}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def _record_direct_result(
        self, chat_id, message, producer, result,
        behavior_mode="useful_answer", *, as_plan=False,
        pending_finalize_user_id=None, persona_usage=None, source_usage=(),
    ):
        event = self._normalized_event(message)
        repository = self.repository(chat_id)
        route = result.action if isinstance(result, MediaDecision) else producer
        if route == "ai":
            route = "llm"
        if as_plan:
            repository.record_routing_event(
                f"route_selected_{route}", event_id=event.event_id
            )
            response_mode = (
                "troll_user" if behavior_mode == "troll_user"
                else "local" if route == "local" else "useful"
            )
            actions = []
            if pending_finalize_user_id is not None:
                actions.append(PendingFinalize(pending_finalize_user_id))
            if isinstance(result, MediaDecision):
                generated_value = result.template_id or result.asset_id or "media"
                actions.append(MediaUsageCommit(result))
                final_producer = (
                    Producer.MEME if result.action == "meme" else Producer.MEDIA
                )
            else:
                generated_value = result
                final_producer = {
                    "llm": Producer.LLM,
                    "local": Producer.LOCAL,
                }.get(route, Producer.LOCAL)
            actions.extend((
                RoutingCommit(route, response_mode),
                GeneratedCommit(generated_value, f"direct_{route}"),
                PolicyTargetCommit(event.user_id, event.message_id),
            ))
            if persona_usage:
                actions.append(PersonaUsageCommit(*persona_usage))
            if source_usage:
                actions.append(SourceUsageCommit(tuple(source_usage)))
            if hasattr(result, "construction_signature"):
                actions.append(StructureUsageCommit(
                    result.construction_signature,
                    result.opening_id,
                    tuple(result.fragment_ids),
                    result.closer_id,
                ))
            if isinstance(result, str) and behavior_mode != "troll_user":
                pending_create = self._pending_create_action(
                    event, result, "substantive"
                )
                if pending_create is not None:
                    actions.append(pending_create)
            provider_key = None
            if final_producer == Producer.LLM:
                provider_key = str(
                    getattr(self.provider_for_chat(chat_id), "provider_key", "llm")
                )
            return self._create_response_plan(
                event, result, final_producer, "direct",
                behavior_mode=behavior_mode, required=True,
                actions=actions, provider_key=provider_key,
            )
        repository.record_routing_event(f"route_{route}")
        response_mode = (
            "troll_user" if behavior_mode == "troll_user"
            else "local" if route == "local" else "useful"
        )
        repository.record_routing_event(f"response_mode_{response_mode}")
        repository.record_generated(
            (result.template_id or result.asset_id or "media")
            if isinstance(result, MediaDecision) else result,
            f"direct_{route}",
        )
        if hasattr(result, "construction_signature"):
            repository.record_response_structure(
                result.construction_signature, result.opening_id,
                result.fragment_ids, result.closer_id,
            )
        self._remember_policy_target(chat_id, event)
        if isinstance(result, str) and behavior_mode != "troll_user":
            self._store_pending_from_response(message, result, "substantive")
        return result

    def _substantive_behavior_mode(self, chat_id):
        """Choose once, locally, before selecting the single response producer."""
        if not self.troll_mode(chat_id):
            return "useful_answer"
        probability = max(0.0, min(1.0, float(self.settings.troll_user_probability)))
        return "troll_user" if self.rng.random() < probability else "useful_answer"

    def _store_pending_from_response(self, message, response, intent):
        event = self._normalized_event(message)
        action = self._pending_create_action(event, response, intent)
        if action is None:
            return False
        self.repository(event.chat_id).save_pending_conversation(
            user_id=action.user_id,
            original_message_id=action.original_message_id,
            original_question=action.original_question,
            clarification_question=action.clarification_question,
            intent=action.intent,
            context=action.context,
            expected_type=action.expected_type,
            pending_mode=action.mode,
        )
        return True

    def pending_conversation(self, chat_id, user_id, current=None):
        snapshot = self._active_context_snapshot(chat_id)
        row = None
        if (
            snapshot is not None
            and snapshot.chat_id == int(chat_id)
            and snapshot.pending is not None
            and snapshot.pending.get("user_id") == user_id
        ):
            candidate = snapshot.pending
            created = self._as_utc(candidate.get("created_at"))
            now = current or datetime.now(timezone.utc)
            now = self._as_utc(now)
            if created and (now - created).total_seconds() <= (
                self.settings.pending_conversation_ttl_seconds
            ):
                row = candidate
        else:
            row = self.repository(chat_id).pending_conversation(
                user_id, self.settings.pending_conversation_ttl_seconds, current,
            )
        if not row:
            return None
        created = self._as_utc(row["created_at"])
        return PendingConversation(
            chat_id=row["chat_id"], user_id=row["user_id"],
            bot_message_id=row["bot_message_id"],
            original_message_id=row["original_message_id"],
            original_question=row["original_question"],
            clarification_question=row["clarification_question"],
            intent=row["intent"], context=row["context"],
            expected_type=row["expected_type"], mode=row["pending_mode"],
            created_at=created,
        )

    def attach_pending_bot_message(self, incoming_message, sent_message):
        event = self._normalized_event(incoming_message)
        user_id = event.user_id
        bot_message_id = getattr(sent_message, "message_id", getattr(sent_message, "id", None))
        if user_id is None or not isinstance(bot_message_id, int):
            return False
        self.repository(event.chat_id).attach_pending_bot_message(
            user_id, bot_message_id
        )
        return True

    def is_pending_continuation(self, message, bot_id=None, current=None):
        event = self._normalized_event(message)
        user_id = event.user_id
        if user_id is None:
            return False
        pending = self.pending_conversation(event.chat_id, user_id, current)
        if pending is None:
            return False
        strong_reply = bool(
            event.reply_to_message_id is not None
            and (
                pending.bot_message_id is not None
                and event.reply_to_message_id == pending.bot_message_id
                or bot_id is not None
                and event.reply_to_user_id == bot_id
            )
        )
        return strong_reply or looks_like_continuation(
            event.effective_text, pending.expected_type, pending.mode
        )

    def _local_continuation_fallback(self, pending, answer):
        if pending.expected_type == "measurements":
            numbers = [float(value.replace(",", ".")) for value in re.findall(r"\d+(?:[.,]\d+)?", answer)]
            if len(numbers) >= 2:
                height, weight = numbers[0], numbers[1]
                if height > 3:
                    height /= 100
                bmi = weight / max(.5, height) ** 2
                return (
                    f"при этих данных имт около {bmi:.1f}; стартуй с +250–350 ккал к поддержанию и 110–135 г белка, две недели без роста — докинь ещё 150–200 ккал"
                )
        if pending.expected_type == "budget":
            return f"с бюджетом {normalize_spaces(answer)} уже можно резать варианты по реальной цене; бери лучший по главному сценарию, а не по маркетинговым мегапикселям"
        if pending.expected_type == "choices":
            alternatives = extract_choice_alternatives(answer)
            if len(alternatives) >= 2:
                return f"между {alternatives[0]} и {alternatives[1]} сначала сравни главный сценарий, цену и что бесит ежедневно; без этого выбор будет чистым глейзингом бренда"
            return "варианты не извлеклись — не буду придумывать выбор из воздуха"
        return f"принял: {normalize_spaces(answer)}. теперь по исходной теме уже можно отвечать без гадания; chairOS контекст не проебал"

    def maybe_pending_continuation(
        self, message, bot_id=None, current=None, _as_plan=False
    ):
        """Consume one fresh same-user pending turn and produce exactly one reply."""
        _as_plan = self._planning_requested(_as_plan)
        event = self._normalized_event(message)
        snapshot = self.context_snapshot(event) if _as_plan else None
        chat_id = event.chat_id
        if not self._enabled(chat_id, "talk"):
            return None
        user_id = event.user_id
        pending = self.pending_conversation(chat_id, user_id, current) if user_id is not None else None
        if pending and pending.expected_type == "choices" and choice_declined(
            event.effective_text
        ):
            self.repository(chat_id).clear_pending_conversation(user_id)
            return None
        if pending is None or not self.is_pending_continuation(event, bot_id, current):
            return None
        repository = self.repository(chat_id)
        if not _as_plan:
            repository.clear_pending_conversation(user_id)
        repository.record_routing_event("pending_continuations")
        repository.record_routing_event("intent_substantive")
        state = self.chat_state_analyzer.analyze(
            repository, incoming_message=event, bot_id=bot_id,
            last_target_user_id=self._last_policy_target_user.get(chat_id),
            answered_message_ids=self._policy_answered_messages.get(chat_id, ()),
            snapshot=snapshot,
        )
        conversation = self.conversation_policy.decide(
            state, addressed=True, local_allowed=True,
            llm_allowed=self.llm_allowed(chat_id), quiet_hours=False,
        )
        self._last_chat_state[chat_id] = state
        self._last_conversation_decision[chat_id] = conversation
        answer = normalize_spaces(event.effective_text)
        context = (
            "Продолжение разговора.\n"
            f"Исходный вопрос пользователя: {pending.original_question[:500]}\n"
            f"Предыдущий ответ CyberChair: {pending.context[:500]}\n"
            f"CyberChair запросил: {pending.clarification_question[:180]}\n"
            f"Новая информация пользователя: {answer[:500]}\n"
            "Продолжи исходную тему и обязательно ответь по существу; не спрашивай, что означает новая информация."
        )
        with self._response_activity(chat_id, "typing", "llm") as action:
            result = None
            if self.llm_allowed(chat_id) and self.provider_available(chat_id):
                result = self.generate_llm(
                    chat_id, context, "reply",
                    conversation_decision=conversation, chat_state=state,
                )
            if result:
                return self._record_direct_result(
                    chat_id, event, "llm", result, as_plan=_as_plan,
                    pending_finalize_user_id=user_id if _as_plan else None,
                )
            repository.record_routing_event("llm_fallback_local")
            result = self._local_continuation_fallback(pending, answer)
            self._ensure_action_visible(action, "local", result)
            return self._record_direct_result(
                chat_id, event, "local", result, as_plan=_as_plan,
                pending_finalize_user_id=user_id if _as_plan else None,
            )

    def prepare_pending_continuation(self, event, bot_id=None, current=None):
        with self.response_planning():
            return self.maybe_pending_continuation(
                event, bot_id=bot_id, current=current, _as_plan=True
            )

    def maybe_direct_reply(
        self, message, bot_id=None, bot_username=None,
        explicit_address=False, _as_plan=False,
    ):
        """Return exactly one final producer result for an explicit address/reply."""
        _as_plan = self._planning_requested(_as_plan)
        event = self._normalized_event(message)
        snapshot = self.context_snapshot(event) if _as_plan else None
        chat_id = event.chat_id
        if not self._enabled(chat_id, "talk"):
            return None
        user_id = event.user_id
        if user_id is not None:
            if self.is_pending_continuation(event, bot_id=bot_id):
                return self.maybe_pending_continuation(
                    event, bot_id=bot_id, _as_plan=_as_plan
                )
            # An explicit new turn supersedes stale conversational ambiguity.
            if not _as_plan:
                self.repository(chat_id).clear_pending_conversation(user_id)
        text = event.effective_text
        direct_reply = event.replies_to_user(bot_id)
        mentioned = bool(bot_username and f"@{bot_username}".casefold() in text.casefold())
        special = any(phrase in text.casefold() for phrase in self.settings.special_phrases)
        if not (explicit_address or direct_reply or mentioned or special):
            return None

        repository = self.repository(chat_id)
        repository.record_routing_event(
            "direct_replies" if direct_reply else "direct_addresses"
        )
        subject = text
        if explicit_address or special:
            subject = self.persona._strip_chair_invocation(subject)
        if mentioned and bot_username:
            subject = re.sub(
                rf"@{re.escape(bot_username)}\b", "", subject, flags=re.I
            ).strip()

        factual_answer = self.date_time_utility.answer(subject)
        if factual_answer:
            repository.record_routing_event("intent_date_time")
            with self._response_activity(chat_id, "typing", "local") as action:
                self._ensure_action_visible(action, "local", factual_answer)
            return self._record_direct_result(
                chat_id, event, "local", factual_answer, "date_time",
                as_plan=_as_plan,
                pending_finalize_user_id=(user_id if _as_plan else None),
            )

        state = self.chat_state_analyzer.analyze(
            repository,
            incoming_message=event,
            bot_id=bot_id,
            last_target_user_id=self._last_policy_target_user.get(chat_id),
            answered_message_ids=self._policy_answered_messages.get(chat_id, ()),
            snapshot=snapshot,
        )
        ai_available = self.llm_allowed(chat_id) and self.provider_available(chat_id)
        budget_exceeded = self._budget_exceeded(chat_id)
        social_ai_useful = (
            direct_reply
            and len(subject.split()) >= 8
            and state.conversation_type in {"serious", "work"}
        )
        route = self.direct_router.decide(
            subject,
            direct_reply=direct_reply,
            ai_available=ai_available,
            budget_exceeded=budget_exceeded,
            social_ai_useful=social_ai_useful,
        )
        detected_intent = (
            question_intent(subject)
            if route.intent == SUBSTANTIVE else route.intent
        )
        behavior_mode = (
            self._substantive_behavior_mode(chat_id)
            if route.intent == SUBSTANTIVE else "chat"
        )
        log.info(
            "DIRECT_ROUTE chat_id=%s intent=%s substantive=%s mode=%s priority=%s producer=%s reason=%s",
            chat_id, detected_intent, route.intent == SUBSTANTIVE,
            behavior_mode, route.priority, route.producer, route.reason,
        )
        repository.record_routing_event(f"intent_{route.intent}")
        self._last_direct_decision[chat_id] = route

        conversation = self.conversation_policy.decide(
            state,
            addressed=True,
            local_allowed=route.producer != "llm",
            llm_allowed=route.producer == "llm",
            quiet_hours=False,
        )
        self._last_chat_state[chat_id] = state
        self._last_conversation_decision[chat_id] = conversation

        # Contextual media is a free final response only for non-substantive
        # turns. A meme decision is skipped here because rendering failure must
        # not consume the guaranteed direct reply.
        if route.intent != SUBSTANTIVE and self.media_enabled(chat_id):
            if snapshot is not None:
                snapshot = self.media_context_snapshot(snapshot)
            day_summary = (
                snapshot.current_summary if snapshot is not None
                else repository.summary_for_day(self.memory.logical_day()) or {}
            )
            callbacks = self.persona.select_callbacks(
                day_summary,
                snapshot.stable_memories if snapshot is not None
                else repository.stable_memories(20),
                subject,
                state.dominant_topic,
            )
            selected_memes = self.meme_lexicon.select(
                subject, {state.conversation_type, conversation.preferred_style},
                conversation.troll_intensity, limit=1,
            )
            media = self.media.decide(
                chat_id, repository, conversation, state,
                snapshot.recent_dialogue if snapshot is not None
                else self.memory.short_term_rows(repository), subject,
                selected_memes, callbacks, self.troll_mode(chat_id),
                self.deterministic_media_roll(chat_id, event.message_id, "direct_media"),
                self.deterministic_media_roll(chat_id, event.message_id, "direct_meme"),
                self.deterministic_media_roll(chat_id, event.message_id, "direct_reaction"),
                media_context=(snapshot.media if snapshot is not None else None),
            )
            if media.action in {"gif", "sticker"}:
                if not _as_plan:
                    self.media.commit(repository, media)
                return self._record_direct_result(
                    chat_id, event, media.action, media, as_plan=_as_plan,
                    pending_finalize_user_id=(user_id if _as_plan else None),
                )

        selected_producer = "llm" if route.producer == "llm" else "local"
        with self._response_activity(chat_id, "typing", selected_producer) as action:
            if route.producer == "llm":
                result = self.generate_llm(
                    chat_id, self._message_context(event),
                    "troll_user" if behavior_mode == "troll_user" else "reply",
                    conversation_decision=conversation, chat_state=state,
                )
                if result:
                    return self._record_direct_result(
                        chat_id, event, "llm", result, behavior_mode,
                        as_plan=_as_plan,
                        pending_finalize_user_id=(user_id if _as_plan else None),
                    )
                repository.record_routing_event("llm_fallback_local")

            result, memes = self.local_responder.respond(
                chat_id, subject, route.intent, repository,
                self.persona._recent_ids[chat_id], self.persona._recent_groups[chat_id],
                self.troll_mode(chat_id),
                conversation.troll_intensity,
                behavior_mode,
                (
                    snapshot.recent_generated_texts[-40:]
                    if snapshot is not None else None
                ),
                snapshot.stable_memories if snapshot is not None
                else repository.stable_memories(20),
                snapshot.recent_dialogue if snapshot is not None
                else repository.recent_messages(40),
                user_id=event.user_id,
                username=event.username,
            )
            persona_usage = None
            if memes:
                persona_usage = (
                    tuple(item.id for item in memes),
                    tuple(item.cooldown_group for item in memes),
                )
            if memes and not _as_plan:
                self.persona.record_usage(
                    chat_id,
                    *persona_usage,
                )
            self._ensure_action_visible(action, "local", result)
            return self._record_direct_result(
                chat_id, event, "local", result, behavior_mode,
                as_plan=_as_plan,
                pending_finalize_user_id=(user_id if _as_plan else None),
                persona_usage=persona_usage,
            )

    def prepare_direct_reply(
        self, event, bot_id=None, bot_username=None, explicit_address=False
    ):
        with self.response_planning():
            return self.maybe_direct_reply(
                event, bot_id=bot_id, bot_username=bot_username,
                explicit_address=explicit_address, _as_plan=True,
            )

    def maybe_reply(
        self, message, bot_id=None, bot_username=None, _as_plan=False
    ):
        _as_plan = self._planning_requested(_as_plan)
        event = self._normalized_event(message)
        snapshot = self.context_snapshot(event) if _as_plan else None
        chat_id = event.chat_id
        if not self._enabled(chat_id, "talk"):
            return None
        text = event.effective_text
        replies_to_bot = event.replies_to_user(bot_id)
        mentioned = bool(bot_username and f"@{bot_username}".casefold() in text.casefold())
        special = any(phrase in text.casefold() for phrase in self.settings.special_phrases)
        addressed = replies_to_bot or mentioned or special
        if addressed:
            return self.maybe_direct_reply(
                event, bot_id=bot_id, bot_username=bot_username,
                explicit_address=False, _as_plan=_as_plan,
            )
        if not self.troll_mode(chat_id) and not addressed:
            return None
        repository = self.repository(chat_id)
        if (
            repository.count() < self.settings.min_training_messages
            and self.triggers.observed_message_count(chat_id) > 0
        ):
            return None
        state = self.chat_state_analyzer.analyze(
            repository,
            incoming_message=event,
            bot_id=bot_id,
            last_target_user_id=self._last_policy_target_user.get(chat_id),
            answered_message_ids=self._policy_answered_messages.get(chat_id, ()),
            snapshot=snapshot,
        )
        local_allowed = (
            False if addressed else self.triggers.allowed(chat_id, "random")
        )
        llm_kind = "addressed" if addressed else "openai_random"
        llm_allowed = self.triggers.allowed(chat_id, llm_kind, addressed=addressed)
        incoming_user_id = event.user_id
        streak_user, streak_count = self._policy_target_streak.get(chat_id, (None, 0))
        if (
            not addressed
            and incoming_user_id is not None
            and incoming_user_id == streak_user
            and streak_count >= 2
        ):
            local_allowed = False
            llm_allowed = False
        decision = self.conversation_policy.decide(
            state,
            addressed=addressed,
            local_allowed=local_allowed,
            llm_allowed=llm_allowed,
            quiet_hours=self.policy_quiet_hours(),
        )
        self._last_chat_state[chat_id] = state
        self._last_conversation_decision[chat_id] = decision
        if decision.action == "none":
            return None
        roll = self.rng.random()
        if roll >= decision.reply_probability:
            return None
        if addressed:
            kind = "addressed"
        elif roll < decision.local_probability:
            kind = "random"
        else:
            kind = "openai_random"
        social_rows = (
            snapshot.recent_dialogue if snapshot is not None
            else repository.recent_messages(self.moment_detector.max_messages)
        )
        moments = self.moment_detector.detect(social_rows, event.message_id)
        moment = moments[0] if moments else None
        relationship = (
            self.relationship_model.current(repository, event.user_id)
            if event.user_id is not None else None
        )
        evidence_candidates = self.evidence_engine.retrieve(
            repository, text, moment, event.user_id, event.message_id,
            current=event.timestamp,
        )
        if snapshot is not None and self.media_enabled(chat_id):
            snapshot = self.media_context_snapshot(snapshot)
        bot_rows = (
            snapshot.recent_generated[-6:] if snapshot is not None
            else repository.recent_generated(6)
        )
        current_at = self._as_utc(event.timestamp) or datetime.now(timezone.utc)
        activity_cutoff = current_at - timedelta(minutes=15)
        recent_bot_activity = []
        for row in bot_rows:
            created = self._as_utc(row.get("created_at"))
            if created is not None and created >= activity_cutoff:
                recent_bot_activity.append(row)
        recent_media_usage = (
            snapshot.media.recent_usage if snapshot is not None and snapshot.media
            else repository.recent_media_usage(6)
        )
        memory_meme = (
            self.media.memory_meme(
                repository, text, event.user_id, event.message_id
            )
            if self.media_enabled(chat_id) and moment is not None
            else None
        )
        # ConversationPolicy has already sampled the established baseline reply
        # rate. With no notable moment, preserve that rate and only let the
        # selector choose TEXT; do not add a second silence lottery.
        legacy_forced = moment is None
        social_selection = self.response_selector.select(
            moment=moment,
            relationship=relationship,
            evidence_candidates=evidence_candidates,
            recent_bot_activity=recent_bot_activity,
            recent_media_usage=recent_media_usage,
            media_enabled=self.media_enabled(chat_id),
            memory_meme_available=memory_meme is not None,
            required=legacy_forced,
        )
        repository.record_routing_event(
            f"social_choice_{social_selection.kind.value}",
            event_id=event.event_id,
        )
        if moment is not None:
            repository.record_routing_event(
                f"social_moment_{moment.moment_type}", event_id=event.event_id
            )
        if social_selection.kind == ResponseKind.SILENCE:
            return None
        if relationship is not None:
            adjusted = decision.troll_intensity + (
                relationship.irritation - relationship.affinity
            ) * .12 + relationship.troll_tendency * .04
            decision = replace(
                decision, troll_intensity=round(max(.15, min(.9, adjusted)), 3)
            )
        if social_selection.kind == ResponseKind.REACTION:
            actions = (
                TriggerCommit(kind),
                GeneratedCommit(social_selection.reaction or "reaction", "social_reaction"),
                PolicyTargetCommit(event.user_id, event.message_id),
            )
            if _as_plan:
                return self.prepare_reaction_response(
                    event, social_selection.reaction, actions=actions
                )
            self.triggers.commit(chat_id, kind)
            repository.record_generated(
                social_selection.reaction or "reaction", "social_reaction"
            )
            self._remember_policy_target(chat_id, event)
            return None
        if social_selection.kind == ResponseKind.EVIDENCE:
            evidence = next(
                (item for item in evidence_candidates
                 if item.id == social_selection.evidence_id), None
            )
            result = self.evidence_engine.callback_text(evidence, text)
            if not result:
                return None
            actions = (
                TriggerCommit(kind),
                GeneratedCommit(result, "social_evidence"),
                EvidenceUsageCommit(evidence.id),
                PolicyTargetCommit(event.user_id, event.message_id),
            )
            if _as_plan:
                return self._create_response_plan(
                    event, result, Producer.EVIDENCE, "social_evidence",
                    actions=actions,
                )
            self.triggers.commit(chat_id, kind)
            repository.record_generated(result, "social_evidence")
            repository.mark_evidence_used(evidence.id)
            self._remember_policy_target(chat_id, event)
            return result
        day_summary = (
            snapshot.current_summary if snapshot is not None
            else repository.summary_for_day(self.memory.logical_day()) or {}
        )
        callbacks = self.persona.select_callbacks(
            day_summary,
            snapshot.stable_memories if snapshot is not None
            else repository.stable_memories(20),
            text,
            state.dominant_topic,
        )
        selected_memes = self.meme_lexicon.select(
            text,
            {state.conversation_type, decision.preferred_style},
            decision.troll_intensity,
            limit=3,
        )
        media_decision = MediaDecision(reason="media_disabled")
        wants_media = social_selection.kind in {
            ResponseKind.GIF, ResponseKind.STICKER, ResponseKind.MEME
        }
        if social_selection.kind == ResponseKind.MEME and memory_meme is not None:
            media_decision = memory_meme
        elif self.media_enabled(chat_id) and (wants_media or legacy_forced):
            media_decision = self.media.decide(
                chat_id=chat_id,
                repository=repository,
                conversation_decision=decision,
                chat_state=state,
                short_term_rows=(
                    snapshot.recent_dialogue if snapshot is not None
                    else self.memory.short_term_rows(repository)
                ),
                target_text=text,
                selected_memes=selected_memes,
                local_callbacks=callbacks,
                troll_mode=self.troll_mode(chat_id),
                probability_roll=self.deterministic_media_roll(
                    chat_id,
                    event.message_id,
                    "media",
                ),
                meme_roll=self.deterministic_media_roll(
                    chat_id,
                    event.message_id,
                    "meme",
                ),
                reaction_roll=self.deterministic_media_roll(
                    chat_id,
                    event.message_id,
                    "reaction",
                ),
                media_context=(snapshot.media if snapshot is not None else None),
            )
            if wants_media and media_decision.action != social_selection.kind.value:
                return None
        if media_decision.action != "none":
            if _as_plan:
                return self._create_response_plan(
                    event, media_decision,
                    Producer.MEME if media_decision.action == "meme" else Producer.MEDIA,
                    "ordinary_media",
                    actions=(
                        TriggerCommit(kind),
                        MediaUsageCommit(media_decision),
                        GeneratedCommit(
                            media_decision.template_id
                            or media_decision.asset_id or "media",
                            "contextual_media",
                        ),
                        PolicyTargetCommit(event.user_id, event.message_id),
                    ),
                )
            self.triggers.commit(chat_id, kind)
            self.media.commit(repository, media_decision)
            repository.record_generated(
                media_decision.template_id or media_decision.asset_id or "media",
                "contextual_media",
            )
            self._remember_policy_target(chat_id, event)
            log.info(
                "Delivery selected chat=%s event=reply action=%s reason=%s",
                chat_id, media_decision.action, media_decision.reason,
            )
            return media_decision
        # Addressed replies use the selected provider. Ordinary free replies
        # use the same contextual responder as direct fallback.
        provider = "llm" if kind in {"addressed", "openai_random"} else "local"
        source_usage = ()
        with self._response_activity(chat_id, "typing", provider) as action:
            if provider == "llm":
                purpose = "reply" if kind == "addressed" else "random_reply"
                result = self.generate_llm(
                    chat_id,
                    self._message_context(event),
                    purpose,
                    conversation_decision=decision,
                    chat_state=state,
                )
            else:
                result, _ = self.local_responder.respond(
                    chat_id, text, SOCIAL, repository,
                    self.persona._recent_ids[chat_id],
                    self.persona._recent_groups[chat_id],
                    self.troll_mode(chat_id), decision.troll_intensity,
                    recent_generated=(snapshot.recent_generated_texts if snapshot else None),
                    stable_memories=(snapshot.stable_memories if snapshot else repository.stable_memories(20)),
                    recent_dialogue=(snapshot.recent_dialogue if snapshot else repository.recent_messages(40)),
                    user_id=event.user_id, username=event.username,
                )
            if result:
                self._ensure_action_visible(action, provider, result)
        if result:
            if _as_plan:
                final_producer = (
                    Producer.LLM if provider == "llm" else Producer.LOCAL
                )
                actions = [
                    TriggerCommit(kind),
                    GeneratedCommit(result, kind),
                    PolicyTargetCommit(event.user_id, event.message_id),
                ]
                if hasattr(result, "construction_signature"):
                    actions.append(StructureUsageCommit(
                        result.construction_signature,
                        result.opening_id,
                        tuple(result.fragment_ids),
                        result.closer_id,
                    ))
                if source_usage:
                    actions.append(SourceUsageCommit(tuple(source_usage)))
                provider_key = None
                if final_producer == Producer.LLM:
                    provider_key = str(getattr(
                        self.provider_for_chat(chat_id), "provider_key", "llm"
                    ))
                return self._create_response_plan(
                    event, result, final_producer, "ordinary",
                    actions=actions, provider_key=provider_key,
                )
            self.triggers.commit(chat_id, kind)
            self.repository(chat_id).record_generated(result, kind)
            if hasattr(result, "construction_signature"):
                repository.record_response_structure(
                    result.construction_signature, result.opening_id,
                    result.fragment_ids, result.closer_id,
                )
            self._remember_policy_target(chat_id, event)
            log.info(
                "Generated reply ready chat=%s trigger=%s generation_path=%s delivery=text",
                chat_id,
                kind,
                provider,
            )
        return result

    def prepare_reply(self, event, bot_id=None, bot_username=None):
        with self.response_planning():
            return self.maybe_reply(
                event, bot_id=bot_id, bot_username=bot_username, _as_plan=True
            )

    def maybe_special_ai(
        self, message, kind, chance, purpose, addressed=True, _as_plan=False
    ):
        _as_plan = self._planning_requested(_as_plan)
        event = self._normalized_event(message)
        chat_id = event.chat_id
        if not self._enabled(chat_id, "talk") or not self.troll_mode(chat_id):
            return None
        if not self.triggers.allowed(chat_id, kind, addressed=addressed):
            return None
        if self.rng.random() >= chance:
            return None
        with self._response_activity(chat_id, "typing", "llm"):
            result = self.generate_llm(
                chat_id, self._message_context(event), purpose
            )
        if result:
            if _as_plan:
                return self._create_response_plan(
                    event, result, Producer.LLM, purpose,
                    actions=(TriggerCommit(kind), GeneratedCommit(result, kind)),
                    provider_key=str(getattr(
                        self.provider_for_chat(chat_id), "provider_key", "llm"
                    )),
                )
            self.triggers.commit(chat_id, kind)
            self.repository(chat_id).record_generated(result, kind)
        return result

    def maybe_stul_cooldown_reply(self, message):
        """Choose exactly one provider for a repeated chair trigger."""
        event = self._normalized_event(message)
        chat_id = event.chat_id
        if not self._enabled(chat_id, "talk") or not self.troll_mode(chat_id):
            return None
        if not self.triggers.allowed(chat_id, "stul_cooldown", addressed=True):
            return None

        # “стул” is an address, never the subject by itself.  A call that also
        # contains a real topic must reach the contextual model even if it has
        # no question mark; otherwise rapid calls degrade into unrelated jokes.
        subject = self.persona._strip_chair_invocation(event.effective_text)
        subject_words = {
            word for word in significant_words(subject)
            if word not in {"еще", "ещё", "раз", "снова", "опять", "второй"}
        }
        if subject_words:
            result = self.generate_llm(
                chat_id,
                self._message_context(event),
                "reply",
            )
            provider = "ai"
            if result:
                self.triggers.commit(chat_id, "stul_cooldown")
                self.repository(chat_id).record_generated(result, "stul_cooldown")
                log.info(
                    "Contextual chair call answered chat=%s generation_path=%s delivery=text",
                    chat_id,
                    provider,
                )
            return result

        frequency = self.triggers.note_chair(chat_id)
        roll = self.rng.random()
        ai_chance = self.settings.reply_to_stul_chance
        local_chance = .50
        invocation = normalize_spaces(event.effective_text).casefold().strip(".,!?…")
        is_bare_invocation = invocation in {"стул", "стульчик"}
        factor = self.settings.bare_stul_reply_factor if is_bare_invocation else 1.0
        total = min(1.0, (ai_chance + local_chance) * factor)
        if roll >= total:
            return None

        provider_roll = roll / total if total else 1.0
        if frequency >= 2:
            provider = "local" if provider_roll < .80 else "ai"
        elif is_bare_invocation:
            # When a rare bare call is accepted, prefer the coherent path; the
            # occurrence probability itself is already sharply reduced.
            provider = "ai" if provider_roll < .8 else "local"
        else:
            provider = "ai" if provider_roll < ai_chance / (ai_chance + local_chance) else "local"

        if provider == "ai":
            result = self.generate_llm(
                chat_id,
                self._message_context(event),
                "stul_cooldown",
            )
            provider = "ai"
        else:
            result, _ = self.local_responder.respond(
                chat_id, event.effective_text, SOCIAL, self.repository(chat_id),
                self.persona._recent_ids[chat_id], self.persona._recent_groups[chat_id],
                self.troll_mode(chat_id), recent_generated=None,
                stable_memories=None, recent_dialogue=None,
                user_id=event.user_id, username=event.username,
            )
            provider = "local"

        if result:
            if self._chair_call_meta_joke_on_cooldown(chat_id, result):
                log.info("Repeated chair-call meta joke blocked chat=%s", chat_id)
                return None
            self.triggers.commit(chat_id, "stul_cooldown")
            self.repository(chat_id).record_generated(result, "stul_cooldown")
            log.info(
                "Repeated chair trigger answered chat=%s generation_path=%s delivery=text",
                chat_id,
                provider,
            )
        return result

    def maybe_voice_story(self, message, _as_plan=False):
        _as_plan = self._planning_requested(_as_plan)
        event = self._normalized_event(message)
        chat_id = event.chat_id
        if not self._enabled(chat_id, "talk") or not self.troll_mode(chat_id):
            return None
        since = (
            datetime.now(timezone.utc)
            - timedelta(seconds=self.settings.voice_story_cooldown)
        ).isoformat()
        if self.repository(chat_id).generated_since(since, "voice_story"):
            return None
        # The invocation is a control command, not story context or memory.
        with self._response_activity(chat_id, "typing", "llm"):
            result = self.generate_llm(chat_id, None, "voice_story")
        if result:
            if _as_plan:
                return self._create_response_plan(
                    event, result, Producer.LLM, "voice_story",
                    actions=(GeneratedCommit(result, "voice_story"),),
                    provider_key=str(getattr(
                        self.provider_for_chat(chat_id), "provider_key", "llm"
                    )),
                )
            self.repository(chat_id).record_generated(result, "voice_story")
        return result

    def maybe_sglypa_reply(self, message, _as_plan=False):
        _as_plan = self._planning_requested(_as_plan)
        event = self._normalized_event(message)
        chat_id = event.chat_id
        if not self.troll_mode(chat_id):
            return None
        if not self.triggers.allowed(chat_id, "sglypa", addressed=True):
            return None
        repository = self.repository(chat_id)
        now = datetime.now(timezone.utc)
        cooldown_since = (
            now - timedelta(seconds=self.settings.sglypa_reply_cooldown)
        ).isoformat()
        if repository.generated_since(cooldown_since, "sglypa"):
            return None
        snapshot = self._active_context_snapshot(chat_id)
        dialogue = snapshot.recent_dialogue if snapshot is not None else repository.recent_messages(20)
        human_activity = sum(
            row.get("speaker") != "cyberchair" and not row.get("is_bot", False)
            for row in dialogue[-12:]
        )
        chance = self.settings.sglypa_reply_chance
        if human_activity < 3:
            chance *= .45
        recent_since = (now - timedelta(hours=6)).isoformat()
        if repository.generated_since(recent_since, "sglypa"):
            chance *= .20
        if self.rng.random() >= chance:
            return None
        with self._response_activity(chat_id, "typing", "llm"):
            result = self.generate_llm(chat_id, event.effective_text, "sglypa")
        if not result:
            return None
        if _as_plan:
            return self._create_response_plan(
                event, result, Producer.LLM, "sglypa", required=True,
                actions=(
                    TriggerCommit("sglypa"), GeneratedCommit(result, "sglypa")
                ),
                provider_key=str(getattr(
                    self.provider_for_chat(chat_id), "provider_key", "llm"
                )),
            )
        self.triggers.commit(chat_id, "sglypa")
        self.repository(chat_id).record_generated(result, "sglypa")
        return result
