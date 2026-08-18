"""Typed response preparation and post-delivery state lifecycle.

This component is the sole interpreter of ResponsePlan commit actions.  It
does not route foreground events and never performs Telegram delivery.
"""

import logging

from .media_service import MediaDecision
from .pending_conversation import (
    expected_answer_type, extract_clarification, is_ambiguous_choice_request,
    pending_mode,
)
from .preprocessing import normalize_spaces
from .response_plan import (
    DeliveryType, GeneratedCommit, ManualMemeCommit, MediaPayload,
    MediaUsageCommit, PendingCreate, PendingFinalize, PersonaUsageCommit,
    PolicyTargetCommit, Producer, ResponsePlan, RoutingCommit,
    SourceUsageCommit, TextPayload, TriggerCommit,
)


log = logging.getLogger("learning.service")


class ResponseLifecycle:
    """Prepare plans and atomically interpret their success/abort intents."""

    def __init__(
        self, *, repository, normalize_event, take_persona_usage, triggers,
        media, persona, meme_renderer, mark_command_meme_sent,
        remember_policy_identity, command_meme_sources, lock,
    ):
        self.repository = repository
        self._normalized_event = normalize_event
        self._take_persona_usage = take_persona_usage
        self.triggers = triggers
        self.media = media
        self.persona = persona
        self.meme_renderer = meme_renderer
        self.mark_command_meme_sent = mark_command_meme_sent
        self._remember_policy_identity = remember_policy_identity
        self._command_meme_sources = command_meme_sources
        self._lock = lock
        self._committed_response_events = set()
        self._aborted_response_events = set()

    def bind_runtime_ports(self, *, mark_command_meme_sent, remember_policy_identity):
        """Refresh compatibility seams without taking a facade back-reference."""
        self.mark_command_meme_sent = mark_command_meme_sent
        self._remember_policy_identity = remember_policy_identity

    def _delivery_type_for_media(self, decision):
        return {
            "gif": DeliveryType.ANIMATION,
            "sticker": DeliveryType.STICKER,
            "meme": DeliveryType.PHOTO,
        }.get(decision.action)

    def _pending_create_action(self, event, response, intent):
        clarification = extract_clarification(response)
        if clarification is None or event.user_id is None:
            return None
        expected_type = expected_answer_type(clarification)
        if expected_type == "choices" and not is_ambiguous_choice_request(
            event.effective_text
        ):
            return None
        return PendingCreate(
            user_id=event.user_id,
            original_message_id=event.message_id,
            original_question=normalize_spaces(event.effective_text),
            clarification_question=clarification,
            intent=intent,
            context=normalize_spaces(response),
            expected_type=expected_type,
            mode=pending_mode(response, clarification),
        )

    def _create_response_plan(
        self, event, result, producer, purpose, *, behavior_mode=None,
        required=False, actions=(), provider_key=None, cleanup_paths=(),
    ):
        if result is None:
            return None
        if isinstance(result, MediaDecision):
            delivery_type = self._delivery_type_for_media(result)
            if delivery_type is None:
                return None
            payload = MediaPayload(result)
        else:
            delivery_type = DeliveryType.TEXT
            payload = TextPayload(str(result))
        actions = list(actions)
        persona_usage = self._take_persona_usage()
        if persona_usage:
            actions.append(PersonaUsageCommit(*persona_usage))
        plan = ResponsePlan(
            event_id=event.event_id,
            chat_id=event.chat_id,
            producer=producer,
            delivery_type=delivery_type,
            payload=payload,
            reply_to_message_id=event.message_id,
            required=required,
            purpose=purpose,
            behavior_mode=behavior_mode,
            provider_key=provider_key,
            commit_actions=tuple(actions),
            cleanup_paths=tuple(cleanup_paths),
        )
        self.repository(event.chat_id).record_routing_event(
            "response_plan_created", event_id=event.event_id,
            provider_key=provider_key, call_type=purpose,
        )
        log.info(
            "RESPONSE_PLAN_CREATED event_id=%s producer=%s delivery_type=%s purpose=%s",
            event.event_id, producer.value, delivery_type.value, purpose,
        )
        return plan

    def prepare_text_response(
        self, event, text, purpose="adapter", *, producer=Producer.LOCAL,
        required=False, actions=(), behavior_mode=None, provider_key=None,
    ):
        """Small adapter facade for already-decided local/system text."""
        return self._create_response_plan(
            self._normalized_event(event), text, producer, purpose,
            required=required, actions=actions, behavior_mode=behavior_mode,
            provider_key=provider_key,
        )

    def prepare_manual_meme_response(
        self, event, decision, prepared_path, cleanup_paths=(),
    ):
        event = self._normalized_event(event)
        plan = ResponsePlan(
            event_id=event.event_id,
            chat_id=event.chat_id,
            producer=Producer.MEME,
            delivery_type=DeliveryType.PHOTO,
            payload=MediaPayload(decision, prepared_path),
            reply_to_message_id=event.message_id,
            required=True,
            purpose="manual_meme",
            commit_actions=(ManualMemeCommit(decision),),
            cleanup_paths=tuple(cleanup_paths),
        )
        self.repository(event.chat_id).record_routing_event(
            "response_plan_created", event_id=event.event_id,
            call_type="manual_meme",
        )
        return plan

    def discard_command_meme_candidate(self, decision):
        """Clear transient source bookkeeping when no plan can be delivered."""
        with self._lock:
            self._command_meme_sources.pop(decision, None)

    def record_delivery_attempt(self, plan):
        self.repository(plan.chat_id).record_routing_event(
            "delivery_attempt", event_id=plan.event_id,
            provider_key=plan.provider_key, call_type=plan.delivery_type.value,
        )

    def _cleanup_response_plan(self, plan):
        for action in plan.commit_actions:
            if isinstance(action, ManualMemeCommit):
                with self._lock:
                    self._command_meme_sources.pop(action.decision, None)
        for path in plan.cleanup_paths:
            self.meme_renderer.cleanup(path)

    def _response_plan_key(self, plan):
        payload_identity = (
            plan.payload.text
            if isinstance(plan.payload, TextPayload)
            else plan.payload.decision
        )
        return (
            plan.event_id, plan.purpose, plan.producer,
            plan.delivery_type, payload_identity,
        )

    def commit_response(self, plan, receipt):
        if not receipt.success or receipt.event_id != plan.event_id:
            return False
        commit_key = self._response_plan_key(plan)
        with self._lock:
            if commit_key in self._committed_response_events:
                self._cleanup_response_plan(plan)
                return False
            self._committed_response_events.add(commit_key)
        repository = self.repository(plan.chat_id)
        active_action = None
        try:
            for action in plan.commit_actions:
                active_action = action
                if isinstance(action, PendingFinalize):
                    repository.clear_pending_conversation(action.user_id)
                elif isinstance(action, PendingCreate):
                    repository.save_pending_conversation(
                        user_id=action.user_id,
                        original_message_id=action.original_message_id,
                        original_question=action.original_question,
                        clarification_question=action.clarification_question,
                        intent=action.intent,
                        context=action.context,
                        expected_type=action.expected_type,
                        pending_mode=action.mode,
                        bot_message_id=receipt.telegram_message_id,
                    )
                elif isinstance(action, GeneratedCommit):
                    repository.record_generated(
                        action.text, action.kind, action.created_at
                    )
                elif isinstance(action, TriggerCommit):
                    self.triggers.commit(plan.chat_id, action.kind)
                elif isinstance(action, RoutingCommit):
                    repository.record_routing_event(
                        f"route_{action.route}", event_id=plan.event_id
                    )
                    if action.response_mode:
                        repository.record_routing_event(
                            f"response_mode_{action.response_mode}",
                            event_id=plan.event_id,
                        )
                elif isinstance(action, MediaUsageCommit):
                    self.media.commit(repository, action.decision)
                elif isinstance(action, PolicyTargetCommit):
                    self._remember_policy_identity(
                        plan.chat_id, action.user_id, action.message_id
                    )
                elif isinstance(action, PersonaUsageCommit):
                    self.persona.record_usage(
                        plan.chat_id, action.meme_ids, action.cooldown_groups
                    )
                elif isinstance(action, SourceUsageCommit):
                    repository.mark_used(action.texts)
                elif isinstance(action, ManualMemeCommit):
                    self.mark_command_meme_sent(plan.chat_id, action.decision)
            repository.record_routing_event(
                "delivery_success", event_id=plan.event_id,
                provider_key=plan.provider_key,
                call_type=plan.delivery_type.value,
            )
            repository.record_routing_event(
                "response_commit_success", event_id=plan.event_id
            )
            log.info(
                "RESPONSE_COMMITTED event_id=%s delivery_type=%s telegram_message_id=%s",
                plan.event_id, plan.delivery_type.value,
                receipt.telegram_message_id,
            )
            return True
        except Exception:
            log.exception(
                "POST_DELIVERY_COMMIT_FAILED event_id=%s action_type=%s "
                "action_count=%s",
                plan.event_id,
                type(active_action).__name__ if active_action is not None else "telemetry",
                len(plan.commit_actions),
            )
            try:
                repository.record_routing_event(
                    "post_delivery_commit_failed", event_id=plan.event_id
                )
            except Exception:
                log.exception(
                    "POST_DELIVERY_COMMIT_FAILURE_TELEMETRY_FAILED event_id=%s",
                    plan.event_id,
                )
            return False
        finally:
            self._cleanup_response_plan(plan)

    def abort_response(self, plan, receipt):
        if receipt.success or receipt.event_id != plan.event_id:
            return False
        abort_key = self._response_plan_key(plan)
        with self._lock:
            if abort_key in self._aborted_response_events:
                self._cleanup_response_plan(plan)
                return False
            self._aborted_response_events.add(abort_key)
        try:
            self.repository(plan.chat_id).record_routing_event(
                "delivery_failure", event_id=plan.event_id,
                provider_key=plan.provider_key,
                call_type=receipt.error_category or plan.delivery_type.value,
            )
            log.warning(
                "RESPONSE_ABORTED event_id=%s delivery_type=%s error_category=%s",
                plan.event_id, plan.delivery_type.value, receipt.error_category,
            )
            return True
        finally:
            self._cleanup_response_plan(plan)

    def finalize_response(self, plan, receipt):
        return (
            self.commit_response(plan, receipt)
            if receipt.success else self.abort_response(plan, receipt)
        )
