"""Optional autonomous-event coordination.

This component owns the existing autonomous policy flow. It does not own
foreground user routing, provider generation, Telegram delivery, or commit.
"""

import logging
from datetime import timedelta, timezone

from .context_snapshot import SnapshotIdentity
from .event_context import EventContext, autonomous_event_id, bind_event, current_event
from .media_service import MediaDecision
from .response_plan import (
    DeliveryType, GeneratedCommit, MediaPayload, MediaUsageCommit, Producer,
    ResponsePlan, TextPayload, TriggerCommit,
)


log = logging.getLogger("learning.service")


class AutonomousCoordinator:
    """Own optional autonomous policy arbitration without Telegram concerns."""

    def __init__(
        self, *, settings, repository, active_snapshot, as_utc,
        context_snapshot_builder, current_context_snapshot,
        set_context_snapshot, reset_context_snapshot, response_planning,
        response_activity, delivery_type_for_media, provider_for_chat,
        provider_available, generate_llm, enabled, troll_mode,
        autonomous_enabled, media_enabled, activity_allows,
        media_context_snapshot, chat_state_analyzer, autonomous_policy,
        media, memory, persona, meme_lexicon, triggers, rng, clock,
        last_policy_target_user, policy_answered_messages,
        last_chat_state, last_conversation_decision, run_autonomous,
    ):
        self.settings=settings
        self.repository=repository
        self._active_context_snapshot=active_snapshot
        self._as_utc=as_utc
        self.context_snapshot_builder=context_snapshot_builder
        self.current_context_snapshot=current_context_snapshot
        self.set_context_snapshot=set_context_snapshot
        self.reset_context_snapshot=reset_context_snapshot
        self.response_planning=response_planning
        self._response_activity=response_activity
        self._delivery_type_for_media=delivery_type_for_media
        self.provider_for_chat=provider_for_chat
        self.provider_available=provider_available
        self.generate_llm=generate_llm
        self._enabled=enabled
        self.troll_mode=troll_mode
        self.autonomous_enabled=autonomous_enabled
        self.media_enabled=media_enabled
        self.activity_allows=activity_allows
        self.media_context_snapshot=media_context_snapshot
        self.chat_state_analyzer=chat_state_analyzer
        self.autonomous_policy=autonomous_policy
        self.media=media
        self.memory=memory
        self.persona=persona
        self.meme_lexicon=meme_lexicon
        self.triggers=triggers
        self.rng=rng
        self._clock=clock
        self._last_policy_target_user=last_policy_target_user
        self._policy_answered_messages=policy_answered_messages
        self._last_chat_state=last_chat_state
        self._last_conversation_decision=last_conversation_decision
        self._last_autonomous_decision={}
        self.run_autonomous=run_autonomous

    def bind_runtime_ports(self, **ports):
        for name, value in ports.items():
            setattr(self, name, value)

    def forget_chat(self, chat_id):
        self._last_autonomous_decision.pop(chat_id, None)

    def autonomous_diagnostics(self, chat_id):
        """Internal state for tests/logging; intentionally not a Telegram command."""
        decision = self._last_autonomous_decision.get(chat_id)
        return decision.debug() if decision else None

    def prepare_autonomous(self, chat_id, current, is_workday=True):
        existing = current_event()
        if existing is not None:
            if self.current_context_snapshot() is None:
                self.set_context_snapshot(
                    self.context_snapshot_builder.build(
                        SnapshotIdentity(existing.event_id, int(chat_id)),
                        self.repository(chat_id), current=current,
                    )
                )
            with self.response_planning():
                return self.run_autonomous(
                    chat_id, current, is_workday, _as_plan=True
                )
        event = EventContext(
            autonomous_event_id(chat_id, current), "autonomous", int(chat_id)
        )
        with bind_event(event):
            snapshot_token = self.set_context_snapshot(
                self.context_snapshot_builder.build(
                    SnapshotIdentity(event.event_id, int(chat_id)),
                    self.repository(chat_id), current=current,
                )
            )
            repository = self.repository(chat_id)
            repository.record_routing_event(
                "autonomous_event", event_id=event.event_id
            )
            try:
                with self.response_planning():
                    result = self.run_autonomous(
                        chat_id, current, is_workday, _as_plan=True
                    )
                log.info(
                    "LLM_EVENT event_id=%s event_type=autonomous llm_calls=%s "
                    "denied_calls=%s",
                    event.event_id, event.permit.call_count, event.permit.denied_count,
                )
                return result
            finally:
                self.reset_context_snapshot(snapshot_token)

    def _autonomous_response_plan(
        self, chat_id, result, producer, kind, actions,
        provider_key=None,
    ):
        context = current_event()
        if context is None:
            raise RuntimeError("autonomous response plan requires EventContext")
        if isinstance(result, MediaDecision):
            delivery_type = self._delivery_type_for_media(result)
            payload = MediaPayload(result)
        else:
            delivery_type = DeliveryType.TEXT
            payload = TextPayload(str(result))
        plan = ResponsePlan(
            event_id=context.event_id,
            chat_id=int(chat_id),
            producer=producer,
            delivery_type=delivery_type,
            payload=payload,
            purpose=kind,
            provider_key=provider_key,
            commit_actions=tuple(actions),
        )
        self.repository(chat_id).record_routing_event(
            "response_plan_created", event_id=context.event_id,
            provider_key=provider_key, call_type=kind,
        )
        return plan

    def _maybe_autonomous(
        self, chat_id, current, is_workday=True, _as_plan=False
    ):
        """Return one local-policy-selected text/media action, or ``None``.

        The policy is deterministic/local.  The provider is called only after
        a roll actually selects a textual autonomous intervention.
        """
        if (
            not self._enabled(chat_id, "talk")
            or not self.troll_mode(chat_id)
            or not self.autonomous_enabled(chat_id)
        ):
            return None
        if not is_workday and not self.settings.autonomous_on_weekends:
            return None
        repository = self.repository(chat_id)
        snapshot = self._active_context_snapshot(chat_id)
        if snapshot is not None and snapshot.chat_id != int(chat_id):
            snapshot = None
        utc_current = current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current.astimezone(timezone.utc)
        quiet = self._quiet_hours_at(current)
        state = self.chat_state_analyzer.analyze(
            repository,
            last_target_user_id=self._last_policy_target_user.get(chat_id),
            answered_message_ids=self._policy_answered_messages.get(chat_id, ()),
            now=utc_current,
            snapshot=snapshot,
        )
        day = self.memory.logical_day(current)
        day_start = current.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()
        generated_rows = (
            list(snapshot.recent_generated) if snapshot is not None else None
        )
        latest_bot_row = (
            generated_rows[-1] if generated_rows
            else (None if generated_rows is not None else repository.latest_generated())
        )
        auto_rows = [
            row for row in (generated_rows or ())
            if row.get("kind") in {"autonomous", "autonomous_media"}
        ]
        latest_auto_row = (
            auto_rows[-1] if auto_rows
            else (None if generated_rows is not None else repository.latest_generated(
                ("autonomous", "autonomous_media")
            ))
        )
        latest_bot = self._as_utc((latest_bot_row or {}).get("created_at"))
        latest_auto = self._as_utc((latest_auto_row or {}).get("created_at"))
        latest_message = snapshot.latest_message if snapshot is not None else None
        if snapshot is None:
            latest_rows = repository.recent_messages(1)
            latest_message = latest_rows[-1] if latest_rows else None
        latest_human = self._as_utc((latest_message or {}).get("created_at"))
        prior_start = (utc_current - timedelta(hours=3)).isoformat()
        day_generated = (
            generated_rows if generated_rows is not None
            else repository.generated_since(day_start)
        )
        autonomous_count = sum(
            1 for row in day_generated
            if str(row.get("created_at") or "") >= day_start
            if row["kind"] in {"autonomous", "autonomous_media"}
        )
        autonomous = self.autonomous_policy.decide(
            state,
            current=utc_current,
            summary=(snapshot.current_summary if snapshot is not None
                     else repository.summary_for_day(day) or {}),
            prior_activity=repository.recent_activity_count(prior_start),
            last_bot_at=latest_bot,
            last_autonomous_at=latest_auto,
            last_human_at=latest_human,
            daily_count=autonomous_count,
            quiet_hours=quiet,
            troll_mode=True,
        )
        self._last_autonomous_decision[chat_id] = autonomous
        self._last_chat_state[chat_id] = state
        self._last_conversation_decision[chat_id] = autonomous.conversation_decision
        if autonomous.action == "none" or self.rng.random() >= autonomous.probability:
            return None
        # The existing activity setting and generic trigger remain hard gates.
        if not self.activity_allows(chat_id) or not self.triggers.allowed(chat_id, "autonomous"):
            return None
        decision = autonomous.conversation_decision
        rows = (
            snapshot.recent_dialogue if snapshot is not None
            else self.memory.short_term_rows(repository)
        )
        target = next((row for row in rows if row.get("message_id") == decision.target_message_id), None)
        target_text = (target or {}).get("text", "")
        summary = (
            snapshot.current_summary if snapshot is not None
            else repository.summary_for_day(day) or {}
        )
        callbacks = self.persona.select_callbacks(
            summary,
            snapshot.stable_memories if snapshot is not None
            else repository.stable_memories(20),
            target_text, state.dominant_topic,
        )
        selected_memes = self.meme_lexicon.select(
            target_text or state.dominant_topic or "",
            {state.conversation_type, decision.preferred_style},
            decision.troll_intensity,
            limit=3,
        )
        media = MediaDecision(reason="media_disabled")
        if self.media_enabled(chat_id):
            if snapshot is not None:
                snapshot = self.media_context_snapshot(snapshot)
            media = self.media.decide(
                chat_id, repository, decision, state, rows, target_text,
                selected_memes, callbacks, troll_mode=True,
                media_context=(snapshot.media if snapshot is not None else None),
            )
        if media.action != "none":
            if _as_plan:
                return self._autonomous_response_plan(
                    chat_id, media,
                    Producer.MEME if media.action == "meme" else Producer.MEDIA,
                    "autonomous_media",
                    (
                        TriggerCommit("autonomous"),
                        MediaUsageCommit(media),
                        GeneratedCommit(
                            media.template_id or media.asset_id or "media",
                            "autonomous_media", utc_current,
                        ),
                    ),
                )
            self.triggers.commit(chat_id, "autonomous")
            self.media.commit(repository, media)
            repository.record_generated(
                media.template_id or media.asset_id or "media", "autonomous_media", utc_current
            )
            log.info(
                "Delivery selected chat=%s event=autonomous action=%s reason=%s",
                chat_id, media.action, media.reason,
            )
            return media
        if not self.provider_available(chat_id):
            return None
        with self._response_activity(chat_id, "typing", "llm"):
            result = self.generate_llm(
                chat_id, target_text or None, "autonomous", decision, state
            )
        if not result:
            return None
        if _as_plan:
            return self._autonomous_response_plan(
                chat_id, result, Producer.LLM, "autonomous",
                (
                    TriggerCommit("autonomous"),
                    GeneratedCommit(result, "autonomous", utc_current),
                ),
                provider_key=str(getattr(
                    self.provider_for_chat(chat_id), "provider_key", "llm"
                )),
            )
        self.triggers.commit(chat_id, "autonomous")
        repository.record_generated(result, "autonomous", utc_current)
        log.info(
            "Generated reply ready chat=%s trigger=autonomous generation_path=ai delivery=text",
            chat_id,
        )
        return result

    def _quiet_hours_at(self, current):
        hour = current.hour
        if self.settings.quiet_start_hour > self.settings.quiet_end_hour:
            return hour >= self.settings.quiet_start_hour or hour < self.settings.quiet_end_hour
        return self.settings.quiet_start_hour <= hour < self.settings.quiet_end_hour
