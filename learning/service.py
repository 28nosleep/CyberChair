"""Composition root and supported compatibility facade for CyberChair core."""

import logging
import random
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone

from .chat_state import ChatStateAnalyzer
from .conversation_policy import ConversationPolicy
from .direct_address import DirectAddressRouter
from .date_time_utility import DateTimeUtility
from .local_responder import LocalResponder
from .cost_diagnostics import TICKS_PER_USD
from .autonomous_policy import AutonomousPolicy
from .media_catalog import MediaCatalog
from .media_service import MediaService
from .memory_service import MemoryService
from .memory_maintenance import MemoryMaintenanceRunner
from .generation_coordinator import GenerationCoordinator
from .response_lifecycle import ResponseLifecycle
from .media_coordinator import MediaCoordinator
from .memory_facade import MemoryFacade
from .foreground_orchestrator import ForegroundOrchestrator
from .autonomous_coordinator import AutonomousCoordinator
from .scheduled_delivery import ScheduledDeliveryCoordinator, ScheduledEventSpec
from .meme_lexicon import MemeLexicon
from .meme_renderer import MemeRenderer
from .meme_sources import MemeSourceSelector
from .persona import PersonaBuilder
from .lexical_diversity import LexicalDiversityTracker
from .response_quality import ResponseQualityGuard
from .relationship import RelationshipModel
from .moment_detector import MomentDetector
from .evidence_engine import EvidenceEngine
from .response_selector import ResponseSelector
from .context_snapshot import ContextSnapshotBuilder
from .concurrency import process_concurrency_controller
from .preprocessing import (
    FOREIGN_BOT_COMMAND_RE,
    VOICE_STORY_COMMAND_RE,
    normalize_spaces,
)
from .repository import ChatRepository
from .provider_factory import create_llm_provider, create_llm_providers
from .triggers import TriggerEngine
from .event_context import (
    EventContext,
    autonomous_event_id,
    bind_event,
    current_event,
    current_event_id,
    implicit_event_id,
    runtime_concurrency,
)
from .normalized_event import (
    NormalizedCallbackEvent,
    NormalizedEvent,
    normalize_telegram_event,
)
from .response_plan import Producer

log = logging.getLogger(__name__)


_response_planning = ContextVar("cyberchair_response_planning", default=False)
_planned_persona_usage = ContextVar(
    "cyberchair_planned_persona_usage", default=None
)
_current_context_snapshot = ContextVar(
    "cyberchair_context_snapshot", default=None
)


def _response_planning_active():
    return bool(_response_planning.get())


def _plan_persona_usage(meme_ids, cooldown_groups):
    _planned_persona_usage.set((tuple(meme_ids), tuple(cooldown_groups)))


def _take_planned_persona_usage():
    value = _planned_persona_usage.get()
    _planned_persona_usage.set(None)
    return value


PHOTO_MEME_CAPTION_RE = re.compile(
    r"^\s*с\s+м\s+стул(?:\s+(?P<hint>.+))?\s*$", re.I
)


class LearningService:
    def __init__(
        self,
        settings,
        openai_client=None,
        rng=None,
        clock=None,
        datetime_clock=None,
        llm_provider=None,
        xai_client=None,
        concurrency_controller=None,
    ):
        self.settings = settings
        self._lock = threading.RLock()
        self._clock = clock or time.monotonic
        self._datetime_clock = datetime_clock
        self.concurrency = concurrency_controller or process_concurrency_controller(
            settings, runtime_concurrency
        )
        self.rng = rng or random
        self.triggers = TriggerEngine(settings, self.rng, clock)
        # Explicit provider/client injection remains a test/integration seam.
        # Runtime construction (no injected clients) always follows config and
        # per-chat selection.
        self._injected_provider = llm_provider
        if self._injected_provider is None and openai_client is not None and xai_client is None:
            self._injected_provider = create_llm_provider(
                settings, openai_client=openai_client, provider_name="openai"
            )
        if self._injected_provider is not None:
            self.providers = {settings.llm_provider.strip().casefold(): self._injected_provider}
            self.llm_provider = self._injected_provider
        else:
            self.providers = create_llm_providers(settings, openai_client, xai_client)
            self.llm_provider = self.providers[settings.llm_provider.strip().casefold()]
        for provider in self.providers.values():
            if hasattr(provider, "_usage_recorder") or provider.__class__.__module__.startswith("learning."):
                provider._usage_recorder = self._record_llm_usage
        self.memory = MemoryService(
            settings, self.llm_provider, self._speaker_name,
            clock=datetime_clock,
            provider_resolver=self.provider_for_chat,
            concurrency_controller=self.concurrency,
        )
        self.memory_maintenance = MemoryMaintenanceRunner(
            self.memory,
            self.concurrency,
            self.provider_for_chat,
            self.llm_allowed,
        )
        self.context_snapshot_builder = ContextSnapshotBuilder(settings, self.memory)
        self.chat_state_analyzer = ChatStateAnalyzer(settings, self.memory)
        self.conversation_policy = ConversationPolicy(settings)
        self.autonomous_policy = AutonomousPolicy(settings, self.conversation_policy)
        self.relationship_model = RelationshipModel()
        self.moment_detector = MomentDetector(settings.context_message_limit * 2)
        self.evidence_engine = EvidenceEngine(
            settings.max_evidence_per_chat,
            settings.evidence_reuse_cooldown_days,
        )
        self.response_selector = ResponseSelector(self.rng)
        self.meme_lexicon = MemeLexicon()
        self.lexical_diversity = LexicalDiversityTracker()
        self.quality_guard = ResponseQualityGuard(self.lexical_diversity)
        self.persona = PersonaBuilder(settings, self.meme_lexicon)
        self.direct_router = DirectAddressRouter()
        self.date_time_utility = DateTimeUtility(
            settings.timezone_name, clock=datetime_clock
        )
        self.local_responder = LocalResponder(
            self.meme_lexicon, self.rng, self.lexical_diversity
        )
        self.media_catalog = MediaCatalog()
        self.media = MediaService(settings, self.media_catalog, self.rng)
        self.meme_renderer = MemeRenderer(
            self.media_catalog, self.settings.data_dir / "generated_media"
        )
        self.meme_sources = MemeSourceSelector(self.rng)
        self.generation = GenerationCoordinator(
            settings=self.settings,
            repository=self.repository,
            active_context_snapshot=self._active_context_snapshot,
            memory=self.memory,
            quality_guard=self.quality_guard,
            lexical_diversity=self.lexical_diversity,
            persona=self.persona,
            concurrency=self.concurrency,
            provider_for_chat=self.provider_for_chat,
            provider_name=self.llm_provider_name,
            injected_provider=self._injected_provider,
            troll_mode=self.troll_mode,
            planning_active=_response_planning_active,
            plan_persona_usage=_plan_persona_usage,
            llm_allowed_check=self.llm_allowed,
            lock=self._lock,
        )
        self._repositories = {}
        self._command_meme_sources = {}
        self.scheduled_delivery = ScheduledDeliveryCoordinator(
            self.repository,
            lease_seconds=settings.scheduled_claim_lease_seconds,
            max_attempts=settings.scheduled_delivery_max_attempts,
            backoff_base_seconds=(
                settings.scheduled_retry_backoff_base_seconds
            ),
            backoff_cap_seconds=settings.scheduled_retry_backoff_cap_seconds,
        )
        self.memory_facade = MemoryFacade(
            settings=self.settings,
            repository=self.repository,
            normalize_event=self._normalized_event,
            enabled=self._enabled,
            triggers=self.triggers,
            invalidate_generation=self.generation.invalidate_chat,
            memory_maintenance=self.memory_maintenance,
            troll_mode=self.troll_mode,
            provider_name=self.llm_provider_name,
            provider_available=self.provider_available,
            activity_percent=self.activity_percent,
            autonomous_enabled=self.autonomous_enabled,
            media_enabled=self.media_enabled,
            relationship_model=self.relationship_model,
            moment_detector=self.moment_detector,
            evidence_engine=self.evidence_engine,
        )
        self.media_coordinator = MediaCoordinator(
            settings=self.settings,
            repository=self.repository,
            normalize_event=self._normalized_event,
            active_snapshot=self._active_context_snapshot,
            media=self.media,
            media_catalog=self.media_catalog,
            meme_renderer=self.meme_renderer,
            meme_sources=self.meme_sources,
            quality_guard=self.quality_guard,
            memory=self.memory,
            persona=self.persona,
            rng=self.rng,
            concurrency=self.concurrency,
            activity_allows=self.activity_allows,
            media_enabled=self.media_enabled,
            troll_mode=self.troll_mode,
            provider_available=self.provider_available,
            generate_llm=self.generate_llm,
            command_meme_sources=self._command_meme_sources,
            lock=self._lock,
            photo_meme_caption_re=PHOTO_MEME_CAPTION_RE,
            evidence_engine=self.evidence_engine,
        )
        self.response_lifecycle = ResponseLifecycle(
            repository=self.repository,
            normalize_event=self._normalized_event,
            take_persona_usage=_take_planned_persona_usage,
            triggers=self.triggers,
            media=self.media,
            persona=self.persona,
            meme_renderer=self.meme_renderer,
            mark_command_meme_sent=self.mark_command_meme_sent,
            remember_policy_identity=self._remember_policy_identity,
            command_meme_sources=self._command_meme_sources,
            lock=self._lock,
        )
        self.foreground = ForegroundOrchestrator(
            settings=self.settings,
            repository=self.repository,
            normalize_event=self._normalized_event,
            active_snapshot=self._active_context_snapshot,
            as_utc=self._as_utc,
            enabled=self._enabled,
            context_snapshot=self.context_snapshot,
            media_context_snapshot=self.media_context_snapshot,
            create_response_plan=self._create_response_plan,
            pending_create_action=self._pending_create_action,
            delivery_type_for_media=self._delivery_type_for_media,
            prepare_reaction_response=self.prepare_reaction_response,
            provider_for_chat=self.provider_for_chat,
            provider_available=self.provider_available,
            generate_llm=self.generate_llm,
            generate_grounded_llm=self.generate_grounded_llm,
            llm_allowed=self.llm_allowed,
            troll_mode=self.troll_mode,
            media_enabled=self.media_enabled,
            budget_exceeded=self._budget_exceeded,
            chair_meta_on_cooldown=self._chair_call_meta_joke_on_cooldown,
            message_context=self._message_context,
            response_planning=self.response_planning,
            planning_requested=self._planning_requested,
            deterministic_media_roll=self._deterministic_media_roll,
            policy_quiet_hours=self._policy_quiet_hours,
            response_activity=self._response_activity,
            ensure_action_visible=self._ensure_action_visible,
            chat_state_analyzer=self.chat_state_analyzer,
            conversation_policy=self.conversation_policy,
            direct_router=self.direct_router,
            local_responder=self.local_responder,
            date_time_utility=self.date_time_utility,
            media=self.media,
            memory=self.memory,
            persona=self.persona,
            meme_lexicon=self.meme_lexicon,
            relationship_model=self.relationship_model,
            moment_detector=self.moment_detector,
            evidence_engine=self.evidence_engine,
            response_selector=self.response_selector,
            triggers=self.triggers,
            rng=self.rng,
            lock=self._lock,
            clock=self._clock,
        )
        self.autonomous = AutonomousCoordinator(
            settings=self.settings,
            repository=self.repository,
            active_snapshot=self._active_context_snapshot,
            as_utc=self._as_utc,
            context_snapshot_builder=self.context_snapshot_builder,
            current_context_snapshot=self.current_context_snapshot,
            set_context_snapshot=_current_context_snapshot.set,
            reset_context_snapshot=_current_context_snapshot.reset,
            response_planning=self.response_planning,
            response_activity=self._response_activity,
            delivery_type_for_media=self._delivery_type_for_media,
            provider_for_chat=self.provider_for_chat,
            provider_available=self.provider_available,
            generate_llm=self.generate_llm,
            enabled=self._enabled,
            troll_mode=self.troll_mode,
            autonomous_enabled=self.autonomous_enabled,
            media_enabled=self.media_enabled,
            activity_allows=self.activity_allows,
            media_context_snapshot=self.media_context_snapshot,
            chat_state_analyzer=self.chat_state_analyzer,
            autonomous_policy=self.autonomous_policy,
            media=self.media,
            memory=self.memory,
            persona=self.persona,
            meme_lexicon=self.meme_lexicon,
            triggers=self.triggers,
            rng=self.rng,
            clock=self._clock,
            last_policy_target_user=self.foreground._last_policy_target_user,
            policy_answered_messages=self.foreground._policy_answered_messages,
            last_chat_state=self.foreground._last_chat_state,
            last_conversation_decision=self.foreground._last_conversation_decision,
            run_autonomous=self._maybe_autonomous,
        )
        self._context_snapshot_metrics = OrderedDict()
        # Optional Telegram-layer UX hook. Core routing remains transport-free.
        self.response_activity = None

    def _response_activity(self, chat_id, action="typing", producer=None):
        if self.response_activity is None:
            return nullcontext(None)
        return self.response_activity(chat_id, action, producer)

    @staticmethod
    def _ensure_action_visible(session, producer, text):
        if session is not None and hasattr(session, "ensure_visible"):
            session.ensure_visible(producer, text)

    def repository(self, chat_id):
        with self._lock:
            if chat_id not in self._repositories:
                repository = ChatRepository(
                    self.settings.data_dir,
                    chat_id,
                    self.settings.max_messages_per_chat,
                    self.settings.max_unsummarized_messages,
                )
                removed = repository.purge_matching_text(FOREIGN_BOT_COMMAND_RE)
                removed += repository.purge_matching_text(VOICE_STORY_COMMAND_RE)
                if removed:
                    log.info("Forbidden memory purged chat=%s rows=%s", chat_id, removed)
                self._repositories[chat_id] = repository
            return self._repositories[chat_id]

    @staticmethod
    def current_context_snapshot():
        return _current_context_snapshot.get()

    @staticmethod
    def _active_context_snapshot(chat_id=None):
        snapshot = _current_context_snapshot.get()
        event = current_event()
        if snapshot is None or event is None or snapshot.event_id != event.event_id:
            return None
        if chat_id is not None and snapshot.chat_id != int(chat_id):
            return None
        return snapshot

    def context_snapshot(self, message_or_event, current=None):
        event = self._normalized_event(message_or_event)
        existing = _current_context_snapshot.get()
        if existing is not None and existing.event_id == event.event_id:
            return existing
        snapshot = self.context_snapshot_builder.build(
            event, self.repository(event.chat_id), current=current
        )
        context = current_event()
        if context is not None and context.event_id != snapshot.event_id:
            raise RuntimeError("ContextSnapshot event identity mismatch")
        _current_context_snapshot.set(snapshot)
        with self._lock:
            self._context_snapshot_metrics[snapshot.event_id] = snapshot.metrics
            while len(self._context_snapshot_metrics) > 500:
                self._context_snapshot_metrics.popitem(last=False)
        return snapshot

    def media_context_snapshot(self, snapshot):
        if snapshot is None or snapshot.media is not None:
            return snapshot
        enriched = self.context_snapshot_builder.enrich_media(
            snapshot, self.repository(snapshot.chat_id)
        )
        _current_context_snapshot.set(enriched)
        with self._lock:
            self._context_snapshot_metrics[enriched.event_id] = enriched.metrics
        return enriched

    def context_snapshot_diagnostics(self):
        with self._lock:
            values = list(self._context_snapshot_metrics.values())
        events = len(values)
        return {
            "events": events,
            "avg_db_connections": (
                sum(item.db_connections for item in values) / events
                if events else 0.0
            ),
            "avg_queries": (
                sum(item.queries for item in values) / events if events else 0.0
            ),
            "avg_build_ms": (
                sum(item.build_ms for item in values) / events if events else 0.0
            ),
            "peak_db_connections": max(
                (item.db_connections for item in values), default=0
            ),
        }

    def format_context_snapshot_diagnostics(self):
        report = self.context_snapshot_diagnostics()
        return "\n".join((
            "CONTEXT SNAPSHOT",
            f"events: {report['events']}",
            f"avg_db_connections_after: {report['avg_db_connections']:.2f}",
            f"avg_queries_after: {report['avg_queries']:.2f}",
            f"avg_build_ms: {report['avg_build_ms']:.3f}",
            f"peak_db_connections: {report['peak_db_connections']}",
        ))

    def _resolved_setting(self, chat_id, key, default=None):
        snapshot = self._active_context_snapshot(chat_id)
        if snapshot is not None:
            return snapshot.setting(key, default)
        return self.repository(chat_id).setting(key, default)

    def activity_percent(self, chat_id):
        raw = self._resolved_setting(
            chat_id, "activity_percent", str(self.settings.default_activity_percent)
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = self.settings.default_activity_percent
        return value if value in {0, 25, 50, 75, 100} else self.settings.default_activity_percent

    def set_activity_percent(self, chat_id, percent):
        percent = int(percent)
        if percent not in {0, 25, 50, 75, 100}:
            raise ValueError("unsupported activity percent")
        self.repository(chat_id).set_setting("activity_percent", percent)

    def activity_allows(self, chat_id):
        return self.rng.random() < self.activity_percent(chat_id) / 100

    def _enabled(self, chat_id, key="learning"):
        if not self.settings.enabled:
            return False
        default = "1"
        return self._resolved_setting(chat_id, key, default) == "1"

    def set_enabled(self, chat_id, kind, enabled):
        if kind not in {"learning", "talk"}:
            raise ValueError("unknown setting")
        self.repository(chat_id).set_setting(kind, "1" if enabled else "0")

    def troll_mode(self, chat_id):
        return self._resolved_setting(chat_id, "troll_mode", "1") == "1"

    def set_troll_mode(self, chat_id, enabled):
        self.repository(chat_id).set_setting("troll_mode", "1" if enabled else "0")

    def autonomous_enabled(self, chat_id):
        return self._resolved_setting(
            chat_id, "autonomous_enabled", "1"
        ) == "1"

    def set_autonomous_enabled(self, chat_id, enabled):
        self.repository(chat_id).set_setting("autonomous_enabled", "1" if enabled else "0")

    def media_enabled(self, chat_id):
        return self._resolved_setting(chat_id, "media_enabled", "1") == "1"

    def set_media_enabled(self, chat_id, enabled):
        self.repository(chat_id).set_setting("media_enabled", "1" if enabled else "0")

    def llm_provider_name(self, chat_id):
        default = self.settings.llm_provider.strip().casefold()
        value = str(
            self._resolved_setting(chat_id, "llm_provider", default)
        ).strip().casefold()
        return value if value in {"grok", "openai"} else default

    def provider_for_chat(self, chat_id):
        if self._injected_provider is not None:
            return self._injected_provider
        return self.providers[self.llm_provider_name(chat_id)]

    def provider_available(self, chat_id, provider_name=None):
        if self._injected_provider is not None:
            return bool(self._injected_provider.available)
        name = (provider_name or self.llm_provider_name(chat_id)).strip().casefold()
        return name in self.providers and bool(self.providers[name].available)

    def provider_unavailable_reason(self, chat_id, provider_name=None):
        if self._injected_provider is not None:
            return None if self._injected_provider.available else "LLM provider unavailable"
        name = (provider_name or self.llm_provider_name(chat_id)).strip().casefold()
        provider = self.providers.get(name)
        return None if provider and provider.available else getattr(
            provider, "unavailable_reason", f"unknown provider: {name}"
        )

    def set_llm_provider(self, chat_id, provider_name):
        name = str(provider_name).strip().casefold()
        if name not in {"grok", "openai"}:
            raise ValueError("unsupported provider")
        if not self.provider_available(chat_id, name):
            return False
        self.repository(chat_id).set_setting("llm_provider", name)
        return True

    @staticmethod
    def _normalized_event(value):
        if isinstance(value, NormalizedEvent):
            return value
        return normalize_telegram_event(value)

    @contextmanager
    def telegram_user_event(self, message_or_event):
        """Bind minimal R0 correlation/budget state around one Telegram message."""
        normalized = (
            message_or_event
            if isinstance(message_or_event, (NormalizedEvent, NormalizedCallbackEvent))
            else self._normalized_event(message_or_event)
        )
        chat_id = int(normalized.chat_id)
        event_type = "callback" if isinstance(normalized, NormalizedCallbackEvent) else "user"
        event = EventContext(normalized.event_id, event_type, chat_id)
        snapshot_token = _current_context_snapshot.set(None)
        with bind_event(event):
            repository = self.repository(chat_id)
            repository.record_routing_event(
                "callback_event" if event_type == "callback" else "user_event",
                event_id=event.event_id,
            )
            try:
                yield event
            finally:
                calls = event.permit.call_count
                if calls > 1:
                    repository.record_routing_event(
                        "llm_invariant_violation", event_id=event.event_id
                    )
                    log.error(
                        "LLM_EVENT_INVARIANT_VIOLATION event_id=%s "
                        "event_type=%s llm_calls=%s",
                        event.event_id, event_type, calls,
                    )
                log.info(
                    "LLM_EVENT event_id=%s event_type=%s llm_calls=%s "
                    "denied_calls=%s",
                    event.event_id, event_type, calls, event.permit.denied_count,
                )
                _current_context_snapshot.reset(snapshot_token)

    def chat_event_slot(self, event, *, background=False):
        """Arbitrate one complete event lifecycle without leaking locks to plans."""
        return self.concurrency.chat_event_slot(
            event.chat_id, event.event_id, background=background
        )

    def autonomous_chat_event_slot(self, chat_id, current):
        return self.concurrency.chat_event_slot(
            chat_id, autonomous_event_id(chat_id, current), background=True
        )

    def media_work_slot(self, chat_id, event_id=None, *, background=False):
        return self.concurrency.media_slot(
            event_id or current_event_id() or implicit_event_id("media", chat_id),
            chat_id,
            background=background,
        )

    @contextmanager
    def response_planning(self):
        """Ask compatibility routing methods for an immutable final plan."""
        token = _response_planning.set(True)
        usage_token = _planned_persona_usage.set(None)
        snapshot_token = (
            _current_context_snapshot.set(None)
            if current_event() is None else None
        )
        try:
            yield
        finally:
            if snapshot_token is not None:
                _current_context_snapshot.reset(snapshot_token)
            _planned_persona_usage.reset(usage_token)
            _response_planning.reset(token)

    @staticmethod
    def _planning_requested(explicit):
        return bool(explicit or _response_planning.get())

    def ingest(self, message, refresh_memory=True):
        return self.memory_facade.ingest(message, refresh_memory)

    def ingest_event(self, event, refresh_memory=True):
        # Keep the historical public seam: callers patching ``ingest`` still
        # observe normalized-event ingestion through this compatibility API.
        return self.ingest(event, refresh_memory=refresh_memory)

    def observe_callback(self, event):
        return self.relationship_model.observe_callback(
            self.repository(event.chat_id), event
        )

    def run_memory_maintenance(self, chat_id, current=None):
        return self.memory_facade.run_memory_maintenance(chat_id, current)

    def persistence_diagnostics(self, chat_id):
        return self.memory_facade.persistence_diagnostics(chat_id)

    def format_persistence_diagnostics(self, chat_id):
        return self.memory_facade.format_persistence_diagnostics(chat_id)

    def _speaker_name(self, row):
        if row["speaker"] == "cyberchair":
            return "Киберстул"
        if (row.get("username") or "").casefold() == self.settings.creator_username:
            return "Харакири (создатель Киберстула)"
        return "Участник"

    def status(self, chat_id):
        return self.memory_facade.status(chat_id)

    def _chair_call_meta_joke_on_cooldown(self, chat_id, text):
        return self.generation._chair_call_meta_joke_on_cooldown(chat_id, text)

    def _message_context(self, message):
        event = self._normalized_event(message)
        text = event.effective_text
        reply_text = normalize_spaces(event.reply_effective_text)
        if reply_text:
            text = f"Сообщение CyberChair, на которое отвечают: {reply_text[:500]}\nОтвет пользователя: {text}"
        username = (event.username or "").casefold()
        if username == self.settings.creator_username:
            return f"Харакири (@{self.settings.creator_username}): {text}"
        return text

    def _budget_exceeded(self, chat_id):
        budget = max(0.0, float(self.settings.xai_daily_chat_budget_usd))
        if not budget:
            return False
        report = self.llm_cost_diagnostics(chat_id, 24)["total"]
        ticks = report.get("cost_usd_ticks")
        return ticks is not None and ticks >= int(budget * TICKS_PER_USD)

    def direct_response_diagnostics(self, chat_id, hours=24, current=None):
        current = current or datetime.now(timezone.utc)
        since = (current - timedelta(hours=hours)).isoformat()
        events = self.repository(chat_id).routing_report(since)
        received = events.get("direct_addresses", 0) + events.get("direct_replies", 0)
        # A message that is both a mention and a reply is counted once as a reply.
        llm_routes = events.get("route_llm", 0) + events.get("route_grok", 0)
        answered = llm_routes + sum(
            events.get(f"route_{name}", 0)
            for name in ("local", "gif", "meme", "sticker")
        )
        llm = self.llm_cost_diagnostics(chat_id, hours)["total"]
        return {
            "received": received,
            "answered": answered,
            "response_rate": 1.0 if received == 0 else min(1.0, answered / received),
            "routes": {
                **{
                    name: events.get(f"route_{name}", 0)
                    for name in ("local", "gif", "meme", "sticker")
                },
                "llm": llm_routes,
            },
            "intents": {name: events.get(f"intent_{name}", 0) for name in ("trivial", "social", "substantive")},
            "llm_fallback_local": (
                events.get("llm_fallback_local", 0)
                + events.get("grok_fallback_local", 0)
            ),
            "llm_share": 0.0 if not answered else llm_routes / answered,
            "cost_usd_ticks": llm.get("cost_usd_ticks"),
        }

    def quality_diagnostics(self, chat_id, hours=24, current=None):
        return self.generation.quality_diagnostics(chat_id, hours, current)

    def generate_free_response(
        self, chat_id, input_text=None, decorate=True, return_sources=False
    ):
        snapshot = self._active_context_snapshot(chat_id)
        repository = self.repository(chat_id)
        result, _ = self.local_responder.respond(
            chat_id, input_text or "", "social", repository,
            recent_generated=(snapshot.recent_generated_texts if snapshot else None),
            stable_memories=(snapshot.stable_memories if snapshot else repository.stable_memories(20)),
            recent_dialogue=(snapshot.recent_dialogue if snapshot else repository.recent_messages(40)),
        )
        return (result, ()) if return_sources else result

    def llm_allowed(self, chat_id):
        return self.generation.llm_allowed(chat_id)

    def _record_llm_usage(self, chat_id, provider, model, call_type, usage):
        return self.generation._record_llm_usage(chat_id, provider, model, call_type, usage)

    def llm_cost_diagnostics(self, chat_id, hours=24, current=None):
        return self.generation.llm_cost_diagnostics(chat_id, hours, current)

    def llm_event_invariant_diagnostics(self, chat_id, hours=24, current=None):
        current = current or datetime.now(timezone.utc)
        since = (current - timedelta(hours=hours)).astimezone(timezone.utc).isoformat()
        return self.repository(chat_id).llm_event_invariant_report(since)

    def format_llm_event_invariant_diagnostics(self, chat_id, hours=24,
                                               current=None):
        report = self.llm_event_invariant_diagnostics(chat_id, hours, current)
        return "\n".join((
            "LLM EVENT INVARIANT",
            f"user_events: {report['user_events']}",
            f"events_with_0_llm: {report['events_with_0_llm']}",
            f"events_with_1_llm: {report['events_with_1_llm']}",
            f"events_with_2plus_llm: {report['events_with_2plus_llm']}",
            f"max_calls_per_user_event: {report['max_calls_per_user_event']}",
        ))

    def memory_lifecycle_diagnostics(self, chat_id, current=None):
        return self.memory_facade.memory_lifecycle_diagnostics(chat_id, current)

    def format_memory_lifecycle_diagnostics(self, chat_id, current=None):
        return self.memory_facade.format_memory_lifecycle_diagnostics(chat_id, current)

    def concurrency_diagnostics(self):
        return self.concurrency.snapshot()

    def format_concurrency_diagnostics(self):
        report = self.concurrency_diagnostics()
        return "\n".join((
            "CONCURRENCY HARDENING",
            f"LLM_MAX_CONCURRENCY={report['llm_max_concurrency']}",
            f"MEDIA_MAX_CONCURRENCY={report['media_max_concurrency']}",
            f"peak_active_llm_after={report.get('peak_active_llm_calls', 0)}",
            f"peak_active_media_after={report.get('peak_active_media_jobs', 0)}",
            f"llm_admission_timeouts={report['llm_admission_timeouts']}",
            f"media_admission_timeouts={report['media_admission_timeouts']}",
            f"chat_gate_wait_p50={report['chat_gate_wait_ms_p50']:.3f}",
            f"chat_gate_wait_max={report['chat_gate_wait_ms_max']:.3f}",
            "autonomous_resource_skips="
            f"{report['autonomous_skipped_chat_busy'] + report['autonomous_skipped_llm_busy'] + report['autonomous_skipped_media_busy']}",
        ))

    def generate_llm(
        self,
        chat_id,
        context=None,
        purpose="reply",
        conversation_decision=None,
        chat_state=None,
    ):
        self.generation.llm_allowed_check = self.llm_allowed
        return self.generation.generate_llm(chat_id, context, purpose, conversation_decision, chat_state)

    def generate_grounded_llm(self, chat_id, request):
        self.generation.llm_allowed_check = self.llm_allowed
        return self.generation.generate_grounded(chat_id, request)

    def _policy_quiet_hours(self):
        self._sync_foreground_runtime_ports()
        return self.foreground._policy_quiet_hours()

    def conversation_diagnostics(self, chat_id):
        self._sync_foreground_runtime_ports()
        return self.foreground.conversation_diagnostics(chat_id)

    def autonomous_diagnostics(self, chat_id):
        return self.autonomous.autonomous_diagnostics(chat_id)

    def _remember_policy_target(self, chat_id, message):
        self._sync_foreground_runtime_ports()
        return self.foreground._remember_policy_target(chat_id, message)

    def _remember_policy_identity(self, chat_id, actual_user, actual_message):
        self._sync_foreground_runtime_ports()
        return self.foreground._remember_policy_identity(chat_id, actual_user, actual_message)

    def _delivery_type_for_media(self, decision):
        return self.response_lifecycle._delivery_type_for_media(decision)

    def _pending_create_action(self, event, response, intent):
        return self.response_lifecycle._pending_create_action(event, response, intent)

    def _create_response_plan(
        self, event, result, producer, purpose, *, behavior_mode=None,
        required=False, actions=(), provider_key=None, cleanup_paths=(),
    ):
        return self.response_lifecycle._create_response_plan(event, result, producer, purpose, behavior_mode=behavior_mode, required=required, actions=actions, provider_key=provider_key, cleanup_paths=cleanup_paths)

    def prepare_text_response(
        self, event, text, purpose="adapter", *, producer=Producer.LOCAL,
        required=False, actions=(), behavior_mode=None, provider_key=None,
    ):
        return self.response_lifecycle.prepare_text_response(event, text, purpose, producer=producer, required=required, actions=actions, behavior_mode=behavior_mode, provider_key=provider_key)

    def prepare_reaction_response(
        self, event, emoji, purpose="social_reaction", *, actions=()
    ):
        return self.response_lifecycle.prepare_reaction_response(
            event, emoji, purpose, actions=actions
        )

    def prepare_manual_meme_response(
        self, event, decision, prepared_path, cleanup_paths=(),
    ):
        return self.response_lifecycle.prepare_manual_meme_response(event, decision, prepared_path, cleanup_paths)

    def discard_command_meme_candidate(self, decision):
        return self.response_lifecycle.discard_command_meme_candidate(decision)

    def record_delivery_attempt(self, plan):
        return self.response_lifecycle.record_delivery_attempt(plan)

    def commit_response(self, plan, receipt):
        self._sync_response_lifecycle_ports()
        return self.response_lifecycle.commit_response(plan, receipt)

    def abort_response(self, plan, receipt):
        self._sync_response_lifecycle_ports()
        return self.response_lifecycle.abort_response(plan, receipt)

    def finalize_response(self, plan, receipt):
        self._sync_response_lifecycle_ports()
        return self.response_lifecycle.finalize_response(plan, receipt)

    def _sync_response_lifecycle_ports(self):
        self.response_lifecycle.bind_runtime_ports(
            mark_command_meme_sent=self.mark_command_meme_sent,
            remember_policy_identity=self._remember_policy_identity,
        )

    def _sync_foreground_runtime_ports(self):
        self.foreground.bind_runtime_ports(
            repository=self.repository,
            _normalized_event=self._normalized_event,
            _active_context_snapshot=self._active_context_snapshot,
            _as_utc=self._as_utc,
            _enabled=self._enabled,
            context_snapshot=self.context_snapshot,
            media_context_snapshot=self.media_context_snapshot,
            _create_response_plan=self._create_response_plan,
            _pending_create_action=self._pending_create_action,
            _delivery_type_for_media=self._delivery_type_for_media,
            prepare_reaction_response=self.prepare_reaction_response,
            provider_for_chat=self.provider_for_chat,
            provider_available=self.provider_available,
            generate_llm=self.generate_llm,
            generate_grounded_llm=self.generate_grounded_llm,
            llm_allowed=self.llm_allowed,
            troll_mode=self.troll_mode,
            media_enabled=self.media_enabled,
            _budget_exceeded=self._budget_exceeded,
            _chair_call_meta_joke_on_cooldown=self._chair_call_meta_joke_on_cooldown,
            _message_context=self._message_context,
            response_planning=self.response_planning,
            _planning_requested=self._planning_requested,
            deterministic_media_roll=self._deterministic_media_roll,
            policy_quiet_hours=self._policy_quiet_hours,
            _response_activity=self._response_activity,
            _ensure_action_visible=self._ensure_action_visible,
            chat_state_analyzer=self.chat_state_analyzer,
            conversation_policy=self.conversation_policy,
            direct_router=self.direct_router,
            local_responder=self.local_responder,
            date_time_utility=self.date_time_utility,
            media=self.media,
            memory=self.memory,
            persona=self.persona,
            meme_lexicon=self.meme_lexicon,
            relationship_model=self.relationship_model,
            moment_detector=self.moment_detector,
            evidence_engine=self.evidence_engine,
            response_selector=self.response_selector,
            triggers=self.triggers,
            rng=self.rng,
        )

    def _sync_autonomous_runtime_ports(self):
        self.autonomous.bind_runtime_ports(
            repository=self.repository,
            _active_context_snapshot=self._active_context_snapshot,
            _as_utc=self._as_utc,
            context_snapshot_builder=self.context_snapshot_builder,
            current_context_snapshot=self.current_context_snapshot,
            set_context_snapshot=_current_context_snapshot.set,
            reset_context_snapshot=_current_context_snapshot.reset,
            response_planning=self.response_planning,
            _response_activity=self._response_activity,
            _delivery_type_for_media=self._delivery_type_for_media,
            provider_for_chat=self.provider_for_chat,
            provider_available=self.provider_available,
            generate_llm=self.generate_llm,
            _enabled=self._enabled,
            troll_mode=self.troll_mode,
            autonomous_enabled=self.autonomous_enabled,
            media_enabled=self.media_enabled,
            activity_allows=self.activity_allows,
            media_context_snapshot=self.media_context_snapshot,
            chat_state_analyzer=self.chat_state_analyzer,
            autonomous_policy=self.autonomous_policy,
            media=self.media,
            memory=self.memory,
            persona=self.persona,
            meme_lexicon=self.meme_lexicon,
            triggers=self.triggers,
            rng=self.rng,
            run_autonomous=self._maybe_autonomous,
        )

    def _deterministic_media_roll(self, chat_id, message_id, salt):
        self._sync_foreground_runtime_ports()
        return self.foreground._deterministic_media_roll(chat_id, message_id, salt)

    def _record_direct_result(
        self, chat_id, message, producer, result,
        behavior_mode="useful_answer", *, as_plan=False,
        pending_finalize_user_id=None, persona_usage=None, source_usage=(),
    ):
        self._sync_foreground_runtime_ports()
        return self.foreground._record_direct_result(chat_id, message, producer, result, behavior_mode, as_plan=as_plan, pending_finalize_user_id=pending_finalize_user_id, persona_usage=persona_usage, source_usage=source_usage)

    def _substantive_behavior_mode(self, chat_id):
        self._sync_foreground_runtime_ports()
        return self.foreground._substantive_behavior_mode(chat_id)

    def _store_pending_from_response(self, message, response, intent):
        self._sync_foreground_runtime_ports()
        return self.foreground._store_pending_from_response(message, response, intent)

    def pending_conversation(self, chat_id, user_id, current=None):
        self._sync_foreground_runtime_ports()
        return self.foreground.pending_conversation(chat_id, user_id, current)

    def attach_pending_bot_message(self, incoming_message, sent_message):
        self._sync_foreground_runtime_ports()
        return self.foreground.attach_pending_bot_message(incoming_message, sent_message)

    def is_pending_continuation(self, message, bot_id=None, current=None):
        self._sync_foreground_runtime_ports()
        return self.foreground.is_pending_continuation(message, bot_id, current)

    def maybe_pending_continuation(
        self, message, bot_id=None, current=None, _as_plan=False
    ):
        self._sync_foreground_runtime_ports()
        return self.foreground.maybe_pending_continuation(message, bot_id, current, _as_plan)

    def prepare_pending_continuation(self, event, bot_id=None, current=None):
        self._sync_foreground_runtime_ports()
        return self.foreground.prepare_pending_continuation(event, bot_id, current)

    def maybe_direct_reply(
        self, message, bot_id=None, bot_username=None,
        explicit_address=False, _as_plan=False,
    ):
        self._sync_foreground_runtime_ports()
        return self.foreground.maybe_direct_reply(message, bot_id, bot_username, explicit_address, _as_plan)

    def prepare_direct_reply(
        self, event, bot_id=None, bot_username=None, explicit_address=False
    ):
        self._sync_foreground_runtime_ports()
        return self.foreground.prepare_direct_reply(event, bot_id, bot_username, explicit_address)

    def maybe_reply(
        self, message, bot_id=None, bot_username=None, _as_plan=False
    ):
        self._sync_foreground_runtime_ports()
        return self.foreground.maybe_reply(message, bot_id, bot_username, _as_plan)

    def prepare_reply(self, event, bot_id=None, bot_username=None):
        self._sync_foreground_runtime_ports()
        return self.foreground.prepare_reply(event, bot_id, bot_username)

    def maybe_special_ai(
        self, message, kind, chance, purpose, addressed=True, _as_plan=False
    ):
        self._sync_foreground_runtime_ports()
        return self.foreground.maybe_special_ai(message, kind, chance, purpose, addressed, _as_plan)

    def maybe_stul_cooldown_reply(self, message):
        self._sync_foreground_runtime_ports()
        return self.foreground.maybe_stul_cooldown_reply(message)

    def stul_cooldown_remaining(self, chat_id):
        return self.triggers.cooldown_remaining(
            chat_id,
            "stul_cooldown",
            addressed=True,
        )

    def take_stul_cooldown_notice(self, chat_id):
        return self.triggers.consume_cooldown_notice(
            chat_id,
            "stul_cooldown",
            addressed=True,
        )

    def note_stul(self, chat_id):
        return self.triggers.note_chair(chat_id)

    def maybe_voice_story(self, message, _as_plan=False):
        self._sync_foreground_runtime_ports()
        return self.foreground.maybe_voice_story(message, _as_plan)

    def _voice_story_cooldown_remaining(self, chat_id):
        since = (
            datetime.now(timezone.utc)
            - timedelta(seconds=self.settings.voice_story_cooldown)
        ).isoformat()
        rows = self.repository(chat_id).generated_since(since, "voice_story")
        if not rows:
            return 0
        last_created_at = max(row["created_at"] for row in rows)
        last = datetime.fromisoformat(last_created_at)
        remaining = self.settings.voice_story_cooldown - (
            datetime.now(timezone.utc) - last
        ).total_seconds()
        return max(0, int(remaining + 0.999))

    def take_voice_story_cooldown_notice(self, chat_id):
        """Return remaining voice cooldown at most once per minute per chat."""
        remaining = self._voice_story_cooldown_remaining(chat_id)
        if remaining <= 0:
            return 0
        now = self._clock()
        with self._lock:
            last_notice = self.foreground._voice_cooldown_notices.get(chat_id)
            if last_notice is not None and now - last_notice < 60:
                return 0
            self.foreground._voice_cooldown_notices[chat_id] = now
        return remaining

    def release_voice_story_cooldown_notice(self, chat_id):
        """Rollback a transient notice claim when Telegram delivery failed."""
        with self._lock:
            self.foreground._voice_cooldown_notices.pop(chat_id, None)

    def claim_scheduled_event(self, chat_id, event_key):
        """Deprecated identity-only compatibility seam; not production delivery."""
        return self.repository(chat_id).claim_scheduled_event(event_key)

    def deliver_scheduled_event(
        self, chat_id, event_key, event_kind, scheduled_at, payload, sender,
        parse_mode=None, current=None,
    ):
        spec = ScheduledEventSpec(
            event_key=str(event_key),
            event_kind=str(event_kind),
            scheduled_at=scheduled_at,
            payload=str(payload),
            parse_mode=parse_mode,
        )
        return self.scheduled_delivery.deliver_event(
            chat_id, spec, sender, current=current
        )

    def deliver_pending_scheduled_events(
        self, chat_id, sender, current=None, limit=10,
    ):
        return self.scheduled_delivery.deliver_pending(
            chat_id, sender, current=current, limit=limit
        )

    def scheduled_delivery_diagnostics(self, chat_id, current=None):
        return self.scheduled_delivery.diagnostics(chat_id, current)

    def format_scheduled_delivery_diagnostics(self, chat_id, current=None):
        return self.scheduled_delivery.format_diagnostics(chat_id, current)

    def maybe_sglypa_reply(self, message, _as_plan=False):
        self._sync_foreground_runtime_ports()
        return self.foreground.maybe_sglypa_reply(message, _as_plan)

    def _as_utc(self, value):
        return self.generation._as_utc(value)

    def prepare_autonomous(self, chat_id, current, is_workday=True):
        self._sync_autonomous_runtime_ports()
        return self.autonomous.prepare_autonomous(chat_id, current, is_workday)

    def _maybe_autonomous(
        self, chat_id, current, is_workday=True, _as_plan=False
    ):
        self._sync_autonomous_runtime_ports()
        return self.autonomous._maybe_autonomous(chat_id, current, is_workday, _as_plan)

    def _quiet_hours_at(self, current):
        return self.autonomous._quiet_hours_at(current)

    def ingest_gif(self, message):
        return self.media_coordinator.ingest_gif(message)

    def ingest_sticker(self, message):
        return self.media_coordinator.ingest_sticker(message)

    def telegram_image_metadata(self, message):
        return self.media_coordinator.telegram_image_metadata(message)

    def ingest_chat_image(self, message):
        return self.media_coordinator.ingest_chat_image(message)

    def render_meme(self, decision, source_path=None, *, background=None):
        return self.media_coordinator.render_meme(decision, source_path, background=background)

    def startup_meme(self, chat_id=None):
        return self.media_coordinator.startup_meme(chat_id)

    def mark_startup_meme_sent(self, decision, chat_id=None):
        return self.media_coordinator.mark_startup_meme_sent(decision, chat_id)

    def meme_command_on_cooldown(self, chat_id):
        return self.media_coordinator.meme_command_on_cooldown(chat_id)

    def fallback_command_meme_background(self, decision, chat_id):
        return self.media_coordinator.fallback_command_meme_background(decision, chat_id)

    def maybe_command_meme(self, chat_or_message, hint=""):
        self._sync_media_runtime_ports()
        return self.media_coordinator.maybe_command_meme(chat_or_message, hint)

    def _sync_media_runtime_ports(self):
        self.media_coordinator.bind_runtime_ports(
            activity_allows=self.activity_allows,
            media_enabled=self.media_enabled,
            troll_mode=self.troll_mode,
            provider_available=self.provider_available,
            generate_llm=self.generate_llm,
            meme_cooldown=self.meme_command_on_cooldown,
            local_caption=self.media_coordinator._local_command_caption,
        )

    def mark_command_meme_sent(self, chat_id, decision):
        return self.media_coordinator.mark_command_meme_sent(chat_id, decision)

    def cleanup_rendered_meme(self, result):
        return self.media_coordinator.cleanup_rendered_meme(result)

    def forget_chat(self, chat_id):
        self.repository(chat_id).clear()
        self.generation.forget_chat(chat_id)
        self.foreground.forget_chat(chat_id)
        self.autonomous.forget_chat(chat_id)
        with self._lock:
            self.persona.clear_chat(chat_id)
            self.meme_sources.clear_chat(chat_id)
            self._command_meme_sources.clear()
        log.info("Chat learning database cleared chat=%s", chat_id)
