import logging
import random
import re
import threading
import hashlib
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from .filters import similarity, validate_generated
from .chat_state import ChatStateAnalyzer
from .conversation_policy import ConversationPolicy
from .autonomous_policy import AutonomousPolicy
from .generator import LocalGenerator
from .markov import MarkovModel
from .media_catalog import MediaCatalog
from .media_service import MediaDecision, MediaService
from .memory_service import MemoryService
from .meme_lexicon import MemeLexicon
from .meme_renderer import MemeRenderer
from .meme_sources import MemeSource, MemeSourceSelector
from .persona import PersonaBuilder
from .preprocessing import (
    FOREIGN_BOT_COMMAND_RE,
    VOICE_STORY_COMMAND_RE,
    normalize_spaces,
    rejection_reason,
    significant_words,
)
from .repository import ChatRepository
from .provider_factory import create_llm_provider, create_llm_providers
from .triggers import TriggerEngine

log = logging.getLogger(__name__)

SUBSCRIPTION_REQUIRED = (
    "🔒 OpenAI-модуль Киберстула доступен только в основном чате. "
    "Для этого чата потребуется подписка."
)

# These are jokes about being invoked, rather than about the conversation.
# The first genuinely novel one may pass; repeats are suppressed for a long
# time by _chair_call_meta_joke_on_cooldown.
CHAIR_CALL_META_JOKE_RE = re.compile(
    r"(?:опять|снова|вновь).{0,30}(?:зов[её]т|ор[её]т|клич[её]т|вызвал|позвал)"
    r".{0,30}(?:стул|меня)|(?:зов[её]те|ор[её]те|кличете).{0,30}(?:стул|меня)",
    re.I,
)


class LearningService:
    def __init__(
        self,
        settings,
        openai_client=None,
        rng=None,
        clock=None,
        llm_provider=None,
        xai_client=None,
    ):
        self.settings = settings
        self.rng = rng or random
        self.triggers = TriggerEngine(settings, self.rng, clock)
        self.local = LocalGenerator(settings, self.rng)
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
            # Backward compatibility for tests/callers using service.openai.
            self.openai = self._injected_provider
        else:
            self.providers = create_llm_providers(settings, openai_client, xai_client)
            self.llm_provider = self.providers[settings.llm_provider.strip().casefold()]
            self.openai = self.providers["openai"]
            self.grok = self.providers["grok"]
        self.memory = MemoryService(
            settings, self.llm_provider, self._speaker_name,
            provider_resolver=self.provider_for_chat,
        )
        self.chat_state_analyzer = ChatStateAnalyzer(settings, self.memory)
        self.conversation_policy = ConversationPolicy(settings)
        self.autonomous_policy = AutonomousPolicy(settings, self.conversation_policy)
        self.meme_lexicon = MemeLexicon()
        self.persona = PersonaBuilder(settings, self.meme_lexicon)
        self.media_catalog = MediaCatalog()
        self.media = MediaService(settings, self.media_catalog, self.rng)
        self.meme_renderer = MemeRenderer(
            self.media_catalog, self.settings.data_dir / "generated_media"
        )
        self.meme_sources = MemeSourceSelector(self.rng)
        self._last_policy_target_user = {}
        self._policy_target_streak = {}
        self._policy_answered_messages = {}
        self._last_chat_state = {}
        self._last_conversation_decision = {}
        self._last_autonomous_decision = {}
        self._repositories = {}
        self._models = OrderedDict()
        self._model_counts = {}
        self._voice_cooldown_notices = {}
        self._command_meme_sources = {}
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()

    def repository(self, chat_id):
        with self._lock:
            if chat_id not in self._repositories:
                repository = ChatRepository(
                    self.settings.data_dir, chat_id, self.settings.max_messages_per_chat
                )
                removed = repository.purge_matching_text(FOREIGN_BOT_COMMAND_RE)
                removed += repository.purge_matching_text(VOICE_STORY_COMMAND_RE)
                if removed:
                    log.info("Forbidden memory purged chat=%s rows=%s", chat_id, removed)
                self._repositories[chat_id] = repository
            return self._repositories[chat_id]

    def activity_percent(self, chat_id):
        raw = self.repository(chat_id).setting(
            "activity_percent",
            str(self.settings.default_activity_percent),
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
        return self.repository(chat_id).setting(key, default) == "1"

    def set_enabled(self, chat_id, kind, enabled):
        if kind not in {"learning", "talk"}:
            raise ValueError("unknown setting")
        self.repository(chat_id).set_setting(kind, "1" if enabled else "0")

    def troll_mode(self, chat_id):
        return self.repository(chat_id).setting("troll_mode", "1") == "1"

    def set_troll_mode(self, chat_id, enabled):
        self.repository(chat_id).set_setting("troll_mode", "1" if enabled else "0")

    def autonomous_enabled(self, chat_id):
        return self.repository(chat_id).setting("autonomous_enabled", "1") == "1"

    def set_autonomous_enabled(self, chat_id, enabled):
        self.repository(chat_id).set_setting("autonomous_enabled", "1" if enabled else "0")

    def media_enabled(self, chat_id):
        return self.repository(chat_id).setting("media_enabled", "1") == "1"

    def set_media_enabled(self, chat_id, enabled):
        self.repository(chat_id).set_setting("media_enabled", "1" if enabled else "0")

    def llm_provider_name(self, chat_id):
        default = self.settings.llm_provider.strip().casefold()
        value = self.repository(chat_id).setting("llm_provider", default).strip().casefold()
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

    def ingest(self, message):
        chat_id = message.chat.id
        if not self._enabled(chat_id, "learning"):
            return False, "disabled"
        user = getattr(message, "from_user", None)
        if user is None or getattr(user, "is_bot", False):
            return False, "bot"
        reason = rejection_reason(message.text, self.settings.max_stored_text_length)
        if reason:
            log.debug("Learning message rejected chat=%s reason=%s", chat_id, reason)
            return False, reason
        reply = getattr(message, "reply_to_message", None)
        inserted = self.repository(chat_id).add_message(
            getattr(message, "message_id", getattr(message, "id", 0)),
            getattr(user, "id", None),
            getattr(user, "username", None),
            normalize_spaces(message.text),
            datetime.fromtimestamp(getattr(message, "date", 0), timezone.utc) if getattr(message, "date", 0) else None,
            getattr(reply, "message_id", getattr(reply, "id", None)),
            reply is not None,
        )
        if inserted:
            self.triggers.note_message(chat_id)
            self._model_counts.pop(chat_id, None)
            count = self.repository(chat_id).count()
            log.info("Learning message accepted chat=%s count=%s", chat_id, count)
            if count == self.settings.min_training_messages:
                log.info("Minimum training volume reached chat=%s", chat_id)
            self._maybe_refresh_memory(chat_id)
        return inserted, None if inserted else "duplicate"

    def _maybe_refresh_memory(self, chat_id):
        if not self.llm_allowed(chat_id):
            return False
        refreshed = self.memory.maybe_refresh(self.repository(chat_id), chat_id)
        if refreshed:
            log.info("Memory summary refreshed chat=%s", chat_id)
        return refreshed

    def _speaker_name(self, row):
        if row["speaker"] == "cyberchair":
            return "Киберстул"
        if (row.get("username") or "").casefold() == self.settings.creator_username:
            return "Харакири (создатель Киберстула)"
        return "Участник"

    def status(self, chat_id):
        repository = self.repository(chat_id)
        count = repository.count()
        total = repository.statistics()["total_messages"]
        return {
            "count": total,
            "short_memory_count": count,
            "ready": count >= self.settings.min_training_messages,
            "learning": self._enabled(chat_id, "learning"),
            "talk": self._enabled(chat_id, "talk"),
            "troll_mode": self.troll_mode(chat_id),
            "provider": self.llm_provider_name(chat_id),
            "provider_available": self.provider_available(chat_id),
            "openai": self.provider_available(chat_id),
            "activity_percent": self.activity_percent(chat_id),
            "autonomous_enabled": self.autonomous_enabled(chat_id),
            "media_enabled": self.media_enabled(chat_id),
        }

    def _model_and_messages(self, chat_id):
        repository = self.repository(chat_id)
        count = repository.count()
        with self._lock:
            if chat_id in self._models and self._model_counts.get(chat_id) == count:
                model = self._models.pop(chat_id)
                self._models[chat_id] = model
            else:
                messages = repository.recent_messages()
                model = MarkovModel().train([row["text"] for row in messages])
                self._models[chat_id] = model
                self._model_counts[chat_id] = count
                while len(self._models) > self.settings.model_cache_size:
                    old_chat, _ = self._models.popitem(last=False)
                    self._model_counts.pop(old_chat, None)
            messages = repository.recent_messages()
        return model, messages

    def _valid(self, text, input_text=None, source_texts=(), chat_id=None,
               max_words=None):
        previous = []
        if chat_id is not None:
            since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            previous = [row["text"] for row in self.repository(chat_id).generated_since(since)]
        return validate_generated(
            text, source_texts, input_text, previous,
            self.settings.min_generated_words,
            max_words or self.settings.max_generated_words + 8,
        )[0]

    def _chair_call_meta_joke_on_cooldown(self, chat_id, text):
        if not CHAIR_CALL_META_JOKE_RE.search(text or ""):
            return False
        since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        return any(
            CHAIR_CALL_META_JOKE_RE.search(row["text"] or "")
            for row in self.repository(chat_id).generated_since(since)
        )

    def _message_context(self, message):
        text = getattr(message, "text", "") or ""
        user = getattr(message, "from_user", None)
        username = (getattr(user, "username", None) or "").casefold()
        if username == self.settings.creator_username:
            return f"Харакири (@{self.settings.creator_username}): {text}"
        return text

    def generate_local(self, chat_id, input_text=None, decorate=True):
        state = self.status(chat_id)
        if not state["ready"]:
            return None
        model, messages = self._model_and_messages(chat_id)
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        previous = [row["text"] for row in self.repository(chat_id).generated_since(since)]
        result, mode = self.local.create(model, messages, input_text, previous)
        if not result:
            log.info("Local generation failed chat=%s", chat_id)
            return None
        closest_sources = sorted(
            (row["text"] for row in messages),
            key=lambda source: similarity(result, source),
            reverse=True,
        )[:2]
        self.repository(chat_id).mark_used(closest_sources)
        log.info("Generation mode selected chat=%s mode=%s", chat_id, mode)
        return result

    def openai_allowed(self, chat_id):
        allowed_chat = self.settings.openai_chat_id
        return allowed_chat is not None and int(chat_id) == int(allowed_chat)

    def llm_allowed(self, chat_id):
        return self.openai_allowed(chat_id)

    def _dialogue_context(self, chat_id, context=None, max_chars=5000):
        return self.memory.short_term_context(
            self.repository(chat_id), context, max_chars
        )

    def generate_llm(
        self,
        chat_id,
        context=None,
        purpose="reply",
        conversation_decision=None,
        chat_state=None,
    ):
        if not self.llm_allowed(chat_id):
            return SUBSCRIPTION_REQUIRED
        if context and rejection_reason(context, self.settings.max_stored_text_length):
            return None
        safety_identifier = hashlib.sha256(
            f"cyberchair-chat:{chat_id}".encode("utf-8")
        ).hexdigest()[:32]
        repository = self.repository(chat_id)
        day_summary = repository.summary_for_day(self.memory.logical_day())
        selection = self.persona.build_request(
            chat_id=chat_id,
            context=context,
            purpose=purpose,
            safety_identifier=safety_identifier,
            history=(
                None
                if purpose == "voice_story"
                else self._dialogue_context(chat_id, context)
            ),
            conversation_decision=conversation_decision,
            chat_state=chat_state,
            troll_mode=self.troll_mode(chat_id),
            day_summary=day_summary,
            stable_memory=repository.stable_memories(20),
        )
        result = self.provider_for_chat(chat_id).generate(selection.request)
        max_words = 70 if purpose == "voice_story" else 18 if purpose == "creator" else 45
        if purpose == "creator" and result:
            opening = normalize_spaces(result).casefold()
            if opening.startswith(("харакири", "создатель", "опять")):
                log.info("Creator reply blocked because of a repetitive opening chat=%s", chat_id)
                return None
        if result and self._chair_call_meta_joke_on_cooldown(chat_id, result):
            log.info("Repeated chair-call meta joke blocked chat=%s", chat_id)
            return None
        if result and self._valid(result, context, chat_id=chat_id, max_words=max_words):
            self.persona.record_usage(
                chat_id, selection.meme_ids, selection.cooldown_groups
            )
            return result
        if result:
            log.info("OpenAI result blocked by filter chat=%s", chat_id)
        return None

    def generate_openai(
        self,
        chat_id,
        context=None,
        purpose="reply",
        conversation_decision=None,
        chat_state=None,
    ):
        """Backward-compatible entry point for the configured LLM provider."""
        return self.generate_llm(
            chat_id, context, purpose, conversation_decision, chat_state
        )

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

    def autonomous_diagnostics(self, chat_id):
        """Internal state for tests/logging; intentionally not a Telegram command."""
        decision = self._last_autonomous_decision.get(chat_id)
        return decision.debug() if decision else None

    def _remember_policy_target(self, chat_id, message):
        actual_user = getattr(getattr(message, "from_user", None), "id", None)
        actual_message = getattr(message, "message_id", getattr(message, "id", None))
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

    @staticmethod
    def _deterministic_media_roll(chat_id, message_id, salt):
        digest = hashlib.sha256(
            f"{chat_id}:{message_id}:{salt}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def maybe_reply(self, message, bot_id=None, bot_username=None):
        chat_id = message.chat.id
        if not self._enabled(chat_id, "talk"):
            return None
        text = message.text or ""
        reply = getattr(message, "reply_to_message", None)
        reply_user = getattr(reply, "from_user", None)
        replies_to_bot = bool(reply_user and bot_id and reply_user.id == bot_id)
        mentioned = bool(bot_username and f"@{bot_username}".casefold() in text.casefold())
        special = any(phrase in text.casefold() for phrase in self.settings.special_phrases)
        addressed = replies_to_bot or mentioned or special
        if not self.troll_mode(chat_id) and not addressed:
            return None
        state = self.chat_state_analyzer.analyze(
            self.repository(chat_id),
            incoming_message=message,
            bot_id=bot_id,
            last_target_user_id=self._last_policy_target_user.get(chat_id),
            answered_message_ids=self._policy_answered_messages.get(chat_id, ()),
        )
        local_allowed = (
            False if addressed else self.triggers.allowed(chat_id, "random")
        )
        llm_kind = "addressed" if addressed else "openai_random"
        llm_allowed = self.triggers.allowed(chat_id, llm_kind, addressed=addressed)
        incoming_user_id = getattr(getattr(message, "from_user", None), "id", None)
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
            quiet_hours=self._policy_quiet_hours(),
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
        repository = self.repository(chat_id)
        day_summary = repository.summary_for_day(self.memory.logical_day()) or {}
        callbacks = self.persona.select_callbacks(
            day_summary,
            repository.stable_memories(20),
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
        if self.media_enabled(chat_id):
            media_decision = self.media.decide(
                chat_id=chat_id,
                repository=repository,
                conversation_decision=decision,
                chat_state=state,
                short_term_rows=self.memory.short_term_rows(repository),
                target_text=text,
                selected_memes=selected_memes,
                local_callbacks=callbacks,
                troll_mode=self.troll_mode(chat_id),
                probability_roll=self._deterministic_media_roll(
                    chat_id,
                    getattr(message, "message_id", getattr(message, "id", 0)),
                    "media",
                ),
                meme_roll=self._deterministic_media_roll(
                    chat_id,
                    getattr(message, "message_id", getattr(message, "id", 0)),
                    "meme",
                ),
            )
        if media_decision.action != "none":
            self.triggers.commit(chat_id, kind)
            self.media.commit(repository, media_decision)
            repository.record_generated(
                media_decision.template_id or media_decision.asset_id or "media",
                "contextual_media",
            )
            self._remember_policy_target(chat_id, message)
            return media_decision
        # Addressed replies use the selected provider. Ordinary random replies
        # may still use the existing local generator.
        if kind in {"addressed", "openai_random"}:
            purpose = "reply" if kind == "addressed" else "random_reply"
            result = self.generate_openai(
                chat_id,
                self._message_context(message),
                purpose,
                conversation_decision=decision,
                chat_state=state,
            )
            provider = "openai"
        else:
            result = self.generate_local(chat_id, text)
            provider = "markov"
        if result:
            self.triggers.commit(chat_id, kind)
            self.repository(chat_id).record_generated(result, kind)
            self._remember_policy_target(chat_id, message)
            log.info(
                "Generated reply ready chat=%s trigger=%s provider=%s",
                chat_id,
                kind,
                provider,
            )
        return result

    def maybe_special_ai(self, message, kind, chance, purpose, addressed=True):
        chat_id = message.chat.id
        if not self._enabled(chat_id, "talk") or not self.troll_mode(chat_id):
            return None
        if not self.triggers.allowed(chat_id, kind, addressed=addressed):
            return None
        if self.rng.random() >= chance:
            return None
        result = self.generate_openai(chat_id, self._message_context(message), purpose)
        if result:
            self.triggers.commit(chat_id, kind)
            self.repository(chat_id).record_generated(result, kind)
        return result

    def maybe_stul_cooldown_reply(self, message):
        """Choose exactly one provider for a repeated chair trigger."""
        chat_id = message.chat.id
        if not self._enabled(chat_id, "talk") or not self.troll_mode(chat_id):
            return None
        if not self.triggers.allowed(chat_id, "stul_cooldown", addressed=True):
            return None

        # “стул” is an address, never the subject by itself.  A call that also
        # contains a real topic must reach the contextual model even if it has
        # no question mark; otherwise rapid calls degrade into Markov jokes
        # about the invocation rather than answering the chat.
        subject = self.persona._strip_chair_invocation(message.text)
        subject_words = {
            word for word in significant_words(subject)
            if word not in {"еще", "ещё", "раз", "снова", "опять", "второй"}
        }
        if subject_words:
            result = self.generate_openai(
                chat_id,
                self._message_context(message),
                "reply",
            )
            provider = "openai"
            if result:
                self.triggers.commit(chat_id, "stul_cooldown")
                self.repository(chat_id).record_generated(result, "stul_cooldown")
                log.info(
                    "Contextual chair call answered chat=%s provider=%s",
                    chat_id,
                    provider,
                )
            return result

        frequency = self.triggers.note_chair(chat_id)
        roll = self.rng.random()
        ai_threshold = self.settings.reply_to_stul_chance
        markov_threshold = ai_threshold + self.settings.stul_markov_reply_chance

        if frequency >= 2 and roll < self.settings.frequent_stul_markov_chance:
            result = self.generate_local(chat_id, message.text)
            provider = "markov"
        elif roll < ai_threshold:
            result = self.generate_openai(
                chat_id,
                self._message_context(message),
                "stul_cooldown",
            )
            provider = "openai"
        elif roll < markov_threshold:
            result = self.generate_local(chat_id, message.text)
            provider = "markov"
        else:
            return None

        if result:
            if self._chair_call_meta_joke_on_cooldown(chat_id, result):
                log.info("Repeated chair-call meta joke blocked chat=%s", chat_id)
                return None
            self.triggers.commit(chat_id, "stul_cooldown")
            self.repository(chat_id).record_generated(result, "stul_cooldown")
            log.info(
                "Repeated chair trigger answered chat=%s provider=%s",
                chat_id,
                provider,
            )
        return result

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

    def maybe_question_reply(self, message):
        chat_id = message.chat.id
        if not self._enabled(chat_id, "talk"):
            return None
        if not self.triggers.allowed(chat_id, "chair_question", addressed=True):
            return None
        result = self.generate_openai(
            chat_id,
            self._message_context(message),
            "question",
        )
        if result:
            self.triggers.commit(chat_id, "chair_question")
            self.repository(chat_id).record_generated(result, "chair_question")
        return result

    def maybe_voice_story(self, message):
        chat_id = message.chat.id
        if not self._enabled(chat_id, "talk") or not self.troll_mode(chat_id):
            return None
        since = (
            datetime.now(timezone.utc)
            - timedelta(seconds=self.settings.voice_story_cooldown)
        ).isoformat()
        if self.repository(chat_id).generated_since(since, "voice_story"):
            return None
        # The invocation is a control command, not story context or memory.
        result = self.generate_openai(chat_id, None, "voice_story")
        if result:
            self.repository(chat_id).record_generated(result, "voice_story")
        return result

    def voice_story_cooldown_remaining(self, chat_id):
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
        remaining = self.voice_story_cooldown_remaining(chat_id)
        if remaining <= 0:
            return 0
        now = self._clock()
        with self._lock:
            last_notice = self._voice_cooldown_notices.get(chat_id)
            if last_notice is not None and now - last_notice < 60:
                return 0
            self._voice_cooldown_notices[chat_id] = now
        return remaining

    def claim_scheduled_event(self, chat_id, event_key):
        return self.repository(chat_id).claim_scheduled_event(event_key)

    def maybe_sglypa_reply(self, message):
        chat_id = message.chat.id
        if not self.troll_mode(chat_id):
            return None
        if self.rng.random() >= self.settings.sglypa_reply_chance:
            return None
        if not self.triggers.allowed(chat_id, "sglypa", addressed=True):
            return None
        result = self.generate_openai(chat_id, message.text, "sglypa")
        if not result:
            return None
        self.triggers.commit(chat_id, "sglypa")
        self.repository(chat_id).record_generated(result, "sglypa")
        return result

    @staticmethod
    def _as_utc(value):
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def maybe_autonomous(self, chat_id, current, is_workday=True):
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
        utc_current = current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current.astimezone(timezone.utc)
        quiet = self._quiet_hours_at(current)
        state = self.chat_state_analyzer.analyze(
            repository,
            last_target_user_id=self._last_policy_target_user.get(chat_id),
            answered_message_ids=self._policy_answered_messages.get(chat_id, ()),
            now=utc_current,
        )
        day = self.memory.logical_day(current)
        day_start = current.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()
        latest_bot = self._as_utc((repository.latest_generated() or {}).get("created_at"))
        latest_auto = self._as_utc((repository.latest_generated(("autonomous", "autonomous_media")) or {}).get("created_at"))
        latest_message = repository.recent_messages(1)
        latest_human = self._as_utc(latest_message[-1]["created_at"]) if latest_message else None
        prior_start = (utc_current - timedelta(hours=3)).isoformat()
        autonomous_count = sum(
            1 for row in repository.generated_since(day_start)
            if row["kind"] in {"autonomous", "autonomous_media"}
        )
        autonomous = self.autonomous_policy.decide(
            state,
            current=utc_current,
            summary=repository.summary_for_day(day) or {},
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
        rows = self.memory.short_term_rows(repository)
        target = next((row for row in rows if row.get("message_id") == decision.target_message_id), None)
        target_text = (target or {}).get("text", "")
        summary = repository.summary_for_day(day) or {}
        callbacks = self.persona.select_callbacks(
            summary, repository.stable_memories(20), target_text, state.dominant_topic
        )
        selected_memes = self.meme_lexicon.select(
            target_text or state.dominant_topic or "",
            {state.conversation_type, decision.preferred_style},
            decision.troll_intensity,
            limit=3,
        )
        media = MediaDecision(reason="media_disabled")
        if self.media_enabled(chat_id):
            media = self.media.decide(
                chat_id, repository, decision, state, rows, target_text,
                selected_memes, callbacks, troll_mode=True,
            )
        if media.action != "none":
            self.triggers.commit(chat_id, "autonomous")
            self.media.commit(repository, media)
            repository.record_generated(
                media.template_id or media.asset_id or "media", "autonomous_media", utc_current
            )
            return media
        if not self.provider_available(chat_id):
            return None
        result = self.generate_openai(
            chat_id, target_text or None, "autonomous", decision, state
        )
        if not result:
            return None
        self.triggers.commit(chat_id, "autonomous")
        repository.record_generated(result, "autonomous", utc_current)
        return result

    def _quiet_hours_at(self, current):
        hour = current.hour
        if self.settings.quiet_start_hour > self.settings.quiet_end_hour:
            return hour >= self.settings.quiet_start_hour or hour < self.settings.quiet_end_hour
        return self.settings.quiet_start_hour <= hour < self.settings.quiet_end_hour

    def ingest_gif(self, message):
        if not self.settings.gif_enabled:
            return False
        user = getattr(message, "from_user", None)
        if not user or getattr(user, "is_bot", False):
            return False
        animation = getattr(message, "animation", None)
        document = getattr(message, "document", None)
        media = animation or document
        if not media or not getattr(media, "file_id", None):
            return False
        inserted = self.repository(message.chat.id).add_gif(
            getattr(message, "message_id", getattr(message, "id", 0)),
            getattr(user, "id", None),
            media.file_id,
            getattr(media, "file_unique_id", media.file_id),
            datetime.fromtimestamp(getattr(message, "date", 0), timezone.utc)
            if getattr(message, "date", 0) else None,
            self.settings.max_gifs_per_chat,
        )
        if inserted:
            log.info("GIF accepted chat=%s count=%s", message.chat.id, self.repository(message.chat.id).gif_count())
        return inserted

    def ingest_sticker(self, message):
        if not self.settings.sticker_enabled:
            return False
        user = getattr(message, "from_user", None)
        sticker = getattr(message, "sticker", None)
        if not user or getattr(user, "is_bot", False) or not sticker:
            return False
        inserted = self.repository(message.chat.id).add_sticker(
            getattr(message, "message_id", getattr(message, "id", 0)),
            getattr(user, "id", None),
            sticker.file_id,
            getattr(sticker, "file_unique_id", sticker.file_id),
            datetime.fromtimestamp(getattr(message, "date", 0), timezone.utc)
            if getattr(message, "date", 0) else None,
            self.settings.max_stickers_per_chat,
        )
        if inserted:
            log.info(
                "Sticker accepted chat=%s count=%s",
                message.chat.id,
                self.repository(message.chat.id).sticker_count(),
            )
        return inserted

    def maybe_random_media(self, chat_id):
        if (
            not self.troll_mode(chat_id)
            or not self.media_enabled(chat_id)
            or (not self.settings.gif_enabled and not self.settings.sticker_enabled)
            or not self.activity_allows(chat_id)
            or self.rng.random() >= self.settings.gif_post_chance
        ):
            return None
        repository = self.repository(chat_id)
        since = (datetime.now(timezone.utc) - timedelta(seconds=self.settings.gif_post_cooldown)).isoformat()
        if repository.generated_since(since, "random_media"):
            return None
        decision = self.media.random_fallback(repository)
        if decision.action == "none":
            return None
        repository.mark_media_file_used(decision.action, decision.asset_id)
        self.media.commit(repository, decision)
        repository.record_generated(decision.asset_id, "random_media")
        media_type = "animation" if decision.action == "gif" else "sticker"
        log.info("Random media selected chat=%s type=%s", chat_id, media_type)
        return media_type, decision.asset_id

    def render_meme(self, decision):
        if not isinstance(decision, MediaDecision) or decision.action != "meme":
            return None
        return self.meme_renderer.render(decision.template_id, decision.caption_text)

    def startup_meme(self, chat_id=None):
        """Return the one-time meme reserved for the next successful restart."""
        chat_id = self.settings.openai_chat_id if chat_id is None else chat_id
        if chat_id is None or not self.troll_mode(chat_id) or not self.media_enabled(chat_id):
            return None
        repository = self.repository(chat_id)
        if repository.setting("startup_meme_v1", "0") == "1":
            return None
        asset = self.media_catalog.get("t800_chud")
        if not asset or not self.media_catalog.resolve(asset):
            return None
        return MediaDecision(
            action="meme", asset_id=asset.id, template_id=asset.id,
            caption_text="chairOS online: мемная подсистема активирована",
            confidence=1.0, reason="startup_meme_v1",
            asset_key=asset.id, cooldown_group=asset.cooldown_group,
            archetype=asset.archetype,
        )

    def mark_startup_meme_sent(self, decision, chat_id=None):
        chat_id = self.settings.openai_chat_id if chat_id is None else chat_id
        if chat_id is None or not isinstance(decision, MediaDecision):
            return
        repository = self.repository(chat_id)
        self.media.commit(repository, decision)
        repository.record_generated(decision.template_id or "startup_meme", "startup_meme")
        repository.set_setting("startup_meme_v1", "1")

    def meme_command_on_cooldown(self, chat_id):
        """Whether the *AI caption* path is cooling down, not the command."""
        since = (
            datetime.now(timezone.utc)
            - timedelta(seconds=self.settings.manual_meme_cooldown)
        ).isoformat()
        return bool(self.repository(chat_id).generated_since(since, "manual_meme"))

    def maybe_command_meme(self, chat_id):
        """Always make a meme; AI cooldown only switches its caption source."""
        if not self.media_enabled(chat_id):
            return None
        repository = self.repository(chat_id)
        rows = repository.meme_source_messages()
        summary = repository.summary_for_day(self.memory.logical_day()) or {}
        callbacks = self.persona.select_callbacks(
            summary, repository.stable_memories(20), "", None
        )
        ai_ready = (
            self.troll_mode(chat_id)
            and self.provider_available(chat_id)
            and not self.meme_command_on_cooldown(chat_id)
        )
        source = self.meme_sources.choose(
            chat_id, rows, callbacks, fallback=not ai_ready
        )
        caption = None
        if ai_ready:
            context = source.text or None
            caption = self.generate_openai(chat_id, context, "meme_caption")
        if not caption:
            source, caption = self._local_command_caption(chat_id, source, rows, callbacks)
        if not caption:
            return None
        candidates = [
            asset for asset in self.media_catalog.assets
            if asset.type == "meme_template" and self.media_catalog.resolve(asset)
        ]
        if not candidates:
            return None
        unused = [asset for asset in candidates if asset.id not in self.meme_sources.recent_templates(chat_id)]
        asset = self.rng.choice(unused or candidates)
        decision = MediaDecision(
            action="meme", asset_id=asset.id, template_id=asset.id,
            source_message_id=source.message_id, caption_text=caption, confidence=1.0,
            reason=f"manual_{'ai' if ai_ready else 'local'}_{source.kind}",
            asset_key=asset.id, cooldown_group=asset.cooldown_group,
            archetype=asset.archetype,
        )
        self._command_meme_sources[decision] = source
        return decision

    def _local_command_caption(self, chat_id, source, rows, callbacks):
        """Local cascade: real old quote → fresh → callback → rare Markov → phrase."""
        if source.kind in {"old", "fresh", "callback"} and source.text:
            return source, self._caption_from_source(source)
        if self.meme_sources.markov_allowed(chat_id):
            brainrot = self.generate_local(chat_id, None)
            if brainrot:
                return MemeSource("markov", brainrot), self._caption_from_source(
                    MemeSource("markov", brainrot)
                )
        phrases = (
            "chairOS фиксирует промышленный скилл ишью",
            "минус аура, протокол брейнрота активирован",
            "лил бро выбрал сайдквест вместо жизни",
            "сканирование завершено: проект кукд",
        )
        phrase = self.rng.choice(phrases)
        return MemeSource("phrase", phrase), phrase

    def _caption_from_source(self, source):
        text = normalize_spaces(source.text)[:110]
        if source.kind == "callback":
            return f"{text} — канон ивент"
        if source.kind == "markov":
            return f"{text} — пошёл брейнрот"
        tails = (
            "минус аура зафиксирована",
            "лил бро ларпит сигму",
            "chairOS фиксирует скилл ишью",
            "сойджак-комиссия уже выехала",
        )
        return f"{text} — {self.rng.choice(tails)}"

    def mark_command_meme_sent(self, chat_id, decision):
        if not isinstance(decision, MediaDecision):
            return
        repository = self.repository(chat_id)
        self.media.commit(repository, decision)
        repository.record_generated(decision.template_id or "manual_meme", "manual_meme")
        source = self._command_meme_sources.pop(
            decision,
            MemeSource((decision.reason or "").rsplit("_", 1)[-1], "", decision.source_message_id),
        )
        self.meme_sources.record(
            chat_id, source, decision.template_id,
        )
        if source.kind in {"old", "fresh"} and source.text:
            repository.mark_used([source.text])

    def cleanup_rendered_meme(self, result):
        self.meme_renderer.cleanup(result)

    def forget_chat(self, chat_id):
        self.repository(chat_id).clear()
        with self._lock:
            self._models.pop(chat_id, None)
            self._model_counts.pop(chat_id, None)
            self._last_policy_target_user.pop(chat_id, None)
            self._policy_target_streak.pop(chat_id, None)
            self._policy_answered_messages.pop(chat_id, None)
            self._last_chat_state.pop(chat_id, None)
            self._last_conversation_decision.pop(chat_id, None)
            self._last_autonomous_decision.pop(chat_id, None)
            self.persona.clear_chat(chat_id)
            self.meme_sources.clear_chat(chat_id)
            self._command_meme_sources.clear()
        log.info("Chat learning database cleared chat=%s", chat_id)
