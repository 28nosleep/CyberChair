"""Media and explicit-meme preparation coordination boundary.

The coordinator joins MediaService, MemeRenderer, MediaCatalog and source
selection.  It owns no Telegram delivery and does not replace those specialized
algorithms.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from .event_context import current_event, implicit_event_id
from .media_service import MediaDecision
from .meme_sources import MemeSource
from .normalized_event import NormalizedEvent, normalize_telegram_event
from .preprocessing import normalize_spaces


log = logging.getLogger("learning.service")

SAFE_CHAT_IMAGE_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/gif", "image/bmp", "image/tiff",
}


class MediaCoordinator:
    """Own media/meme orchestration over shared stateful collaborators."""

    def __init__(
        self, *, settings, repository, normalize_event, active_snapshot,
        media, media_catalog, meme_renderer, meme_sources, quality_guard,
        memory, persona, rng, concurrency, activity_allows, media_enabled,
        troll_mode, provider_available, generate_llm, generate_local,
        command_meme_sources, lock, photo_meme_caption_re,
    ):
        self.settings=settings
        self.repository=repository
        self._normalized_event=normalize_event
        self._active_context_snapshot=active_snapshot
        self.media=media
        self.media_catalog=media_catalog
        self.meme_renderer=meme_renderer
        self.meme_sources=meme_sources
        self.quality_guard=quality_guard
        self.memory=memory
        self.persona=persona
        self.rng=rng
        self.concurrency=concurrency
        self.activity_allows=activity_allows
        self.media_enabled=media_enabled
        self.troll_mode=troll_mode
        self.provider_available=provider_available
        self.generate_llm=generate_llm
        self.generate_local=generate_local
        self._command_meme_sources=command_meme_sources
        self._lock=lock
        self._photo_meme_caption_re=photo_meme_caption_re
        self._meme_cooldown = self.meme_command_on_cooldown
        self._local_caption = self._local_command_caption

    def bind_runtime_ports(
        self, *, activity_allows, media_enabled, troll_mode,
        provider_available, generate_llm, generate_local, meme_cooldown,
        local_caption,
    ):
        """Refresh injected facade ports without taking a service back-reference."""
        self.activity_allows = activity_allows
        self.media_enabled = media_enabled
        self.troll_mode = troll_mode
        self.provider_available = provider_available
        self.generate_llm = generate_llm
        self.generate_local = generate_local
        self._meme_cooldown = meme_cooldown
        self._local_caption = local_caption

    def forget_chat(self, chat_id):
        with self._lock:
            self.meme_sources.clear_chat(chat_id)
            self._command_meme_sources.clear()

    def ingest_gif(self, message):
        event = self._normalized_event(message)
        if not self.settings.gif_enabled:
            return False
        if event.user_id is None or event.user_is_bot:
            return False
        media = event.media
        if not media or media.kind not in {"animation", "document"}:
            return False
        inserted = self.repository(event.chat_id).add_gif(
            event.message_id,
            event.user_id,
            media.file_id,
            media.file_unique_id,
            event.timestamp,
            self.settings.max_gifs_per_chat,
        )
        if inserted:
            log.info(
                "GIF accepted chat=%s count=%s", event.chat_id,
                self.repository(event.chat_id).gif_count(),
            )
        return inserted

    def ingest_sticker(self, message):
        event = self._normalized_event(message)
        if not self.settings.sticker_enabled:
            return False
        sticker = event.media
        if (
            event.user_id is None or event.user_is_bot
            or not sticker or sticker.kind != "sticker"
        ):
            return False
        inserted = self.repository(event.chat_id).add_sticker(
            event.message_id,
            event.user_id,
            sticker.file_id,
            sticker.file_unique_id,
            event.timestamp,
            self.settings.max_stickers_per_chat,
        )
        if inserted:
            log.info(
                "Sticker accepted chat=%s count=%s",
                event.chat_id,
                self.repository(event.chat_id).sticker_count(),
            )
        return inserted

    def telegram_image_metadata(self, message):
        """Compatibility facade over the adapter-normalized media facts."""
        if message is None:
            return None
        event = (
            message if isinstance(message, NormalizedEvent)
            else normalize_telegram_event(message)
        )
        return self._event_image_metadata(event)

    def _event_image_metadata(self, event, reply=False):
        media = event.reply_media if reply else event.media
        if not media or media.kind not in {"photo", "document"}:
            return None
        mime_type = media.mime_type or ""
        if media.kind == "document" and mime_type not in SAFE_CHAT_IMAGE_MIME_TYPES:
            return None
        return {
            "message_id": (
                event.reply_to_message_id if reply else event.message_id
            ),
            "user_id": event.reply_to_user_id if reply else event.user_id,
            "file_id": media.file_id,
            "file_unique_id": media.file_unique_id,
            "media_type": media.kind,
            "mime_type": mime_type,
            "caption": normalize_spaces(
                event.reply_effective_text if reply else event.caption
            ),
            "file_size": media.file_size,
            "width": media.width,
            "height": media.height,
            "from_bot": event.reply_to_user_is_bot if reply else event.user_is_bot,
            "created_at": event.reply_timestamp if reply else event.timestamp,
        }

    def ingest_chat_image(self, message):
        event = self._normalized_event(message)
        metadata = self._event_image_metadata(event)
        if not metadata:
            return False
        inserted = self.repository(event.chat_id).add_chat_image(
            **metadata, max_images=self.settings.max_chat_images_per_chat
        )
        if inserted:
            log.info(
                "Chat image metadata accepted chat=%s message=%s from_bot=%s",
                event.chat_id, metadata["message_id"], metadata["from_bot"],
            )
        return inserted

    def render_meme(self, decision, source_path=None, *, background=None):
        if not isinstance(decision, MediaDecision) or decision.action != "meme":
            return None
        event = current_event()
        if background is None:
            background = bool(event is not None and event.event_type == "autonomous")
        chat_id = event.chat_id if event is not None else self.settings.openai_chat_id or 0
        event_id = (
            event.event_id if event is not None
            else implicit_event_id("media", chat_id)
        )
        with self.concurrency.media_slot(
            event_id, chat_id, background=background
        ) as admission:
            if not admission:
                return None
            if source_path is not None and decision.background_file_id:
                return self.meme_renderer.render_image(
                    source_path,
                    decision.caption_text,
                    decision.render_profile or "top_caption",
                    max_bytes=self.settings.max_chat_image_bytes,
                    max_dimension=self.settings.max_chat_image_dimension,
                    max_pixels=self.settings.max_chat_image_pixels,
                )
            return self.meme_renderer.render(
                decision.template_id, decision.caption_text
            )

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
        rows = repository.meme_source_messages()
        real_caption = next((
            normalize_spaces(row.get("text", ""))[:110]
            for row in reversed(rows) if normalize_spaces(row.get("text", ""))
        ), None)
        if not real_caption:
            return None
        return MediaDecision(
            action="meme", asset_id=asset.id, template_id=asset.id,
            caption_text=real_caption,
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
        repository.record_routing_event("meme_caption_source_recent_quote")
        repository.set_setting("startup_meme_v1", "1")

    def meme_command_on_cooldown(self, chat_id):
        """Whether the *AI caption* path is cooling down, not the command."""
        since = (
            datetime.now(timezone.utc)
            - timedelta(seconds=self.settings.manual_meme_cooldown)
        ).isoformat()
        snapshot = self._active_context_snapshot(chat_id)
        if snapshot is not None:
            return any(
                row.get("kind") == "manual_meme"
                and str(row.get("created_at") or "") >= since
                for row in snapshot.recent_generated
            )
        return bool(self.repository(chat_id).generated_since(since, "manual_meme"))

    def _curated_command_background(self, recent_templates=()):
        candidates = [
            asset for asset in self.media_catalog.assets
            if asset.type == "meme_template" and self.media_catalog.resolve(asset)
        ]
        if not candidates:
            return None
        unused = [asset for asset in candidates if asset.id not in recent_templates]
        return self.rng.choice(unused or candidates)

    def _command_background_decision(self, decision, asset):
        if not asset:
            return None
        values = decision.debug()
        values.update({
            "asset_id": asset.id,
            "template_id": asset.id,
            "asset_key": asset.id,
            "cooldown_group": asset.cooldown_group,
            "archetype": asset.archetype,
            "background_file_id": None,
            "background_file_unique_id": None,
            "background_media_type": None,
            "background_mime_type": None,
            "background_user_id": None,
            "background_message_id": None,
            "background_explicit": False,
            "render_profile": None,
        })
        return MediaDecision(**values)

    def fallback_command_meme_background(self, decision, chat_id):
        """Choose a curated template only after a chat image failed validation."""
        asset = self._curated_command_background(
            self.meme_sources.recent_templates(chat_id)
        )
        fallback = self._command_background_decision(decision, asset)
        with self._lock:
            if fallback is not None and decision in self._command_meme_sources:
                self._command_meme_sources[fallback] = self._command_meme_sources.pop(decision)
        return fallback

    def maybe_command_meme(self, chat_or_message, hint=""):
        """Always make a meme; AI cooldown only switches its caption source."""
        event = (
            self._normalized_event(chat_or_message)
            if not isinstance(chat_or_message, int) else None
        )
        chat_id = event.chat_id if event is not None else int(chat_or_message)
        if not self.media_enabled(chat_id):
            return None
        repository = self.repository(chat_id)
        caption_match = self._photo_meme_caption_re.fullmatch(
            event.caption
        ) if event is not None and event.has_photo else None
        explicit_target_present = bool(
            event is not None
            and (caption_match or event.reply_to_message_id is not None)
        )
        explicit_image = None
        if event is not None:
            explicit_image = self._event_image_metadata(
                event, reply=not bool(caption_match)
            )
        if caption_match:
            repository.record_routing_event("photo_caption_meme_trigger")
            hint = hint or (caption_match.group("hint") or "")
        if explicit_image:
            # Preserve metadata even when an old update was missed. Bot media is
            # stored for audit/regression purposes but is never selectable.
            repository.add_chat_image(
                **explicit_image, max_images=self.settings.max_chat_images_per_chat
            )
        explicit_usable = bool(
            explicit_image
            and not explicit_image["from_bot"]
            and (explicit_image.get("file_size") or 0) <= self.settings.max_chat_image_bytes
            and max(explicit_image.get("width") or 0, explicit_image.get("height") or 0)
                <= self.settings.max_chat_image_dimension
        )
        # A reply to a non-image/bot image is an explicit but unusable choice:
        # safely fall back to curated media rather than substituting an unrelated
        # image from chat history.
        force_curated = explicit_target_present and not explicit_usable
        hint = normalize_spaces(hint)
        background_context = normalize_spaces(
            f"{hint} {(explicit_image or {}).get('caption', '')}"
        )
        rows = repository.meme_source_messages()
        snapshot = self._active_context_snapshot(chat_id)
        if snapshot is not None and snapshot.chat_id != int(chat_id):
            snapshot = None
        summary = (
            snapshot.current_summary if snapshot is not None
            else repository.summary_for_day(self.memory.logical_day()) or {}
        )
        callbacks = self.persona.select_callbacks(
            summary,
            snapshot.stable_memories if snapshot is not None
            else repository.stable_memories(20),
            background_context, hint,
        )
        ai_ready = (
            self.troll_mode(chat_id)
            and self.provider_available(chat_id)
            and not self._meme_cooldown(chat_id)
        )
        source = self.meme_sources.choose(
            chat_id, rows, callbacks, current_text=background_context,
            topic=hint, fallback=not ai_ready
        )
        caption = None
        caption_source = None
        if ai_ready:
            context = normalize_spaces(
                " | ".join(value for value in (
                    f"Пожелание к мему: {hint}" if hint else "",
                    f"Подпись исходной картинки: {explicit_image.get('caption')}"
                    if explicit_image and explicit_image.get("caption") else "",
                    source.text,
                ) if value)
            ) or None
            caption = self.generate_llm(chat_id, context, "meme_caption")
            if caption:
                caption_source = "ai"
        if not caption:
            # A failed/incomplete/invalid AI caption may use one local fallback;
            # this never performs a second LLM call and still has one final meme.
            source, caption = self._local_caption(
                chat_id, source, rows, callbacks, snapshot=snapshot
            )
            caption_source = source.kind if caption else None
        if not caption:
            return None
        base = MediaDecision(
            action="meme",
            source_message_id=source.message_id, caption_text=caption, confidence=1.0,
            reason=("manual_ai" if caption_source == "ai" else f"manual_local_{caption_source}"),
        )
        image = explicit_image if explicit_usable else None
        if image is None and not force_curated:
            chance = max(0.0, min(1.0, self.settings.chat_image_background_chance))
            if self.rng.random() < chance:
                ranked = self.media.score_chat_images(
                    repository, background_context or source.text,
                    event.user_id if event is not None else None,
                )
                caption_hash = hashlib.sha256(caption.casefold().encode("utf-8")).hexdigest()[:20]
                image = next((
                    row for row in ranked
                    if not repository.chat_image_caption_used(
                        row["file_unique_id"], caption_hash
                    )
                ), None)
        if image is not None:
            profile = self.rng.choice(("top_caption", "bottom_caption", "top_bottom"))
            values = base.debug()
            values.update({
                "background_file_id": image["file_id"],
                "background_file_unique_id": image["file_unique_id"],
                "background_media_type": image["media_type"],
                "background_mime_type": image.get("mime_type"),
                "background_user_id": image.get("user_id"),
                "background_message_id": image.get("message_id"),
                "background_explicit": bool(explicit_usable),
                "render_profile": profile,
                "asset_key": f"chat_image:{image['file_unique_id']}",
                "cooldown_group": "chat_image",
                "archetype": "chat_image",
            })
            decision = MediaDecision(**values)
        else:
            asset = self._curated_command_background(
                self.meme_sources.recent_templates(chat_id)
            )
            decision = self._command_background_decision(base, asset)
            if decision is None:
                return None
        with self._lock:
            self._command_meme_sources[decision] = source
        return decision

    def _local_command_caption(self, chat_id, source, rows, callbacks,
                               snapshot=None):
        """One ready local fallback; it never chains into another producer."""
        if source.kind in {"old", "fresh", "callback"} and source.text:
            return source, self._caption_from_source(source)
        if self.meme_sources.markov_allowed(chat_id):
            generated = self.generate_local(chat_id)
            if isinstance(generated, str) and generated:
                candidate = MemeSource("markov", generated)
                quality = self.quality_guard.check(
                    generated,
                    (
                        list(snapshot.recent_generated_texts[-40:])
                        if snapshot is not None
                        else [
                            row["text"]
                            for row in self.repository(chat_id).recent_generated(40)
                        ]
                    ),
                    local=True, image_meme=True,
                )
                if quality.accepted:
                    return candidate, self._caption_from_source(candidate)
        # Last resort is still a real chat utterance, never a phrase bank.
        for row in reversed(rows):
            text = normalize_spaces(row.get("text", ""))
            if text:
                candidate = MemeSource(
                    "fresh", text, row.get("message_id"), row.get("user_id")
                )
                return candidate, self._caption_from_source(candidate)
        return MemeSource("none", ""), None

    def _caption_from_source(self, source):
        # Quotes/callbacks/Markov are the caption itself. Appending a stock tail
        # turns real chat history back into a visible canned phrase bank.
        return normalize_spaces(source.text)[:110]

    def mark_command_meme_sent(self, chat_id, decision):
        if not isinstance(decision, MediaDecision):
            return
        repository = self.repository(chat_id)
        self.media.commit(repository, decision)
        repository.record_generated(decision.template_id or "manual_meme", "manual_meme")
        if decision.background_file_unique_id:
            caption_hash = hashlib.sha256(
                (decision.caption_text or "").casefold().encode("utf-8")
            ).hexdigest()[:20]
            repository.mark_chat_image_used(
                decision.background_file_unique_id,
                caption_hash,
                decision.background_user_id,
            )
        with self._lock:
            source = self._command_meme_sources.pop(
                decision,
                MemeSource((decision.reason or "").rsplit("_", 1)[-1], "", decision.source_message_id),
            )
        self.meme_sources.record(
            chat_id, source, decision.template_id,
        )
        reason_source = (decision.reason or "").removeprefix("manual_local_").removeprefix("manual_")
        source_name = {
            "old": "historical_quote", "fresh": "recent_quote",
            "callback": "callback", "markov": "markov", "ai": "ai",
        }.get(reason_source, source.kind)
        repository.record_routing_event(f"meme_caption_source_{source_name}")
        if source.kind in {"old", "fresh"} and source.text:
            repository.mark_used([source.text])

    def cleanup_rendered_meme(self, result):
        self.meme_renderer.cleanup(result)
