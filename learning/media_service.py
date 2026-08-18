import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from .repository import normalize_memory


@dataclass(frozen=True)
class MediaDecision:
    action: str = "none"
    asset_id: str | None = None
    template_id: str | None = None
    source_message_id: int | None = None
    caption_text: str | None = None
    confidence: float = 0.0
    reason: str = ""
    asset_key: str | None = None
    cooldown_group: str | None = None
    archetype: str | None = None
    background_file_id: str | None = None
    background_file_unique_id: str | None = None
    background_media_type: str | None = None
    background_mime_type: str | None = None
    background_user_id: int | None = None
    background_message_id: int | None = None
    background_explicit: bool = False
    render_profile: str | None = None

    def debug(self):
        return asdict(self)


class MediaService:
    """Local selector. It neither calls an LLM nor sends anything to Telegram."""

    def __init__(self, settings, catalog, rng):
        self.settings = settings
        self.catalog = catalog
        self.rng = rng

    @staticmethod
    def none(reason):
        return MediaDecision(reason=reason)

    def _recent(self, repository, media_context=None):
        if media_context is not None:
            return list(media_context.recent_usage[: self.settings.media_recent_limit])
        return repository.recent_media_usage(self.settings.media_recent_limit)

    @staticmethod
    def _as_utc(value):
        if not value:
            return None
        try:
            moment = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    def score_chat_images(self, repository, current_text="", current_user_id=None,
                          current=None):
        """Rank reusable human images without turning the archive into roulette."""
        current = current or datetime.now(timezone.utc)
        rows = repository.chat_image_candidates()
        recent_messages = repository.recent_messages(40)
        current_terms = set(normalize_memory(current_text).split())
        for row in recent_messages[-12:]:
            current_terms.update(normalize_memory(row.get("text") or "").split())
        active_users = {row.get("user_id") for row in recent_messages[-12:]}
        recent_usage = repository.recent_chat_image_usage(8)
        recent_ids = [row["file_unique_id"] for row in recent_usage]
        recent_authors = [row.get("user_id") for row in recent_usage[:3]]
        scored = []
        for row in rows:
            if row.get("from_bot") or not row.get("file_id"):
                continue
            created = self._as_utc(row.get("created_at"))
            age_days = (current - created).total_seconds() / 86400 if created else 3650
            if age_days <= 1:
                score = 5.0
            elif age_days <= 7:
                score = 3.5
            elif age_days <= 30:
                score = 1.5
            else:
                score = -1.0
            caption_terms = set(normalize_memory(row.get("caption") or "").split())
            overlap = sum(
                1 for word in caption_terms
                if any(
                    word == term
                    or (min(len(word), len(term)) >= 5 and word[:5] == term[:5])
                    for term in current_terms
                )
            )
            score += min(9.0, overlap * 2.25)
            score += min(4.0, float(row.get("reply_count") or 0) * 1.25)
            score += min(3.0, float(row.get("nearby_message_count") or 0) * .25)
            if row.get("user_id") in active_users:
                score += 2.0
            if current_user_id is not None and row.get("user_id") == current_user_id:
                score += 1.0
            used_count = int(row.get("used_count") or 0)
            if not used_count:
                score += 4.0
            else:
                score -= min(6.0, used_count * 1.4)
            last_used = self._as_utc(row.get("last_used_at"))
            if last_used:
                hours = (current - last_used).total_seconds() / 3600
                score += -8.0 if hours < 24 else -3.0 if hours < 168 else 1.0
            if recent_ids and row["file_unique_id"] == recent_ids[0]:
                continue
            score -= recent_ids.count(row["file_unique_id"]) * 4.0
            score -= recent_authors.count(row.get("user_id")) * 2.0
            item = dict(row)
            item["score"] = score
            item["topic_overlap"] = overlap
            scored.append(item)
        return sorted(
            scored,
            key=lambda item: (item["score"], item.get("created_at") or ""),
            reverse=True,
        )

    def select_chat_image(self, repository, current_text="", current_user_id=None):
        ranked = self.score_chat_images(repository, current_text, current_user_id)
        if not ranked:
            return None
        # Keep a little variety among genuinely competitive candidates while
        # making relevance, recency and anti-repeat penalties decisive.
        best = ranked[0]["score"]
        pool = [row for row in ranked[:5] if row["score"] >= best - 2.5]
        return self.rng.choice(pool)

    def _cooldown_active(self, repository, action=None, asset_key=None, group=None,
                         seconds=None, media_context=None):
        seconds = self.settings.media_cooldown if seconds is None else seconds
        since = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
        if media_context is not None:
            return any(
                str(row.get("created_at") or "") >= since
                and (action is None or row.get("action") == action)
                and (asset_key is None or row.get("asset_key") == asset_key)
                and (group is None or row.get("cooldown_group") == group)
                for row in media_context.recent_usage
            )
        return repository.has_media_usage_since(
            since, action=action, asset_key=asset_key, cooldown_group=group
        )

    def _media_probability(self, state, intensity):
        chance = self.settings.media_reply_chance + intensity * .06
        if state.conversation_type == "humor":
            chance += self.settings.media_humor_bonus
        elif state.conversation_type == "argument":
            chance += self.settings.media_argument_bonus
        elif state.conversation_type == "serious":
            chance *= .25
        elif state.conversation_type == "work":
            chance *= .75
        if state.activity_level == "burst":
            chance += .03
        return max(0.0, min(.45, chance))

    def _quote(self, rows, target_message_id):
        user_rows = [
            row for row in rows
            if row.get("speaker") != "cyberchair" and row.get("text")
        ]
        target = next(
            (row for row in user_rows if row.get("message_id") == target_message_id),
            None,
        )
        source = target or (user_rows[-1] if user_rows else None)
        if not source:
            return None, None
        text = re.sub(r"\s+", " ", str(source["text"])).strip()
        if not text or len(text) > self.settings.meme_quote_hard_limit:
            return None, None
        maximum = self.settings.meme_quote_max_chars
        if len(text) > maximum:
            sentence = re.split(r"(?<=[.!?])\s+", text)[0]
            if 15 <= len(sentence) <= maximum:
                text = sentence
            else:
                shortened = text[: maximum - 1].rsplit(" ", 1)[0].strip()
                text = (shortened or text[: maximum - 1]).rstrip(".,;:") + "…"
        return text, source.get("message_id")

    def _score_templates(self, state, decision, text, selected_memes, callbacks, recent):
        normalized = normalize_memory(text)
        words = set(normalized.split())
        topic_words = set(normalize_memory(state.dominant_topic or "").split())
        callback_words = set()
        for callback in callbacks or ():
            callback_words.update(normalize_memory(callback).split())
        meme_terms = {
            normalize_memory(getattr(item, "id", item)) for item in selected_memes or ()
        } | {
            normalize_memory(getattr(item, "output", "")) for item in selected_memes or ()
        }
        recent_assets = {row["asset_key"] for row in recent}
        recent_groups = {row["cooldown_group"] for row in recent[:4]}
        scored = []
        for asset in self.catalog.assets:
            if asset.type != "meme_template" or decision.troll_intensity < asset.intensity_min:
                continue
            template_path = self.catalog.resolve(asset)
            if not template_path or not template_path.exists():
                continue
            if asset.id in recent_assets or asset.cooldown_group in recent_groups:
                continue
            contexts = set(asset.contexts)
            tags = {normalize_memory(tag) for tag in asset.tags}
            score = asset.weight
            if state.conversation_type in contexts:
                score += 2.2
            if decision.preferred_style in contexts:
                score += 1.4
            tag_overlap = tags & (words | topic_words)
            meme_overlap = tags & meme_terms or normalize_memory(asset.archetype) in meme_terms
            callback_overlap = bool(callback_words and tags & callback_words and words & callback_words)
            score += min(2.5, len(tag_overlap) * 1.0)
            if meme_overlap:
                score += 3.5
            if callback_overlap:
                score += 2.0
            if state.activity_level == "burst" and "burst" in contexts:
                score += .6
            semantic_match = bool(tag_overlap or meme_overlap or callback_overlap)
            if score >= 2.8 and (
                semantic_match or state.conversation_type in {"humor", "argument"}
            ):
                scored.append((score, asset.weight, asset.id, asset))
        return sorted(scored, reverse=True)

    def _tagged_reaction(self, repository, state, decision, text, selected_memes,
                         recent, media_context=None):
        signals = set(normalize_memory(text).split())
        signals.update(normalize_memory(state.dominant_topic or "").split())
        signals.update(normalize_memory(getattr(item, "output", item)) for item in selected_memes or ())
        signals.add(state.conversation_type)
        signals.add(decision.preferred_style)
        recent_assets = {row["asset_key"] for row in recent}
        scored = []
        for kind, enabled in (("gif", self.settings.gif_enabled), ("sticker", self.settings.sticker_enabled)):
            if not enabled:
                continue
            rows = (
                media_context.tagged_gifs if media_context is not None and kind == "gif"
                else media_context.tagged_stickers if media_context is not None
                else repository.tagged_media(kind)
            )
            for row in rows:
                key = f"{kind}:{row['file_unique_id']}"
                if key in recent_assets:
                    continue
                overlap = len(set(row["tags"]) & signals)
                if overlap:
                    # Equal contextual matches used to prefer "sticker" only
                    # because it sorts after "gif" lexicographically.
                    kind_bias = 1 if kind == "gif" else 0
                    scored.append((overlap, kind_bias, row.get("last_used_at") is None, key, kind, row))
        if not scored:
            return None
        _, _, _, key, kind, row = sorted(scored, reverse=True)[0]
        return MediaDecision(
            action=kind, asset_id=row["file_id"], confidence=.65,
            reason=f"contextual_{kind}",
            asset_key=key, cooldown_group=f"{kind}_reaction", archetype=kind,
        ), key, f"{kind}_reaction"

    def _contextual_pool_reaction(self, repository, state, text, recent, roll):
        """Use learned untagged reactions only in clearly informal contexts."""
        if state.conversation_type not in {"humor", "argument"}:
            return None
        if len(normalize_memory(text).split()) < 2:
            return None
        gif_share = max(0.0, min(1.0, self.settings.media_gif_share))
        first = "gif" if roll < gif_share else "sticker"
        order = (first, "sticker" if first == "gif" else "gif")
        recent_assets = {row["asset_key"] for row in recent}
        for kind in order:
            if kind == "gif" and not self.settings.gif_enabled:
                continue
            if kind == "sticker" and not self.settings.sticker_enabled:
                continue
            row = repository.random_gif() if kind == "gif" else repository.random_sticker()
            if not row:
                continue
            key = f"{kind}:{row['file_unique_id']}"
            if key in recent_assets:
                continue
            return MediaDecision(
                action=kind, asset_id=row["file_id"], confidence=.38,
                reason=f"contextual_{kind}_pool", asset_key=key,
                cooldown_group=f"{kind}_reaction", archetype=kind,
            )
        return None

    def decide(self, chat_id, repository, conversation_decision, chat_state,
               short_term_rows, target_text=None, selected_memes=(), local_callbacks=(),
               troll_mode=True, probability_roll=None, meme_roll=None,
               reaction_roll=None, media_context=None):
        if not troll_mode:
            return self.none("troll_mode_off")
        if conversation_decision.action == "none":
            return self.none("conversation_policy_none")
        if self._cooldown_active(repository, media_context=media_context):
            return self.none("media_cooldown")
        chance = self._media_probability(chat_state, conversation_decision.troll_intensity)
        roll = self.rng.random() if probability_roll is None else probability_roll
        if roll >= chance:
            return self.none("text_preferred")
        quote, source_id = self._quote(
            short_term_rows, conversation_decision.target_message_id
        )
        text = quote or target_text or ""
        recent = self._recent(repository, media_context)
        templates = self._score_templates(
            chat_state, conversation_decision, text, selected_memes,
            local_callbacks, recent,
        )
        meme_probability_roll = self.rng.random() if meme_roll is None else meme_roll
        if quote and templates and meme_probability_roll < self.settings.media_meme_chance:
            asset = templates[0][-1]
            if self._cooldown_active(
                repository, action="meme", seconds=self.settings.meme_render_cooldown,
                media_context=media_context,
            ):
                templates = []
            elif self._cooldown_active(
                repository, group=asset.cooldown_group,
                seconds=self.settings.media_template_cooldown,
                media_context=media_context,
            ):
                templates = []
            else:
                confidence = min(.98, .5 + templates[0][0] / 10)
                return MediaDecision(
                    "meme", asset.id, asset.id, source_id, quote,
                    confidence, f"contextual_template:{asset.cooldown_group}",
                    asset.id, asset.cooldown_group, asset.archetype,
                )
        reaction = self._tagged_reaction(
            repository, chat_state, conversation_decision, text,
            selected_memes, recent, media_context,
        )
        if reaction:
            return reaction[0]
        pool_roll = self.rng.random() if reaction_roll is None else reaction_roll
        reaction = self._contextual_pool_reaction(
            repository, chat_state, text, recent, pool_roll
        )
        if reaction:
            return reaction
        return self.none("no_contextual_asset")

    def commit(self, repository, decision):
        if decision.action == "none":
            return
        asset = self.catalog.get(decision.template_id) if decision.template_id else None
        key = decision.asset_key or (
            f"{decision.action}:{decision.asset_id}"
            if decision.action != "meme" else decision.template_id
        )
        repository.record_media_usage(
            action=decision.action,
            asset_key=key,
            template_id=decision.template_id,
            cooldown_group=(decision.cooldown_group or (
                asset.cooldown_group if asset else f"{decision.action}_reaction"
            )),
            archetype=decision.archetype or (asset.archetype if asset else decision.action),
        )
