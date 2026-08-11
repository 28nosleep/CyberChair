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

    def _recent(self, repository):
        return repository.recent_media_usage(self.settings.media_recent_limit)

    def _cooldown_active(self, repository, action=None, asset_key=None, group=None, seconds=None):
        seconds = self.settings.media_cooldown if seconds is None else seconds
        since = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
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

    def _tagged_reaction(self, repository, state, decision, text, selected_memes, recent):
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
            for row in repository.tagged_media(kind):
                key = f"{kind}:{row['file_unique_id']}"
                if key in recent_assets:
                    continue
                overlap = len(set(row["tags"]) & signals)
                if overlap:
                    scored.append((overlap, row.get("last_used_at") or "", key, kind, row))
        if not scored:
            return None
        _, _, key, kind, row = sorted(scored, reverse=True)[0]
        return MediaDecision(
            action=kind, asset_id=row["file_id"], confidence=.65,
            reason=f"contextual_{kind}",
            asset_key=key, cooldown_group=f"{kind}_reaction", archetype=kind,
        ), key, f"{kind}_reaction"

    def decide(self, chat_id, repository, conversation_decision, chat_state,
               short_term_rows, target_text=None, selected_memes=(), local_callbacks=(),
               troll_mode=True, probability_roll=None, meme_roll=None):
        if not troll_mode:
            return self.none("troll_mode_off")
        if conversation_decision.action == "none":
            return self.none("conversation_policy_none")
        if self._cooldown_active(repository):
            return self.none("media_cooldown")
        chance = self._media_probability(chat_state, conversation_decision.troll_intensity)
        roll = self.rng.random() if probability_roll is None else probability_roll
        if roll >= chance:
            return self.none("text_preferred")
        quote, source_id = self._quote(
            short_term_rows, conversation_decision.target_message_id
        )
        text = quote or target_text or ""
        recent = self._recent(repository)
        templates = self._score_templates(
            chat_state, conversation_decision, text, selected_memes,
            local_callbacks, recent,
        )
        meme_probability_roll = self.rng.random() if meme_roll is None else meme_roll
        if quote and templates and meme_probability_roll < self.settings.media_meme_chance:
            asset = templates[0][-1]
            if self._cooldown_active(
                repository, action="meme", seconds=self.settings.meme_render_cooldown
            ):
                templates = []
            elif self._cooldown_active(
                repository, group=asset.cooldown_group,
                seconds=self.settings.media_template_cooldown,
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
            selected_memes, recent,
        )
        if reaction:
            return reaction[0]
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

    def random_fallback(self, repository):
        candidates = []
        gif = repository.random_gif() if self.settings.gif_enabled else None
        sticker = repository.random_sticker() if self.settings.sticker_enabled else None
        if gif:
            candidates.append(("gif", gif))
        if sticker:
            candidates.append(("sticker", sticker))
        if not candidates:
            return self.none("no_random_assets")
        kind, row = self.rng.choice(candidates)
        return MediaDecision(
            action=kind, asset_id=row["file_id"], confidence=.2,
            reason="legacy_random_fallback",
            asset_key=f"{kind}:{row['file_unique_id']}",
            cooldown_group=f"{kind}_random", archetype=kind,
        )
