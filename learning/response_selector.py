"""One local arbitration point for optional social reactions."""

from dataclasses import asdict, dataclass
from enum import Enum


class ResponseKind(str, Enum):
    SILENCE = "silence"
    REACTION = "reaction"
    TEXT = "text"
    GIF = "gif"
    STICKER = "sticker"
    MEME = "meme"
    EVIDENCE = "evidence"


REACTIONS = {
    "self_own": "🤡",
    "absurd_statement": "🤯",
    "unexpected_agreement": "🤝",
    "meme_opportunity": "🔥",
    "pile_on": "👀",
}


@dataclass(frozen=True)
class ResponseSelection:
    kind: ResponseKind
    reason: str
    reaction: str | None = None
    evidence_id: int | None = None

    def debug(self):
        value = asdict(self)
        value["kind"] = self.kind.value
        return value


class ResponseSelector:
    """Prefer relevance and silence; this class never calls a provider."""

    def __init__(self, rng):
        self.rng = rng

    def select(
        self, *, moment, relationship, evidence_candidates=(),
        recent_bot_activity=(), recent_media_usage=(), media_enabled=True,
        memory_meme_available=False, required=False,
    ):
        if required:
            return ResponseSelection(ResponseKind.TEXT, "required")
        if moment is None:
            return ResponseSelection(ResponseKind.SILENCE, "no_moment")
        recent_bot = len(tuple(recent_bot_activity)[-6:])
        if recent_bot >= 3:
            return ResponseSelection(ResponseKind.SILENCE, "bot_activity_cap")
        if moment.moment_type == "message_burst":
            return ResponseSelection(ResponseKind.SILENCE, "burst_anti_spam")

        affinity = getattr(relationship, "affinity", .5)
        irritation = getattr(relationship, "irritation", 0.0)
        familiarity = getattr(relationship, "familiarity", 0.0)
        interest = getattr(relationship, "interest", .25)
        probability = max(
            .03,
            min(.32, .06 + (moment.score - .5) * .5 + familiarity * .05 + interest * .03),
        )
        probability += min(.04, max(0.0, irritation - affinity) * .05)
        if self.rng.random() >= probability:
            return ResponseSelection(ResponseKind.SILENCE, "social_sampling")

        evidence = evidence_candidates[0] if evidence_candidates else None
        if evidence and moment.moment_type in {
            "contradiction", "self_own", "callback_opportunity"
        }:
            return ResponseSelection(
                ResponseKind.EVIDENCE, "grounded_callback", evidence_id=evidence.id
            )
        if memory_meme_available and moment.moment_type in {
            "meme_opportunity", "self_own", "callback_opportunity"
        } and moment.score >= .68:
            return ResponseSelection(ResponseKind.MEME, "memory_meme")
        recent_actions = [row.get("action") for row in tuple(recent_media_usage)[-4:]]
        if media_enabled and moment.moment_type in {"meme_opportunity", "pile_on"}:
            if recent_actions.count("gif") <= recent_actions.count("sticker"):
                return ResponseSelection(ResponseKind.GIF, "contextual_media")
            return ResponseSelection(ResponseKind.STICKER, "contextual_media")
        if moment.moment_type in REACTIONS and moment.score >= .65:
            return ResponseSelection(
                ResponseKind.REACTION, "reaction_sufficient",
                reaction=REACTIONS[moment.moment_type],
            )

        return ResponseSelection(ResponseKind.TEXT, "contextual_text")
