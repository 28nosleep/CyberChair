"""Immutable response intent, delivery receipt and post-delivery actions."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from .media_service import MediaDecision


class Producer(str, Enum):
    LLM = "llm"
    LOCAL = "local"
    MEDIA = "media"
    MEME = "meme"
    SYSTEM = "system"
    REACTION = "reaction"
    EVIDENCE = "evidence"


class DeliveryType(str, Enum):
    TEXT = "text"
    PHOTO = "photo"
    ANIMATION = "animation"
    STICKER = "sticker"
    REACTION = "reaction"


@dataclass(frozen=True)
class TextPayload:
    text: str


@dataclass(frozen=True)
class MediaPayload:
    decision: MediaDecision
    prepared_path: Path | None = None


@dataclass(frozen=True)
class ReactionPayload:
    emoji: str


@dataclass(frozen=True)
class GeneratedCommit:
    text: str
    kind: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class TriggerCommit:
    kind: str


@dataclass(frozen=True)
class RoutingCommit:
    route: str
    response_mode: str | None = None


@dataclass(frozen=True)
class MediaUsageCommit:
    decision: MediaDecision


@dataclass(frozen=True)
class PolicyTargetCommit:
    user_id: int | None
    message_id: int


@dataclass(frozen=True)
class PendingFinalize:
    user_id: int


@dataclass(frozen=True)
class PendingCreate:
    user_id: int
    original_message_id: int
    original_question: str
    clarification_question: str
    intent: str
    context: str
    expected_type: str
    mode: str


@dataclass(frozen=True)
class PersonaUsageCommit:
    meme_ids: tuple[str, ...]
    cooldown_groups: tuple[str, ...]


@dataclass(frozen=True)
class SourceUsageCommit:
    texts: tuple[str, ...]


@dataclass(frozen=True)
class ManualMemeCommit:
    decision: MediaDecision


@dataclass(frozen=True)
class EvidenceUsageCommit:
    evidence_id: int


@dataclass(frozen=True)
class StructureUsageCommit:
    construction_signature: str
    opening_id: str | None = None
    fragment_ids: tuple[str, ...] = ()
    closer_id: str | None = None


CommitAction = (
    GeneratedCommit
    | TriggerCommit
    | RoutingCommit
    | MediaUsageCommit
    | PolicyTargetCommit
    | PendingFinalize
    | PendingCreate
    | PersonaUsageCommit
    | SourceUsageCommit
    | ManualMemeCommit
    | EvidenceUsageCommit
    | StructureUsageCommit
)


@dataclass(frozen=True)
class ResponsePlan:
    """One immutable final response decision, not a delivery success claim."""

    event_id: str
    chat_id: int
    producer: Producer
    delivery_type: DeliveryType
    payload: TextPayload | MediaPayload | ReactionPayload
    reply_to_message_id: int | None = None
    required: bool = False
    purpose: str = "reply"
    behavior_mode: str | None = None
    provider_key: str | None = None
    commit_actions: tuple[CommitAction, ...] = ()
    cleanup_paths: tuple[Path, ...] = ()

    def __post_init__(self):
        if self.delivery_type == DeliveryType.TEXT and not isinstance(self.payload, TextPayload):
            raise ValueError("text delivery requires TextPayload")
        if self.delivery_type == DeliveryType.REACTION and not isinstance(
            self.payload, ReactionPayload
        ):
            raise ValueError("reaction delivery requires ReactionPayload")
        if self.delivery_type not in {DeliveryType.TEXT, DeliveryType.REACTION} and not isinstance(
            self.payload, MediaPayload
        ):
            raise ValueError("media delivery requires MediaPayload")


@dataclass(frozen=True)
class DeliveryReceipt:
    event_id: str
    success: bool
    delivery_type: DeliveryType
    telegram_message_id: int | None = None
    error_category: str | None = None
