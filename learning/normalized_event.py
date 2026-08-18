"""Immutable Telegram-to-domain event normalization boundary."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .event_context import callback_event_id, telegram_event_id
from .preprocessing import normalize_spaces


class EventKind(str, Enum):
    TEXT = "text"
    PHOTO = "photo"
    IMAGE_DOCUMENT = "image_document"
    ANIMATION = "animation"
    STICKER = "sticker"
    OTHER = "other"


@dataclass(frozen=True)
class NormalizedMedia:
    kind: str
    file_id: str
    file_unique_id: str
    mime_type: str | None = None
    file_size: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class NormalizedEvent:
    """Immutable runtime facts extracted once from a Telegram message."""

    event_id: str
    event_kind: EventKind
    chat_id: int
    message_id: int
    user_id: int | None
    username: str | None
    display_name: str
    first_name: str
    user_is_bot: bool
    text: str
    caption: str
    effective_text: str
    normalized_text: str
    effective_text_source: str
    content_type: str
    timestamp: datetime | None
    reply_to_message_id: int | None = None
    reply_to_user_id: int | None = None
    reply_to_user_is_bot: bool = False
    reply_effective_text: str = ""
    reply_content_type: str | None = None
    reply_timestamp: datetime | None = None
    media: NormalizedMedia | None = None
    reply_media: NormalizedMedia | None = None
    is_command: bool = False
    command_name: str | None = None

    @property
    def has_photo(self):
        return self.event_kind == EventKind.PHOTO

    @property
    def has_image_document(self):
        return self.event_kind == EventKind.IMAGE_DOCUMENT

    @property
    def has_animation(self):
        return self.event_kind == EventKind.ANIMATION

    @property
    def has_sticker(self):
        return self.event_kind == EventKind.STICKER

    @property
    def reply_has_photo(self):
        return bool(self.reply_media and self.reply_media.kind == "photo")

    @property
    def reply_has_image_document(self):
        return bool(self.reply_media and self.reply_media.kind == "document")

    @property
    def telegram_file_id(self):
        return self.media.file_id if self.media else None

    @property
    def telegram_file_unique_id(self):
        return self.media.file_unique_id if self.media else None

    @property
    def mime_type(self):
        return self.media.mime_type if self.media else None

    def replies_to_user(self, bot_id):
        return bool(bot_id is not None and self.reply_to_user_id == bot_id)


@dataclass(frozen=True)
class NormalizedCallbackEvent:
    event_id: str
    event_kind: str
    callback_id: str
    data: str
    chat_id: int
    message_id: int
    user_id: int | None
    username: str | None


def _integer(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _timestamp(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, timezone.utc)
    return None


def _content_type(message):
    declared = str(getattr(message, "content_type", "") or "").casefold()
    if declared:
        return declared
    if getattr(message, "photo", None):
        return "photo"
    if getattr(message, "animation", None):
        return "animation"
    if getattr(message, "sticker", None):
        return "sticker"
    if getattr(message, "document", None):
        return "document"
    return "text" if getattr(message, "text", None) is not None else "other"


def _largest_photo(photos):
    return max(
        photos,
        key=lambda item: (
            int(getattr(item, "width", 0) or 0)
            * int(getattr(item, "height", 0) or 0),
            int(getattr(item, "file_size", 0) or 0),
        ),
    )


def _media(message, content_type):
    photos = tuple(getattr(message, "photo", None) or ())
    if photos:
        value = _largest_photo(photos)
        kind, mime_type = "photo", "image/jpeg"
    elif content_type == "animation" and getattr(message, "animation", None):
        value = message.animation
        kind = "animation"
        mime_type = str(getattr(value, "mime_type", "") or "image/gif").casefold()
    elif content_type == "sticker" and getattr(message, "sticker", None):
        value = message.sticker
        kind = "sticker"
        mime_type = str(getattr(value, "mime_type", "") or "").casefold() or None
    elif getattr(message, "document", None):
        value = message.document
        kind = "document"
        mime_type = str(getattr(value, "mime_type", "") or "").casefold() or None
    else:
        return None
    file_id = str(getattr(value, "file_id", "") or "")
    if not file_id:
        return None
    return NormalizedMedia(
        kind=kind,
        file_id=file_id,
        file_unique_id=str(getattr(value, "file_unique_id", "") or file_id),
        mime_type=mime_type,
        file_size=_integer(getattr(value, "file_size", None)),
        width=_integer(getattr(value, "width", None)),
        height=_integer(getattr(value, "height", None)),
    )


def _event_kind(content_type, media):
    if content_type == "text":
        return EventKind.TEXT
    if media and media.kind == "photo":
        return EventKind.PHOTO
    if media and media.kind == "document" and str(media.mime_type or "").startswith("image/"):
        return EventKind.IMAGE_DOCUMENT
    if media and media.kind == "animation":
        return EventKind.ANIMATION
    if media and media.kind == "sticker":
        return EventKind.STICKER
    return EventKind.OTHER


def _effective_text(message, content_type):
    if content_type == "text":
        return str(getattr(message, "text", "") or ""), "text"
    if content_type in {"photo", "document"}:
        caption = str(getattr(message, "caption", "") or "")
        return caption, "caption" if caption else "none"
    return "", "none"


def normalize_telegram_event(message):
    content_type = _content_type(message)
    media = _media(message, content_type)
    kind = _event_kind(content_type, media)
    effective_text, source = _effective_text(message, content_type)
    text = str(getattr(message, "text", "") or "")
    caption = str(getattr(message, "caption", "") or "")
    user = getattr(message, "from_user", None)
    first_name = str(getattr(user, "first_name", "") or "").strip()
    last_name = str(getattr(user, "last_name", "") or "").strip()
    display_name = normalize_spaces(f"{first_name} {last_name}")
    reply = getattr(message, "reply_to_message", None)
    reply_user = getattr(reply, "from_user", None)
    reply_content_type = _content_type(reply) if reply is not None else None
    reply_effective_text = (
        _effective_text(reply, reply_content_type)[0] if reply is not None else ""
    )
    reply_media = _media(reply, reply_content_type) if reply is not None else None
    normalized = normalize_spaces(effective_text)
    command_name = None
    if normalized.startswith("/"):
        command_name = normalized.split(maxsplit=1)[0][1:].split("@", 1)[0].casefold()
    return NormalizedEvent(
        event_id=telegram_event_id(message),
        event_kind=kind,
        chat_id=int(getattr(getattr(message, "chat", None), "id", 0)),
        message_id=int(getattr(message, "message_id", getattr(message, "id", 0)) or 0),
        user_id=_integer(getattr(user, "id", None)),
        username=getattr(user, "username", None) or None,
        display_name=display_name,
        first_name=first_name,
        user_is_bot=bool(getattr(user, "is_bot", False)),
        text=text,
        caption=caption,
        effective_text=effective_text,
        normalized_text=normalized,
        effective_text_source=source,
        content_type=content_type,
        timestamp=_timestamp(getattr(message, "date", None)),
        reply_to_message_id=_integer(
            getattr(reply, "message_id", getattr(reply, "id", None))
            if reply is not None else None
        ),
        reply_to_user_id=_integer(getattr(reply_user, "id", None)),
        reply_to_user_is_bot=bool(getattr(reply_user, "is_bot", False)),
        reply_effective_text=reply_effective_text,
        reply_content_type=reply_content_type,
        reply_timestamp=(
            _timestamp(getattr(reply, "date", None)) if reply is not None else None
        ),
        media=media,
        reply_media=reply_media,
        is_command=command_name is not None,
        command_name=command_name,
    )


def normalize_callback_event(call):
    message = getattr(call, "message", None)
    chat_id = int(getattr(getattr(message, "chat", None), "id", 0))
    message_id = int(getattr(message, "message_id", getattr(message, "id", 0)) or 0)
    callback_id = str(getattr(call, "id", "") or "")
    user = getattr(call, "from_user", None)
    return NormalizedCallbackEvent(
        event_id=callback_event_id(chat_id, message_id, callback_id),
        event_kind="callback",
        callback_id=callback_id,
        data=str(getattr(call, "data", "") or ""),
        chat_id=chat_id,
        message_id=message_id,
        user_id=_integer(getattr(user, "id", None)),
        username=getattr(user, "username", None) or None,
    )
