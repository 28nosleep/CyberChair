import re

URL_RE = re.compile(r"(?:https?://|www\.)\S+|(?:t\.me/\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
TOKEN_RE = re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b")
SECRET_RE = re.compile(r"\b(?:password|passwd|пароль|api[_ -]?key|secret|token)\s*[:=]\s*\S+", re.I)
ONLY_EMOJI_RE = re.compile(r"^[\W_]+$", re.UNICODE)
ONLY_LINK_RE = re.compile(r"^\s*(?:https?://|www\.|t\.me/)\S+\s*$", re.I)
WORD_RE = re.compile(r"[\wёЁ'-]+|[.!?]+", re.UNICODE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{5,32}\b")
FOREIGN_BOT_COMMAND_RE = re.compile(
    r"(?<![A-Za-z0-9_])s\s+g\s+[md](?![A-Za-z0-9_])",
    re.I,
)
VOICE_STORY_COMMAND_RE = re.compile(r"\bстул\s+голос\b", re.I)

STOP_WORDS = {
    "а", "без", "бы", "был", "была", "были", "в", "во", "вот", "вы", "да",
    "для", "до", "его", "ее", "если", "есть", "же", "за", "и", "из", "или",
    "к", "как", "ко", "ли", "мы", "на", "не", "но", "ну", "о", "он", "она",
    "они", "от", "по", "при", "с", "со", "так", "то", "у", "что", "это", "я",
}


def normalize_spaces(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def rejection_reason(text, max_length=2000):
    clean = normalize_spaces(text)
    if not clean:
        return "empty"
    if clean.startswith("/"):
        return "command"
    if len(clean) < 3:
        return "too_short"
    if len(clean) > max_length:
        return "too_long"
    if ONLY_EMOJI_RE.fullmatch(clean):
        return "emoji_only"
    if ONLY_LINK_RE.fullmatch(clean) or URL_RE.search(clean):
        return "link"
    if EMAIL_RE.search(clean):
        return "email"
    if PHONE_RE.search(clean):
        return "phone"
    if TOKEN_RE.search(clean) or SECRET_RE.search(clean):
        return "secret"
    if FOREIGN_BOT_COMMAND_RE.search(clean):
        return "foreign_bot_command"
    if VOICE_STORY_COMMAND_RE.search(clean):
        return "bot_command"
    return None


def tokenize(text):
    return WORD_RE.findall(normalize_spaces(text))


def significant_words(text):
    return [
        word.lower() for word in tokenize(text)
        if word[0].isalnum() and len(word) > 2 and word.lower() not in STOP_WORDS
    ]


def lexical_stem(word):
    """Conservative prefix hint for common Russian inflection."""
    value = str(word or "").casefold().replace("ё", "е")
    if len(value) < 3:
        return value
    return value[:3] if len(value) <= 4 else value[:5]


def strip_mentions(text):
    return normalize_spaces(MENTION_RE.sub("кто-то", text))
