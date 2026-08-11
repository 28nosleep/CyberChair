import re
from difflib import SequenceMatcher

from .preprocessing import (
    EMAIL_RE,
    FOREIGN_BOT_COMMAND_RE,
    PHONE_RE,
    SECRET_RE,
    TOKEN_RE,
    URL_RE,
    VOICE_STORY_COMMAND_RE,
    normalize_spaces,
)


def similarity(left, right):
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def validate_generated(text, source_texts=(), input_text=None, previous_bot_texts=(), min_words=3, max_words=25):
    clean = normalize_spaces(text)
    if not clean:
        return False, "empty"
    if clean.startswith("/"):
        return False, "command"
    if URL_RE.search(clean) or EMAIL_RE.search(clean) or PHONE_RE.search(clean):
        return False, "personal_or_link"
    if TOKEN_RE.search(clean) or SECRET_RE.search(clean):
        return False, "secret"
    if FOREIGN_BOT_COMMAND_RE.search(clean):
        return False, "foreign_bot_command"
    if VOICE_STORY_COMMAND_RE.search(clean):
        return False, "bot_command"
    words = re.findall(r"[\wёЁ'-]+", clean)
    if len(words) < min_words or len(words) > max_words:
        return False, "word_count"
    lowered = [word.casefold() for word in words]
    if max(lowered.count(word) for word in set(lowered)) > max(3, len(words) // 2):
        return False, "repetition"
    if input_text and clean.casefold() == normalize_spaces(input_text).casefold():
        return False, "input_copy"
    if any(clean.casefold() == normalize_spaces(old).casefold() for old in previous_bot_texts):
        return False, "bot_copy"
    if any(similarity(clean, source) > 0.85 for source in source_texts):
        return False, "source_copy"
    if len(set("".join(words))) < 3:
        return False, "nonsense"
    return True, None
