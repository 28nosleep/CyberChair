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
    def normalized(value):
        return " ".join(re.findall(r"[\wёЁ'-]+", value.casefold().replace("ё", "е")))

    return SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def _word_tokens(text):
    return re.findall(r"[\wёЁ'-]+", normalize_spaces(text).casefold().replace("ё", "е"))


def _longest_word_overlap(left, right):
    """Length of the longest verbatim contiguous word run."""
    a, b = _word_tokens(left), _word_tokens(right)
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    longest = 0
    for left_word in a:
        current = [0]
        for index, right_word in enumerate(b, 1):
            value = previous[index - 1] + 1 if left_word == right_word else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest


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
    if input_text:
        target_words = _word_tokens(input_text)
        overlap = _longest_word_overlap(clean, input_text)
        if clean.casefold() == normalize_spaces(input_text).casefold():
            return False, "input_copy"
        if overlap >= 6 or (overlap >= 4 and overlap >= len(target_words) * .7):
            return False, "input_overlap"
        for quoted in re.findall(r"[\"«](.*?)[\"»]", clean):
            if _longest_word_overlap(quoted, input_text) >= 3:
                return False, "input_quote"
    if any(similarity(clean, old) > .78 for old in previous_bot_texts):
        return False, "bot_copy"
    if any(similarity(clean, source) > 0.78 for source in source_texts):
        return False, "source_copy"
    if len(set("".join(words))) < 3:
        return False, "nonsense"
    return True, None
