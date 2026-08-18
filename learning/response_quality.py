"""Cheap final response checks; deliberately contains no generation calls."""

from dataclasses import dataclass
import re

from .preprocessing import normalize_spaces


TECHNICAL_MARKER_RE = re.compile(
    r"\b(?:ACCESS_DENIED|CHAIR_PROCESS|SYSTEM_PROMPT|finish_reason|max_output_tokens)\b|"
    r"\[/?(?:assistant|system|tool)\]",
    re.I,
)


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: str | None = None
    lexical_phrases: tuple[str, ...] = ()
    incomplete_reason: str | None = None


class ResponseQualityGuard:
    CANNED_MEME_CAPTIONS = (
        "сойджак-комиссия", "chairOS фиксирует промышленный скилл ишью",
        "протокол брейнрота активирован", "лил бро выбрал сайдквест",
        "сканирование завершено: проект кукд",
    )

    def __init__(self, lexical_tracker):
        self.lexical_tracker = lexical_tracker

    def check(self, text, recent_texts=(), *, incomplete_reason=None,
              local=False, image_meme=False):
        clean = normalize_spaces(text)
        if not clean:
            return QualityResult(False, "empty", incomplete_reason=incomplete_reason)
        if TECHNICAL_MARKER_RE.search(clean):
            return QualityResult(False, "technical_marker", incomplete_reason=incomplete_reason)
        if image_meme and any(value.casefold() in clean.casefold() for value in self.CANNED_MEME_CAPTIONS):
            return QualityResult(False, "canned_meme_caption", incomplete_reason=incomplete_reason)
        score, phrases = self.lexical_tracker.score(clean, recent_texts)
        # AI is never regenerated for style. Local callers may choose another option.
        if local and phrases and score >= 4.0:
            return QualityResult(False, "lexical_tick", phrases, incomplete_reason)
        return QualityResult(True, lexical_phrases=phrases, incomplete_reason=incomplete_reason)
