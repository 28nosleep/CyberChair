"""Local recency penalties for conspicuous CyberChair verbal tics."""

from collections import Counter
from dataclasses import dataclass
import re

from .preprocessing import STOP_WORDS, normalize_spaces


TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
CONSTRUCTIONS = (
    ("это не x это y", re.compile(r"\bэто\s+не\b.{0,80}\bэто\b", re.I)),
    ("протокол x активирован", re.compile(r"\bпротокол\b.{0,60}\bактивирован", re.I)),
    ("chairOS фиксирует", re.compile(r"\bchairos\s+фиксирует\b", re.I)),
    ("ебать ты", re.compile(r"\bебать\s+ты\b", re.I)),
)


@dataclass(frozen=True)
class LexicalPenalty:
    phrase: str
    count: int
    penalty: float
    opening: bool = False


class LexicalDiversityTracker:
    """Scores noticeable recent words/phrases without permanently banning them."""

    def __init__(self, window=40, strong_count=3):
        self.window = max(10, int(window))
        self.strong_count = max(2, int(strong_count))

    @staticmethod
    def _tokens(text):
        return [token.casefold().replace("ё", "е") for token in TOKEN_RE.findall(text or "")]

    @staticmethod
    def _noticeable_word(word):
        return len(word) >= 6 and word not in STOP_WORDS and not word.isdigit()

    def features(self, text):
        clean = normalize_spaces(text).casefold().replace("ё", "е")
        words = self._tokens(clean)
        phrases = set()
        for word in words:
            if self._noticeable_word(word):
                phrases.add(word)
        for size in (2, 3):
            for index in range(max(0, len(words) - size + 1)):
                gram = words[index:index + size]
                if sum(self._noticeable_word(word) for word in gram) >= 1:
                    phrases.add(" ".join(gram))
        openings = {
            " ".join(words[:size]) for size in (1, 2, 3)
            if len(words) >= size and any(self._noticeable_word(word) for word in words[:size])
        }
        phrases.update(openings)
        for label, pattern in CONSTRUCTIONS:
            if pattern.search(clean):
                phrases.add(label)
        return phrases, openings

    def penalties(self, recent_texts):
        texts = [normalize_spaces(value) for value in recent_texts if normalize_spaces(value)][-self.window:]
        counts, opening_counts = Counter(), Counter()
        for age, text in enumerate(reversed(texts)):
            features, openings = self.features(text)
            # Recency fades smoothly; phrases naturally become available again.
            weight = max(.15, 1.0 - age / self.window)
            for phrase in features:
                counts[phrase] += weight
            for phrase in openings:
                opening_counts[phrase] += weight
        result = []
        for phrase, count in counts.items():
            if count < 1.55:
                continue
            penalty = (count - 1.0) * (2.0 if count >= self.strong_count - .25 else .8)
            if phrase in opening_counts:
                penalty *= 1.35
            result.append(LexicalPenalty(phrase, round(count), penalty, phrase in opening_counts))
        return sorted(result, key=lambda item: (item.penalty, len(item.phrase)), reverse=True)

    def score(self, candidate, recent_texts):
        features, openings = self.features(candidate)
        score = 0.0
        matched = []
        for item in self.penalties(recent_texts):
            if item.phrase in features:
                value = item.penalty * (1.5 if item.phrase in openings else 1.0)
                score += value
                matched.append(item.phrase)
        return score, tuple(matched)

    def prompt_penalties(self, recent_texts, limit=8):
        return tuple(item.phrase for item in self.penalties(recent_texts)[:limit])

    def top_phrases(self, texts, limit=10):
        counts = Counter()
        for text in texts:
            features, _ = self.features(text)
            counts.update(features)
        return counts.most_common(max(0, int(limit)))
