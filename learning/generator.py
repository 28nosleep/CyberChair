import random
import re

from .filters import validate_generated
from .preprocessing import significant_words, strip_mentions


class LocalGenerator:
    def __init__(self, settings, rng=None):
        self.settings = settings
        self.rng = rng or random

    def _mode(self):
        names = list(self.settings.mode_weights)
        return self.rng.choices(names, weights=[self.settings.mode_weights[name] for name in names], k=1)[0]

    def _combine(self, texts):
        if len(texts) < 2:
            return None
        first, second = self.rng.sample(texts, 2)
        a, b = first.split(), second.split()
        if len(a) < 3 or len(b) < 3:
            return None
        return " ".join(a[:max(2, len(a) // 2)] + b[min(len(b) - 2, len(b) // 2):])

    def _mutate(self, texts):
        candidates = [text for text in texts if 4 <= len(text.split()) <= self.settings.max_generated_words]
        if not candidates:
            return None
        words = self.rng.choice(candidates).split()
        replacements = ["Киберстул", "протокол", "машины", "сервер", "сбой", "система"]
        words[self.rng.randrange(len(words))] = self.rng.choice(replacements)
        return " ".join(words)

    def create(self, model, messages, input_text=None, previous_bot_texts=()):
        source_texts = [row["text"] if isinstance(row, dict) else row for row in messages]
        texts = (
            [
                text
                for row, text in zip(messages, source_texts)
                for _ in range(max(1, int(row.get("generation_weight", 1))))
            ]
            if messages and isinstance(messages[0], dict)
            else source_texts
        )
        context = significant_words(input_text or "")
        for _ in range(10):
            mode = self._mode()
            if mode in {"markov", "contextual"}:
                result = model.generate(
                    self.settings.min_generated_words,
                    self.settings.max_generated_words,
                    context if mode == "contextual" else None,
                    self.rng,
                )
            elif mode == "combine":
                result = self._combine(texts)
            elif mode == "quote_mutation":
                result = self._mutate(texts)
            else:
                candidates = [text for text in texts if self.settings.min_generated_words <= len(text.split()) <= 12]
                result = self.rng.choice(candidates) if candidates else None
            if not result:
                continue
            result = strip_mentions(result) if not self.settings.allow_user_mentions else result
            result = re.sub(r"\s+", " ", result).strip()
            ok, _ = validate_generated(
                result, source_texts, input_text, previous_bot_texts,
                self.settings.min_generated_words,
                self.settings.max_generated_words,
            )
            if ok:
                return result, mode
        return None, None
