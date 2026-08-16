import random
from collections import defaultdict

from .preprocessing import tokenize


class MarkovModel:
    def __init__(self):
        self.transitions = defaultdict(list)
        self.starts = []

    def train(self, texts):
        self.transitions.clear()
        self.starts.clear()
        for item in texts:
            text, weight = item if isinstance(item, tuple) else (item, 1)
            for _ in range(max(1, int(weight))):
                self.add(text)
        return self

    def add(self, text):
        words = tokenize(text)
        if len(words) < 3:
            return
        self.starts.append((words[0], words[1]))
        for index in range(len(words) - 2):
            self.transitions[(words[index].lower(), words[index + 1].lower())].append(words[index + 2])

    def generate(self, min_words=3, max_words=25, context_words=None, rng=None):
        rng = rng or random
        if not self.starts:
            return None
        candidates = self.starts
        wanted = set(context_words or ())
        if wanted:
            contextual = [pair for pair in self.transitions if wanted.intersection(pair)]
            if contextual:
                start_key = rng.choice(contextual)
                start = next((pair for pair in self.starts if tuple(w.lower() for w in pair) == start_key), start_key)
            else:
                start = rng.choice(candidates)
        else:
            start = rng.choice(candidates)
        result = list(start)
        while len(result) < max_words:
            options = self.transitions.get((result[-2].lower(), result[-1].lower()))
            if not options:
                break
            word = rng.choice(options)
            result.append(word)
            if len(result) >= min_words and word in {".", "!", "?"}:
                break
        text = " ".join(result)
        return text.replace(" .", ".").replace(" !", "!").replace(" ?", "?")
