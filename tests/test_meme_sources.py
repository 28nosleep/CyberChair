import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning.meme_sources import MemeSource, MemeSourceSelector
from learning.service import LearningService
from learning.settings import LearningSettings


class MemeSourceSelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = MemeSourceSelector(random.Random(4))
        self.rows = [
            {"message_id": i, "user_id": i % 3, "text": text, "is_reply": i == 2,
             "reply_count": 4 if i == 2 else 0, "last_used_at": None}
            for i, text in enumerate((
                "длинное обычное сообщение без особой цитатности про релиз и дела",
                "я никогда не буду деплоить в пятницу",
                "это точно займёт пять минут!!!",
                "ну всё, проект готов",
                "сервер опять лежит",
                "я просто посмотрю логи",
                "ещё один маленький фикс",
                "релиз в прод прямо сейчас",
                "всё под контролем",
                "с м стул не считается",
            ), 1)
        ]

    def test_old_quotes_are_ranked_not_random(self):
        ranked = self.selector.rank_old_quotes(self.rows[:-2], "релиз опять упал", "релиз")
        self.assertEqual(ranked[0].message_id, 2)
        self.assertGreater(ranked[0].score, ranked[-1].score)

    def test_old_quote_can_be_selected_and_not_repeated(self):
        source = self.selector.choose(-1, self.rows, fallback=True)
        self.assertEqual(source.kind, "old")
        self.selector.record(-1, source)
        next_source = self.selector.choose(-1, self.rows, fallback=True)
        self.assertNotEqual(next_source.message_id, source.message_id)

class LocalMemeFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = LearningService(LearningSettings(
            data_dir=Path(self.temp.name), openai_chat_id=-1, manual_meme_cooldown=120,
            min_training_messages=1,
        ), rng=random.Random(2))
        repo = self.service.repository(-1)
        for index in range(12):
            repo.add_message(index + 1, index % 3 + 1, None, f"старая цитата {index}: я никогда не ошибаюсь")

    def tearDown(self):
        self.temp.cleanup()

    def test_local_fallback_prefers_old_real_quote(self):
        with patch.object(self.service, "meme_command_on_cooldown", return_value=True):
            decision = self.service.maybe_command_meme(-1)
        self.assertEqual(decision.action, "meme")
        self.assertIn("manual_local_old", decision.reason)

    def test_no_canned_phrase_exists_after_real_quotes(self):
        with patch.object(self.service.meme_sources, "choose", return_value=MemeSource("none", "")):
            source, caption = self.service.media_coordinator._local_command_caption(
                -1, MemeSource("none", ""), [], []
            )
        self.assertEqual(source.kind, "none")
        self.assertIsNone(caption)
