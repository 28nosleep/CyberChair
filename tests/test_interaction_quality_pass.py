import random
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from PIL import Image

from learning import (
    ChatState, ConversationDecision, LearningService, LearningSettings,
    MediaCatalog, MediaService,
)
from learning.date_time_utility import DateTimeUtility
from learning.meme_renderer import MemeRenderer
from learning.repository import ChatRepository


def incoming(text, message_id=1, user_id=7):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=message_id, text=text,
        date=1_700_000_000 + message_id,
        from_user=SimpleNamespace(id=user_id, username="саня", is_bot=False),
        reply_to_message=None,
    )


def settings(path, **overrides):
    values = dict(
        data_dir=Path(path), openai_chat_id=-999,
        addressed_cooldown=0, generated_cooldown=0,
        max_generated_per_hour=1000, media_cooldown=0,
        meme_render_cooldown=0, media_template_cooldown=0,
    )
    values.update(overrides)
    return LearningSettings(**values)


class DateTimeUtilityTests(unittest.TestCase):
    def setUp(self):
        self.fixed = datetime(2026, 8, 20, 21, 53, tzinfo=timezone.utc)
        self.utility = DateTimeUtility(
            "Europe/Moscow", clock=lambda: self.fixed
        )

    def test_date_weekday_and_time_are_programmatic(self):
        self.assertEqual(
            self.utility.answer("какое сегодня число?"),
            "сегодня 21 августа 2026, пятница",
        )
        self.assertEqual(
            self.utility.answer("какой сегодня день?"), "сегодня пятница"
        )
        self.assertEqual(
            self.utility.answer("сколько сейчас времени?"), "сейчас 00:53"
        )
        self.assertEqual(self.utility.answer("который час?"), "сейчас 00:53")

    def test_midnight_relative_dates_use_application_timezone(self):
        before = DateTimeUtility(
            "Europe/Moscow",
            clock=lambda: datetime(2026, 8, 20, 20, 59, tzinfo=timezone.utc),
        )
        after = DateTimeUtility(
            "Europe/Moscow",
            clock=lambda: datetime(2026, 8, 20, 21, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(
            before.answer("что завтра за день?"),
            "завтра 21 августа 2026, пятница",
        )
        self.assertEqual(
            after.answer("какое сегодня число?"),
            "сегодня 21 августа 2026, пятница",
        )

    def test_semantic_and_schedule_questions_are_not_intercepted(self):
        for text in (
            "почему время быстро идёт?",
            "как получить время в Python?",
            "во сколько заканчивается рабочий день?",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self.utility.answer(text))

    def test_direct_date_answer_never_calls_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(
                settings(directory, openai_chat_id=-1),
                datetime_clock=lambda: self.fixed,
            )
            service.set_media_enabled(-1, False)
            with patch.object(service, "generate_llm") as provider:
                answer = service.maybe_direct_reply(
                    incoming("стул, какое сегодня число?"),
                    explicit_address=True,
                )
            self.assertEqual(answer, "сегодня 21 августа 2026, пятница")
            provider.assert_not_called()


class FreeResponseCalibrationTests(unittest.TestCase):
    def test_ten_forced_free_questions_remain_contextual_and_call_no_llm(self):
        questions = (
            "почему docker опять упал?", "как выбрать монитор?",
            "зачем саня удалил тесты?", "что делать с релизом?",
            "почему айфон перегрелся?", "как чинить роутер?",
            "куда делся бюджет?", "почему кофе закончился?",
            "как вернуть backup?", "что случилось с nginx?",
        )
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(settings(directory), rng=random.Random(7))
            service.set_media_enabled(-1, False)
            with patch.object(service, "generate_llm") as provider:
                answers = [
                    service.maybe_direct_reply(
                        incoming(f"стул {question}", index),
                        explicit_address=True,
                    )
                    for index, question in enumerate(questions, 1)
                ]
            provider.assert_not_called()
            self.assertTrue(all(answers))
            for question, answer in zip(questions, answers):
                topic = next(
                    word for word in question.rstrip("?").split()
                    if word in {"docker", "монитор", "тесты", "релизом", "айфон", "роутер", "бюджет", "кофе", "backup", "nginx"}
                )
                self.assertIn(topic, answer.casefold())
            self.assertGreaterEqual(len(set(answers)), 9)

    def test_callback_is_used_for_same_topic(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(settings(directory), rng=random.Random(2))
            service.set_media_enabled(-1, False)
            service.repository(-1).remember_stable(["айфон для нормисов"])
            answer = service.maybe_direct_reply(
                incoming("стул думаю купить айфон"), explicit_address=True
            )
            self.assertIn("айфон для нормисов", answer)

    def test_one_hundred_varied_free_responses_have_no_visible_tick_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(settings(directory), rng=random.Random(11))
            repository = service.repository(-1)
            outputs = []
            for index in range(100):
                result, _ = service.local_responder.respond(
                    -1, f"я снова сломал проект номер {index}", "social", repository,
                    recent_generated=outputs[-40:], user_id=7, username="саня",
                )
                outputs.append(result)
            self.assertGreaterEqual(len(set(outputs)), 95)
            joined = " ".join(outputs).casefold()
            self.assertLessEqual(joined.count("классика"), 1)
            self.assertLessEqual(joined.count("chairos"), 3)


class MediaAndMemeCalibrationTests(unittest.TestCase):
    @staticmethod
    def state(kind, activity="normal"):
        return ChatState(
            activity, 8, kind, "релиз", .7,
            .8 if kind == "humor" else .1, .1,
            .8 if kind == "serious" else .1,
            .8 if kind == "work" else .1, .4, 3, 10, 7, .8,
        )

    @staticmethod
    def decision():
        return ConversationDecision(
            "reply", 1, .8, 30, "absurd_short", 10, 7, "fixture", 0, 1
        )

    def test_humor_media_rate_exceeds_serious_and_work(self):
        with tempfile.TemporaryDirectory() as directory:
            media = MediaService(settings(directory), MediaCatalog(), random.Random(1))
            humor = media._media_probability(self.state("humor", "burst"), .8)
            serious = media._media_probability(self.state("serious"), .8)
            work = media._media_probability(self.state("work"), .8)
            self.assertGreater(humor, work)
            self.assertGreater(work, serious)

    def test_sglypa_spam_cannot_create_consecutive_replies(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(settings(
                directory, openai_chat_id=-1, sglypa_reply_chance=1.0,
                sglypa_reply_cooldown=1800,
            ))
            for index in range(3):
                service.repository(-1).add_message(
                    index + 1, index + 1, None, f"активность людей {index}"
                )
            event = incoming("сглыпа вещает")
            with patch.object(
                service, "generate_llm", return_value="короткая реакция по смыслу"
            ) as provider:
                first = service.maybe_sglypa_reply(event)
                second = service.maybe_sglypa_reply(event)
            self.assertTrue(first)
            self.assertIsNone(second)
            provider.assert_called_once()

    def test_automatic_meme_prefers_recent_unused_chat_image(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(directory, automatic_chat_image_chance=1.0)
            repository = ChatRepository(Path(directory), -1, 50)
            repository.add_chat_image(
                20, 7, "chat-file", "chat-unique", "photo",
                caption="саня опять уронил прод", width=1200, height=900,
            )
            media = MediaService(cfg, MediaCatalog(), random.Random(0))
            result = media.decide(
                -1, repository, self.decision(), self.state("humor"),
                [{"message_id": 10, "user_id": 7, "speaker": "human", "text": "саня опять уронил прод"}],
                "саня опять уронил прод", probability_roll=0, meme_roll=0,
            )
            self.assertEqual(result.action, "meme")
            self.assertEqual(result.background_file_id, "chat-file")
            self.assertEqual(result.reason, "contextual_chat_image")

    def test_unrelated_ai_caption_is_rejected(self):
        from learning.media_coordinator import MediaCoordinator

        self.assertTrue(MediaCoordinator._caption_grounded(
            "САНЯ СНОВА УРОНИЛ ПРОД", ("саня опять уронил прод",)
        ))
        self.assertFalse(MediaCoordinator._caption_grounded(
            "DEVOPS ИЗ ПЯТЁРОЧКИ ВЫШЕЛ НА СМЕНУ", ("саня опять уронил прод",)
        ))
        self.assertFalse(MediaCoordinator._caption_grounded(
            "DEVOPS САНЯ УРОНИЛ ПРОД", ("саня опять уронил прод",)
        ))

    def test_render_profiles_are_large_uppercase_cyrillic_and_not_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (900, 600), "#667788").save(source)
            renderer = MemeRenderer(MediaCatalog(), root / "out")
            for profile, caption in (
                ("top_caption", "ты опять уронил прод"),
                ("bottom_caption", "релиз жив"),
                ("center", "пов"),
                ("top_bottom", "саня опять | уронил прод"),
                ("top_center", "очень смешно"),
                ("center_bottom", "ну приехали"),
            ):
                with self.subTest(profile=profile):
                    result = renderer.render_image(source, caption, profile)
                    self.assertIsNotNone(result)
                    left, top, right, bottom = result.text_box
                    self.assertGreaterEqual(left, 0)
                    self.assertGreaterEqual(top, 0)
                    self.assertLessEqual(right, 900)
                    self.assertLessEqual(bottom, 600)
                    self.assertGreaterEqual(result.font_size, 38)
                    self.assertTrue(result.font_name)
                    renderer.cleanup(result)

    def test_portrait_and_small_images_do_not_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renderer = MemeRenderer(MediaCatalog(), root / "out")
            for size in ((600, 1000), (260, 180)):
                source = root / f"source-{size[0]}-{size[1]}.png"
                Image.new("RGB", size, "#445566").save(source)
                result = renderer.render_image(
                    source, "ТЫ ОПЯТЬ УРОНИЛ ПРОД", "bottom_caption"
                )
                self.assertIsNotNone(result)
                left, top, right, bottom = result.text_box
                self.assertGreaterEqual(left, 0)
                self.assertGreaterEqual(top, 0)
                self.assertLessEqual(right, size[0])
                self.assertLessEqual(bottom, size[1])
                renderer.cleanup(result)


if __name__ == "__main__":
    unittest.main()
