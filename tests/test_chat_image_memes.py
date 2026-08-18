import random
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from learning import LearningService, LearningSettings, MediaCatalog, MediaDecision, MemeRenderer


class ZeroRandom(random.Random):
    def random(self):
        return 0.0

    def choice(self, values):
        return values[0]


def user(user_id=7, is_bot=False):
    return SimpleNamespace(id=user_id, username=f"user{user_id}", is_bot=is_bot)


def photo(file_id, unique_id, width, height, size=1000):
    return SimpleNamespace(
        file_id=file_id, file_unique_id=unique_id, width=width, height=height,
        file_size=size,
    )


def image_message(message_id=10, photos=None, document=None, is_bot=False,
                  caption="", created=1_700_000_000):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=message_id, text=None,
        caption=caption, photo=photos or [], document=document, date=created,
        from_user=user(99 if is_bot else 7, is_bot), reply_to_message=None,
    )


def command(reply=None, text="с м стул", message_id=20):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=message_id, text=text,
        date=1_700_000_100, from_user=user(), reply_to_message=reply,
    )


class ChatImageMemeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = LearningSettings(
            data_dir=Path(self.temp.name), openai_chat_id=-1,
            min_training_messages=1, manual_meme_cooldown=120,
            chat_image_background_chance=.35,
        )
        self.service = LearningService(self.settings, rng=ZeroRandom())
        self.service.repository(-1).add_message(
            1, 7, None, "реальная реплика из чата для подписи"
        )
        self.cooldown = patch.object(
            self.service, "meme_command_on_cooldown", return_value=True
        )
        self.cooldown.start()

    def tearDown(self):
        self.cooldown.stop()
        self.temp.cleanup()

    def test_explicit_photo_uses_exact_largest_variant(self):
        reply = image_message(10, [
            photo("thumb", "u-thumb", 90, 90),
            photo("original", "u-original", 1600, 1200),
            photo("medium", "u-medium", 800, 600),
        ], caption="сломанная машина")
        decision = self.service.maybe_command_meme(command(reply), "про серёгу")
        self.assertTrue(decision.background_explicit)
        self.assertEqual(decision.background_file_id, "original")
        self.assertEqual(decision.background_file_unique_id, "u-original")
        self.assertIsNone(decision.template_id)

    def test_explicit_image_document_works(self):
        document = SimpleNamespace(
            file_id="png-file", file_unique_id="png-unique",
            mime_type="image/png", file_size=2000,
        )
        decision = self.service.maybe_command_meme(
            command(image_message(document=document))
        )
        self.assertEqual(decision.background_file_id, "png-file")
        self.assertEqual(decision.background_media_type, "document")

    def test_non_image_document_safely_uses_curated_template(self):
        document = SimpleNamespace(
            file_id="pdf", file_unique_id="pdf-u", mime_type="application/pdf",
            file_size=1000,
        )
        decision = self.service.maybe_command_meme(
            command(image_message(document=document))
        )
        self.assertIsNone(decision.background_file_id)
        self.assertIsNotNone(decision.template_id)

    def test_bot_image_is_never_a_background(self):
        reply = image_message(10, [photo("bot-file", "bot-u", 800, 600)], is_bot=True)
        decision = self.service.maybe_command_meme(command(reply))
        self.assertFalse(decision.background_explicit)
        self.assertIsNone(decision.background_file_id)
        self.assertIsNotNone(decision.template_id)
        self.assertEqual(self.service.repository(-1).chat_image_count(), 1)
        self.assertEqual(self.service.repository(-1).chat_image_candidates(), [])

    def test_explicit_reply_precedes_template_selector_and_anti_repeat(self):
        reply = image_message(10, [photo("chosen", "chosen-u", 800, 600)])
        repository = self.service.repository(-1)
        repository.add_chat_image(**self.service.telegram_image_metadata(reply))
        repository.mark_chat_image_used("chosen-u", "old-caption", 7)
        with patch.object(
            self.service.media_coordinator, "_curated_command_background"
        ) as curated:
            decision = self.service.maybe_command_meme(command(reply))
        curated.assert_not_called()
        self.assertEqual(decision.background_file_id, "chosen")

    def test_user_image_is_stored_and_duplicate_unique_id_is_upserted(self):
        first = image_message(10, [photo("old-token", "same-u", 800, 600)])
        second = image_message(11, [photo("fresh-token", "same-u", 800, 600)])
        self.assertTrue(self.service.ingest_chat_image(first))
        self.assertFalse(self.service.ingest_chat_image(second))
        repository = self.service.repository(-1)
        self.assertEqual(repository.chat_image_count(), 1)
        self.assertEqual(repository.chat_image_by_unique_id("same-u")["file_id"], "fresh-token")

    def test_old_relevant_image_can_beat_recent_unrelated_image(self):
        repository = self.service.repository(-1)
        old = datetime.now(timezone.utc) - timedelta(days=10)
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        repository.add_chat_image(
            1, 7, "car", "car-u", "photo", "image/jpeg",
            "серёга сломанная машина эвакуатор", created_at=old,
        )
        repository.add_chat_image(
            2, 8, "lunch", "lunch-u", "photo", "image/jpeg",
            "обычный обед", created_at=recent,
        )
        ranked = self.service.media.score_chat_images(
            repository, "серёга снова обсуждает сломанную машину"
        )
        self.assertEqual(ranked[0]["file_unique_id"], "car-u")

    def test_recently_used_image_gets_penalty_and_no_consecutive_repeat(self):
        repository = self.service.repository(-1)
        for number in (1, 2):
            repository.add_chat_image(
                number, number, f"file-{number}", f"unique-{number}",
                "photo", "image/jpeg", "одинаковая тема",
            )
        repository.mark_chat_image_used("unique-1", "caption", 1)
        ranked = self.service.media.score_chat_images(repository, "одинаковая тема")
        self.assertEqual(ranked[0]["file_unique_id"], "unique-2")
        self.assertNotIn("unique-1", [row["file_unique_id"] for row in ranked])

    def test_no_reply_can_choose_history_or_curated_when_history_is_absent(self):
        self.service.ingest_chat_image(
            image_message(10, [photo("history", "history-u", 800, 600)])
        )
        history = self.service.maybe_command_meme(command())
        self.assertEqual(history.background_file_id, "history")

        empty_service = LearningService(self.settings, rng=ZeroRandom())
        empty_service.repository(-2).add_message(
            1, 7, None, "другая реальная реплика из истории"
        )
        with patch.object(empty_service, "meme_command_on_cooldown", return_value=True):
            curated = empty_service.maybe_command_meme(-2)
        self.assertIsNone(curated.background_file_id)
        self.assertIsNotNone(curated.template_id)

    def test_cooldown_keeps_explicit_image_and_uses_no_ai_or_markov(self):
        reply = image_message(10, [photo("chosen", "chosen-u", 800, 600)])
        with (
            patch.object(self.service, "generate_llm") as ai,
            patch.object(self.service, "generate_local") as markov,
        ):
            decision = self.service.maybe_command_meme(command(reply))
        self.assertEqual(decision.background_file_id, "chosen")
        self.assertTrue(decision.reason.startswith("manual_local_"))
        ai.assert_not_called()
        markov.assert_not_called()

    def test_hint_is_passed_into_single_ai_caption_context(self):
        reply = image_message(
            10, [photo("chosen", "chosen-u", 800, 600)], caption="офис"
        )
        with (
            patch.object(self.service, "meme_command_on_cooldown", return_value=False),
            patch.object(self.service, "provider_available", return_value=True),
            patch.object(self.service, "generate_llm", return_value="короткая подпись") as ai,
        ):
            decision = self.service.maybe_command_meme(command(reply), "про серёгу")
        self.assertEqual(ai.call_count, 1)
        self.assertIn("про серёгу", ai.call_args.args[1])
        self.assertEqual(decision.background_file_id, "chosen")


class ArbitraryImageRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.renderer = MemeRenderer(MediaCatalog(), self.root / "out")

    def tearDown(self):
        self.temp.cleanup()

    def make_image(self, name, size, image_format):
        path = self.root / name
        Image.new("RGB", size, (34, 90, 140)).save(path, image_format)
        return path

    def test_jpeg_png_portrait_landscape_and_cyrillic(self):
        cases = (
            ("landscape.jpg", (1200, 600), "JPEG", "top_caption"),
            ("portrait.png", (500, 1000), "PNG", "bottom_caption"),
        )
        for name, size, image_format, profile in cases:
            with self.subTest(name=name):
                source = self.make_image(name, size, image_format)
                result = self.renderer.render_image(
                    source, "Серёга снова чинит прод", profile
                )
                self.assertIsNotNone(result)
                with Image.open(result.path) as rendered:
                    self.assertAlmostEqual(
                        rendered.width / rendered.height, size[0] / size[1], places=2
                    )
                self.renderer.cleanup(result)

    def test_long_caption_is_wrapped_or_shortened_without_covering_half(self):
        source = self.make_image("long.png", (1000, 700), "PNG")
        result = self.renderer.render_image(
            source, "очень длинная кириллическая подпись " * 24, "top_caption"
        )
        self.assertIsNotNone(result)
        self.assertLess(result.text_box[3] - result.text_box[1], 350)
        self.renderer.cleanup(result)

    def test_corrupted_and_oversized_images_fail_safely(self):
        corrupted = self.root / "broken.jpg"
        corrupted.write_bytes(b"not an image")
        self.assertIsNone(self.renderer.render_image(corrupted, "подпись"))
        oversized = self.make_image("big.png", (200, 100), "PNG")
        self.assertIsNone(self.renderer.render_image(
            oversized, "подпись", max_dimension=100
        ))

    def test_large_image_is_downscaled_without_aspect_ratio_change(self):
        source = self.make_image("large.jpg", (3200, 1600), "JPEG")
        result = self.renderer.render_image(source, "подпись")
        self.assertIsNotNone(result)
        with Image.open(result.path) as rendered:
            self.assertEqual(rendered.size, (1600, 800))
        self.renderer.cleanup(result)


class ChatImageRoutingAndCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    @contextmanager
    def activity(*_args, **_kwargs):
        yield None

    def test_special_route_stops_before_direct_router_and_passes_hint(self):
        import bot as bot_module

        incoming = command(text="с м стул про серёгу")
        with (
            patch.object(bot_module.chat_action_manager, "activity", side_effect=self.activity),
            patch.object(bot_module, "send_manual_meme", return_value=True) as send,
            patch.object(bot_module.learning_service, "maybe_direct_reply") as direct,
            patch.object(bot_module.bot, "reply_to") as reply,
        ):
            bot_module.handle_message(incoming)
        send.assert_called_once()
        self.assertEqual(send.call_args.args, (incoming,))
        self.assertEqual(send.call_args.kwargs["hint"], "про серёгу")
        self.assertEqual(send.call_args.kwargs["event"].message_id, incoming.message_id)
        direct.assert_not_called()
        reply.assert_not_called()

    def test_source_and_rendered_files_cleanup_after_send_and_exception(self):
        import bot as bot_module

        for raises in (False, True):
            source = self.root / f"source-{raises}.jpg"
            Image.new("RGB", (400, 300), "navy").save(source, "JPEG")
            rendered_path = self.root / f"rendered-{raises}.png"
            Image.new("RGB", (400, 300), "navy").save(rendered_path, "PNG")
            rendered = SimpleNamespace(path=rendered_path)
            decision = MediaDecision(
                action="meme", caption_text="подпись",
                background_file_id="telegram-file",
                background_file_unique_id="telegram-unique",
            )
            send_effect = RuntimeError("send failed") if raises else None
            with (
                patch.object(bot_module.chat_action_manager, "activity", side_effect=self.activity),
                patch.object(bot_module, "download_chat_image", return_value=source),
                patch.object(bot_module.learning_service, "render_meme", return_value=rendered),
                patch.object(bot_module.learning_service, "cleanup_rendered_meme", side_effect=lambda value: value.path.unlink(missing_ok=True)),
                patch.object(bot_module.learning_service, "mark_command_meme_sent"),
                patch.object(bot_module.bot, "send_photo", side_effect=send_effect),
            ):
                result = bot_module.send_manual_meme(command(), decision)
            self.assertEqual(result, not raises)
            self.assertFalse(source.exists())
            self.assertFalse(rendered_path.exists())

    def test_corrupted_explicit_download_falls_back_to_curated(self):
        import bot as bot_module

        broken = self.root / "broken.jpg"
        broken.write_bytes(b"broken")
        fallback = MediaDecision(
            action="meme", template_id="doomer_wojak", caption_text="подпись"
        )
        rendered_path = self.root / "fallback.png"
        rendered_path.write_bytes(b"png")
        rendered = SimpleNamespace(path=rendered_path)
        explicit = MediaDecision(
            action="meme", caption_text="подпись", background_file_id="bad"
        )
        with (
            patch.object(bot_module.chat_action_manager, "activity", side_effect=self.activity),
            patch.object(bot_module, "download_chat_image", return_value=broken),
            patch.object(bot_module.learning_service, "render_meme", side_effect=[None, rendered]) as render,
            patch.object(bot_module.learning_service, "fallback_command_meme_background", return_value=fallback),
            patch.object(bot_module.learning_service, "cleanup_rendered_meme", side_effect=lambda value: value.path.unlink(missing_ok=True)),
            patch.object(bot_module.learning_service, "mark_command_meme_sent") as mark,
            patch.object(bot_module.bot, "send_photo"),
        ):
            self.assertTrue(bot_module.send_manual_meme(command(), explicit))
        self.assertEqual(render.call_count, 2)
        mark.assert_called_once_with(-1, fallback)
        self.assertFalse(broken.exists())


if __name__ == "__main__":
    unittest.main()
