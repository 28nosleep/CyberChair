import json
import inspect
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageChops

from learning import (
    ChatState,
    ConversationDecision,
    LearningService,
    LearningSettings,
    MediaCatalog,
    MediaDecision,
    MediaService,
    MemeLexicon,
    MemeRenderer,
)
from learning.repository import ChatRepository


class ZeroRandom:
    def random(self):
        return 0.0

    def choice(self, values):
        return values[0]


class Provider:
    available = True

    def __init__(self):
        self.generate_calls = []

    def generate(self, request):
        self.generate_calls.append(request)
        return "обычный текстовый ответ"

    def summarize(self, request):
        return None


def chat_state(kind="humor", activity="high", topic="соя"):
    return ChatState(
        activity, 5, kind, topic, .8,
        .85 if kind == "humor" else .1,
        .85 if kind == "argument" else .1,
        .85 if kind == "serious" else .1,
        .85 if kind == "work" else .1,
        .5, 4, 10, 7, .9,
    )


def policy(intensity=.8, style="absurd_short", action="reply"):
    return ConversationDecision(
        action, 1.0 if action != "none" else 0.0, intensity, 30, style,
        10, 7, "test", 0.0, 1.0,
    )


def rows(text="это какой-то соевый кринж", target=10):
    return [{
        "message_id": target, "user_id": 7, "speaker": "user", "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }]


class MediaServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = LearningSettings(
            data_dir=self.root, openai_chat_id=-1, min_training_messages=1,
            media_cooldown=0, meme_render_cooldown=0,
            media_template_cooldown=0, media_meme_chance=1.0,
            addressed_cooldown=0, generated_cooldown=0,
        )
        self.catalog = MediaCatalog()
        self.repository = ChatRepository(self.root, -1, 50)
        self.service = MediaService(self.settings, self.catalog, ZeroRandom())

    def tearDown(self):
        self.temp.cleanup()

    def decide(self, **overrides):
        values = dict(
            chat_id=-1, repository=self.repository,
            conversation_decision=policy(), chat_state=chat_state(),
            short_term_rows=rows(), target_text="это какой-то соевый кринж",
            selected_memes=MemeLexicon().recognize("soyjak"),
            local_callbacks=(), troll_mode=True,
            probability_roll=0.0, meme_roll=0.0,
        )
        values.update(overrides)
        return self.service.decide(**values)

    def test_troll_mode_off_forbids_all_media(self):
        self.assertEqual(self.decide(troll_mode=False).action, "none")
        self.assertEqual(self.decide(troll_mode=False).reason, "troll_mode_off")

    def test_troll_mode_on_allows_contextual_media(self):
        self.assertEqual(self.decide().action, "meme")

    def test_humor_increases_media_probability(self):
        humor = self.service._media_probability(chat_state("humor"), .8)
        casual = self.service._media_probability(chat_state("casual"), .8)
        self.assertGreater(humor, casual)

    def test_serious_reduces_media_probability(self):
        serious = self.service._media_probability(chat_state("serious"), .35)
        normal = self.service._media_probability(chat_state("casual"), .35)
        self.assertLess(serious, normal)

    def test_unrelated_context_does_not_choose_asset(self):
        result = self.decide(
            conversation_decision=policy(.5, "chatty"),
            chat_state=chat_state("casual", "normal", None),
            short_term_rows=rows("сегодня хорошая погода"),
            target_text="сегодня хорошая погода", selected_memes=(),
        )
        self.assertEqual(result.action, "none")

    def test_recent_asset_is_not_repeated(self):
        first = self.decide()
        self.service.commit(self.repository, first)
        second = self.decide()
        self.assertNotEqual(first.template_id, second.template_id)

    def test_specific_template_cooldown_is_enforced(self):
        settings = replace(self.settings, media_template_cooldown=7200)
        service = MediaService(settings, self.catalog, ZeroRandom())
        first = service.decide(
            -1, self.repository, policy(), chat_state(), rows(),
            "соевый кринж", MemeLexicon().recognize("soyjak"), (), True, 0, 0,
        )
        service.commit(self.repository, first)
        second = service.decide(
            -1, self.repository, policy(), chat_state(), rows(),
            "соевый кринж", MemeLexicon().recognize("soyjak"), (), True, 0, 0,
        )
        self.assertNotEqual(first.template_id, second.template_id)

    def test_untagged_old_gif_remains_random_fallback(self):
        self.repository.add_gif(1, 7, "gif-file", "gif-unique")
        decision = self.service.random_fallback(self.repository)
        self.assertEqual((decision.action, decision.asset_id), ("gif", "gif-file"))

    def test_memelelexicon_signal_changes_template_ranking(self):
        result = self.decide(
            target_text="sigma moment",
            short_term_rows=rows("sigma moment"),
            selected_memes=MemeLexicon().recognize("sigma"),
        )
        self.assertEqual(result.template_id, "gigachad_approval")

    def test_local_callback_affects_selection(self):
        result = self.decide(
            conversation_decision=policy(.7, "work_sarcastic"),
            chat_state=chat_state("work", "normal", "сервер"),
            target_text="серега снова чинит сервер",
            short_term_rows=rows("серега снова чинит сервер"),
            selected_memes=(),
            local_callbacks=("серега опять чинит сервер",),
        )
        self.assertEqual(result.template_id, "doomer_wojak")

    def test_bot_message_cannot_be_quote_source(self):
        quote, source = self.service._quote([
            {"message_id": 10, "speaker": "cyberchair", "text": "ответ бота"}
        ], 10)
        self.assertIsNone(quote)
        self.assertIsNone(source)

    def test_target_message_is_preferred_quote_source(self):
        quote, source = self.service._quote([
            {"message_id": 10, "speaker": "user", "text": "целевая реплика"},
            {"message_id": 11, "speaker": "user", "text": "более свежая реплика"},
        ], 10)
        self.assertEqual((quote, source), ("целевая реплика", 10))

    def test_long_quote_is_limited_without_changing_claim(self):
        original = "это содержательное начало сообщения " + "лишний хвост " * 20
        quote, source = self.service._quote(rows(original), 10)
        self.assertLessEqual(len(quote), self.settings.meme_quote_max_chars)
        self.assertTrue(quote.startswith("это содержательное начало"))
        self.assertEqual(source, 10)

    def test_extreme_quote_length_falls_back(self):
        result = self.decide(short_term_rows=rows("очень " * 200), target_text="соя")
        self.assertEqual(result.action, "none")

    def test_tagged_gif_can_be_selected_contextually(self):
        self.repository.add_gif(1, 7, "fail-file", "fail-unique")
        self.repository.set_media_tags("gif", "fail-unique", ["провал", "work"])
        result = self.decide(
            conversation_decision=policy(.5, "work_sarcastic"),
            chat_state=chat_state("work", "normal", "провал"),
            short_term_rows=[], target_text="провал", selected_memes=(),
            meme_roll=1.0,
        )
        self.assertEqual((result.action, result.asset_id), ("gif", "fail-file"))

    def test_policy_none_cannot_create_media(self):
        result = self.decide(conversation_decision=policy(action="none"))
        self.assertEqual(result.reason, "conversation_policy_none")

    def test_missing_template_falls_back_to_none_without_placeholder(self):
        catalog_path = self.root / "missing-catalog.json"
        catalog_path.write_text(json.dumps({"version": "test", "assets": [{
            "id": "missing", "type": "meme_template", "file": "nope.png",
            "contexts": ["humor"], "tags": ["соя"], "intensity_min": 0,
            "weight": 10, "cooldown_group": "missing", "archetype": "soyjak",
            "render_profile": "top_caption", "text_box": [.05, .05, .95, .2]
        }]}), encoding="utf-8")
        service = MediaService(self.settings, MediaCatalog(catalog_path), ZeroRandom())
        result = service.decide(
            -1, self.repository, policy(), chat_state(), rows(), "соя",
            MemeLexicon().recognize("soyjak"), (), True, 0, 0,
        )
        self.assertEqual(result.action, "none")

    def test_media_service_has_no_llm_dependency(self):
        self.assertFalse(hasattr(self.service, "llm_provider"))
        self.decide()

    def test_one_decision_contains_only_one_action(self):
        result = self.decide()
        self.assertIn(result.action, {"none", "gif", "sticker", "meme"})
        self.assertIsInstance(result, MediaDecision)

    def test_contextual_media_skips_text_llm_call(self):
        provider = Provider()
        learning = LearningService(self.settings, llm_provider=provider, rng=ZeroRandom())
        learning.repository(-1).add_message(
            10, 7, "tester", "это соевый кринж", datetime.now(timezone.utc)
        )
        with (
            patch.object(learning.chat_state_analyzer, "analyze", return_value=chat_state()),
            patch.object(learning.conversation_policy, "decide", return_value=policy()),
            patch.object(learning, "_deterministic_media_roll", return_value=0.0),
        ):
            result = learning.maybe_reply(SimpleNamespace(
                chat=SimpleNamespace(id=-1), message_id=10,
                text="это соевый кринж",
                from_user=SimpleNamespace(id=7), reply_to_message=None,
            ))
        self.assertIsInstance(result, MediaDecision)
        self.assertEqual(provider.generate_calls, [])

    def test_troll_mode_off_keeps_scheduler_utility(self):
        learning = LearningService(self.settings, llm_provider=Provider())
        learning.set_troll_mode(-1, False)
        self.assertTrue(learning.claim_scheduled_event(-1, "utility:start"))

    def test_startup_meme_is_persistent_one_shot(self):
        learning = LearningService(self.settings, llm_provider=Provider())
        first = learning.startup_meme()
        self.assertEqual((first.action, first.template_id), ("meme", "t800_chud"))
        learning.mark_startup_meme_sent(first)
        self.assertIsNone(learning.startup_meme())

    def test_catalog_is_data_driven_and_versioned(self):
        self.assertEqual(self.catalog.version, "2.2.0")
        self.assertGreaterEqual(len(self.catalog.assets), 17)
        self.assertTrue(all(asset.contexts and asset.tags and asset.source_url for asset in self.catalog.assets))
        self.assertTrue(all(self.catalog.resolve(asset).is_file() for asset in self.catalog.assets))


class MemeRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = MediaCatalog()
        self.renderer = MemeRenderer(self.catalog, Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_renderer_supports_cyrillic(self):
        result = self.renderer.render("soyjak_pointing", "сервер опять упал в проде")
        self.assertIsNotNone(result)
        self.assertEqual(result.rendered_text, "сервер опять упал в проде")
        with Image.open(result.path) as image:
            self.assertEqual(image.format, "PNG")
        self.renderer.cleanup(result)

    def test_production_templates_are_original_image_assets(self):
        self.assertTrue(all(asset.source_url for asset in self.catalog.assets))
        self.assertTrue(all(self.catalog.resolve(asset).is_file() for asset in self.catalog.assets))
        self.assertFalse(any(
            path.name in {"soyjak.png", "wojak.png", "enterprise.png", "office.png"}
            for path in self.catalog.root.joinpath("templates").iterdir()
        ))

    def test_renderer_does_not_draw_characters_or_archetype_name(self):
        source = inspect.getsource(MemeRenderer)
        self.assertNotIn("ellipse(", source)
        self.assertNotIn("rounded_rectangle(", source)
        self.assertNotIn("asset.archetype", source)

    def test_template_composition_is_preserved_outside_safe_area(self):
        asset = self.catalog.get("soyjak_pointing")
        result = self.renderer.render(asset.id, "это настоящий исходный шаблон")
        with Image.open(self.catalog.resolve(asset)) as original:
            original = original.convert("RGB")
            if max(original.size) > 1600:
                original.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        with Image.open(result.path) as rendered:
            difference = ImageChops.difference(original, rendered.convert("RGB"))
        left, top, right, bottom = result.safe_box
        outside = (
            difference.crop((0, 0, difference.width, top)),
            difference.crop((0, top, left, bottom)),
            difference.crop((right, top, difference.width, bottom)),
            difference.crop((0, bottom, difference.width, difference.height)),
        )
        self.assertTrue(all(part.getbbox() is None for part in outside))
        self.renderer.cleanup(result)

    def test_top_caption_profile_with_cyrillic(self):
        result = self.renderer.render("soyjak_pointing", "давайте это вынесем в микросервис")
        self.assertEqual(result.render_profile, "top_caption")
        self.assertEqual(result.font_name, "Impact")
        self.assertIn("микросервис", result.rendered_text)
        self.renderer.cleanup(result)

    def test_suitable_template_is_not_resized(self):
        asset = self.catalog.get("doomer_wojak")
        with Image.open(self.catalog.resolve(asset)) as original:
            original_size = original.size
        result = self.renderer.render(asset.id, "деплой готов")
        with Image.open(result.path) as rendered:
            self.assertEqual(rendered.size, original_size)
        self.renderer.cleanup(result)

    def test_oversized_template_is_resized_with_preserved_aspect_ratio(self):
        asset = self.catalog.get("soyjak_pointing")
        with Image.open(self.catalog.resolve(asset)) as original:
            original_ratio = original.width / original.height
        result = self.renderer.render(asset.id, "микросервис")
        with Image.open(result.path) as rendered:
            self.assertLessEqual(max(rendered.size), 1600)
            self.assertAlmostEqual(rendered.width / rendered.height, original_ratio, places=3)
        self.renderer.cleanup(result)

    def test_bottom_caption_profile(self):
        result = self.renderer.render("doomer_wojak", "пятничный деплой готов")
        self.assertEqual(result.render_profile, "bottom_caption")
        self.assertGreater(result.text_box[1], 600)
        self.renderer.cleanup(result)

    def test_top_bottom_profile(self):
        fixture = Path(self.temp.name) / "fixture.png"
        shutil.copyfile(self.catalog.resolve(self.catalog.get("doomer_wojak")), fixture)
        catalog_path = Path(self.temp.name) / "top_bottom.json"
        catalog_path.write_text(json.dumps({"version": "test", "assets": [{
            "id": "fixture", "type": "meme_template", "file": "fixture.png",
            "contexts": ["humor"], "tags": ["fixture"], "intensity_min": 0,
            "weight": 1, "cooldown_group": "fixture", "archetype": "fixture",
            "render_profile": "top_bottom", "text_box": [.05, .05, .95, .95]
        }]}), encoding="utf-8")
        renderer = MemeRenderer(MediaCatalog(catalog_path), Path(self.temp.name) / "out")
        result = renderer.render("fixture", "верхняя фраза | нижняя фраза")
        self.assertEqual(result.render_profile, "top_bottom")
        self.assertIn("верхняя", result.rendered_text)
        self.assertIn("нижняя", result.rendered_text)
        self.renderer.cleanup(result)

    def test_overlay_uses_template_safe_area(self):
        result = self.renderer.render("gigachad_approval", "бейсд релиз")
        left, top, right, bottom = result.text_box
        safe_left, safe_top, safe_right, safe_bottom = result.safe_box
        self.assertGreaterEqual(left, safe_left)
        self.assertGreaterEqual(top, safe_top)
        self.assertLessEqual(right, safe_right)
        self.assertLessEqual(bottom, safe_bottom)
        self.renderer.cleanup(result)

    def test_phone_templates_draw_text_inside_phone_screen(self):
        for template_id in ("chudjak_phone_scream", "feraljak_phone_rage"):
            result = self.renderer.render(template_id, "ОНА ПРОЧИТАЛА И МОЛЧИТ")
            self.assertIsNotNone(result)
            self.assertIn("phone_screen", result.render_profile)
            left, top, right, bottom = result.text_box
            safe_left, safe_top, safe_right, safe_bottom = result.safe_box
            self.assertGreaterEqual(left, safe_left)
            self.assertGreaterEqual(top, safe_top)
            self.assertLessEqual(right, safe_right)
            self.assertLessEqual(bottom, safe_bottom)
            self.renderer.cleanup(result)

    def test_caption_uses_visible_outline(self):
        source = inspect.getsource(MemeRenderer._draw_caption)
        self.assertIn("stroke_width=stroke", source)
        self.assertIn("stroke_fill=\"black\"", source)

    def test_long_caption_wraps(self):
        result = self.renderer.render(
            "soyjak_pointing",
            "давайте ради одного маленького эндпоинта вынесем всё в отдельный "
            "микросервис и позовём архитектора на три созвона",
        )
        self.assertGreater(result.line_count, 1)
        self.assertLessEqual(result.line_count, 4)
        self.renderer.cleanup(result)

    def test_text_stays_inside_safe_area(self):
        result = self.renderer.render(
            "soyjak_pointing", "ебать ты конечно архитектор микросервисной платформы"
        )
        self.assertIsNotNone(result)
        left, top, right, bottom = result.text_box
        safe_left, safe_top, safe_right, safe_bottom = result.safe_box
        self.assertGreaterEqual(left, safe_left)
        self.assertGreaterEqual(top, safe_top)
        self.assertLessEqual(right, safe_right)
        self.assertLessEqual(bottom, safe_bottom)
        self.renderer.cleanup(result)

    def test_missing_template_does_not_raise(self):
        raw = json.loads(self.catalog.path.read_text(encoding="utf-8"))
        raw["assets"][0]["file"] = "templates/does-not-exist.png"
        path = Path(self.temp.name) / "catalog.json"
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        renderer = MemeRenderer(MediaCatalog(path), Path(self.temp.name) / "out")
        self.assertIsNone(renderer.render("soyjak_pointing", "кириллица"))

    def test_temporary_file_is_deleted(self):
        result = self.renderer.render("doomer_wojak", "пятничный деплой готов")
        self.assertTrue(result.path.exists())
        self.renderer.cleanup(result)
        self.assertFalse(result.path.exists())

    def test_renderer_has_no_llm_dependency(self):
        self.assertFalse(hasattr(self.renderer, "llm_provider"))

    def test_sender_deletes_file_after_success(self):
        import bot as bot_module

        result = self.renderer.render("doomer_wojak", "релиз опять упал")
        incoming = SimpleNamespace(chat=SimpleNamespace(id=-1), message_id=10)
        decision = MediaDecision("meme", "doomer_wojak", "doomer_wojak", 10, "релиз опять упал", .8, "test")
        with (
            patch.object(bot_module.learning_service, "render_meme", return_value=result),
            patch.object(bot_module.bot, "send_photo"),
            patch.object(bot_module.learning_service, "cleanup_rendered_meme", side_effect=self.renderer.cleanup),
        ):
            self.assertTrue(bot_module.send_contextual_response(incoming, decision))
        self.assertFalse(result.path.exists())


if __name__ == "__main__":
    unittest.main()
