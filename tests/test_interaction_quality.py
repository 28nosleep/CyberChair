import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    GenerateRequest,
    GrokProvider,
    LearningService,
    LearningSettings,
    LexicalDiversityTracker,
    MemeLexicon,
    PersonaBuilder,
    ResponseQualityGuard,
)
from learning.meme_sources import MemeSource, MemeSourceSelector


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class ZeroRandom(random.Random):
    def random(self):
        return 0.0

    def choice(self, values):
        return values[0]


def settings(path, **overrides):
    values = dict(data_dir=Path(path), openai_chat_id=-1, min_training_messages=1)
    values.update(overrides)
    return LearningSettings(**values)


def request(builder, text, purpose="question"):
    return builder.build_request(
        -1, context=text, purpose=purpose, troll_mode=True
    ).request


class DynamicBudgetAndCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = settings(self.temp.name)
        self.builder = PersonaBuilder(self.settings, MemeLexicon())

    def tearDown(self):
        self.temp.cleanup()

    def test_budgets_follow_response_purpose(self):
        social = request(self.builder, "ну чё", "random_reply")
        troll = request(self.builder, "стул как выбрать vps", "troll_user")
        useful = request(self.builder, "стул почему небо синее")
        recipe = request(self.builder, "дай рецепт харчо")
        complex_answer = request(self.builder, "объясни настройку Docker и DNS")
        meme = request(self.builder, "серёга в офисе", "meme_caption")
        self.assertLess(social.max_output_tokens, useful.max_output_tokens)
        self.assertLess(troll.max_output_tokens, recipe.max_output_tokens)
        self.assertEqual(recipe.max_output_tokens, self.settings.recipe_max_output_tokens)
        self.assertEqual(complex_answer.max_output_tokens, self.settings.complex_max_output_tokens)
        self.assertLess(meme.max_output_tokens, troll.max_output_tokens)
        self.assertEqual(recipe.metadata["response_purpose"], "recipe_instruction")

    def test_provider_incomplete_is_logged_metered_and_never_retried(self):
        usage = SimpleNamespace(output_tokens=100, input_tokens=20)
        response = SimpleNamespace(
            output_text="лук снять с огня",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            usage=usage,
        )
        client = SimpleNamespace(responses=FakeResponses(response))
        service = LearningService(self.settings, xai_client=client)
        with self.assertLogs("learning.responses_provider", level="WARNING") as logs:
            result = service.generate_llm(-1, "как приготовить суп", "question")
        self.assertEqual(result, "лук снять с огня")
        self.assertEqual(len(client.responses.calls), 1)
        self.assertIn("LLM_INCOMPLETE", "\n".join(logs.output))
        report = service.quality_diagnostics(-1)
        self.assertEqual(report["llm_incomplete"], 1)
        self.assertEqual(report["llm_incomplete_reason"]["max_output_tokens"], 1)
        self.assertEqual(report["llm_truncated_total"], 1)
        self.assertEqual(report["llm_truncated_by_purpose"]["recipe_instruction"], 1)

    def test_multiline_useful_answer_is_not_sliced_by_postprocessing(self):
        response = SimpleNamespace(
            output_text="шаг один готов\nшаг два готов\nшаг три готов\nшаг четыре готов",
            status="completed", usage=SimpleNamespace(output_tokens=20),
        )
        provider = GrokProvider(self.settings, SimpleNamespace(responses=FakeResponses(response)))
        result = provider.generate(GenerateRequest(
            "persona", "context", 360,
            metadata={"purpose": "question", "response_purpose": "useful_answer"},
        ))
        self.assertEqual(len(result.splitlines()), 4)


class PhotoCaptionSpecialRouteTests(unittest.TestCase):
    @staticmethod
    def photo_message(caption="с м стул"):
        variants = [
            SimpleNamespace(file_id="small", file_unique_id="small-u", width=90, height=90, file_size=500),
            SimpleNamespace(file_id="exact", file_unique_id="exact-u", width=1200, height=900, file_size=2000),
        ]
        return SimpleNamespace(
            chat=SimpleNamespace(id=-1), message_id=77, text=None,
            caption=caption, photo=variants, document=None, date=1_700_000_000,
            from_user=SimpleNamespace(id=7, username="user", is_bot=False),
            reply_to_message=None,
        )

    def test_photo_caption_uses_current_exact_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(settings(directory), rng=ZeroRandom())
            service.repository(-1).add_message(1, 7, None, "реальная подпись из чата")
            with patch.object(service, "meme_command_on_cooldown", return_value=True):
                decision = service.maybe_command_meme(self.photo_message())
            self.assertEqual(decision.background_file_id, "exact")
            self.assertTrue(decision.background_explicit)
            self.assertEqual(service.quality_diagnostics(-1)["photo_caption_meme_trigger"], 1)

    def test_photo_handler_is_single_special_producer(self):
        import bot as bot_module

        incoming = self.photo_message("с м стул про релиз")
        with (
            patch.object(bot_module, "send_manual_meme", return_value=True) as meme,
            patch.object(bot_module.learning_service, "ingest_chat_image") as ingest,
            patch.object(bot_module.learning_service, "maybe_direct_reply") as direct,
        ):
            bot_module.remember_photo(incoming)
        meme.assert_called_once()
        self.assertEqual(meme.call_args.args, (incoming,))
        self.assertEqual(meme.call_args.kwargs["hint"], "про релиз")
        self.assertEqual(meme.call_args.kwargs["event"].message_id, incoming.message_id)
        ingest.assert_not_called()
        direct.assert_not_called()


class LexicalDiversityTests(unittest.TestCase):
    def setUp(self):
        self.tracker = LexicalDiversityTracker(window=40)

    def test_single_use_allowed_repetition_strong_and_window_recovers(self):
        self.assertEqual(self.tracker.score("классика жанра", ["ну классика"])[0], 0)
        recent = ["классика приехала", "ну классика", "классика жанра"]
        self.assertGreater(self.tracker.score("классика снова", recent)[0], 3)
        aged = recent + [f"обычная содержательная реплика номер {i}" for i in range(45)]
        self.assertEqual(self.tracker.score("классика снова", aged)[0], 0)

    def test_opening_construction_and_stopwords(self):
        recent = ["chairOS фиксирует очередной крашаут"] * 3
        score, phrases = self.tracker.score("chairOS фиксирует новый прикол", recent)
        self.assertGreater(score, 3)
        self.assertIn("chairos фиксирует", phrases)
        features, _ = self.tracker.features("и но если это можно")
        self.assertFalse(features)

    def test_quality_guard_only_rejects_strong_local_tick(self):
        guard = ResponseQualityGuard(self.tracker)
        recent = ["ну классика приехала"] * 3
        self.assertFalse(guard.check("ну классика опять", recent, local=True).accepted)
        self.assertTrue(guard.check("ну классика опять", recent, local=False).accepted)


class MemeAndContextQualityTests(unittest.TestCase):
    def test_image_meme_selector_has_no_phrase_source_or_canned_tail(self):
        selector = MemeSourceSelector(random.Random(1))
        self.assertEqual(selector.choose(-1, [], fallback=True).kind, "none")
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(settings(directory))
            caption = service.media_coordinator._caption_from_source(
                MemeSource("old", "я больше не пишу рэп")
            )
            self.assertEqual(caption, "я больше не пишу рэп")
            self.assertNotIn("сойджак-комиссия", caption)
            self.assertFalse(service.quality_guard.check(
                "сойджак-комиссия уже выехала", image_meme=True
            ).accepted)

    def test_relevant_old_callback_wins_and_unrelated_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = PersonaBuilder(settings(directory), MemeLexicon())
            memories = (
                "рэп это хуйня, больше не пишу",
                "я купил зелёный зонт в электричке",
            )
            rap = builder.select_callbacks({}, memories, "как прославиться в рэпе", "музыка")
            rice = builder.select_callbacks({}, memories, "как сварить рис", "кухня")
            self.assertEqual(rap, (memories[0],))
            self.assertEqual(rice, ())


if __name__ == "__main__":
    unittest.main()
