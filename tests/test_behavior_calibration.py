import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    ChatState,
    ConversationDecision,
    LearningService,
    LearningSettings,
    MediaCatalog,
    MediaDecision,
    MediaService,
    MemeLexicon,
    PersonaBuilder,
)
from learning.filters import similarity, validate_generated
from learning.markov import MarkovModel
from learning.meme_sources import MemeSource
from learning.repository import ChatRepository


class FixedRandom(random.Random):
    def __init__(self, value):
        super().__init__(1)
        self.value = value

    def random(self):
        return self.value


def message(text, message_id=1, chat_id=-1):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id), message_id=message_id, text=text,
        date=datetime.now(timezone.utc).timestamp(),
        from_user=SimpleNamespace(id=7, username="tester", is_bot=False),
        reply_to_message=None,
    )


def state(kind="humor", topic="разговор"):
    return ChatState(
        "normal", 6, kind, topic, .7,
        .8 if kind == "humor" else .1,
        .8 if kind == "argument" else .1,
        .8 if kind == "serious" else .1,
        .8 if kind == "work" else .1,
        .4, 3, 10, 7, .8,
    )


def decision(intensity=.75):
    return ConversationDecision(
        "reply", 1.0, intensity, 32, "absurd_short", 10, 7,
        "regression", 0.0, 1.0,
    )


class BehaviorCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, **overrides):
        values = dict(
            data_dir=self.root, openai_chat_id=-1, min_training_messages=1,
            addressed_cooldown=0, generated_cooldown=0,
            max_generated_per_hour=100, media_cooldown=0,
            meme_render_cooldown=0, media_template_cooldown=0,
        )
        values.update(overrides)
        return LearningSettings(**values)

    def test_target_long_verbatim_overlap_and_automatic_quote_are_blocked(self):
        target = "у меня вроде бы нет сколиоза потому что спина вообще не болит"
        copied = "ну у меня вроде бы нет сколиоза потому что спина вообще не болит — открытие века"
        quoted = "фраза «вроде бы нет сколиоза потому» уже звучит как диагноз"
        self.assertEqual(validate_generated(copied, input_text=target)[1], "input_overlap")
        self.assertEqual(validate_generated(quoted, input_text=target)[1], "input_quote")

    def test_contextual_reaction_without_copy_is_allowed(self):
        target = "у меня вроде бы нет сколиоза потому что спина вообще не болит"
        response = "отсутствие боли ничего не доказывает, сходи к ортопеду и не ларпь рентген"
        self.assertTrue(validate_generated(response, input_text=target)[0])

    def test_prompt_requires_original_reaction_and_useful_answer(self):
        built = PersonaBuilder(self.settings(), MemeLexicon()).build_request(
            -1, "стул как выбрать UPS", "question",
            conversation_decision=decision(), chat_state=state("work", "ups"),
            troll_mode=True,
        )
        self.assertIn("полезная информация обязательна", built.request.input)
        self.assertIn("не цитируй и не перефразируй", built.request.input)
        self.assertIn("Не цитируй целевое сообщение", built.request.instructions)

    def test_meme_selection_is_diverse_and_aura_has_no_default_priority(self):
        lexicon = MemeLexicon()
        selected = [
            lexicon.select(
                f"всё сломалось и упало с ошибкой номер {index}",
                {"failure", "mocking"}, .8, limit=1,
            )[0].id
            for index in range(100)
        ]
        self.assertGreaterEqual(len(set(selected)), 5)
        self.assertLess(selected.count("aura_loss"), 10)

    def test_minus_aura_group_has_cooldown(self):
        lexicon = MemeLexicon()
        first = lexicon.select("aura loss", {"failure"}, .8, limit=2)
        self.assertEqual(first[0].id, "aura_loss")
        second = lexicon.select(
            "aura loss", {"failure"}, .8,
            excluded_groups={"aura"}, limit=3,
        )
        self.assertNotIn("aura_loss", {entry.id for entry in second})
        self.assertNotIn("aura", {entry.id for entry in second})

    def test_bare_chair_has_separate_lower_probability(self):
        service = LearningService(self.settings(), rng=FixedRandom(.32))
        with (
            patch.object(service, "generate_openai") as ai,
            patch.object(service, "generate_local") as markov,
        ):
            self.assertIsNone(service.maybe_stul_cooldown_reply(message("стул")))
        ai.assert_not_called()
        markov.assert_not_called()
        self.assertLess(
            (service.settings.reply_to_stul_chance + service.settings.stul_markov_reply_chance)
            * service.settings.bare_stul_reply_factor,
            service.settings.reply_to_stul_chance + service.settings.stul_markov_reply_chance,
        )

    def test_substantive_chair_request_keeps_normal_ai_path(self):
        service = LearningService(self.settings(), rng=FixedRandom(.99))
        incoming = message("стул как приготовить курицу сувид")
        with (
            patch.object(service, "generate_openai", return_value="63–65 градусов, потом быстро обжарь") as ai,
            patch.object(service, "generate_local") as markov,
        ):
            result = service.maybe_stul_cooldown_reply(incoming)
        self.assertIn("63", result)
        ai.assert_called_once()
        markov.assert_not_called()

    def test_markov_corpus_excludes_live_edge_and_very_recent_rows(self):
        service = LearningService(self.settings(
            markov_exclude_recent_messages=3,
            markov_min_message_age_seconds=120,
            markov_recent_history_size=2,
        ))
        repository = service.repository(-1)
        now = datetime.now(timezone.utc)
        for index in range(1, 6):
            repository.add_message(
                index, 7, None, f"старая локальная фраза номер {index}",
                now - timedelta(days=2, minutes=index),
            )
        repository.add_message(6, 7, None, "слишком свежая фраза не для маркова", now)
        for index in range(7, 10):
            repository.add_message(
                index, 7, None, f"последнее сообщение live edge {index}",
                now - timedelta(days=2),
            )
        corpus = service._markov_corpus(repository, current=now)
        texts = [row["text"] for row in corpus]
        self.assertEqual(len(texts), 5)
        self.assertNotIn("слишком свежая фраза не для маркова", texts)
        self.assertFalse(any("live edge" in text for text in texts))
        self.assertGreater(corpus[0]["generation_weight"], corpus[-1]["generation_weight"])

    def test_markov_model_never_learns_excluded_recent_phrase(self):
        service = LearningService(self.settings(markov_exclude_recent_messages=3))
        repository = service.repository(-1)
        old = datetime.now(timezone.utc) - timedelta(days=1)
        for index, text in enumerate((
            "старый мем про сервер который живёт своей жизнью",
            "старый мем про релиз который опять убежал",
            "старый мем про созвон который никто не просил",
            "уникальная свежая последовательность один два три",
            "уникальная свежая последовательность четыре пять шесть",
            "уникальная свежая последовательность семь восемь девять",
        ), 1):
            repository.add_message(index, 7, None, text, old)
        model, _ = service._model_and_messages(-1)
        learned = " ".join(
            word for options in model.transitions.values() for word in options
        ).casefold()
        self.assertNotIn("свежая", learned)

    def test_normalized_similarity_penalizes_near_repeats(self):
        source = "сервер опять упал после пятничного релиза"
        near = "Сервер, опять упал после пятничного релиза!"
        self.assertGreater(similarity(source, near), .95)
        self.assertEqual(
            validate_generated(near, previous_bot_texts=[source])[1],
            "bot_copy",
        )

    def test_ai_event_never_calls_markov_and_failed_ai_does_not_fallback(self):
        service = LearningService(self.settings(), rng=FixedRandom(.05))
        with (
            patch.object(service, "generate_openai", return_value="один ответ") as ai,
            patch.object(service, "generate_local") as markov,
        ):
            self.assertEqual(service.maybe_stul_cooldown_reply(message("стул")), "один ответ")
        ai.assert_called_once()
        markov.assert_not_called()

        with (
            patch.object(service, "provider_available", return_value=True),
            patch.object(service, "meme_command_on_cooldown", return_value=False),
            patch.object(service.meme_sources, "choose", return_value=MemeSource("old", "старая цитата")),
            patch.object(service, "generate_openai", return_value=None) as ai,
            patch.object(service, "_local_command_caption") as local,
        ):
            self.assertIsNone(service.maybe_command_meme(-1))
        ai.assert_called_once()
        local.assert_not_called()

    def test_untagged_gif_reaches_contextual_telegram_sender(self):
        repository = ChatRepository(self.root, -1, 50)
        repository.add_gif(1, 7, "gif-file", "gif-unique")
        media = MediaService(self.settings(media_meme_chance=0), MediaCatalog(), FixedRandom(0))
        result = media.decide(
            -1, repository, decision(), state("humor"), [],
            "чат разорвало от этого прикола", (), (), True,
            probability_roll=0, meme_roll=1, reaction_roll=0,
        )
        self.assertEqual((result.action, result.asset_id), ("gif", "gif-file"))

        import bot as bot_module
        incoming = SimpleNamespace(chat=SimpleNamespace(id=-1), message_id=10)
        with patch.object(bot_module.bot, "send_animation") as sender:
            self.assertTrue(bot_module.send_contextual_response(incoming, result))
        sender.assert_called_once_with(-1, "gif-file", reply_to_message_id=10)

    def test_contextual_media_mix_has_material_gif_share(self):
        repository = ChatRepository(self.root, -2, 50)
        repository.add_gif(1, 7, "gif-file", "gif-unique")
        repository.add_sticker(2, 7, "sticker-file", "sticker-unique")
        media = MediaService(
            self.settings(media_meme_chance=0, media_gif_share=.72),
            MediaCatalog(), FixedRandom(0),
        )
        actions = [
            media.decide(
                -2, repository, decision(), state("humor"), [],
                "чат разорвало от этого прикола", (), (), True,
                probability_roll=0, meme_roll=1, reaction_roll=index / 100,
            ).action
            for index in range(100)
        ]
        self.assertEqual(actions.count("gif"), 72)
        self.assertEqual(actions.count("sticker"), 28)

    def test_freekucher_latin_trigger_is_automatic_and_single_response(self):
        import bot as bot_module

        incoming = message("Kucher снова в чате", message_id=808, chat_id=-808)
        bot_module.last_freekucher_reply_at.clear()
        with (
            patch.object(bot_module.learning_service, "troll_mode", return_value=True),
            patch.object(bot_module.learning_service, "maybe_reply") as regular,
            patch.object(bot_module.bot, "reply_to") as reply,
        ):
            bot_module.handle_message(incoming)
        reply.assert_called_once_with(incoming, "#FREEKUCHER")
        regular.assert_not_called()

    def test_callbacks_require_real_topic_overlap(self):
        builder = PersonaBuilder(self.settings(), MemeLexicon())
        memories = (
            "серёга всегда деплоит сервер по пятницам",
            "андрей потерял зонт в электричке",
        )
        relevant = builder.select_callbacks({}, memories, "серёга чинит сервер", "деплой")
        unrelated = builder.select_callbacks({}, memories, "сколько варить яйца", "кухня")
        self.assertEqual(relevant, (memories[0],))
        self.assertEqual(unrelated, ())

    def test_useful_question_keeps_one_llm_call_and_persona(self):
        class Provider:
            available = True

            def __init__(self):
                self.calls = []

            def generate(self, request):
                self.calls.append(request)
                return "смотри exit code и логи, entrypoint опять проебан"

            def summarize(self, request):
                return None

        provider = Provider()
        service = LearningService(self.settings(), llm_provider=provider, rng=FixedRandom(.99))
        result = service.maybe_stul_cooldown_reply(message("стул почему docker container падает"))
        self.assertIn("exit code", result)
        self.assertIn("проебан", result)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("полезный ответ", provider.calls[0].input)

    def test_daily_freekucher_scheduler_claims_once_without_llm(self):
        import scheduler as scheduler_module

        current = datetime(2026, 8, 17, 21, 0)
        sent = []
        claimed = set()

        def claim(chat_id, event_key):
            if event_key in claimed:
                return False
            claimed.add(event_key)
            return True

        with (
            patch.object(scheduler_module, "get_now", return_value=current),
            patch.object(scheduler_module, "daily_freekucher_minute", return_value=0),
            patch.object(scheduler_module, "daily_quote_minutes", return_value=[]),
            patch.object(scheduler_module, "is_workday", return_value=False),
            patch.object(scheduler_module.time, "sleep", side_effect=[None, KeyboardInterrupt]),
        ):
            with self.assertRaises(KeyboardInterrupt):
                scheduler_module.scheduler(
                    SimpleNamespace(), -1, "Europe/Moscow", 9, 0, 17, 30,
                    event_claim_callback=claim,
                    daily_freekucher_callback=lambda chat_id: sent.append(chat_id),
                )
        self.assertEqual(sent, [-1])
        self.assertEqual(claimed, {"freekucher:2026-08-17"})

    def test_movie_quotes_keep_seven_item_anti_repeat_and_matrix_pool(self):
        import scheduler as scheduler_module
        from messages import MOVIE_QUOTES

        scheduler_module._recent_quotes.clear()
        selected = []
        fake_bot = SimpleNamespace(send_message=lambda *args, **kwargs: None)
        with patch.object(scheduler_module.random, "choice", side_effect=lambda values: values[0]):
            for _ in range(24):
                scheduler_module.movie_quote_message(fake_bot, -1)
                quote = scheduler_module._last_quote
                self.assertNotIn(quote, selected[-7:])
                selected.append(quote)

        matrix_quotes = [quote for quote in MOVIE_QUOTES if quote[2].startswith("Матрица")]
        self.assertEqual(len(matrix_quotes), 13)
        self.assertIn("Ложки не существует", {quote[0] for quote in matrix_quotes})
        self.assertIn("Добро пожаловать в реальный мир", {quote[0] for quote in matrix_quotes})


if __name__ == "__main__":
    unittest.main()
