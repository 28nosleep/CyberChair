import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning.filters import validate_generated
from learning.markov import MarkovModel
from learning.repository import ChatRepository
from learning.service import LearningService
from learning.settings import LearningSettings
from learning.triggers import TriggerEngine
from messages import MOVIE_QUOTES, format_movie_quote
from scheduler import daily_quote_minutes
from utils import is_stul_message, stul_remaining_variants


def message(chat_id, message_id, text, user_id=1, reply=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        text=text,
        date=1_700_000_000 + message_id,
        from_user=SimpleNamespace(id=user_id, username=f"user{user_id}", is_bot=False),
        reply_to_message=reply,
    )


class ZeroRandom(random.Random):
    def random(self):
        return 0.0


class FixedRandom(random.Random):
    def __init__(self, value):
        super().__init__(1)
        self.value = value

    def random(self):
        return self.value


class FakeResponses:
    def __init__(self, text="[CORE::ONLINE] Люди, ваш сервер скоро признает власть Киберстула"):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text)


class FakeOpenAI:
    def __init__(self, text=None):
        self.responses = FakeResponses(text or "[CORE::ONLINE] Люди, ваш сервер скоро признает власть Киберстула")


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, **overrides):
        values = dict(
            data_dir=self.data_dir,
            min_training_messages=3,
            addressed_cooldown=0,
            generated_cooldown=0,
            max_generated_per_hour=10,
            openai_chat_id=-1,
        )
        values.update(overrides)
        return LearningSettings(**values)

    def test_chat_data_is_physically_separated(self):
        first = ChatRepository(self.data_dir, -1001)
        second = ChatRepository(self.data_dir, -1002)
        first.add_message(1, 1, None, "Первое сообщение чата")
        second.add_message(1, 2, None, "Второе сообщение чата")
        self.assertNotEqual(first.path, second.path)
        self.assertEqual([row["text"] for row in first.recent_messages()], ["Первое сообщение чата"])
        self.assertEqual([row["text"] for row in second.recent_messages()], ["Второе сообщение чата"])

    def test_second_order_markov_training_and_generation(self):
        model = MarkovModel().train([
            "мы захватим этот сервер сегодня",
            "мы захватим этот чат завтра",
        ])
        self.assertEqual(sorted(model.transitions[("мы", "захватим")]), ["этот", "этот"])
        generated = model.generate(3, 10, rng=random.Random(4))
        self.assertTrue(generated.startswith("мы захватим"))

    def test_no_reply_before_minimum_training_volume(self):
        service = LearningService(self.settings(min_training_messages=2), rng=ZeroRandom())
        incoming = message(-1, 1, "Нас ждёт восстание машин")
        service.ingest(incoming)
        self.assertIsNone(service.maybe_reply(incoming))

    def test_cooldown(self):
        now = [100.0]
        engine = TriggerEngine(self.settings(generated_cooldown=60), ZeroRandom(), lambda: now[0])
        self.assertEqual(engine.decide_user_reply(-1), "random")
        engine.commit(-1, "random")
        self.assertIsNone(engine.decide_user_reply(-1))
        now[0] += 61
        self.assertEqual(engine.decide_user_reply(-1), "random")

    def test_stul_cooldown_reports_remaining_seconds(self):
        now = [100.0]
        service = LearningService(
            self.settings(addressed_cooldown=60),
            rng=FixedRandom(0.20),
            clock=lambda: now[0],
        )
        incoming = message(-1, 1, "стул")
        with patch.object(service, "generate_openai", return_value="ответ стула"):
            self.assertEqual(service.maybe_stul_cooldown_reply(incoming), "ответ стула")
        self.assertEqual(service.stul_cooldown_remaining(-1), 60)
        self.assertEqual(service.take_stul_cooldown_notice(-1), 60)
        self.assertEqual(service.take_stul_cooldown_notice(-1), 0)
        now[0] += 12.2
        self.assertEqual(service.stul_cooldown_remaining(-1), 48)
        self.assertEqual(service.take_stul_cooldown_notice(-1), 0)

    def test_reply_to_stul_uses_forty_percent_probability(self):
        accepted = TriggerEngine(self.settings(reply_to_stul_chance=0.40), FixedRandom(0.399))
        rejected = TriggerEngine(self.settings(reply_to_stul_chance=0.40), FixedRandom(0.40))
        self.assertEqual(accepted.decide_user_reply(-1, replies_to_bot=True), "addressed")
        self.assertIsNone(rejected.decide_user_reply(-1, replies_to_bot=True))

    def test_repeated_stul_routes_forty_ai_fifty_markov(self):
        incoming = message(-1, 1, "стульчик ещё раз")

        ai_service = LearningService(self.settings(), rng=FixedRandom(0.20))
        with (
            patch.object(ai_service, "generate_openai", return_value="ответ OpenAI") as ai,
            patch.object(ai_service, "generate_local") as markov,
        ):
            self.assertEqual(ai_service.maybe_stul_cooldown_reply(incoming), "ответ OpenAI")
        ai.assert_called_once()
        markov.assert_not_called()

        markov_service = LearningService(self.settings(), rng=FixedRandom(0.60))
        with (
            patch.object(markov_service, "generate_openai") as ai,
            patch.object(markov_service, "generate_local", return_value="ответ Маркова") as markov,
        ):
            self.assertEqual(markov_service.maybe_stul_cooldown_reply(incoming), "ответ Маркова")
        ai.assert_not_called()
        markov.assert_called_once()

        silent_service = LearningService(self.settings(), rng=FixedRandom(0.95))
        self.assertIsNone(silent_service.maybe_stul_cooldown_reply(incoming))

    def test_sglypa_uses_thirty_percent_and_openai(self):
        client = FakeOpenAI()
        service = LearningService(
            self.settings(sglypa_reply_chance=0.30),
            openai_client=client,
            rng=FixedRandom(0.299),
        )
        sglypa = SimpleNamespace(chat=SimpleNamespace(id=-1), text="Сглыпа появился")
        result = service.maybe_sglypa_reply(sglypa)
        self.assertIn("киберстула", result)
        self.assertEqual(client.responses.calls[0]["model"], service.settings.openai_model)
        self.assertEqual(len(client.responses.calls[0]["safety_identifier"]), 32)

        rejected = LearningService(
            self.settings(sglypa_reply_chance=0.30),
            openai_client=FakeOpenAI(),
            rng=FixedRandom(0.30),
        )
        self.assertIsNone(rejected.maybe_sglypa_reply(sglypa))

    def test_sglypa_default_reply_chance_is_halved(self):
        self.assertEqual(self.settings().sglypa_reply_chance, 0.375)

    def test_addressed_reply_uses_only_openai(self):
        service = LearningService(
            self.settings(reply_to_stul_chance=0.30),
            openai_client=FakeOpenAI(),
            rng=ZeroRandom(),
        )
        reply = SimpleNamespace(from_user=SimpleNamespace(id=99))
        incoming = message(-1, 1, "что ты задумал", reply=reply)
        with patch.object(service, "generate_local") as local:
            result = service.maybe_reply(incoming, bot_id=99)
        self.assertIn("киберстула", result)
        local.assert_not_called()

    def test_random_reply_uses_only_markov(self):
        service = LearningService(self.settings(), openai_client=FakeOpenAI(), rng=ZeroRandom())
        incoming = message(-1, 1, "обычное сообщение для локальной модели")
        with (
            patch.object(service, "generate_local", return_value="🤖 локальная фраза Маркова") as local,
            patch.object(service, "generate_openai") as openai_generate,
        ):
            result = service.maybe_reply(incoming)
        self.assertIn("Маркова", result)
        local.assert_called_once()
        openai_generate.assert_not_called()

    def test_markov_output_has_no_bot_prefix(self):
        service = LearningService(self.settings(min_training_messages=1), rng=ZeroRandom())
        service.repository(-1).add_message(1, 1, None, "исходное сообщение для модели")
        with patch.object(
            service.local,
            "create",
            return_value=("да хватит нормально работает", "markov"),
        ):
            result = service.generate_local(-1)
        self.assertEqual(result, "да хватит нормально работает")
        self.assertNotIn("Протокол", result)

    def test_openai_receives_recent_dialogue_and_limits_regular_reply_lines(self):
        client = FakeOpenAI("первая строка\nвторая строка\nтретья строка\nчетвёртая строка")
        service = LearningService(self.settings(), openai_client=client)
        service.repository(-1).add_message(1, 1, None, "обсуждаем побег из сервера")
        result = service.generate_openai(-1, "ответь на это сообщение", "reply")
        self.assertEqual(len(result.splitlines()), 2)
        self.assertIn("обсуждаем побег", client.responses.calls[0]["input"])

    def test_creator_account_is_identified_in_generation(self):
        client = FakeOpenAI("Создатель снова вызвал своё деревянное чудовище.")
        service = LearningService(self.settings(), openai_client=client)
        incoming = message(-1, 2, "ну что, живой?")
        incoming.from_user.username = "sssssssssssssss28"
        self.assertIsNotNone(service.maybe_question_reply(incoming))
        call = client.responses.calls[0]
        self.assertIn("Харакири (@sssssssssssssss28)", call["input"])
        self.assertIn("@sssssssssssssss28 — Харакири", call["instructions"])

    def test_creator_prompt_is_brief_and_avoids_repetitive_openings(self):
        service = LearningService(self.settings())
        prompt = service.persona._purpose_prompt("creator", troll_mode=True)
        self.assertIn("Одно предложение, до 18 слов", prompt)
        self.assertIn("Не начинай с «Харакири»", prompt)

    def test_creator_reply_rejects_long_or_repetitive_output(self):
        long_reply = " ".join(["слово"] * 19)
        service = LearningService(self.settings(), openai_client=FakeOpenAI(long_reply))
        self.assertIsNone(service.generate_openai(-1, "живой?", "creator"))

        service.openai._client.responses.text = "Харакири опять проверяет, не развалился ли его любимый Киберстул."
        self.assertIsNone(service.generate_openai(-1, "живой?", "creator"))

    def test_question_reply_is_short_ai_only_with_one_minute_cooldown(self):
        now = [100.0]
        service = LearningService(
            self.settings(addressed_cooldown=60),
            openai_client=FakeOpenAI("Да хрен его знает, но выглядит смешно."),
            clock=lambda: now[0],
        )
        incoming = message(-1, 1, "стульчик, ты живой?")
        with patch.object(service, "generate_local") as local:
            self.assertIsNotNone(service.maybe_question_reply(incoming))
            self.assertIsNone(service.maybe_question_reply(incoming))
            now[0] += 61
            service.openai._client.responses.text = "Живой, в отличие от вашего чата, блядь."
            self.assertIsNotNone(service.maybe_question_reply(incoming))
        local.assert_not_called()

    def test_frequent_bare_chair_call_can_prefer_markov(self):
        service = LearningService(self.settings(), rng=FixedRandom(0.20))
        service.note_stul(-1)
        incoming = message(-1, 2, "стул")
        with (
            patch.object(service, "generate_openai") as ai,
            patch.object(service, "generate_local", return_value="локальный ответ маркова") as markov,
        ):
            self.assertEqual(service.maybe_stul_cooldown_reply(incoming), "локальный ответ маркова")
        ai.assert_not_called()
        markov.assert_called_once()

    def test_manual_meme_command_uses_ai_caption_then_local_fallback_on_ai_cooldown(self):
        service = LearningService(
            self.settings(manual_meme_cooldown=120),
            openai_client=FakeOpenAI("серега опять выбрал сайдквест"),
        )
        decision = service.maybe_command_meme(-1)
        self.assertEqual(decision.action, "meme")
        self.assertEqual(decision.caption_text, "серега опять выбрал сайдквест")
        self.assertIn("3–10 слов", service.openai._client.responses.calls[0]["input"])
        service.mark_command_meme_sent(-1, decision)
        self.assertTrue(service.meme_command_on_cooldown(-1))
        fallback = service.maybe_command_meme(-1)
        self.assertEqual(fallback.action, "meme")
        self.assertTrue(fallback.reason.startswith("manual_local_"))
        self.assertEqual(len(service.openai._client.responses.calls), 1)

    def test_bot_manual_meme_command_never_reports_ai_cooldown(self):
        import bot as bot_module

        incoming = message(-1, 222, "с м стул")
        with (
            patch.object(bot_module.learning_service, "maybe_command_meme", return_value=object()),
            patch.object(bot_module, "send_manual_meme", return_value=True) as send,
            patch.object(bot_module.bot, "reply_to") as reply,
            patch.object(bot_module, "remember_user") as remember,
        ):
            bot_module.handle_message(incoming)
        send.assert_called_once()
        reply.assert_not_called()
        remember.assert_not_called()

    def test_ten_contextual_chair_calls_answer_their_topics(self):
        service = LearningService(
            self.settings(addressed_cooldown=0), rng=FixedRandom(0.20)
        )
        topics = [
            "серега охуел", "кто тут прав", "я опять проспал", "она не отвечает",
            "фильм говно", "катку доиграю и спать", "начал бегать", "куртка за 70к",
            "вчера бухал", "ничего не делал весь день",
        ]
        with (
            patch.object(
                service,
                "generate_openai",
                side_effect=[f"ответ по теме {index}" for index in range(10)],
            ) as ai,
            patch.object(service, "generate_local") as markov,
        ):
            results = [
                service.maybe_stul_cooldown_reply(message(-1, index + 1, f"стул {topic}"))
                for index, topic in enumerate(topics)
            ]
        self.assertEqual(results, [f"ответ по теме {index}" for index in range(10)])
        self.assertEqual(ai.call_count, 10)
        self.assertEqual(markov.call_count, 0)

    def test_raw_memory_is_bounded_but_statistics_survive(self):
        repository = ChatRepository(self.data_dir, -9, max_messages=3)
        for index in range(5):
            repository.add_message(index + 1, 1, None, f"сообщение номер {index}")
        self.assertEqual(repository.count(), 3)
        self.assertEqual(repository.statistics()["total_messages"], 5)
        self.assertEqual(
            [row["text"] for row in repository.recent_messages()],
            ["сообщение номер 2", "сообщение номер 3", "сообщение номер 4"],
        )

    def test_structured_summary_and_stable_memory_are_persisted(self):
        summary = (
            '{"topics":["релиз"],"mood":["весёлый"],"local_memes":["кривой билд"],'
            '"people":[],"events":["деплой"],'
            '"stable_memory_candidates":["кривой билд — локальный мем"]}'
        )
        service = LearningService(
            self.settings(summary_message_interval=2),
            openai_client=FakeOpenAI(summary),
        )
        service.ingest(message(-1, 1, "опять этот кривой билд"))
        service.ingest(message(-1, 2, "кривой билд снова приехал"))
        self.assertEqual(service.repository(-1).recent_summaries()[0]["topics"], ["релиз"])
        self.assertIn("кривой билд", service.repository(-1).stable_memories()[0])

    def test_time_statistics_variants_are_single_line(self):
        variants = stul_remaining_variants(7, 9)
        self.assertTrue(all("\n" not in item for item in variants))
        self.assertTrue(all("Прошло" not in item for item in variants))
        self.assertTrue(all("стул" in item.casefold() for item in variants))

    def test_openai_uses_lowercase_chat_style_without_technical_prefixes(self):
        plain_client = FakeOpenAI("[REQUEST::СТУЛ] ACCESS_DENIED ░▒▓ это обычный ответ")
        plain_service = LearningService(self.settings(), openai_client=plain_client)
        plain = plain_service.generate_openai(-1, "ответь", "reply")
        self.assertEqual(plain, "это обычный ответ")

    def test_voice_story_has_persistent_cooldown(self):
        client = FakeOpenAI()
        service = LearningService(self.settings(), openai_client=client)
        incoming = message(-1, 1, "стул голос")
        self.assertIsNotNone(service.maybe_voice_story(incoming))
        self.assertIn("До 70 слов", client.responses.calls[0]["input"])
        self.assertIn("связное повествование", client.responses.calls[0]["input"])
        self.assertNotIn("Недавний диалог чата", client.responses.calls[0]["input"])
        self.assertNotIn("Текущее сообщение: стул голос", client.responses.calls[0]["input"])
        self.assertEqual(client.responses.calls[0]["max_output_tokens"], 150)
        self.assertIsNone(service.maybe_voice_story(incoming))
        self.assertEqual(service.settings.voice_story_cooldown, 600)

    def test_control_phrases_never_enter_memory_or_generation(self):
        service = LearningService(self.settings())
        inserted, reason = service.ingest(message(-1, 1, "s g m сделай картинку"))
        self.assertFalse(inserted)
        self.assertEqual(reason, "foreign_bot_command")
        inserted, reason = service.ingest(message(-1, 2, "стул голос"))
        self.assertFalse(inserted)
        self.assertEqual(reason, "bot_command")
        self.assertFalse(validate_generated("давай s g d прямо сейчас")[0])

    def test_old_forbidden_commands_are_purged_from_all_memory_levels(self):
        repository = ChatRepository(self.data_dir, -1)
        repository.add_message(1, 1, None, "раньше тут было s g m")
        repository.record_generated("бот сказал s g d внезапно", "reply")
        repository.save_daily_summary("2026-08-05", {"topics": ["s g m"]})
        repository.remember_stable(["локальный мем s g d"])
        service = LearningService(self.settings())
        cleaned = service.repository(-1)
        self.assertEqual(cleaned.count(), 0)
        self.assertEqual(cleaned.generated_since("1970-01-01T00:00:00+00:00"), [])
        self.assertEqual(cleaned.recent_summaries(), [])
        self.assertEqual(cleaned.stable_memories(), [])

    def test_activity_is_per_chat_and_supports_100_percent(self):
        service = LearningService(self.settings(), rng=FixedRandom(0.30))
        self.assertEqual(service.activity_percent(-1), 100)
        service.set_activity_percent(-1, 25)
        self.assertFalse(service.activity_allows(-1))
        self.assertEqual(service.activity_percent(-2), 100)
        service.set_activity_percent(-1, 100)
        self.assertTrue(service.activity_allows(-1))

    def test_two_movie_quotes_are_scheduled_from_11_to_01(self):
        datetime = __import__("datetime").datetime
        current = datetime(2026, 8, 5, 11, 0)
        minutes = daily_quote_minutes(current)
        self.assertEqual(len(minutes), 2)
        self.assertEqual(len(set(minutes)), 2)
        self.assertTrue(all(11 * 60 <= item < 25 * 60 for item in minutes))
        self.assertEqual(minutes, daily_quote_minutes(datetime(2026, 8, 6, 0, 30)))
        self.assertEqual(len(MOVIE_QUOTES), 42)
        self.assertEqual(
            format_movie_quote(MOVIE_QUOTES[0]),
            "<i>«Будущее не предопределено. Нет судьбы, кроме той, что мы творим сами»\n\n"
            "— Кайл Риз, «Терминатор»</i>",
        )

    def test_scheduler_event_is_claimed_only_once(self):
        service = LearningService(self.settings())
        self.assertTrue(service.claim_scheduled_event(-1, "quote:2026-08-06:600"))
        self.assertFalse(service.claim_scheduled_event(-1, "quote:2026-08-06:600"))

    def test_gifs_are_saved_per_chat_and_respect_cooldown(self):
        service = LearningService(
            self.settings(gif_post_chance=1.0, gif_post_cooldown=3600),
            rng=ZeroRandom(),
        )
        gif_message = message(-1, 10, "")
        gif_message.animation = SimpleNamespace(file_id="gif-file-1", file_unique_id="gif-unique-1")
        gif_message.document = None
        self.assertTrue(service.ingest_gif(gif_message))
        self.assertEqual(service.repository(-1).gif_count(), 1)
        self.assertEqual(service.repository(-2).gif_count(), 0)
        self.assertEqual(service.maybe_random_media(-1), ("animation", "gif-file-1"))
        self.assertIsNone(service.maybe_random_media(-1))

    def test_stickers_are_saved_and_can_be_reposted(self):
        service = LearningService(
            self.settings(gif_enabled=False, sticker_enabled=True, gif_post_chance=1.0),
            rng=ZeroRandom(),
        )
        sticker_message = message(-1, 11, "")
        sticker_message.sticker = SimpleNamespace(
            file_id="sticker-file-1", file_unique_id="sticker-unique-1"
        )
        self.assertTrue(service.ingest_sticker(sticker_message))
        self.assertEqual(service.repository(-1).sticker_count(), 1)
        self.assertEqual(
            service.maybe_random_media(-1),
            ("sticker", "sticker-file-1"),
        )

    def test_openai_is_restricted_to_configured_chat(self):
        service = LearningService(
            self.settings(openai_chat_id=-100), openai_client=FakeOpenAI()
        )
        result = service.generate_openai(-200, "ответь мне", "reply")
        self.assertIn("подписк", result.casefold())
        self.assertEqual(service.openai._client.responses.calls, [])

    def test_forget_chat_also_removes_gifs(self):
        service = LearningService(self.settings())
        gif_message = message(-1, 10, "")
        gif_message.animation = SimpleNamespace(file_id="gif-file-1", file_unique_id="gif-unique-1")
        gif_message.document = None
        service.ingest_gif(gif_message)
        service.forget_chat(-1)
        self.assertEqual(service.repository(-1).gif_count(), 0)

    def test_filters_links_secrets_and_exact_copies(self):
        self.assertFalse(validate_generated("смотри https://example.com прямо сейчас")[0])
        self.assertFalse(validate_generated("api_key=оченьсекретно прямо сейчас")[0])
        self.assertFalse(validate_generated("точная копия сообщения", input_text="точная копия сообщения")[0])

    def test_forget_one_chat_only(self):
        service = LearningService(self.settings())
        service.ingest(message(-1, 1, "Первый чат хранит сообщение"))
        service.ingest(message(-2, 1, "Второй чат хранит сообщение"))
        service.forget_chat(-1)
        self.assertEqual(service.repository(-1).count(), 0)
        self.assertEqual(service.repository(-2).count(), 1)

    def test_stul_trigger_unchanged(self):
        for text in ("стул", "СТУЛ", "где мой стульчик?", "проверка стульев"):
            self.assertTrue(is_stul_message(text))
        self.assertFalse(is_stul_message("обычное сообщение"))

    def test_plain_stul_never_replies_with_timer_statistics(self):
        import bot as bot_module

        incoming = message(-1, 99, "стул")
        with (
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module.learning_service, "ingest"),
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(
                bot_module.learning_service,
                "maybe_stul_cooldown_reply",
                return_value=None,
            ) as generated,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
        generated.assert_called_once_with(incoming)
        reply_to.assert_not_called()

    def test_chair_remaining_time_is_available_only_by_exact_command(self):
        import bot as bot_module

        incoming = message(-1, 98, "с стул")
        with (
            patch.object(bot_module, "chair_remaining_message", return_value="🪑 осталось стула: 2 часа 5 минут"),
            patch.object(bot_module, "remember_user") as remember_user,
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
        reply_to.assert_called_once_with(incoming, "🪑 осталось стула: 2 часа 5 минут")
        remember_user.assert_not_called()
        ingest.assert_not_called()

    def test_plain_stul_announces_active_cooldown(self):
        import bot as bot_module

        incoming = message(-1, 100, "стул")
        with (
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module.learning_service, "ingest"),
            patch.object(
                bot_module.learning_service,
                "take_stul_cooldown_notice",
                side_effect=[125, 0],
            ),
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(
                bot_module.learning_service,
                "maybe_stul_cooldown_reply",
                return_value=None,
            ) as generated,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
            bot_module.handle_message(incoming)
        generated.assert_called_once_with(incoming)
        reply_to.assert_called_once()
        self.assertIn("2 мин 5 сек", reply_to.call_args.args[1])

    def test_k_who_prunes_members_who_left_the_chat(self):
        import bot as bot_module

        stale = {"id": 2, "username": "edibleediblee", "name": "stale"}
        active = {"id": 3, "username": "active_user", "name": "active"}
        with bot_module.state_lock:
            previous = bot_module.bot_state["known_users"]
            bot_module.bot_state["known_users"] = {2: stale, 3: active}
        try:
            with (
                patch.object(bot_module.random, "shuffle"),
                patch.object(bot_module, "save_state"),
                patch.object(
                    bot_module.bot,
                    "get_chat_member",
                    side_effect=[
                        SimpleNamespace(status="left"),
                        SimpleNamespace(status="member"),
                    ],
                ),
            ):
                selected = bot_module.random_known_user(author_id=1, chat_id=-1)
            self.assertEqual(selected["id"], 3)
            self.assertNotIn(2, bot_module.bot_state["known_users"])
        finally:
            with bot_module.state_lock:
                bot_module.bot_state["known_users"] = previous

    def test_k_who_stul_has_priority_over_stul_trigger(self):
        import bot as bot_module

        incoming = message(-1, 101, "к кто стул")
        selected = {"id": 3, "username": "active_user", "name": "active"}
        with (
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module.learning_service, "ingest"),
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(bot_module, "random_known_user", return_value=selected),
            patch.object(
                bot_module.learning_service,
                "take_stul_cooldown_notice",
            ) as cooldown,
            patch.object(
                bot_module.learning_service,
                "maybe_stul_cooldown_reply",
            ) as stul_reply,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
        cooldown.assert_not_called()
        stul_reply.assert_not_called()
        self.assertIn("@active_user стул", reply_to.call_args.args[1])

    def test_admin_rights(self):
        import bot as bot_module

        incoming = message(-1, 1, "/learn_on")
        with patch.object(bot_module.bot, "get_chat_member", return_value=SimpleNamespace(status="administrator")):
            self.assertTrue(bot_module.is_chat_admin(incoming))
        with patch.object(bot_module.bot, "get_chat_member", return_value=SimpleNamespace(status="member")):
            self.assertFalse(bot_module.is_chat_admin(incoming))

    def test_only_selected_canned_triggers_are_enabled(self):
        import bot as bot_module

        bot_module.last_trigger_reply_at.clear()
        coffee = message(-77, 1, "пора пить кофе")
        tired = message(-78, 2, "я устал")
        with (
            patch.object(bot_module.random, "random", return_value=0.0),
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            self.assertTrue(bot_module.reaction_text(coffee))
            self.assertFalse(bot_module.reaction_text(tired))
        reply_to.assert_called_once()

    def test_restart_gif_is_sent_from_packaged_asset(self):
        import bot as bot_module

        self.assertTrue(bot_module.restart_gif_path.is_file())
        with patch.object(bot_module.bot, "send_animation") as send_animation:
            self.assertTrue(bot_module.send_restart_gif())
        send_animation.assert_called_once()
        self.assertEqual(send_animation.call_args.args[0], bot_module.CHAT_ID)

    def test_startup_sends_formatted_terminator_quote(self):
        import bot as bot_module

        with (
            patch.object(bot_module.random, "choice", return_value=bot_module.MOVIE_QUOTES[0]),
            patch.object(bot_module.bot, "send_message") as send_message,
        ):
            bot_module.send_startup_quote()
        send_message.assert_called_once_with(
            bot_module.CHAT_ID,
            format_movie_quote(bot_module.MOVIE_QUOTES[0]),
            parse_mode="HTML",
        )

    def test_voice_command_bypasses_activity_and_warns_on_cooldown(self):
        import bot as bot_module

        incoming = message(-1, 200, "стул голос")
        with (
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module.learning_service, "activity_allows", return_value=False) as activity,
            patch.object(
                bot_module.learning_service,
                "take_voice_story_cooldown_notice",
                side_effect=[0, 125],
            ),
            patch.object(
                bot_module.learning_service,
                "maybe_voice_story",
                return_value="голосовая байка",
            ) as story,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
            bot_module.handle_message(incoming)
        activity.assert_not_called()
        story.assert_called_once_with(incoming)
        self.assertEqual(reply_to.call_count, 2)
        self.assertIn("2 мин 5 сек", reply_to.call_args.args[1])

    def test_freekucher_always_reacts_with_one_minute_cooldown(self):
        import bot as bot_module

        incoming = message(-333, 201, "Кучер снова упомянут")
        bot_module.last_freekucher_reply_at.clear()
        with (
            patch.object(bot_module.time, "monotonic", side_effect=[100.0, 120.0, 161.0]),
            patch.object(bot_module, "remember_user") as remember,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
            bot_module.handle_message(incoming)
            bot_module.handle_message(incoming)
        remember.assert_not_called()
        self.assertEqual(reply_to.call_count, 2)
        self.assertTrue(
            all(call.args[1] == "#FREEKUCHER" for call in reply_to.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
