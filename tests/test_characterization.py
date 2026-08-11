import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from learning import (
    GenerateRequest,
    LLMProvider,
    LearningService,
    LearningSettings,
    GrokProvider,
    OpenAIGenerator,
    SummarizeRequest,
    create_llm_provider,
)
from learning.repository import ChatRepository


def message(chat_id, message_id, text, user_id=1, username=None, is_bot=False):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        text=text,
        date=1_700_000_000 + message_id,
        from_user=SimpleNamespace(
            id=user_id,
            username=username or f"user{user_id}",
            first_name=f"User {user_id}",
            is_bot=is_bot,
        ),
        reply_to_message=None,
    )


class FixedRandom:
    def random(self):
        return 0.0

    def choice(self, values):
        return values[0]


class FailingResponses:
    def create(self, **kwargs):
        raise RuntimeError("provider unavailable")


class RecordingProvider:
    available = True

    def __init__(self):
        self.generate_requests = []
        self.summarize_requests = []

    def generate(self, request):
        self.generate_requests.append(request)
        return "нормальный короткий ответ стула"

    def summarize(self, request):
        self.summarize_requests.append(request)
        return {
            "topics": ["релиз"],
            "mood": [],
            "local_memes": [],
            "people": [],
            "events": [],
            "stable_memory_candidates": [],
        }


class CharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, **overrides):
        values = {
            "data_dir": self.data_dir,
            "min_training_messages": 1,
            "addressed_cooldown": 0,
            "generated_cooldown": 0,
            "max_generated_per_hour": 10,
            "openai_chat_id": -1,
        }
        values.update(overrides)
        return LearningSettings(**values)

    def test_telegram_routing_keeps_freekucher_ahead_of_foreign_command(self):
        import bot as bot_module

        incoming = message(-1, 1, "Кучер, s g m нарисуй это")
        bot_module.last_freekucher_reply_at.clear()
        with (
            patch.object(bot_module.time, "monotonic", return_value=100.0),
            patch.object(bot_module, "remember_user") as remember,
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
        reply_to.assert_called_once_with(incoming, "#FREEKUCHER")
        remember.assert_not_called()
        ingest.assert_not_called()

    def test_ordinary_message_is_remembered_ingested_and_routed_to_reply(self):
        import bot as bot_module

        incoming = message(-1, 2, "обычное сообщение без специальных слов")
        with (
            patch.object(bot_module, "remember_user") as remember,
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(bot_module, "reaction_text", return_value=False),
            patch.object(bot_module, "get_bot_identity", return_value={"id": 7, "username": "chair"}),
            patch.object(bot_module.learning_service, "maybe_reply", return_value="обычный ответ") as maybe_reply,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
        remember.assert_called_once_with(incoming)
        ingest.assert_called_once_with(incoming)
        maybe_reply.assert_called_once_with(incoming, bot_id=7, bot_username="chair")
        reply_to.assert_called_once_with(incoming, "обычный ответ")

    def test_creator_messages_use_the_normal_random_reply_limit(self):
        import bot as bot_module

        incoming = message(
            -1, 4, "ну что, живой?", username=bot_module.learning_settings.creator_username
        )
        with (
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module.learning_service, "ingest"),
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(bot_module.learning_service, "maybe_special_ai", return_value="коротко") as special,
            patch.object(bot_module.bot, "reply_to") as reply_to,
        ):
            bot_module.handle_message(incoming)
        special.assert_called_once_with(
            incoming,
            "random",
            bot_module.learning_settings.random_reply_chance,
            "creator",
            addressed=False,
        )
        reply_to.assert_called_once_with(incoming, "коротко")

    def test_creator_mentions_do_not_trigger_a_privileged_reply(self):
        import bot as bot_module

        mention = message(-1, 5, "Харакири, ты тут?", username="another_user")
        self.assertFalse(bot_module.is_creator_message(mention))

    def test_special_sglypa_route_does_not_enter_regular_message_flow(self):
        import bot as bot_module

        incoming = message(-1, 3, "реплика Сглыпы", username=bot_module.SGLYPA_USERNAME, is_bot=True)
        with (
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(bot_module, "sglypa_reaction", return_value=True) as reaction,
            patch.object(bot_module, "remember_user") as remember,
            patch.object(bot_module.learning_service, "ingest") as ingest,
        ):
            bot_module.handle_message(incoming)
        reaction.assert_called_once_with(incoming)
        remember.assert_not_called()
        ingest.assert_not_called()

    def test_main_startup_initializes_activity_messages_scheduler_and_polling(self):
        import bot as bot_module

        fake_thread = Mock()
        with (
            patch.object(bot_module, "TOKEN", "123:configured"),
            patch.dict(bot_module.os.environ, {"XAI_API_KEY": "configured"}),
            patch.object(bot_module, "send_startup_quote") as quote,
            patch.object(bot_module, "send_restart_gif") as gif,
            patch.object(bot_module.threading, "Thread", return_value=fake_thread) as thread,
            patch.object(bot_module.bot, "infinity_polling", side_effect=KeyboardInterrupt) as polling,
        ):
            with self.assertRaises(KeyboardInterrupt):
                bot_module.main()
        quote.assert_not_called()
        gif.assert_not_called()
        thread.assert_called_once()
        self.assertTrue(thread.call_args.kwargs["daemon"])
        fake_thread.start.assert_called_once_with()
        polling.assert_called_once_with(skip_pending=True, timeout=30, long_polling_timeout=30)

    def _run_scheduler_at(self, current):
        import scheduler as scheduler_module

        scheduler_module._last_event = None
        scheduler_module._last_quote_event = None
        scheduler_module._next_random_at = current + timedelta(hours=1)
        fake_bot = Mock()
        with (
            patch.object(scheduler_module, "get_now", return_value=current),
            patch.object(scheduler_module, "is_workday", return_value=True),
            patch.object(scheduler_module, "daily_quote_minutes", return_value=[]),
            patch.object(scheduler_module.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(KeyboardInterrupt):
                scheduler_module.scheduler(
                    fake_bot,
                    -1,
                    "Europe/Moscow",
                    9,
                    0,
                    17,
                    30,
                )
        return fake_bot

    def test_scheduler_start_event_sends_start_message(self):
        bot = self._run_scheduler_at(
            datetime(2026, 8, 11, 9, 0, tzinfo=ZoneInfo("Europe/Moscow"))
        )
        bot.send_message.assert_called_once()

    def test_scheduler_end_event_sends_end_message(self):
        bot = self._run_scheduler_at(
            datetime(2026, 8, 11, 17, 30, tzinfo=ZoneInfo("Europe/Moscow"))
        )
        bot.send_message.assert_called_once()

    def test_scheduler_never_sends_remaining_time_automatically(self):
        current = datetime(2026, 8, 11, 12, 15, tzinfo=ZoneInfo("Europe/Moscow"))
        bot = self._run_scheduler_at(current)
        bot.send_message.assert_not_called()

    def test_openai_failure_returns_none_without_local_fallback(self):
        client = SimpleNamespace(responses=FailingResponses())
        service = LearningService(self.settings(), openai_client=client)
        with patch.object(service, "generate_local") as local:
            self.assertIsNone(service.generate_openai(-1, "ответь на вопрос", "reply"))
        local.assert_not_called()

    def test_local_generation_uses_only_local_generator(self):
        service = LearningService(self.settings(), rng=FixedRandom())
        service.ingest(message(-1, 10, "исходное нормальное сообщение для модели"))
        with (
            patch.object(service.local, "create", return_value=("локальный нормальный ответ", "markov")),
            patch.object(service.llm_provider, "generate") as llm_generate,
        ):
            self.assertEqual(service.generate_local(-1), "локальный нормальный ответ")
        llm_generate.assert_not_called()

    def test_existing_hour_limit_blocks_after_configured_count(self):
        service = LearningService(self.settings(max_generated_per_hour=1), rng=FixedRandom())
        engine = service.triggers
        self.assertEqual(engine.decide_user_reply(-1), "random")
        engine.commit(-1, "random")
        self.assertIsNone(engine.decide_user_reply(-1))

    def test_memory_save_and_read_round_trip(self):
        repository = ChatRepository(self.data_dir, -50)
        repository.add_message(1, 9, "tester", "память сохраняет эту реплику")
        repository.save_daily_summary("2026-08-11", {"topics": ["релиз"]})
        repository.remember_stable(["старый локальный мем"])
        self.assertEqual(repository.recent_messages()[0]["text"], "память сохраняет эту реплику")
        self.assertEqual(repository.recent_summaries()[0]["topics"], ["релиз"])
        self.assertEqual(repository.stable_memories(), ["старый локальный мем"])

    def test_provider_boundary_receives_prepared_requests(self):
        provider = RecordingProvider()
        self.assertIsInstance(provider, LLMProvider)
        service = LearningService(self.settings(summary_message_interval=1), llm_provider=provider)
        service.ingest(message(-1, 20, "обсуждаем релиз и локальный контекст"))
        result = service.generate_llm(-1, "что с релизом", "reply")
        self.assertEqual(result, "нормальный короткий ответ стула")
        self.assertIsInstance(provider.generate_requests[0], GenerateRequest)
        self.assertIn("Сжатая память чата", provider.generate_requests[0].input)
        self.assertIsInstance(provider.summarize_requests[0], SummarizeRequest)

    def test_provider_factory_selects_default_grok_and_rejects_unknown_provider(self):
        provider = create_llm_provider(self.settings())
        self.assertIsInstance(provider, GrokProvider)
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            create_llm_provider(self.settings(llm_provider="unknown"))


if __name__ == "__main__":
    unittest.main()
