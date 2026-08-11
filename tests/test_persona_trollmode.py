import inspect
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    ChatState,
    ConversationDecision,
    LearningService,
    LearningSettings,
    MemeLexicon,
    PersonaBuilder,
)
from learning.openai_generator import OpenAIGenerator
from learning.repository import ChatRepository


class RecordingProvider:
    available = True

    def __init__(self):
        self.generate_requests = []
        self.summarize_requests = []

    def generate(self, request):
        self.generate_requests.append(request)
        return "релиз переносим на пятницу"

    def summarize(self, request):
        self.summarize_requests.append(request)
        return None


def state(kind="humor", activity="high", topic="релиз"):
    return ChatState(
        activity, 10, kind, topic, .8, .8 if kind == "humor" else .1,
        .8 if kind == "argument" else .1, .8 if kind == "serious" else .1,
        .8 if kind == "work" else .1, .4, 4, 123, 7, .85,
    )


def decision(intensity=.8, style="absurd_short"):
    return ConversationDecision(
        "reply", .3, intensity, 30, style, 123, 7, "synthetic", 0, .3
    )


def message(chat_id, message_id, text):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id), message_id=message_id, text=text,
        date=datetime.now(timezone.utc).timestamp() + message_id,
        from_user=SimpleNamespace(id=7, username="tester", is_bot=False),
        reply_to_message=None,
    )


class PersonaAndTrollModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.settings = LearningSettings(
            data_dir=self.data_dir, openai_chat_id=-1, min_training_messages=1,
            addressed_cooldown=0, generated_cooldown=0, max_generated_per_hour=10,
        )

    def tearDown(self):
        self.temp.cleanup()

    def builder(self):
        return PersonaBuilder(self.settings, MemeLexicon())

    def build(self, **kwargs):
        values = dict(
            chat_id=-1, context="релиз опять сломался", purpose="reply",
            conversation_decision=decision(), chat_state=state(), troll_mode=True,
        )
        values.update(kwargs)
        return self.builder().build_request(**values)

    def test_troll_mode_on_uses_cyberchair_persona(self):
        result = self.build()
        self.assertIn("CyberChair / chairOS by id:28", result.request.instructions)

    def test_chair_address_is_not_sent_as_the_subject(self):
        request = self.build(context="стул серега охуел?").request
        self.assertIn("Смысл целевого сообщения: серега охуел?", request.input)
        self.assertNotIn("Целевое сообщение: стул", request.input)
        self.assertIn("не строй реплику вокруг того, что тебя позвали", request.instructions)

    def test_sglypa_reply_prompt_forbids_stock_introduction(self):
        result = self.build(purpose="sglypa")
        self.assertIn("Не начинай с", result.request.input)
        self.assertIn("сразу бей по сути", result.request.input)
        self.assertEqual(result.request.max_output_tokens, 50)

    def test_meme_caption_is_short_and_has_no_emoji(self):
        request = self.build(purpose="meme_caption").request
        self.assertIn("3–8 слов", request.input)
        self.assertIn("без emoji", request.input)
        self.assertEqual(request.max_output_tokens, 30)
        self.assertIn("не заканчивай ими сообщения автоматически", request.instructions)

    def test_troll_mode_off_uses_neutral_persona(self):
        result = self.build(troll_mode=False)
        self.assertNotIn("офисный стул-киборг", result.request.instructions)
        self.assertIn("краткий участник рабочего", result.request.instructions)

    def test_troll_mode_off_does_not_pass_imageboard_lexicon(self):
        result = self.build(context="sigma based soyjak", troll_mode=False)
        self.assertNotIn("Доступные русифицированные мемы", result.request.input)
        self.assertEqual(result.meme_ids, ())

    def test_troll_mode_off_preserves_llm_work_reply(self):
        provider = RecordingProvider()
        service = LearningService(self.settings, llm_provider=provider)
        service.set_troll_mode(-1, False)
        self.assertEqual(
            service.generate_llm(-1, "стул что решили по релизу", "question"),
            "релиз переносим на пятницу",
        )
        self.assertIn("troll_off", provider.generate_requests[0].input)

    def test_troll_mode_off_routes_explicit_chair_address_without_question_mark(self):
        import bot as bot_module

        incoming = message(-1, 55, "стул что решили по релизу")
        with (
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module.learning_service, "ingest"),
            patch.object(bot_module.learning_service, "troll_mode", return_value=False),
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(bot_module.learning_service, "maybe_question_reply", return_value="релиз в пятницу") as useful,
            patch.object(bot_module.bot, "reply_to") as reply,
        ):
            bot_module.handle_message(incoming)
        useful.assert_called_once_with(incoming)
        reply.assert_called_once_with(incoming, "релиз в пятницу")

    def test_low_and_high_intensity_have_different_instructions(self):
        low = self.build(conversation_decision=decision(.2)).request.instructions
        high = self.build(conversation_decision=decision(.9)).request.instructions
        self.assertNotEqual(low, high)
        self.assertIn("почти нейтральная", low)
        self.assertIn("Максимальная интенсивность", high)

    def test_preferred_style_enters_generation_context(self):
        request = self.build(conversation_decision=decision(.8, "direct_mocking")).request
        self.assertIn("предпочтительный стиль: direct_mocking", request.input)
        self.assertIn("прицельно высмей", request.instructions)

    def test_conversation_type_enters_generation_context(self):
        request = self.build(chat_state=state("argument")).request
        self.assertIn("тип разговора: argument", request.input)
        self.assertEqual(request.metadata["conversation_type"], "argument")

    def test_local_callback_precedes_generic_meme(self):
        result = self.build(
            context="серега снова чинит принтер и релиз",
            day_summary={"callback_jokes": ["серега опять чинит принтер"]},
        )
        callback_at = result.request.input.index("Локальные callbacks")
        meme_at = result.request.input.find("Доступные русифицированные мемы")
        self.assertTrue(meme_at == -1 or callback_at < meme_at)
        self.assertLessEqual(len(result.meme_ids), 1)

    def test_recent_meme_receives_cooldown(self):
        builder = self.builder()
        kwargs = dict(
            chat_id=-1, context="релиз сломался и нужен роллбек", purpose="reply",
            conversation_decision=decision(.8, "work_sarcastic"),
            chat_state=state("work"), troll_mode=True,
        )
        first = builder.build_request(**kwargs)
        builder.record_usage(-1, first.meme_ids, first.cooldown_groups)
        second = builder.build_request(**kwargs)
        self.assertTrue(set(first.meme_ids).isdisjoint(second.meme_ids))

    def test_service_does_not_repeat_same_generic_meme_consecutively(self):
        provider = RecordingProvider()
        service = LearningService(self.settings, llm_provider=provider)
        for _ in range(2):
            service.generate_llm(
                -1, "релиз сломался нужен роллбек", "reply",
                decision(.8, "work_sarcastic"), state("work"),
            )
        first = set(provider.generate_requests[0].metadata["selected_meme_ids"])
        second = set(provider.generate_requests[1].metadata["selected_meme_ids"])
        self.assertTrue(first.isdisjoint(second))

    def test_lexicon_filters_by_context(self):
        lexicon = MemeLexicon()
        entries = lexicon.select("", {"work", "overengineering"}, .5, limit=8)
        self.assertTrue(entries)
        self.assertTrue(all(set(item.contexts) & {"work", "overengineering"} for item in entries))

    def test_whole_lexicon_is_not_sent_to_llm(self):
        result = self.build()
        self.assertLessEqual(len(result.meme_ids), 2)
        self.assertGreater(len(MemeLexicon().entries), 40)

    def test_english_alias_is_recognized(self):
        ids = {item.id for item in MemeLexicon().recognize("bro is absolutely cooked")}
        self.assertIn("cooked", ids)

    def test_generation_uses_russian_output_not_alias(self):
        result = self.build(context="bro is absolutely cooked")
        meme_section = result.request.input.split("Доступные русифицированные мемы", 1)[1]
        self.assertIn("кукд", meme_section)
        self.assertNotIn("cooked:", meme_section)

    def test_sigma_output_is_russian(self):
        self.assertEqual(MemeLexicon().recognize("sigma")[0].output, "сигма")

    def test_based_output_is_russian(self):
        self.assertEqual(MemeLexicon().recognize("based")[0].output, "бейсд")

    def test_soyjak_output_is_russian(self):
        self.assertEqual(MemeLexicon().recognize("soyjak")[0].output, "сойджак")

    def test_gigachad_output_is_russian(self):
        self.assertEqual(MemeLexicon().recognize("gigachad")[0].output, "гигачад")

    def test_brainrot_output_is_russian(self):
        self.assertEqual(MemeLexicon().recognize("brainrot")[0].output, "брейнрот")

    def test_troll_mode_survives_repository_reopen(self):
        service = LearningService(self.settings, llm_provider=RecordingProvider())
        service.set_troll_mode(-1, False)
        reopened = LearningService(self.settings, llm_provider=RecordingProvider())
        self.assertFalse(reopened.troll_mode(-1))

    def test_old_database_without_troll_setting_defaults_on(self):
        repository = ChatRepository(self.data_dir, -2)
        repository.set_setting("learning", "1")
        service = LearningService(self.settings, llm_provider=RecordingProvider())
        self.assertTrue(service.troll_mode(-2))
        self.assertEqual(service.repository(-2).setting("learning"), "1")

    def test_troll_mode_off_does_not_disable_scheduler_utility_state(self):
        service = LearningService(self.settings, llm_provider=RecordingProvider())
        service.set_troll_mode(-1, False)
        self.assertTrue(service.claim_scheduled_event(-1, "start:2026-08-11"))
        self.assertFalse(service.claim_scheduled_event(-1, "start:2026-08-11"))

    def test_troll_mode_off_disables_troll_media_only(self):
        service = LearningService(self.settings, llm_provider=RecordingProvider())
        service.set_troll_mode(-1, False)
        self.assertIsNone(service.maybe_random_media(-1))

    def test_troll_mode_off_does_not_disable_memory_service(self):
        service = LearningService(self.settings, llm_provider=RecordingProvider())
        service.set_troll_mode(-1, False)
        self.assertTrue(service.ingest(message(-1, 1, "обсуждаем рабочий релиз"))[0])
        self.assertIn("рабочий релиз", service._dialogue_context(-1, "релиз"))

    def test_openai_adapter_has_no_persona_business_logic(self):
        source = inspect.getsource(OpenAIGenerator)
        self.assertNotIn("CyberChair", source)
        self.assertNotIn("TrollMode", source)
        self.assertNotIn("MemeLexicon", source)

    def test_generation_adds_no_llm_calls(self):
        provider = RecordingProvider()
        service = LearningService(self.settings, llm_provider=provider)
        result = service.generate_llm(
            -1, "что с релизом", "reply", decision(.35, "dry_sarcastic"), state("serious")
        )
        self.assertEqual(result, "релиз переносим на пятницу")
        self.assertEqual(len(provider.generate_requests), 1)
        self.assertEqual(len(provider.summarize_requests), 0)


if __name__ == "__main__":
    unittest.main()
