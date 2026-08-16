import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from learning import LearningService, LearningSettings
from learning.pending_conversation import question_intent


class CountingProvider:
    available = True

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self.responses.pop(0) if self.responses else "продолженный полезный ответ, без потери контекста"

    def summarize(self, request):
        return None


class FixedRandom:
    def random(self):
        return 0.5


def message(text, message_id=1, user_id=7, reply=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=message_id, text=text, date=0,
        from_user=SimpleNamespace(id=user_id, username=f"u{user_id}", is_bot=False),
        reply_to_message=reply,
    )


def bot_message(message_id, text="между чем выбираешь"):
    return SimpleNamespace(
        message_id=message_id, text=text,
        from_user=SimpleNamespace(id=99, username="chair", is_bot=True),
    )


class PendingConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, responses=None, **overrides):
        values = dict(
            data_dir=Path(self.temp.name), openai_chat_id=-1,
            min_training_messages=20, direct_social_markov_share=0.0,
            pending_conversation_ttl_seconds=1200,
        )
        values.update(overrides)
        provider = CountingProvider(responses)
        service = LearningService(
            LearningSettings(**values), llm_provider=provider, rng=FixedRandom()
        )
        service.set_media_enabled(-1, False)
        return service, provider

    def test_answer_first_policy_and_useful_mode_are_in_prompt(self):
        service, provider = self.service(["ешь с небольшим профицитом и качайся, лил бро"])
        result = service.maybe_direct_reply(
            message("стул как набрать вес"), explicit_address=True
        )
        self.assertIn("профицитом", result)
        self.assertEqual(len(provider.calls), 1)
        request = provider.calls[0]
        self.assertIn("ANSWER FIRST", request.input)
        self.assertEqual(request.metadata["behavior_mode"], "useful_answer")
        self.assertNotIn("уточни контекст и желаемый результат — иначе", result)

    def test_opinion_question_is_substantive_and_calls_provider_once(self):
        service, provider = self.service(["я бы следил за x и y: у первого звук злее, второй быстрее растёт"])
        result = service.maybe_direct_reply(
            message("стул кто самый перспективный русский саундклауд рэпер"),
            explicit_address=True,
        )
        self.assertIn("следил", result)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(service._last_direct_decision[-1].intent, "substantive")

    def test_ambiguous_choice_creates_pending(self):
        service, provider = self.service(["между чем выбираешь, лил бро?"])
        service.maybe_direct_reply(message("стул что выбрать"), explicit_address=True)
        pending = service.pending_conversation(-1, 7)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.expected_type, "choices")
        self.assertEqual(pending.mode, "hard")
        self.assertEqual(len(provider.calls), 1)

    def test_how_to_wins_over_model_choice_clarification(self):
        service, provider = self.service(["между чем выбираешь, лил бро?"])
        result = service.maybe_direct_reply(
            message("так и как по итогу прославиться в рэпе, стул?"),
            explicit_address=True,
        )
        self.assertIn("между чем", result)
        self.assertEqual(question_intent("так и как по итогу прославиться в рэпе"), "how_to")
        self.assertEqual(service._last_direct_decision[-1].intent, "substantive")
        self.assertIsNone(service.pending_conversation(-1, 7))
        self.assertEqual(len(provider.calls), 1)

    def test_original_dialogue_no_longer_has_choice_continuation(self):
        service, provider = self.service([
            "сначала сделай узнаваемый звук, регулярно выпускай сниппеты и треки, ищи коллабы своего размера и играй лайвы",
        ])
        first = service.maybe_direct_reply(
            message("так и как по итогу прославиться в рэпе, стул?"),
            explicit_address=True,
        )
        self.assertIn("узнаваемый звук", first)
        self.assertIsNone(service.pending_conversation(-1, 7))
        self.assertIsNone(service.maybe_pending_continuation(message("между ничем", 2), bot_id=99))
        self.assertEqual(len(provider.calls), 1)

    def test_how_to_choice_wording_never_creates_choices_pending(self):
        for text in ("стул как набрать вес?", "стул как выбрать микрофон?"):
            with self.subTest(text=text):
                service, _ = self.service(["между чем выбираешь?"])
                service.maybe_direct_reply(message(text), explicit_address=True)
                self.assertIsNone(service.pending_conversation(-1, 7))

    def test_concrete_choice_does_not_create_hard_choice_pending(self):
        for text in ("стул айфон или пиксель?", "стул что лучше X или Y?"):
            with self.subTest(text=text):
                service, _ = self.service(["между чем выбираешь?"])
                service.maybe_direct_reply(message(text), explicit_address=True)
                self.assertIsNone(service.pending_conversation(-1, 7))

    def test_choice_continuation_without_reply_is_required_and_one_call(self):
        service, provider = self.service([
            "между чем выбираешь, лил бро?",
            "пиксель бери ради камеры и чистого android, айфон — если экосистема уже засосала",
        ])
        service.maybe_direct_reply(message("стул что выбрать"), explicit_address=True)
        result = service.maybe_pending_continuation(message("айфон и пиксель", 2), bot_id=99)
        self.assertIn("пиксель", result)
        self.assertEqual(len(provider.calls), 2)
        self.assertIsNone(service.pending_conversation(-1, 7))
        self.assertIn("Исходный вопрос пользователя", provider.calls[-1].input)
        self.assertIn("айфон и пиксель", provider.calls[-1].input)

    def test_choice_decline_closes_pending_without_phantom_continuation(self):
        service, provider = self.service(["между чем выбираешь?"])
        service.maybe_direct_reply(message("стул что выбрать"), explicit_address=True)
        self.assertFalse(service.is_pending_continuation(message("между ничем", 2), bot_id=99))
        self.assertIsNone(service.maybe_pending_continuation(message("между ничем", 2), bot_id=99))
        self.assertIsNone(service.pending_conversation(-1, 7))
        self.assertEqual(len(provider.calls), 1)

    def test_local_choice_fallback_never_claims_unparsed_options(self):
        service, _ = self.service()
        pending = service.pending_conversation(-1, 7)
        self.assertIsNone(pending)
        from learning.pending_conversation import PendingConversation
        fake = PendingConversation(-1, 7, None, 1, "что выбрать", "между чем?", "substantive", "", "choices", "hard", datetime.now(timezone.utc))
        self.assertNotIn("из этих вариантов", service._local_continuation_fallback(fake, "первый"))

    def test_thirty_direct_question_intent_matrix(self):
        cases = (
            # how-to
            ("как прославиться в рэпе", "how_to"),
            ("как приготовить курицу", "how_to"),
            ("как поднять сервер", "how_to"),
            ("как познакомиться с девушкой", "how_to"),
            ("как набрать вес", "how_to"),
            ("как начать программировать", "how_to"),
            ("как починить docker", "how_to"),
            ("как найти работу", "how_to"),
            ("как мне заработать", "how_to"),
            ("так и как по итогу стать артистом", "how_to"),
            # factual / explanation
            ("почему docker падает", "factual"),
            ("что такое dns", "factual"),
            ("кто придумал биткоин", "factual"),
            ("где лежат логи nginx", "factual"),
            ("когда выйдет релиз", "factual"),
            ("сколько белка нужно", "factual"),
            ("зачем нужен redis", "factual"),
            # choice
            ("что выбрать", "choice"),
            ("какой выбрать", "choice"),
            ("айфон или пиксель", "choice"),
            ("что лучше X или Y", "choice"),
            ("какой из этих взять", "choice"),
            ("выбрать X или Y", "choice"),
            # open advice / opinion
            ("посоветуй микрофон", "open_advice"),
            ("что думаешь про новый альбом", "open_advice"),
            ("стоит ли брать подписку", "open_advice"),
            ("помоги с резюме", "open_advice"),
            ("какой ноутбук для работы", "open_advice"),
            ("как выбрать микрофон", "how_to"),
            ("что купить для записи", "open_advice"),
        )
        self.assertEqual(len(cases), 30)
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(question_intent(text), expected)

    def test_measurements_continue_original_weight_topic(self):
        service, provider = self.service([
            "держи +300 ккал и силовые. хочешь точнее — рост/вес кидай",
            "при 181/68 начни примерно с 2600 ккал и смотри среднюю за неделю",
        ])
        service.maybe_direct_reply(message("стул как набрать вес"), explicit_address=True)
        self.assertEqual(service.pending_conversation(-1, 7).mode, "soft")
        result = service.maybe_pending_continuation(message("181/68", 2), bot_id=99)
        self.assertIn("181/68", result)
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("как набрать вес", provider.calls[-1].input)

    def test_reply_to_pending_bot_message_is_strong_signal(self):
        service, provider = self.service(["между чем выбираешь?", "выбирай первый, он честнее по цене"])
        first = message("стул что выбрать")
        service.maybe_direct_reply(first, explicit_address=True)
        service.attach_pending_bot_message(first, bot_message(501))
        followup = message("первый вариант выглядит проще", 2, reply=bot_message(501))
        self.assertTrue(service.is_pending_continuation(followup, bot_id=99))
        self.assertTrue(service.maybe_pending_continuation(followup, bot_id=99))
        self.assertEqual(len(provider.calls), 2)

    def test_pending_is_isolated_by_user(self):
        service, provider = self.service(["между чем выбираешь?"])
        service.maybe_direct_reply(message("стул что выбрать", user_id=7), explicit_address=True)
        other = message("айфон и пиксель", 2, user_id=8)
        self.assertFalse(service.is_pending_continuation(other, bot_id=99))
        self.assertIsNone(service.maybe_pending_continuation(other, bot_id=99))
        self.assertEqual(len(provider.calls), 1)

    def test_expired_pending_is_not_continuation(self):
        service, _ = self.service()
        old = datetime.now(timezone.utc) - timedelta(minutes=21)
        service.repository(-1).save_pending_conversation(
            7, 1, "что выбрать", "между чем выбираешь?", "substantive",
            expected_type="choices", created_at=old,
        )
        self.assertFalse(service.is_pending_continuation(message("айфон и пиксель", 2), bot_id=99))
        self.assertIsNone(service.pending_conversation(-1, 7))

    def test_new_explicit_question_does_not_get_captured(self):
        service, _ = self.service()
        service.repository(-1).save_pending_conversation(
            7, 1, "расскажи", "какой вариант?", "substantive",
            expected_type="short_answer",
        )
        self.assertFalse(service.is_pending_continuation(
            message("стул как приготовить пасту", 2), bot_id=99
        ))

    def test_new_direct_topic_clears_old_pending(self):
        service, provider = self.service(["варианты кидай", "вари макароны 9–11 минут, не убивай их в кашу"])
        service.maybe_direct_reply(message("стул что выбрать"), explicit_address=True)
        service.maybe_direct_reply(
            message("стул как приготовить пасту", 2), explicit_address=True,
        )
        self.assertIsNone(service.pending_conversation(-1, 7))
        self.assertEqual(len(provider.calls), 2)

    def test_soft_pending_ignores_unrelated_chat_message(self):
        service, provider = self.service([
            "держи +300 ккал и силовые. хочешь точнее — рост/вес кидай",
        ])
        service.maybe_direct_reply(message("стул как набрать вес"), explicit_address=True)
        unrelated = message("ахах серёга опять уснул", 2)
        self.assertFalse(service.is_pending_continuation(unrelated, bot_id=99))
        self.assertIsNone(service.maybe_pending_continuation(unrelated, bot_id=99))
        self.assertEqual(len(provider.calls), 1)

    def test_provider_failure_still_answers_continuation(self):
        service, provider = self.service(["рост/вес кидай", None])
        service.maybe_direct_reply(message("стул как набрать вес"), explicit_address=True)
        result = service.maybe_pending_continuation(message("181/68", 2), bot_id=99)
        self.assertIn("имт", result)
        self.assertEqual(len(provider.calls), 2)


if __name__ == "__main__":
    unittest.main()
