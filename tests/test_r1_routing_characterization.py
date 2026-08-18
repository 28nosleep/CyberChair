import unittest
from types import SimpleNamespace
from unittest.mock import patch

from learning.normalized_event import NormalizedEvent


FIELDS = (
    "event_type", "effective_text", "direct", "reply_to_chair", "special",
    "pending_candidate", "ingested", "possible_llm", "final_producer",
    "early_return", "priority_rank",
)


# Frozen from the live pre-R1 bot.py order. These are observations, not a new
# routing policy. Some values are conditional and are deliberately written as
# such instead of pretending the adapter already made a domain decision.
CHARACTERIZATION_MATRIX = (
    dict(event_type="admin_command", effective_text="command text", direct=False, reply_to_chair=False, special=True, pending_candidate=False, ingested=False, possible_llm="/generate only", final_producer="admin text/local/llm", early_return=True, priority_rank=0),
    dict(event_type="callback_query", effective_text="", direct=False, reply_to_chair=False, special=True, pending_candidate=False, ingested=False, possible_llm=False, final_producer="callback UI", early_return=True, priority_rank=0),
    dict(event_type="freekucher", effective_text="message.text", direct="irrelevant", reply_to_chair="irrelevant", special=True, pending_candidate=False, ingested=False, possible_llm=False, final_producer="local text", early_return=True, priority_rank=1),
    dict(event_type="foreign_control", effective_text="message.text", direct="irrelevant", reply_to_chair="irrelevant", special=True, pending_candidate=False, ingested=False, possible_llm=False, final_producer="none", early_return=True, priority_rank=2),
    dict(event_type="manual_meme_text", effective_text="message.text", direct="not evaluated", reply_to_chair="not evaluated", special=True, pending_candidate=False, ingested=False, possible_llm=True, final_producer="meme", early_return=True, priority_rank=3),
    dict(event_type="manual_meme_photo_caption", effective_text="message.caption", direct="not evaluated", reply_to_chair=False, special=True, pending_candidate=False, ingested=False, possible_llm=True, final_producer="meme", early_return=True, priority_rank=3),
    dict(event_type="manual_meme_reply_image", effective_text="message.text", direct="not evaluated", reply_to_chair=False, special=True, pending_candidate=False, ingested=False, possible_llm=True, final_producer="meme", early_return=True, priority_rank=3),
    dict(event_type="chair_remaining", effective_text="message.text", direct="not evaluated", reply_to_chair=False, special=True, pending_candidate=False, ingested=False, possible_llm=False, final_producer="local text", early_return=True, priority_rank=4),
    dict(event_type="sglypa", effective_text="message.text", direct=False, reply_to_chair=False, special=True, pending_candidate=False, ingested=False, possible_llm=True, final_producer="sglypa text/none", early_return=True, priority_rank=5),
    dict(event_type="voice_story", effective_text="message.text", direct=False, reply_to_chair=False, special=True, pending_candidate=False, ingested=False, possible_llm=True, final_producer="text/none", early_return=True, priority_rank=6),
    dict(event_type="who_command", effective_text="message.text", direct=False, reply_to_chair=False, special=True, pending_candidate=False, ingested=True, possible_llm=False, final_producer="local text/none", early_return=True, priority_rank=7),
    dict(event_type="pending", effective_text="message.text", direct=False, reply_to_chair=False, special=False, pending_candidate=True, ingested=True, possible_llm=True, final_producer="llm/local text", early_return=True, priority_rank=8),
    dict(event_type="explicit_direct", effective_text="message.text", direct=True, reply_to_chair=False, special=False, pending_candidate=False, ingested=True, possible_llm=True, final_producer="llm/local/media", early_return=True, priority_rank=9),
    dict(event_type="reply_to_chair", effective_text="message.text", direct=True, reply_to_chair=True, special=False, pending_candidate="resolved inside direct lane", ingested=True, possible_llm=True, final_producer="pending/direct llm/local/media", early_return=True, priority_rank=9),
    dict(event_type="activity_sampling", effective_text="message.text", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested=True, possible_llm=False, final_producer="none when denied", early_return="conditional", priority_rank=10),
    dict(event_type="creator_lane", effective_text="message.text", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested=True, possible_llm=True, final_producer="text/none", early_return=True, priority_rank=11),
    dict(event_type="rare_trigger", effective_text="message.text", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested=True, possible_llm=False, final_producer="local text/none", early_return="when selected", priority_rank=12),
    dict(event_type="ordinary_policy", effective_text="message.text", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested=True, possible_llm=True, final_producer="none/local/llm/media", early_return=False, priority_rank=13),
    dict(event_type="photo", effective_text="message.caption or empty", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested="image metadata", possible_llm=False, final_producer="none", early_return=True, priority_rank=1),
    dict(event_type="image_document", effective_text="message.caption or empty", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested="image metadata", possible_llm=False, final_producer="none", early_return=True, priority_rank=1),
    dict(event_type="gif_document", effective_text="message.caption or empty", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested="image + GIF metadata", possible_llm=False, final_producer="none", early_return=True, priority_rank=1),
    dict(event_type="animation", effective_text="", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested="GIF metadata", possible_llm=False, final_producer="none", early_return=True, priority_rank=1),
    dict(event_type="sticker", effective_text="", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested="sticker metadata", possible_llm=False, final_producer="none", early_return=True, priority_rank=1),
    dict(event_type="unsupported", effective_text="", direct=False, reply_to_chair=False, special=False, pending_candidate=False, ingested=False, possible_llm=False, final_producer="none", early_return=True, priority_rank=99),
)


def user(user_id=7, username="tester", is_bot=False):
    return SimpleNamespace(
        id=user_id, username=username, first_name="Tester", is_bot=is_bot
    )


def message(text="обычный текст", message_id=1, *, caption=None, reply=None,
            content_type="text", username="tester", photo=None, document=None,
            animation=None, sticker=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1, type="supergroup"),
        message_id=message_id,
        text=text,
        caption=caption,
        content_type=content_type,
        date=1_776_000_000 + message_id,
        from_user=user(username=username),
        reply_to_message=reply,
        photo=photo,
        document=document,
        animation=animation,
        sticker=sticker,
    )


def chair_reply(message_id=900, *, photo=None, document=None):
    return SimpleNamespace(
        message_id=message_id,
        text="ответ CyberChair",
        caption=None,
        content_type="photo" if photo else "document" if document else "text",
        from_user=user(99, "chair", True),
        photo=photo,
        document=document,
    )


class R1RoutingCharacterizationTests(unittest.TestCase):
    def assert_normalized_call(self, mocked, source, **expected_kwargs):
        mocked.assert_called_once()
        event = mocked.call_args.args[0]
        self.assertIsInstance(event, NormalizedEvent)
        self.assertEqual(event.chat_id, source.chat.id)
        self.assertEqual(event.message_id, source.message_id)
        self.assertEqual(mocked.call_args.kwargs, expected_kwargs)
        return event

    def test_matrix_covers_every_required_event_class_and_column(self):
        self.assertEqual(len(CHARACTERIZATION_MATRIX), 24)
        for row in CHARACTERIZATION_MATRIX:
            with self.subTest(event_type=row["event_type"]):
                self.assertEqual(tuple(row), FIELDS)

    def test_text_priority_is_frozen(self):
        expected = (
            "admin_command", "callback_query", "freekucher", "foreign_control",
            "manual_meme_text", "manual_meme_photo_caption",
            "manual_meme_reply_image", "chair_remaining", "sglypa",
            "voice_story", "who_command", "pending", "explicit_direct",
            "reply_to_chair", "activity_sampling", "creator_lane",
            "rare_trigger", "ordinary_policy",
        )
        rows = sorted(CHARACTERIZATION_MATRIX[:18], key=lambda row: row["priority_rank"])
        self.assertEqual(tuple(row["event_type"] for row in rows), expected)

    def test_freekucher_wins_over_foreign_and_every_regular_lane(self):
        import bot as bot_module

        incoming = message("Кучер, s g m стул с м стул")
        with (
            patch.object(bot_module, "freekucher_reaction", return_value=True) as winner,
            patch.object(bot_module, "send_manual_meme") as meme,
            patch.object(bot_module, "remember_user") as remember,
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.learning_service, "maybe_reply") as ordinary,
        ):
            bot_module.handle_message(incoming)
        winner.assert_called_once()
        self.assertIsInstance(winner.call_args.args[1], NormalizedEvent)
        meme.assert_not_called()
        remember.assert_not_called()
        ingest.assert_not_called()
        ordinary.assert_not_called()

    def test_manual_meme_wins_over_pending_direct_and_ingest(self):
        import bot as bot_module

        incoming = message("с м стул", reply=chair_reply())
        with (
            patch.object(bot_module, "freekucher_reaction", return_value=False),
            patch.object(bot_module, "send_manual_meme", return_value=True) as meme,
            patch.object(bot_module.learning_service, "is_pending_continuation") as pending,
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.learning_service, "maybe_direct_reply") as direct,
        ):
            bot_module.handle_message(incoming)
        meme.assert_called_once()
        pending.assert_not_called()
        ingest.assert_not_called()
        direct.assert_not_called()

    def test_who_is_ingested_before_its_early_return(self):
        import bot as bot_module

        incoming = message("к кто сегодня отвечает")
        calls = []
        with (
            patch.object(bot_module, "freekucher_reaction", return_value=False),
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module, "get_bot_identity", return_value={"id": 99, "username": "chair"}),
            patch.object(bot_module.learning_service, "is_pending_continuation", return_value=False),
            patch.object(bot_module.learning_service, "ingest", side_effect=lambda value, **kwargs: calls.append("ingest")),
            patch.object(bot_module.learning_service, "troll_mode", return_value=True),
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(
                bot_module, "handle_who",
                side_effect=lambda value, text, event: calls.append("who"),
            ),
            patch.object(bot_module.learning_service, "maybe_reply") as ordinary,
        ):
            bot_module.handle_message(incoming)
        self.assertEqual(calls, ["ingest", "who"])
        ordinary.assert_not_called()

    def test_pending_beats_activity_creator_rare_and_ordinary(self):
        import bot as bot_module

        incoming = message("айфон и пиксель", username=bot_module.learning_settings.creator_username)
        with (
            patch.object(bot_module, "freekucher_reaction", return_value=False),
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module, "get_bot_identity", return_value={"id": 99, "username": "chair"}),
            patch.object(bot_module.learning_service, "is_pending_continuation", return_value=True),
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.learning_service, "maybe_pending_continuation", return_value="pending") as pending,
            patch.object(bot_module, "send_contextual_response") as send,
            patch.object(bot_module.learning_service, "activity_allows") as activity,
            patch.object(bot_module, "reaction_text") as rare,
            patch.object(bot_module.learning_service, "maybe_reply") as ordinary,
        ):
            bot_module.handle_message(incoming)
        self.assert_normalized_call(ingest, incoming, refresh_memory=False)
        pending_event = self.assert_normalized_call(pending, incoming, bot_id=99)
        send.assert_called_once_with(incoming, "pending", pending_event)
        activity.assert_not_called()
        rare.assert_not_called()
        ordinary.assert_not_called()

    def test_reply_to_chair_enters_direct_lane_before_activity(self):
        import bot as bot_module

        incoming = message("без слова-триггера", reply=chair_reply())
        with (
            patch.object(bot_module, "freekucher_reaction", return_value=False),
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module, "get_bot_identity", return_value={"id": 99, "username": "chair"}),
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.learning_service, "maybe_direct_reply", return_value="direct") as direct,
            patch.object(bot_module, "send_contextual_response") as send,
            patch.object(bot_module.learning_service, "activity_allows") as activity,
        ):
            bot_module.handle_message(incoming)
        self.assert_normalized_call(ingest, incoming, refresh_memory=False)
        direct.assert_called_once()
        direct_event = direct.call_args.args[0]
        self.assertIsInstance(direct_event, NormalizedEvent)
        send.assert_called_once_with(incoming, "direct", direct_event)
        activity.assert_not_called()

    def test_activity_denial_stops_creator_rare_and_ordinary_after_ingest(self):
        import bot as bot_module

        incoming = message("обычный текст", username=bot_module.learning_settings.creator_username)
        with (
            patch.object(bot_module, "freekucher_reaction", return_value=False),
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module, "get_bot_identity", return_value={"id": 99, "username": "chair"}),
            patch.object(bot_module.learning_service, "is_pending_continuation", return_value=False),
            patch.object(bot_module.learning_service, "ingest") as ingest,
            patch.object(bot_module.learning_service, "activity_allows", return_value=False),
            patch.object(bot_module.learning_service, "maybe_special_ai") as creator,
            patch.object(bot_module, "reaction_text") as rare,
            patch.object(bot_module.learning_service, "maybe_reply") as ordinary,
        ):
            bot_module.handle_message(incoming)
        self.assert_normalized_call(ingest, incoming, refresh_memory=True)
        creator.assert_not_called()
        rare.assert_not_called()
        ordinary.assert_not_called()

    def test_creator_beats_rare_and_ordinary(self):
        import bot as bot_module

        incoming = message("обычный текст", username=bot_module.learning_settings.creator_username)
        with (
            patch.object(bot_module, "freekucher_reaction", return_value=False),
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module, "get_bot_identity", return_value={"id": 99, "username": "chair"}),
            patch.object(bot_module.learning_service, "is_pending_continuation", return_value=False),
            patch.object(bot_module.learning_service, "ingest"),
            patch.object(bot_module.learning_service, "activity_allows", return_value=True),
            patch.object(bot_module.learning_service, "maybe_special_ai", return_value="creator") as creator,
            patch.object(bot_module.bot, "reply_to") as reply,
            patch.object(bot_module, "reaction_text") as rare,
            patch.object(bot_module.learning_service, "maybe_reply") as ordinary,
        ):
            bot_module.handle_message(incoming)
        creator.assert_called_once()
        reply.assert_called_once_with(incoming, "creator")
        rare.assert_not_called()
        ordinary.assert_not_called()

    def test_photo_caption_meme_owns_event_and_plain_photo_only_ingests_image(self):
        import bot as bot_module

        photo = [SimpleNamespace(file_id="large", file_unique_id="u", width=10, height=10)]
        command = message(None, 20, caption="с м стул", content_type="photo", photo=photo)
        plain = message(None, 21, caption="обычная подпись", content_type="photo", photo=photo)
        with (
            patch.object(bot_module, "send_manual_meme", return_value=True) as meme,
            patch.object(bot_module.learning_service, "ingest_chat_image") as ingest_image,
        ):
            bot_module.remember_photo(command)
            bot_module.remember_photo(plain)
        meme.assert_called_once()
        self.assert_normalized_call(ingest_image, plain)

    def test_image_document_gif_animation_and_sticker_keep_metadata_routes(self):
        import bot as bot_module

        gif_document = SimpleNamespace(
            file_id="gif-doc", file_unique_id="gif-u", mime_type="image/gif"
        )
        doc_message = message(None, 30, content_type="document", document=gif_document)
        animation = message(
            None, 31, content_type="animation",
            animation=SimpleNamespace(file_id="anim", file_unique_id="anim-u"),
        )
        sticker = message(
            None, 32, content_type="sticker",
            sticker=SimpleNamespace(file_id="sticker", file_unique_id="sticker-u"),
        )
        with (
            patch.object(bot_module.learning_service, "ingest_chat_image") as image_ingest,
            patch.object(bot_module.learning_service, "ingest_gif") as gif_ingest,
            patch.object(bot_module.learning_service, "ingest_sticker") as sticker_ingest,
        ):
            bot_module.remember_image_document(doc_message)
            bot_module.remember_animation(animation)
            bot_module.remember_sticker(sticker)
        self.assert_normalized_call(image_ingest, doc_message)
        self.assertIsInstance(gif_ingest.call_args_list[0].args[0], NormalizedEvent)
        self.assertEqual(gif_ingest.call_args_list[0].args[0].message_id, 30)
        self.assertIsInstance(gif_ingest.call_args_list[1].args[0], NormalizedEvent)
        self.assertEqual(gif_ingest.call_args_list[1].args[0].message_id, 31)
        self.assert_normalized_call(sticker_ingest, sticker)

    def test_callback_query_is_separate_ui_route_and_never_ingests(self):
        import bot as bot_module

        call = SimpleNamespace(
            id="callback-1", data="chair:close", from_user=user(),
            message=SimpleNamespace(chat=SimpleNamespace(id=-1, type="supergroup"), message_id=77),
        )
        with (
            patch.object(bot_module, "is_user_chat_admin", return_value=True),
            patch.object(bot_module.bot, "edit_message_text") as edit,
            patch.object(bot_module.bot, "answer_callback_query") as answer,
            patch.object(bot_module.learning_service, "ingest") as ingest,
        ):
            bot_module.chair_settings_callback(call)
        edit.assert_called_once()
        answer.assert_called_once_with(call.id)
        ingest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
