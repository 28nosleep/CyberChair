import dataclasses
import unittest
from types import SimpleNamespace

from learning import (
    EventKind,
    normalize_callback_event,
    normalize_telegram_event,
    telegram_event_id,
)


def user(user_id=7, username="tester", is_bot=False):
    return SimpleNamespace(
        id=user_id, username=username, first_name="Test", last_name="User",
        is_bot=is_bot,
    )


def media(file_id, *, unique=None, mime=None, width=None, height=None, size=None):
    return SimpleNamespace(
        file_id=file_id,
        file_unique_id=unique or f"{file_id}-unique",
        mime_type=mime,
        width=width,
        height=height,
        file_size=size,
    )


def message(message_id=1, *, text="hello", caption=None, content_type="text",
            photo=None, document=None, animation=None, sticker=None, reply=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-100), message_id=message_id, text=text,
        caption=caption, content_type=content_type, date=1_776_000_000,
        from_user=user(), reply_to_message=reply, photo=photo,
        document=document, animation=animation, sticker=sticker,
    )


class R1EventNormalizationTests(unittest.TestCase):
    def test_normalized_event_and_media_are_immutable(self):
        event = normalize_telegram_event(message())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.text = "changed"

    def test_effective_text_has_one_canonical_source(self):
        cases = (
            (message(text="стул привет", caption=None), "стул привет", "text"),
            (message(text=None, caption="с м стул", content_type="photo", photo=[media("p")]), "с м стул", "caption"),
            (message(text=None, caption=None, content_type="photo", photo=[media("p")]), "", "none"),
        )
        for incoming, expected, source in cases:
            with self.subTest(source=source, expected=expected):
                event = normalize_telegram_event(incoming)
                self.assertEqual(event.effective_text, expected)
                self.assertEqual(event.effective_text_source, source)
                self.assertNotEqual(event.effective_text, "None")

    def test_normalized_text_reuses_existing_space_normalization(self):
        event = normalize_telegram_event(message(text="  С   М  Стул!!  "))
        self.assertEqual(event.effective_text, "  С   М  Стул!!  ")
        self.assertEqual(event.normalized_text, "С М Стул!!")

    def test_photo_uses_largest_variant_without_raw_bytes(self):
        incoming = message(
            text=None, caption="с м стул", content_type="photo",
            photo=[
                media("small", width=10, height=10, size=100),
                media("large", width=100, height=80, size=1000),
            ],
        )
        event = normalize_telegram_event(incoming)
        self.assertEqual(event.display_name, "Test User")
        self.assertEqual(event.first_name, "Test")
        self.assertEqual(event.event_kind, EventKind.PHOTO)
        self.assertTrue(event.has_photo)
        self.assertEqual(event.telegram_file_id, "large")
        self.assertEqual(event.mime_type, "image/jpeg")
        self.assertFalse(hasattr(event.media, "bytes"))

    def test_image_document_animation_and_sticker_metadata(self):
        cases = (
            (
                message(text=None, content_type="document", document=media("doc", mime="image/png")),
                EventKind.IMAGE_DOCUMENT, "doc", "image/png",
            ),
            (
                message(text=None, content_type="animation", animation=media("anim", mime="video/mp4")),
                EventKind.ANIMATION, "anim", "video/mp4",
            ),
            (
                message(text=None, content_type="sticker", sticker=media("sticker")),
                EventKind.STICKER, "sticker", None,
            ),
        )
        for incoming, kind, file_id, mime_type in cases:
            with self.subTest(kind=kind):
                event = normalize_telegram_event(incoming)
                self.assertEqual(event.event_kind, kind)
                self.assertEqual(event.telegram_file_id, file_id)
                self.assertEqual(event.mime_type, mime_type)

    def test_reply_facts_include_text_identity_and_explicit_image(self):
        reply = message(
            90, text=None, caption="исходная картинка", content_type="photo",
            photo=[media("reply-photo", width=20, height=20)],
        )
        reply.from_user = user(99, "chair", True)
        incoming = message(91, text="с м стул", reply=reply)
        event = normalize_telegram_event(incoming)
        self.assertEqual(event.reply_to_message_id, 90)
        self.assertEqual(event.reply_to_user_id, 99)
        self.assertTrue(event.reply_to_user_is_bot)
        self.assertTrue(event.reply_has_photo)
        self.assertEqual(event.reply_media.file_id, "reply-photo")
        self.assertEqual(event.reply_effective_text, "исходная картинка")
        self.assertTrue(event.replies_to_user(99))

    def test_command_facts_do_not_classify_domain_intent(self):
        command = normalize_telegram_event(message(text="/activity@chair 50"))
        direct = normalize_telegram_event(message(2, text="стул как сварить харчо?"))
        self.assertTrue(command.is_command)
        self.assertEqual(command.command_name, "activity")
        self.assertFalse(direct.is_command)
        self.assertFalse(hasattr(direct, "producer"))
        self.assertFalse(hasattr(direct, "should_reply"))

    def test_r0_event_identity_is_reused_exactly(self):
        incoming = message(123, text="same")
        first = normalize_telegram_event(incoming)
        incoming.content_type = "photo"
        incoming.text = None
        incoming.caption = "same"
        incoming.photo = [media("photo")]
        second = normalize_telegram_event(incoming)
        self.assertEqual(first.event_id, telegram_event_id(incoming))
        self.assertEqual(first.event_id, second.event_id)
        self.assertNotEqual(
            first.event_id, normalize_telegram_event(message(124)).event_id
        )

    def test_callback_has_deterministic_separate_identity(self):
        call = SimpleNamespace(
            id="callback-7", data="chair:status", from_user=user(),
            message=SimpleNamespace(
                chat=SimpleNamespace(id=-100), message_id=55
            ),
        )
        first = normalize_callback_event(call)
        second = normalize_callback_event(call)
        self.assertEqual(first, second)
        self.assertTrue(first.event_id.startswith("cb_"))
        self.assertNotEqual(first.event_id, normalize_telegram_event(message(55)).event_id)
        self.assertEqual(first.data, "chair:status")

    def test_unsupported_event_has_empty_text_and_no_media(self):
        incoming = message(text=None, content_type="location")
        event = normalize_telegram_event(incoming)
        self.assertEqual(event.event_kind, EventKind.OTHER)
        self.assertEqual(event.effective_text, "")
        self.assertIsNone(event.media)


if __name__ == "__main__":
    unittest.main()
