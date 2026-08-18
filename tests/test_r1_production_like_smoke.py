import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import LearningService, LearningSettings, normalize_telegram_event


class SmokeProvider:
    available = True
    provider_key = "r1-smoke"
    _usage_recorder = None

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return "один содержательный ответ для synthetic события"

    def summarize(self, request):
        raise AssertionError("smoke does not schedule maintenance summaries")


def user(user_id=7, username="user", is_bot=False):
    return SimpleNamespace(
        id=user_id, username=username, first_name="User", is_bot=is_bot
    )


def media(file_id, mime_type=None):
    return SimpleNamespace(
        file_id=file_id, file_unique_id=f"{file_id}-unique",
        mime_type=mime_type, width=640, height=480, file_size=1000,
    )


def message(message_id, text="", *, reply=None, caption=None,
            content_type="text", photo=None, document=None, animation=None,
            sticker=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-500, type="supergroup"),
        message_id=message_id, text=text, caption=caption,
        content_type=content_type, date=1_776_100_000 + message_id,
        from_user=user(), reply_to_message=reply, photo=photo,
        document=document, animation=animation, sticker=sticker,
    )


def chair_reply(message_id):
    value = message(message_id, "предыдущий ответ")
    value.from_user = user(99, "chair", True)
    return value


class R1ProductionLikeSmokeTests(unittest.TestCase):
    def test_eighty_mixed_events_match_characterization_and_one_call_limit(self):
        import bot as bot_module

        fixtures = []
        next_id = 1000

        def add(count, route, producer, ingested, factory):
            nonlocal next_id
            for offset in range(count):
                fixtures.append(
                    (factory(next_id, offset), route, producer, ingested)
                )
                next_id += 1

        add(20, "ordinary", "llm", True,
            lambda mid, i: message(mid, f"обычный разговор про релиз номер {i}"))
        add(10, "direct", "llm", True,
            lambda mid, i: message(mid, f"стул как исправить релиз номер {i}?"))
        add(10, "reply_to_chair", "llm", True,
            lambda mid, i: message(mid, f"продолжение темы номер {i}", reply=chair_reply(mid + 10_000)))
        add(10, "pending", "llm", True,
            lambda mid, i: message(mid, f"вариант продолжения номер {i}"))
        add(4, "photo_metadata", "none", False,
            lambda mid, i: message(mid, None, caption=f"фото {i}", content_type="photo", photo=[media(f"p-{mid}")]))
        add(2, "image_document_metadata", "none", False,
            lambda mid, i: message(mid, None, caption=f"док {i}", content_type="document", document=media(f"d-{mid}", "image/png")))
        add(2, "animation_metadata", "none", False,
            lambda mid, i: message(mid, None, content_type="animation", animation=media(f"a-{mid}", "video/mp4")))
        add(2, "sticker_metadata", "none", False,
            lambda mid, i: message(mid, None, content_type="sticker", sticker=media(f"s-{mid}")))
        add(2, "freekucher", "local", False,
            lambda mid, i: message(mid, f"Кучер снова в чате {i}"))
        add(2, "foreign_control", "none", False,
            lambda mid, i: message(mid, f"s g m нарисуй это {i}"))
        add(2, "manual_meme", "llm", False,
            lambda mid, i: message(mid, f"с м стул про релиз {i}"))
        add(1, "chair_remaining", "local", False,
            lambda mid, i: message(mid, "с стул"))
        add(1, "voice_story", "llm", False,
            lambda mid, i: message(mid, "стул голос"))
        add(2, "who_command", "local", True,
            lambda mid, i: message(mid, f"к кто отвечает за релиз {i}"))
        add(10, "unsupported", "none", False,
            lambda mid, i: message(mid, None, content_type="location"))
        self.assertEqual(len(fixtures), 80)

        with tempfile.TemporaryDirectory() as directory:
            provider = SmokeProvider()
            service = LearningService(
                LearningSettings(
                    data_dir=Path(directory), openai_chat_id=-500,
                    min_training_messages=1000, summary_message_interval=1000,
                    summary_time_interval=3600, addressed_cooldown=0,
                    generated_cooldown=0, max_generated_per_hour=1000,
                ),
                llm_provider=provider,
                rng=SimpleNamespace(random=lambda: 0.0, choice=lambda values: values[0]),
            )
            ingested_ids = set()
            media_ids = set()

            def ingest(event, **_kwargs):
                ingested_ids.add(event.event_id)
                return True, None

            def generate(event, *_args, **_kwargs):
                return service.generate_llm(
                    event.chat_id, event.effective_text or "media", "reply"
                )

            def generate_voice(event, *_args, **_kwargs):
                return service.generate_llm(event.chat_id, None, "voice_story")

            def freekucher(raw, event):
                if not bot_module.is_freekucher_message(event.effective_text):
                    return False
                bot_module.bot.reply_to(raw, "#FREEKUCHER")
                return True

            def manual_meme(raw, decision=None, hint="", event=None):
                service.generate_llm(event.chat_id, hint or "мем", "meme_caption")
                bot_module.bot.send_photo(event.chat_id, b"synthetic")
                return True

            def who(raw, _text, _event):
                bot_module.bot.reply_to(raw, "локальный выбор")
                return True

            def media_ingest(event):
                media_ids.add(event.event_id)
                return True

            delivered = SimpleNamespace(message_id=88_888)
            with ExitStack() as stack:
                stack.enter_context(patch.object(bot_module, "learning_service", service))
                stack.enter_context(patch.object(bot_module, "remember_user"))
                stack.enter_context(patch.object(
                    bot_module, "get_bot_identity",
                    return_value={"id": 99, "username": "chair"},
                ))
                freekucher_mock = stack.enter_context(patch.object(
                    bot_module, "freekucher_reaction", side_effect=freekucher,
                ))
                meme_mock = stack.enter_context(patch.object(
                    bot_module, "send_manual_meme", side_effect=manual_meme,
                ))
                who_mock = stack.enter_context(patch.object(
                    bot_module, "handle_who", side_effect=who,
                ))
                stack.enter_context(patch.object(
                    bot_module, "reaction_text", return_value=False,
                ))
                stack.enter_context(patch.object(service, "ingest", side_effect=ingest))
                stack.enter_context(patch.object(
                    service, "is_pending_continuation",
                    side_effect=lambda event, **kwargs: event.effective_text.startswith(
                        "вариант продолжения"
                    ),
                ))
                pending_mock = stack.enter_context(patch.object(
                    service, "maybe_pending_continuation", side_effect=generate,
                ))
                direct_mock = stack.enter_context(patch.object(
                    service, "maybe_direct_reply", side_effect=generate,
                ))
                ordinary_mock = stack.enter_context(patch.object(
                    service, "maybe_reply", side_effect=generate,
                ))
                voice_mock = stack.enter_context(patch.object(
                    service, "maybe_voice_story", side_effect=generate_voice,
                ))
                stack.enter_context(patch.object(
                    service, "take_voice_story_cooldown_notice", return_value=0,
                ))
                stack.enter_context(patch.object(
                    service, "activity_allows", return_value=True,
                ))
                stack.enter_context(patch.object(
                    service, "troll_mode", return_value=True,
                ))
                stack.enter_context(patch.object(
                    service, "ingest_chat_image", side_effect=media_ingest,
                ))
                stack.enter_context(patch.object(
                    service, "ingest_gif", side_effect=media_ingest,
                ))
                stack.enter_context(patch.object(
                    service, "ingest_sticker", side_effect=media_ingest,
                ))
                reply_mock = stack.enter_context(patch.object(
                    bot_module.bot, "reply_to", return_value=delivered,
                ))
                photo_mock = stack.enter_context(patch.object(
                    bot_module.bot, "send_photo",
                ))
                rows = []
                seen_ids = set()
                for raw, expected_route, expected_producer, expected_ingested in fixtures:
                    event = normalize_telegram_event(raw)
                    before_llm = len(provider.requests)
                    before_reply = reply_mock.call_count
                    before_photo = photo_mock.call_count
                    before_routes = {
                        "ordinary": ordinary_mock.call_count,
                        "direct": direct_mock.call_count,
                        "pending": pending_mock.call_count,
                        "manual_meme": meme_mock.call_count,
                        "freekucher": freekucher_mock.call_count,
                        "voice_story": voice_mock.call_count,
                        "who_command": who_mock.call_count,
                    }
                    if expected_route == "unsupported":
                        pass
                    elif event.content_type == "photo":
                        bot_module.remember_photo(raw)
                    elif event.content_type == "document":
                        bot_module.remember_image_document(raw)
                    elif event.content_type == "animation":
                        bot_module.remember_animation(raw)
                    elif event.content_type == "sticker":
                        bot_module.remember_sticker(raw)
                    else:
                        bot_module.handle_message(raw)

                    llm_calls = len(provider.requests) - before_llm
                    text_deliveries = reply_mock.call_count - before_reply
                    photo_deliveries = photo_mock.call_count - before_photo
                    deliveries = text_deliveries + photo_deliveries
                    self.assertLessEqual(llm_calls, 1)
                    self.assertEqual(
                        llm_calls, 1 if expected_producer == "llm" else 0,
                        f"unexpected LLM count for route={expected_route}",
                    )
                    self.assertLessEqual(deliveries, 1)
                    self.assertEqual(event.event_id in ingested_ids, expected_ingested)
                    if llm_calls:
                        self.assertEqual(
                            provider.requests[-1].metadata["event_id"], event.event_id
                        )
                    route_key = {
                        "reply_to_chair": "direct",
                    }.get(expected_route, expected_route)
                    if route_key in before_routes:
                        after = {
                            "ordinary": ordinary_mock.call_count,
                            "direct": direct_mock.call_count,
                            "pending": pending_mock.call_count,
                            "manual_meme": meme_mock.call_count,
                            "freekucher": freekucher_mock.call_count,
                            "voice_story": voice_mock.call_count,
                            "who_command": who_mock.call_count,
                        }[route_key]
                        self.assertEqual(after - before_routes[route_key], 1)
                    delivery_type = (
                        "photo" if photo_deliveries
                        else "text" if text_deliveries else "none"
                    )
                    expected_delivery = (
                        "photo" if expected_route == "manual_meme"
                        else "none" if expected_producer == "none" else "text"
                    )
                    self.assertEqual(delivery_type, expected_delivery)
                    rows.append({
                        "event_kind": event.event_kind.value,
                        "event_id": event.event_id,
                        "effective_text_source": event.effective_text_source,
                        "route": expected_route,
                        "ingested": expected_ingested,
                        "llm_calls": llm_calls,
                        "producer": expected_producer,
                        "delivery_type": delivery_type,
                    })
                    seen_ids.add(event.event_id)

            self.assertEqual(len(rows), 80)
            self.assertEqual(len(seen_ids), 80)
            self.assertEqual(max(row["llm_calls"] for row in rows), 1)
            self.assertEqual(sum(row["llm_calls"] for row in rows), 53)
            self.assertEqual(sum(row["ingested"] for row in rows), 52)
            self.assertEqual(
                sum(row["route"].endswith("metadata") for row in rows), 10
            )
            self.assertEqual(len(media_ids), 10)
            invariant = service.llm_event_invariant_diagnostics(
                -500, hours=24 * 365 * 10
            )
            self.assertEqual(invariant["user_events"], 70)
            self.assertEqual(invariant["events_with_0_llm"], 17)
            self.assertEqual(invariant["events_with_1_llm"], 53)
            self.assertEqual(invariant["events_with_2plus_llm"], 0)
            self.assertEqual(invariant["max_calls_per_user_event"], 1)


if __name__ == "__main__":
    unittest.main()
