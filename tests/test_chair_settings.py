import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bot as bot_module
from learning import LearningService, LearningSettings


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="ответ")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def message(user_id=7, chat_type="supergroup"):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-100, type=chat_type),
        from_user=SimpleNamespace(id=user_id),
        message_id=44,
        text="/chair_settings",
    )


def callback(data, user_id=7):
    return SimpleNamespace(
        id="callback-1",
        data=data,
        from_user=SimpleNamespace(id=user_id),
        message=message(user_id=999),
    )


class ChairSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.grok = FakeClient()
        self.openai = FakeClient()
        settings = LearningSettings(
            data_dir=Path(self.temp.name), llm_provider="grok", openai_chat_id=-100,
            default_activity_percent=50,
        )
        self.service = LearningService(
            settings, openai_client=self.openai, xai_client=self.grok
        )
        self.service_patch = patch.object(bot_module, "learning_service", self.service)
        self.service_patch.start()

    def tearDown(self):
        self.service_patch.stop()
        self.temp.cleanup()

    def admin(self, status="administrator"):
        return patch.object(
            bot_module.bot, "get_chat_member", return_value=SimpleNamespace(status=status)
        )

    def test_command_is_group_only_and_forbidden_to_non_admin(self):
        with (
            self.admin("member"),
            patch.object(bot_module.bot, "reply_to") as reply,
            patch.object(bot_module.bot, "send_message") as send,
        ):
            bot_module.chair_settings_command(message())
        send.assert_not_called()
        self.assertIn("права администратора", reply.call_args.args[1])

        with patch.object(bot_module.bot, "reply_to") as private_reply:
            bot_module.chair_settings_command(message(chat_type="private"))
        self.assertIn("только внутри", private_reply.call_args.args[1])

    def test_admin_gets_current_inline_menu(self):
        with self.admin(), patch.object(bot_module.bot, "send_message") as send:
            bot_module.chair_settings_command(message())
        text = send.call_args.args[1]
        keyboard = send.call_args.kwargs["reply_markup"]
        self.assertIn("chairOS // панель управления", text)
        self.assertIn("мозги: grok-4.5", text)
        data = [button.callback_data for row in keyboard.keyboard for button in row]
        self.assertIn("chair:troll", data)
        self.assertIn("chair:close", data)

    def test_callback_rechecks_admin(self):
        with (
            self.admin("member"),
            patch.object(bot_module.bot, "answer_callback_query") as answer,
            patch.object(bot_module.bot, "edit_message_text") as edit,
        ):
            bot_module.chair_settings_callback(callback("chair:troll"))
        edit.assert_not_called()
        self.assertTrue(answer.call_args.kwargs["show_alert"])
        self.assertTrue(self.service.troll_mode(-100))

    def test_troll_autonomous_and_media_toggle_persist(self):
        for action in ("chair:troll", "chair:auto", "chair:media"):
            with (
                self.admin(),
                patch.object(bot_module.bot, "answer_callback_query"),
                patch.object(bot_module.bot, "edit_message_text"),
            ):
                bot_module.chair_settings_callback(callback(action))
        self.assertFalse(self.service.troll_mode(-100))
        self.assertFalse(self.service.autonomous_enabled(-100))
        self.assertFalse(self.service.media_enabled(-100))
        reopened = LearningService(
            self.service.settings, openai_client=FakeClient(), xai_client=FakeClient()
        )
        self.assertFalse(reopened.autonomous_enabled(-100))
        self.assertFalse(reopened.media_enabled(-100))

    def test_provider_submenu_and_switch(self):
        with (
            self.admin(),
            patch.object(bot_module.bot, "answer_callback_query"),
            patch.object(bot_module.bot, "edit_message_text") as edit,
        ):
            bot_module.chair_settings_callback(callback("chair:provider"))
            bot_module.chair_settings_callback(callback("chair:provider:openai"))
        self.assertEqual(self.service.llm_provider_name(-100), "openai")
        self.assertEqual(edit.call_count, 2)
        self.assertIn("выбор мозга", edit.call_args.args[0])

    def test_unavailable_grok_does_not_change_selection(self):
        settings = LearningSettings(
            data_dir=Path(self.temp.name) / "missing", llm_provider="openai",
            openai_chat_id=-100,
        )
        with patch.dict(bot_module.os.environ, {"XAI_API_KEY": "", "OPENAI_API_KEY": "set"}):
            unavailable = LearningService(settings)
            with (
                patch.object(bot_module, "learning_service", unavailable),
                self.admin(),
                patch.object(bot_module.bot, "answer_callback_query") as answer,
                patch.object(bot_module.bot, "edit_message_text") as edit,
            ):
                bot_module.chair_settings_callback(callback("chair:provider:grok"))
        self.assertEqual(unavailable.llm_provider_name(-100), "openai")
        edit.assert_not_called()
        self.assertIn("XAI_API_KEY", answer.call_args.args[1])

    def test_activity_changes_and_callback_edits_same_message(self):
        with (
            self.admin(),
            patch.object(bot_module.bot, "answer_callback_query"),
            patch.object(bot_module.bot, "edit_message_text") as edit,
            patch.object(bot_module.bot, "send_message") as send,
        ):
            bot_module.chair_settings_callback(callback("chair:activity:75"))
        self.assertEqual(self.service.activity_percent(-100), 75)
        edit.assert_called_once()
        self.assertEqual(edit.call_args.kwargs["message_id"], 44)
        send.assert_not_called()

    def test_close_replaces_panel_without_keyboard(self):
        with (
            self.admin(),
            patch.object(bot_module.bot, "answer_callback_query"),
            patch.object(bot_module.bot, "edit_message_text") as edit,
        ):
            bot_module.chair_settings_callback(callback("chair:close"))
        self.assertIn("панель закрыта", edit.call_args.args[0])
        self.assertNotIn("reply_markup", edit.call_args.kwargs)

    def test_status_contains_no_paths_or_secrets(self):
        text, keyboard = bot_module.chair_status_screen(-100)
        self.assertIn("chairOS // статус", text)
        self.assertIn("stable memories", text)
        self.assertNotIn(str(self.temp.name), text)
        self.assertNotIn("API_KEY", text)
        self.assertIsNotNone(keyboard)

    def test_menu_does_not_call_any_llm(self):
        with (
            self.admin(),
            patch.object(bot_module.bot, "answer_callback_query"),
            patch.object(bot_module.bot, "edit_message_text"),
        ):
            bot_module.chair_settings_callback(callback("chair:troll"))
            bot_module.chair_settings_callback(callback("chair:status"))
        self.assertEqual(self.grok.responses.calls, [])
        self.assertEqual(self.openai.responses.calls, [])

    def test_startup_reports_missing_selected_grok_key(self):
        with (
            patch.object(bot_module, "TOKEN", "123:configured"),
            patch.object(bot_module.learning_service, "llm_provider_name", return_value="grok"),
            patch.object(bot_module.learning_service, "provider_available", return_value=False),
            # main() now owns the canonical process shutdown path.  Keep this
            # startup-failure characterization from shutting down the shared
            # process-wide R4 controller used by later tests.
            patch.object(bot_module.learning_service.concurrency, "shutdown"),
            patch.object(bot_module.chat_action_manager, "shutdown"),
        ):
            with self.assertRaisesRegex(RuntimeError, "XAI_API_KEY"):
                bot_module.main()

if __name__ == "__main__":
    unittest.main()
