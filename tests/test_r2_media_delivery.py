import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning import LearningService, LearningSettings, MediaDecision
from learning.event_context import EventContext, bind_event
from learning.meme_sources import MemeSource
from learning.response_plan import (
    DeliveryReceipt,
    MediaUsageCommit,
    Producer,
    GeneratedCommit,
    TriggerCommit,
)
from test_r2_response_plan import event


class MediaDeliveryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = LearningService(
            LearningSettings(data_dir=Path(self.temp.name))
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_media_usage_is_success_only(self):
        decision = MediaDecision(action="gif", asset_id="gif-file", asset_key="gif")
        failed = self.service._create_response_plan(
            event(40), decision, Producer.MEDIA, "contextual_media",
            actions=(MediaUsageCommit(decision),),
        )
        with patch.object(self.service.media, "commit") as commit:
            self.service.finalize_response(
                failed,
                DeliveryReceipt(
                    failed.event_id, False, failed.delivery_type,
                    error_category="telegram_network",
                ),
            )
            commit.assert_not_called()

            success = self.service._create_response_plan(
                event(41), decision, Producer.MEDIA, "contextual_media",
                actions=(MediaUsageCommit(decision),),
            )
            self.service.finalize_response(
                success,
                DeliveryReceipt(
                    success.event_id, True, success.delivery_type,
                    telegram_message_id=55,
                ),
            )
            commit.assert_called_once_with(self.service.repository(-1), decision)

    def test_manual_meme_abort_clears_transient_source_and_temp_files(self):
        decision = MediaDecision(
            action="meme", template_id="template", asset_key="template",
            caption_text="подпись",
        )
        self.service._command_meme_sources[decision] = MemeSource("fresh", "цитата")
        rendered = Path(self.temp.name) / "rendered.png"
        source = Path(self.temp.name) / "source.jpg"
        rendered.write_bytes(b"png")
        source.write_bytes(b"jpg")
        plan = self.service.prepare_manual_meme_response(
            event(50), decision, rendered, (rendered, source)
        )
        self.service.finalize_response(
            plan,
            DeliveryReceipt(
                plan.event_id, False, plan.delivery_type,
                error_category="telegram_timeout",
            ),
        )
        self.assertNotIn(decision, self.service._command_meme_sources)
        self.assertFalse(rendered.exists())
        self.assertFalse(source.exists())

    def test_autonomous_state_is_committed_only_after_delivery(self):
        with bind_event(EventContext("auto_r2_fail", "autonomous", -1)):
            failed = self.service.autonomous._autonomous_response_plan(
                -1, "автономный ответ", Producer.LLM, "autonomous",
                (
                    TriggerCommit("autonomous"),
                    GeneratedCommit("автономный ответ", "autonomous"),
                ),
            )
        self.assertEqual(self.service.repository(-1).recent_generated(10), [])
        self.service.finalize_response(
            failed,
            DeliveryReceipt(
                failed.event_id, False, failed.delivery_type,
                error_category="telegram_network",
            ),
        )
        self.assertEqual(self.service.repository(-1).recent_generated(10), [])

        with bind_event(EventContext("auto_r2_success", "autonomous", -1)):
            success = self.service.autonomous._autonomous_response_plan(
                -1, "автономный ответ", Producer.LLM, "autonomous",
                (
                    TriggerCommit("autonomous"),
                    GeneratedCommit("автономный ответ", "autonomous"),
                ),
            )
        self.service.finalize_response(
            success,
            DeliveryReceipt(
                success.event_id, True, success.delivery_type,
                telegram_message_id=700,
            ),
        )
        self.assertEqual(len(self.service.repository(-1).recent_generated(10)), 1)


if __name__ == "__main__":
    unittest.main()
