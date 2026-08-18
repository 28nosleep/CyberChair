import tempfile
import unittest
from pathlib import Path

from learning import LearningService, LearningSettings
from learning.response_plan import (
    DeliveryReceipt,
    PendingCreate,
    PendingFinalize,
)
from test_r2_response_plan import event


class PendingDeliveryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = LearningService(
            LearningSettings(data_dir=Path(self.temp.name))
        )
        self.repository = self.service.repository(-1)

    def tearDown(self):
        self.temp.cleanup()

    def save_existing(self):
        self.repository.save_pending_conversation(
            user_id=7, original_message_id=1,
            original_question="что выбрать", clarification_question="между чем?",
            intent="choice", context="между чем выбираешь?",
            expected_type="choices", pending_mode="hard", bot_message_id=400,
        )

    def receipt(self, plan, success, message_id=None):
        return DeliveryReceipt(
            plan.event_id, success, plan.delivery_type,
            telegram_message_id=message_id,
            error_category=None if success else "telegram_timeout",
        )

    def test_existing_pending_survives_failure_and_finalizes_after_success(self):
        self.save_existing()
        plan = self.service.prepare_text_response(
            event(2), "бери первый", "pending",
            actions=(PendingFinalize(7),),
        )
        self.service.finalize_response(plan, self.receipt(plan, False))
        self.assertIsNotNone(self.repository.pending_conversation(7, 1200))
        self.service.finalize_response(plan, self.receipt(plan, True, 501))
        self.assertIsNone(self.repository.pending_conversation(7, 1200))

    def test_new_pending_is_created_only_after_success_with_real_bot_id(self):
        action = PendingCreate(
            user_id=7, original_message_id=3,
            original_question="что выбрать", clarification_question="между чем?",
            intent="choice", context="между чем выбираешь?",
            expected_type="choices", mode="hard",
        )
        failed_plan = self.service.prepare_text_response(
            event(3), "между чем выбираешь?", "direct", actions=(action,)
        )
        self.service.finalize_response(
            failed_plan, self.receipt(failed_plan, False)
        )
        self.assertIsNone(self.repository.pending_conversation(7, 1200))

        success_plan = self.service.prepare_text_response(
            event(4), "между чем выбираешь?", "direct", actions=(action,)
        )
        self.service.finalize_response(
            success_plan, self.receipt(success_plan, True, 90210)
        )
        pending = self.repository.pending_conversation(7, 1200)
        self.assertEqual(pending["bot_message_id"], 90210)


if __name__ == "__main__":
    unittest.main()
