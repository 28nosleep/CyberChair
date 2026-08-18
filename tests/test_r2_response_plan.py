import dataclasses
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from learning import LearningService, LearningSettings
from learning.normalized_event import normalize_telegram_event
from learning.response_plan import (
    DeliveryReceipt,
    DeliveryType,
    GeneratedCommit,
    PersonaUsageCommit,
    Producer,
    ResponsePlan,
    TextPayload,
)


def event(message_id=1, text="стул привет"):
    message = SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=message_id, text=text,
        caption=None, content_type="text", date=1_776_000_000 + message_id,
        from_user=SimpleNamespace(
            id=7, username="user7", first_name="User", is_bot=False,
        ),
        reply_to_message=None,
    )
    return normalize_telegram_event(message)


class ResponsePlanModelTests(unittest.TestCase):
    def test_plan_and_receipt_are_immutable_typed_data(self):
        plan = ResponsePlan(
            event_id="tg_test", chat_id=-1, producer=Producer.LOCAL,
            delivery_type=DeliveryType.TEXT, payload=TextPayload("ответ"),
        )
        receipt = DeliveryReceipt(
            "tg_test", True, DeliveryType.TEXT, telegram_message_id=501
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.required = True
        with self.assertRaises(dataclasses.FrozenInstanceError):
            receipt.success = False
        self.assertFalse(any(callable(value) for value in plan.commit_actions))

    def test_payload_must_match_delivery_type(self):
        with self.assertRaises(ValueError):
            ResponsePlan(
                event_id="tg_bad", chat_id=-1, producer=Producer.LOCAL,
                delivery_type=DeliveryType.PHOTO, payload=TextPayload("нет"),
            )

    def test_success_commit_is_idempotent_and_failure_never_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(LearningSettings(data_dir=Path(directory)))
            normalized = event(10)
            plan = service.prepare_text_response(
                normalized, "доставленный ответ", "test",
                actions=(GeneratedCommit("доставленный ответ", "test"),),
            )
            failure = DeliveryReceipt(
                plan.event_id, False, plan.delivery_type,
                error_category="telegram_timeout",
            )
            self.assertTrue(service.abort_response(plan, failure))
            self.assertEqual(
                service.repository(-1).generated_since(
                    datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
                ),
                [],
            )

            success = DeliveryReceipt(
                plan.event_id, True, plan.delivery_type,
                telegram_message_id=777,
            )
            self.assertTrue(service.commit_response(plan, success))
            self.assertFalse(service.commit_response(plan, success))
            rows = service.repository(-1).generated_since(
                datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
            )
            self.assertEqual([row["text"] for row in rows], ["доставленный ответ"])

    def test_persona_cooldown_usage_is_success_only(self):
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(LearningSettings(data_dir=Path(directory)))
            action = PersonaUsageCommit(("meme-r2",), ("group-r2",))
            failed = service.prepare_text_response(
                event(20), "ответ", "direct", actions=(action,)
            )
            service.finalize_response(
                failed,
                DeliveryReceipt(
                    failed.event_id, False, failed.delivery_type,
                    error_category="telegram_timeout",
                ),
            )
            self.assertNotIn("meme-r2", service.persona._recent_ids[-1])

            success = service.prepare_text_response(
                event(21), "ответ", "direct", actions=(action,)
            )
            service.finalize_response(
                success,
                DeliveryReceipt(
                    success.event_id, True, success.delivery_type,
                    telegram_message_id=100,
                ),
            )
            self.assertIn("meme-r2", service.persona._recent_ids[-1])


if __name__ == "__main__":
    unittest.main()
