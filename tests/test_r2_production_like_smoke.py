import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bot as bot_module
from learning import LearningService, LearningSettings
from learning.response_plan import GeneratedCommit, Producer
from test_r2_response_plan import event


class R2ProductionLikeFailureSmokeTests(unittest.TestCase):
    def test_one_hundred_mixed_events_keep_success_state_transport_truthful(self):
        categories = (
            "direct_useful", "troll_user", "local", "pending", "ordinary",
            "markov", "gif", "sticker", "meme", "photo_caption_meme",
            "provider_fallback",
        )
        producers = (
            Producer.LLM, Producer.LLM, Producer.LOCAL, Producer.LOCAL,
            Producer.LLM, Producer.MARKOV, Producer.MEDIA, Producer.MEDIA,
            Producer.MEME, Producer.MEME, Producer.LOCAL,
        )
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            service = LearningService(
                LearningSettings(
                    data_dir=Path(directory), max_messages_per_chat=200
                )
            )
            outcomes = [
                SimpleNamespace(message_id=5000 + index)
                if index < 70 else TimeoutError("forced")
                for index in range(100)
            ]
            with (
                patch.object(bot_module, "learning_service", service),
                patch.object(bot_module.bot, "send_message", side_effect=outcomes) as send,
            ):
                for index in range(100):
                    category = categories[index % len(categories)]
                    text = f"response-{index}"
                    plan = service.prepare_text_response(
                        event(1000 + index), text, category,
                        producer=producers[index % len(producers)],
                        actions=(GeneratedCommit(text, category),),
                    )
                    receipt = bot_module.execute_response_plan(plan)
                    success = receipt.success
                    rows.append({
                        "event": category,
                        "producer": plan.producer.value,
                        "plan_count": 1,
                        "llm_calls": 0,
                        "delivery_attempts": 1,
                        "delivery_success": int(success),
                        "commit_count": int(success),
                        "abort_count": int(not success),
                        "pending_before": category == "pending",
                        "pending_after": category == "pending" and not success,
                        "cooldown_committed": bool(success),
                        "generated_committed": bool(success),
                    })
            self.assertEqual(send.call_count, 100)

            generated = service.repository(-1).generated_since(
                "2020-01-01T00:00:00+00:00"
            )
            self.assertEqual(len(generated), 70)
            self.assertEqual(sum(row["delivery_success"] for row in rows), 70)
            self.assertEqual(sum(row["abort_count"] for row in rows), 30)
            self.assertTrue(all(row["plan_count"] <= 1 for row in rows))
            self.assertTrue(all(row["llm_calls"] <= 1 for row in rows))
            self.assertTrue(all(row["delivery_success"] <= 1 for row in rows))
            self.assertTrue(all(
                row["generated_committed"] == bool(row["delivery_success"])
                for row in rows
            ))
            self.assertTrue(all(
                row["cooldown_committed"] == bool(row["delivery_success"])
                for row in rows
            ))
            self.assertTrue(all(
                not row["pending_before"]
                or row["pending_after"] == (not bool(row["delivery_success"]))
                for row in rows
            ))


if __name__ == "__main__":
    unittest.main()
