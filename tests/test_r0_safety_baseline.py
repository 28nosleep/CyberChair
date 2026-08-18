import os
import re
import subprocess
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from env_loader import load_environment
from learning import (
    ConversationDecision,
    EventLLMPermit,
    LearningService,
    LearningSettings,
    telegram_event_id,
)
from learning.chat_state import ChatState
from learning.event_context import runtime_concurrency
from learning.media_service import MediaDecision


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def message(message_id, text="обычное сообщение про релиз", chat_id=-1, user_id=7):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        text=text,
        caption=None,
        date=1_776_000_000 + message_id,
        from_user=SimpleNamespace(
            id=user_id, username=f"user{user_id}", first_name="User", is_bot=False
        ),
        reply_to_message=None,
    )


def chat_state(message_id=1, user_id=7):
    return ChatState(
        activity_level="normal",
        silence_seconds=30,
        conversation_type="work",
        dominant_topic="релиз",
        topic_strength=0.8,
        humor_score=0.1,
        argument_score=0.1,
        serious_score=0.6,
        work_score=0.9,
        reply_density=0.1,
        participant_count=2,
        target_message_id=message_id,
        target_user_id=user_id,
        confidence=0.9,
    )


def ai_decision(message_id=1, user_id=7):
    return ConversationDecision(
        action="reply",
        reply_probability=1.0,
        troll_intensity=0.5,
        max_reply_length=36,
        preferred_style="work_sarcastic",
        target_message_id=message_id,
        target_user_id=user_id,
        reason="r0_test",
        local_probability=0.0,
        llm_probability=1.0,
    )


class RecordingProvider:
    available = True
    provider_key = "test-provider"
    _usage_recorder = None

    def __init__(self, generate_result="релиз откати и проверь логи до повторной выкладки"):
        self.generate_result = generate_result
        self.generate_requests = []
        self.summarize_requests = []
        self._lock = threading.Lock()

    def _record(self, request):
        recorder = self._usage_recorder
        if recorder:
            metadata = request.metadata or {}
            recorder(
                metadata["chat_id"], self.provider_key, "test-model",
                metadata["call_type"],
                {"input_tokens": 10, "output_tokens": 5},
            )

    def generate(self, request):
        with self._lock:
            self.generate_requests.append(request)
        self._record(request)
        return self.generate_result

    def summarize(self, request):
        with self._lock:
            self.summarize_requests.append(request)
        self._record(request)
        return {
            "main_topics": ["релиз"],
            "current_mood": "рабочий",
            "active_conflicts": [],
            "inside_jokes": [],
            "frequently_mentioned_people": [],
            "notable_events": [],
            "repeated_phrases": [],
            "callback_jokes": [],
            "memory_candidates": [],
        }


class SecurityBaselineTests(unittest.TestCase):
    SECRET_KEYS = ("TELEGRAM_BOT_TOKEN", "XAI_API_KEY", "OPENAI_API_KEY")

    def test_tracked_example_contains_only_documented_secret_placeholders(self):
        values = {}
        for raw_line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            key, separator, value = raw_line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
        for key in self.SECRET_KEYS:
            value = values.get(key, "")
            placeholder = value.startswith("your_") and value.endswith("_here")
            self.assertTrue(
                not value or placeholder,
                f"SECRET_PATTERN_FOUND file=.env.example key={key}",
            )

    def test_tracked_deployment_config_has_no_known_credential_shapes(self):
        patterns = {
            "TELEGRAM_BOT_TOKEN": re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
            "XAI_API_KEY": re.compile(r"\bxai-[A-Za-z0-9_-]{20,}\b"),
            "OPENAI_API_KEY": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        }
        for relative in (
            ".env.example", ".dockerignore", "Dockerfile",
            "docker-compose.yml", "env_loader.py",
        ):
            content = (ROOT / relative).read_text(encoding="utf-8")
            for key, pattern in patterns.items():
                self.assertIsNone(
                    pattern.search(content),
                    f"SECRET_PATTERN_FOUND file={relative} key={key}",
                )

    def test_private_env_is_not_tracked_or_in_docker_build_context(self):
        tracked = subprocess.run(
            ["git", "ls-files", ".env"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        self.assertNotIn(".env", tracked)
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", dockerignore)
        self.assertIn(".env.*", dockerignore)

    def test_compose_and_loader_use_private_env_not_example(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("- .env", compose)
        self.assertNotIn("- .env.example", compose)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / ".env.example").write_text(
                "XAI_API_KEY=documentation_only\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_environment(base)
                self.assertEqual(loaded, [])
                self.assertNotIn("XAI_API_KEY", os.environ)


class EventPermitUnitTests(unittest.TestCase):
    def test_permit_is_one_way_after_failure_semantics(self):
        permit = EventLLMPermit("tg_test")
        self.assertTrue(permit.try_acquire("reply", "test-provider"))
        self.assertFalse(permit.try_acquire("summary", "test-provider"))
        self.assertEqual(permit.call_count, 1)
        self.assertEqual(permit.denied_count, 1)

    def test_event_id_is_stable_private_and_unique(self):
        first = message(10, "secret text", user_id=101)
        same_identity = message(10, "different text", user_id=202)
        second = message(11, "secret text", user_id=101)
        first_id = telegram_event_id(first)
        self.assertEqual(first_id, telegram_event_id(same_identity))
        self.assertNotEqual(first_id, telegram_event_id(second))
        self.assertNotIn("secret", first_id)
        self.assertNotIn("user", first_id)


class R0EventIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def service(self, provider=None, **overrides):
        values = {
            "data_dir": Path(self.temp.name),
            "openai_chat_id": -1,
            "min_training_messages": 1,
            "summary_message_interval": 1,
            "summary_time_interval": 0,
            "generated_cooldown": 0,
            "addressed_cooldown": 0,
            "max_generated_per_hour": 100,
        }
        values.update(overrides)
        provider = provider or RecordingProvider()
        service = LearningService(
            LearningSettings(**values), llm_provider=provider,
            rng=SimpleNamespace(random=lambda: 0.0, choice=lambda values: values[0]),
        )
        service.set_media_enabled(-1, False)
        return service, provider

    def test_acceptance_summary_due_plus_ordinary_ai_is_one_total_call(self):
        service, provider = self.service()
        incoming = message(100)
        state = chat_state(100)
        decision = ai_decision(100)
        with (
            patch.object(service.chat_state_analyzer, "analyze", return_value=state),
            patch.object(service.conversation_policy, "decide", return_value=decision),
            patch.object(service.triggers, "allowed", return_value=True),
            patch.object(service, "_policy_quiet_hours", return_value=False),
            service.telegram_user_event(incoming) as event,
        ):
            inserted, reason = service.ingest(incoming)
            result = service.maybe_reply(incoming)
        self.assertTrue(inserted)
        self.assertIsNone(reason)
        self.assertTrue(result)
        self.assertEqual(len(provider.generate_requests), 1)
        self.assertEqual(len(provider.summarize_requests), 0)
        self.assertEqual(event.permit.call_count, 1)
        summary_state = service.repository(-1).summary_state()
        self.assertEqual(summary_state["last_message_row_id"], 0)
        self.assertIsNotNone(summary_state["pending_since"])
        self.assertEqual(service.repository(-1).count(), 1)

    def test_ordinary_no_response_and_not_due_has_zero_llm_calls(self):
        service, provider = self.service(
            summary_message_interval=50, summary_time_interval=1200
        )
        incoming = message(98)
        with service.telegram_user_event(incoming) as event:
            service.ingest(incoming)
        self.assertEqual(event.permit.call_count, 0)
        self.assertEqual(provider.generate_requests, [])
        self.assertEqual(provider.summarize_requests, [])

    def test_ordinary_summary_due_makes_no_foreground_summary_call(self):
        service, provider = self.service()
        incoming = message(99)
        with service.telegram_user_event(incoming) as event:
            service.ingest(incoming)
        self.assertEqual(event.permit.call_count, 0)
        self.assertEqual(provider.generate_requests, [])
        self.assertEqual(provider.summarize_requests, [])
        self.assertIsNotNone(service.repository(-1).summary_state()["pending_since"])

    def test_trivial_local_direct_response_spends_no_llm_permit(self):
        service, provider = self.service(summary_message_interval=50)
        incoming = message(97, "стул")
        with service.telegram_user_event(incoming) as event:
            result = service.maybe_direct_reply(incoming, explicit_address=True)
        self.assertTrue(result)
        self.assertEqual(event.permit.call_count, 0)
        self.assertEqual(provider.generate_requests, [])

    def test_full_telegram_handler_reproduces_and_fixes_summary_plus_reply(self):
        import bot as bot_module

        service, provider = self.service()
        incoming = message(103)
        state = chat_state(103)
        decision = ai_decision(103)
        with (
            patch.object(bot_module, "learning_service", service),
            patch.object(bot_module, "remember_user"),
            patch.object(bot_module, "get_bot_identity", return_value={"id": 99, "username": "chair"}),
            patch.object(bot_module, "reaction_text", return_value=False),
            patch.object(bot_module.bot, "reply_to", return_value=SimpleNamespace(message_id=900)),
            patch.object(service, "activity_allows", return_value=True),
            patch.object(service.chat_state_analyzer, "analyze", return_value=state),
            patch.object(service.conversation_policy, "decide", return_value=decision),
            patch.object(service.triggers, "allowed", return_value=True),
            patch.object(service, "_policy_quiet_hours", return_value=False),
        ):
            bot_module.handle_message(incoming)
        self.assertEqual(len(provider.generate_requests), 1)
        self.assertEqual(len(provider.summarize_requests), 0)
        self.assertEqual(service.repository(-1).count(), 1)
        self.assertEqual(service.repository(-1).summary_state()["last_message_row_id"], 0)

    def test_same_event_id_correlates_routing_attempt_and_usage(self):
        service, provider = self.service(summary_message_interval=50)
        incoming = message(101)
        with service.telegram_user_event(incoming) as event:
            self.assertTrue(service.generate_llm(-1, "что с релизом", "reply"))
        rows = service.repository(-1).correlated_event_rows(event.event_id)
        self.assertIn("user_event", [row["event_type"] for row in rows["routing"]])
        self.assertIn("llm_call_attempt", [row["event_type"] for row in rows["routing"]])
        self.assertEqual(rows["usage"][0]["event_id"], event.event_id)
        self.assertEqual(rows["usage"][0]["provider"], provider.provider_key)

    def test_memory_summary_has_separate_budget_and_does_not_block_generation(self):
        service, provider = self.service()
        incoming = message(102)
        with service.telegram_user_event(incoming) as event:
            service.ingest(incoming, refresh_memory=False)
            self.assertTrue(service.memory.maybe_refresh(service.repository(-1), -1))
            self.assertIsNotNone(service.generate_llm(-1, "ответь", "reply"))
        self.assertEqual(len(provider.summarize_requests), 1)
        self.assertEqual(len(provider.generate_requests), 1)
        self.assertEqual(event.permit.call_count, 1)
        summary_event_id = provider.summarize_requests[0].metadata["event_id"]
        self.assertTrue(summary_event_id.startswith("mem_"))
        self.assertNotEqual(summary_event_id, event.event_id)

    def test_provider_failure_and_refusal_never_open_second_permit(self):
        for result in (None, "я не могу помочь с этим запросом"):
            with self.subTest(result=result):
                service, provider = self.service(RecordingProvider(result))
                incoming = message(200 if result is None else 201)
                with service.telegram_user_event(incoming) as event:
                    self.assertIsNone(service.generate_llm(-1, "ответь", "reply"))
                    self.assertIsNone(service.generate_llm(-1, "ещё раз", "reply"))
                self.assertEqual(len(provider.generate_requests), 1)
                self.assertEqual(event.permit.call_count, 1)

    def test_autonomous_event_does_not_reuse_user_permit(self):
        service, provider = self.service()
        incoming = message(300)
        with service.telegram_user_event(incoming) as user_event:
            self.assertTrue(service.generate_llm(-1, "user", "reply"))
        with patch.object(
            service, "_maybe_autonomous",
            side_effect=lambda chat_id, current, workday, _as_plan=False: service.generate_llm(
                chat_id, "autonomous", "autonomous"
            ),
        ):
            self.assertTrue(service.prepare_autonomous(-1, NOW))
        self.assertEqual(len(provider.generate_requests), 2)
        event_ids = [request.metadata["event_id"] for request in provider.generate_requests]
        self.assertEqual(event_ids[0], user_event.event_id)
        self.assertNotEqual(event_ids[0], event_ids[1])

    def test_concurrent_events_have_independent_permits_and_correlation(self):
        barrier = threading.Barrier(2)

        class ConcurrentProvider(RecordingProvider):
            def generate(self, request):
                with self._lock:
                    self.generate_requests.append(request)
                barrier.wait(timeout=5)
                self._record(request)
                return self.generate_result

        service, provider = self.service(ConcurrentProvider())
        runtime_concurrency.reset_peaks_for_test()
        results = {}

        def worker(identifier):
            incoming = message(identifier)
            with service.telegram_user_event(incoming) as event:
                first = service.generate_llm(-1, f"event {identifier}", "reply")
                second = service.generate_llm(-1, "second", "reply")
                results[identifier] = (event.event_id, bool(first), second)

        threads = [threading.Thread(target=worker, args=(value,)) for value in (401, 402)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(provider.generate_requests), 2)
        self.assertNotEqual(results[401][0], results[402][0])
        self.assertEqual((results[401][1], results[401][2]), (True, None))
        self.assertEqual((results[402][1], results[402][2]), (True, None))
        telemetry = service.concurrency_diagnostics()
        self.assertGreaterEqual(telemetry["peak_active_llm_calls"], 2)
        # Test rows use wall-clock timestamps, so ask across a wide enough range.
        report = service.llm_event_invariant_diagnostics(
            -1, hours=24 * 365 * 10, current=NOW
        )
        self.assertEqual(report["events_with_2plus_llm"], 0)
        self.assertLessEqual(report["max_calls_per_user_event"], 1)
        diagnostic = service.format_llm_event_invariant_diagnostics(
            -1, hours=24 * 365 * 10, current=NOW
        )
        self.assertIn("LLM EVENT INVARIANT", diagnostic)
        self.assertIn("events_with_2plus_llm: 0", diagnostic)

    def test_concurrent_media_jobs_are_serialized_by_r4_admission(self):
        service, _ = self.service()
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def render(template_id, text):
            calls.append(template_id)
            entered.set()
            release.wait(timeout=5)
            return None

        runtime_concurrency.reset_peaks_for_test()
        decision = MediaDecision(
            action="meme", template_id="synthetic", caption_text="caption"
        )
        with patch.object(service.meme_renderer, "render", side_effect=render):
            threads = [
                threading.Thread(target=service.render_meme, args=(decision,))
                for _ in range(2)
            ]
            for thread in threads:
                thread.start()
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(len(calls), 1)
            release.set()
            for thread in threads:
                thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(calls), 2)
        telemetry = service.concurrency_diagnostics()
        self.assertEqual(telemetry["peak_active_media_jobs"], 1)
        self.assertGreater(telemetry["peak_observed_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
