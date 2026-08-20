import importlib
import pkgutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import learning
from learning import (
    LearningService,
    LearningSettings,
)
from learning.autonomous_coordinator import AutonomousCoordinator
from learning.foreground_orchestrator import ForegroundOrchestrator
from learning.generation_coordinator import GenerationCoordinator
from learning.media_coordinator import MediaCoordinator
from learning.memory_facade import MemoryFacade
from learning.response_lifecycle import ResponseLifecycle


class Provider:
    available = True
    provider_key = "r6-fake"
    _usage_recorder = None

    def __init__(self, result=None):
        self.result = result or (
            "сначала проверь журнал ошибки, затем откати последний релиз и "
            "зафиксируй минимальный воспроизводимый сценарий"
        )
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self.result

    def summarize(self, request):
        raise AssertionError("foreground decomposition must not summarize")


def message(message_id=1, text="стул почему docker падает?", chat_id=-1):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        text=text,
        caption=None,
        content_type="text",
        date=1_776_000_000 + message_id,
        from_user=SimpleNamespace(
            id=7, username="tester", first_name="Tester", is_bot=False
        ),
        reply_to_message=None,
    )


def service(tmp_path, provider=None):
    return LearningService(
        LearningSettings(
            data_dir=Path(tmp_path),
            openai_chat_id=-1,
            min_training_messages=1,
            generated_cooldown=0,
            addressed_cooldown=0,
            max_generated_per_hour=1000,
        ),
        llm_provider=provider or Provider(),
    )


def test_composition_root_uses_specialized_owners_and_shared_instances(tmp_path):
    provider = Provider()
    learning_service = service(tmp_path, provider)

    assert isinstance(learning_service.foreground, ForegroundOrchestrator)
    assert isinstance(learning_service.autonomous, AutonomousCoordinator)
    assert isinstance(learning_service.generation, GenerationCoordinator)
    assert isinstance(learning_service.media_coordinator, MediaCoordinator)
    assert isinstance(learning_service.memory_facade, MemoryFacade)
    assert isinstance(learning_service.response_lifecycle, ResponseLifecycle)

    assert learning_service.generation.persona is learning_service.persona
    assert learning_service.foreground.local_responder is learning_service.local_responder
    assert learning_service.generation.concurrency is learning_service.concurrency
    assert learning_service.media_coordinator.media is learning_service.media
    assert learning_service.media_coordinator.meme_sources is learning_service.meme_sources
    assert learning_service.response_lifecycle.media is learning_service.media
    assert learning_service.foreground.media is learning_service.media
    assert learning_service.foreground.rng is learning_service.rng
    assert learning_service.media_coordinator.rng is learning_service.rng
    assert learning_service.autonomous.media is learning_service.media
    assert (
        learning_service.autonomous._last_policy_target_user
        is learning_service.foreground._last_policy_target_user
    )
    assert learning_service.provider_for_chat(-1) is provider
    assert learning_service.generation.provider_for_chat(-1) is provider

    # Explicit dependencies only: no extracted component keeps a service backref.
    for component in (
        learning_service.foreground,
        learning_service.autonomous,
        learning_service.generation,
        learning_service.media_coordinator,
        learning_service.memory_facade,
        learning_service.response_lifecycle,
    ):
        assert learning_service not in component.__dict__.values()


def test_public_facade_methods_delegate_to_their_single_owner(tmp_path):
    learning_service = service(tmp_path)
    incoming = message()

    with patch.object(learning_service.memory_facade, "ingest", return_value=(True, None)) as owner:
        assert learning_service.ingest(incoming) == (True, None)
        owner.assert_called_once()
    with patch.object(learning_service.local_responder, "respond", return_value=("local", ())) as owner:
        assert learning_service.generate_free_response(-1, "x") == "local"
        owner.assert_called_once()
    with patch.object(learning_service.media_coordinator, "ingest_gif", return_value=True) as owner:
        assert learning_service.ingest_gif(incoming) is True
        owner.assert_called_once_with(incoming)
    with patch.object(learning_service.foreground, "maybe_reply", return_value="reply") as owner:
        assert learning_service.maybe_reply(incoming) == "reply"
        owner.assert_called_once()
    with patch.object(learning_service.autonomous, "autonomous_diagnostics", return_value={"ok": True}) as owner:
        assert learning_service.autonomous_diagnostics(-1) == {"ok": True}
        owner.assert_called_once_with(-1)


def test_direct_plan_uses_one_snapshot_one_provider_and_existing_event_id(tmp_path):
    provider = Provider()
    learning_service = service(tmp_path, provider)
    learning_service.set_troll_mode(-1, False)
    learning_service.set_media_enabled(-1, False)
    incoming = message()

    with learning_service.telegram_user_event(incoming) as context:
        plan = learning_service.prepare_direct_reply(
            incoming, explicit_address=True
        )
        assert plan.event_id == context.event_id
        assert context.permit.call_count == 1

    assert plan is not None
    assert plan.producer.value == "llm"
    assert len(provider.calls) == 1
    metrics = learning_service._context_snapshot_metrics[plan.event_id]
    assert metrics.db_connections <= 1


def test_import_all_learning_modules_has_no_runtime_cycle():
    imported = []
    for module in pkgutil.iter_modules(learning.__path__, learning.__name__ + "."):
        imported.append(importlib.import_module(module.name).__name__)
    assert "learning.service" in imported
    assert "learning.foreground_orchestrator" in imported
    assert "learning.generation_coordinator" in imported
