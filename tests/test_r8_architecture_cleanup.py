import ast
import re
from pathlib import Path

import learning
from learning import LearningService, LearningSettings


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_COMPONENTS = (
    "autonomous_coordinator.py",
    "foreground_orchestrator.py",
    "generation_coordinator.py",
    "media_coordinator.py",
    "memory_facade.py",
    "memory_maintenance.py",
    "response_lifecycle.py",
)


class Provider:
    available = True
    provider_key = "r8-fake"

    def generate(self, request):
        return "готовый ответ"

    def summarize(self, request):
        raise AssertionError("construction must not summarize")


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def test_component_dependency_direction_and_transport_boundary():
    for filename in INTERNAL_COMPONENTS:
        path = ROOT / "learning" / filename
        assert "learning.service" not in imported_modules(path)
        assert ".service" not in imported_modules(path)

    for path in (ROOT / "learning").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        telegram_sends = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in {
                "send_message", "send_photo", "send_animation", "send_sticker"
            }
        }
        assert telegram_sends == set(), path


def test_composition_keeps_single_provider_rng_and_state_owners(tmp_path):
    provider = Provider()
    service = LearningService(
        LearningSettings(data_dir=tmp_path, openai_chat_id=-1),
        llm_provider=provider,
    )
    assert service.provider_for_chat(-1) is provider
    assert service.generation.provider_for_chat(-1) is provider
    assert service.foreground.rng is service.rng
    assert service.media_coordinator.rng is service.rng
    assert service.autonomous.rng is service.rng
    assert not hasattr(service, "_models")
    assert not hasattr(service, "_last_direct_decision")
    assert not hasattr(service, "_committed_response_events")


def test_removed_legacy_symbols_are_not_exposed():
    assert not hasattr(LearningService, "generate_openai")
    assert not hasattr(LearningService, "openai_allowed")
    assert not hasattr(LearningService, "_maybe_refresh_memory")
    assert not hasattr(LearningService, "maybe_question_reply")
    assert not hasattr(LearningService, "maybe_random_media")
    assert not hasattr(LearningService, "maybe_autonomous")
    assert not hasattr(learning, "ForegroundOrchestrator")
    assert not hasattr(learning, "ResponseLifecycle")

    repository_source = (ROOT / "learning" / "repository.py").read_text(
        encoding="utf-8"
    )
    assert "_legacy_initialize_bootstrap" not in repository_source
    assert "_legacy_clear_rows" not in repository_source


def test_env_example_matches_supported_configuration():
    sources = [*ROOT.glob("*.py"), *(ROOT / "learning").glob("*.py")]
    source = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    supported = set(re.findall(
        r"(?:os\.)?(?:getenv|environ\.get)\(\s*[\"']([A-Z][A-Z0-9_]*)",
        source,
    ))
    settings_source = (ROOT / "learning" / "settings.py").read_text(
        encoding="utf-8"
    )
    supported.update(re.findall(
        r"_(?:bool|int|float|optional_int)\(\s*[\"']([A-Z][A-Z0-9_]*)",
        settings_source,
    ))
    example = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if re.match(r"^[A-Z][A-Z0-9_]*=", line)
    }
    legacy_aliases = {"BOT_TOKEN", "OPENAI_RANDOM_REPLY_CHANCE"}
    assert example <= supported
    assert supported - example == legacy_aliases


def test_current_architecture_document_records_hard_invariants():
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for phrase in (
        "at most one LLM network call",
        "at most one final `ResponsePlan`",
        "after Telegram success",
        "one base `ContextSnapshot`",
        "Same-chat foreground lifecycles are serial",
        "Summary generation never runs inside a foreground Telegram event",
        "forward-only and versioned",
        "physically replaces the per-chat database",
    ):
        assert phrase in architecture
