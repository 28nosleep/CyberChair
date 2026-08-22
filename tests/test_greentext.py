from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from learning import LearningService, LearningSettings, TextPayload
from learning.greentext import is_greentext_command


NOW = datetime(2026, 8, 22, 12, tzinfo=timezone.utc)


class Provider:
    available = True
    provider_key = "test"

    def __init__(self, result):
        self.result, self.calls = result, []

    def generate(self, request):
        self.calls.append(request)
        return self.result

    def summarize(self, request):
        return None


def event(mid, text, user=7):
    return SimpleNamespace(
        chat=SimpleNamespace(id=-1), message_id=mid, text=text, caption=None,
        date=int((NOW + timedelta(seconds=mid)).timestamp()),
        from_user=SimpleNamespace(id=user, username=f"u{user}", is_bot=False),
        reply_to_message=None,
    )


def service(tmp_path, output):
    provider = Provider(output)
    instance = LearningService(LearningSettings(
        data_dir=Path(tmp_path), openai_chat_id=-1, min_training_messages=1,
        addressed_cooldown=0, generated_cooldown=0, max_generated_per_hour=999,
    ), llm_provider=provider)
    return instance, provider


def plan_for(instance):
    for item in (
        event(1, "я больше энергетики не пью", 7),
        event(2, "да ты вчера то же самое говорил", 8),
        event(3, "третий монстр пошёл", 7),
        event(4, "ну конечно", 8),
    ):
        instance.ingest_event(item, refresh_memory=False)
    command = event(5, "стул грин текст", 9)
    instance.ingest_event(command, refresh_memory=False)
    with instance.response_planning():
        instance.context_snapshot(command)
        return instance.foreground.prepare_greentext(command)


def test_command_spellings_are_explicit():
    assert all(is_greentext_command(text) for text in (
        "стул грин текст", "стул гринтекст", "стул greentext", "стул green text",
    ))


def test_grounded_greentext_is_one_llm_call_and_valid_shape(tmp_path):
    output = "\n".join((
        ">я больше энергетики не пью", ">энергетики не пью",
        ">третий монстр пошёл", ">третий монстр", ">монстр пошёл", ">не пью",
    ))
    instance, provider = service(tmp_path, output)
    plan = plan_for(instance)
    assert plan.payload.text == output
    assert len(provider.calls) == 1
    assert provider.calls[0].metadata["call_type"] == "greentext"
    assert "linked_evidence_ids=" in provider.calls[0].input


def test_fake_llm_event_is_rejected_to_local_fallback_without_retry(tmp_path):
    instance, provider = service(tmp_path, ">улететь на марс\n>позвать васю\n>взорвать луну\n>стать президентом\n>купить ракету")
    plan = plan_for(instance)
    assert plan.producer.value == "local"
    assert len(provider.calls) == 1
    assert all(line.startswith(">") for line in plan.payload.text.splitlines())
    assert "марс" not in plan.payload.text


def test_unavailable_llm_uses_grounded_fallback(tmp_path):
    instance, provider = service(tmp_path, "ignored")
    provider.available = False
    plan = plan_for(instance)
    assert plan.producer.value == "local"
    assert provider.calls == []
    lines = plan.payload.text.splitlines()
    assert 5 <= len(lines) <= 10 and all(line.startswith(">") for line in lines)


def test_repeat_prefers_other_episode(tmp_path):
    instance, provider = service(tmp_path, None)
    first = plan_for(instance)
    # Persist commits in production; commit only the generated marker here.
    for action in first.commit_actions:
        if action.__class__.__name__ == "GeneratedCommit":
            instance.repository(-1).record_generated(action.text, action.kind)
    # An absurd later episode gives the selector a distinct choice.
    instance.ingest_event(event(6, "рептилоиды опять обсуждают вечный двигатель", 10), refresh_memory=False)
    command = event(7, "стул green text", 9)
    instance.ingest_event(command, refresh_memory=False)
    with instance.response_planning():
        instance.context_snapshot(command)
        second = instance.foreground.prepare_greentext(command)
    assert second.commit_actions[1].kind != first.commit_actions[1].kind
