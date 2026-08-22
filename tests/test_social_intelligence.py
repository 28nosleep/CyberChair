import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from learning import (
    CURRENT_SCHEMA_VERSION,
    DeliveryReceipt,
    DeliveryType,
    EvidenceEngine,
    LearningService,
    LearningSettings,
    Moment,
    MomentDetector,
    RelationshipModel,
    ResponseKind,
    ResponseSelector,
)
from learning.conversation_policy import ConversationDecision
from learning.normalized_event import normalize_telegram_event
from learning.repository import ChatRepository


NOW = datetime(2026, 8, 22, 12, tzinfo=timezone.utc)


class UnavailableProvider:
    available = False
    provider_key = "none"


class ZeroRng:
    @staticmethod
    def random():
        return 0.0

    @staticmethod
    def choice(values):
        return values[0]


def message(message_id, text, user_id=7, chat_id=-1, when=None, reply_to=None):
    when = when or NOW + timedelta(seconds=message_id)
    reply = None
    if reply_to is not None:
        reply = SimpleNamespace(
            message_id=reply_to,
            text="предыдущая реплика",
            caption=None,
            date=int(when.timestamp()) - 1,
            from_user=SimpleNamespace(
                id=99, username="other", first_name="Other", is_bot=False
            ),
        )
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id), message_id=message_id,
        text=text, caption=None, date=int(when.timestamp()),
        from_user=SimpleNamespace(
            id=user_id, username=f"u{user_id}", first_name="User", is_bot=False
        ),
        reply_to_message=reply,
    )


def row(message_id, text, user_id=7, seconds=0, reply_to=None):
    return {
        "message_id": message_id,
        "text": text,
        "user_id": user_id,
        "created_at": (NOW + timedelta(seconds=seconds)).isoformat(),
        "reply_to_message_id": reply_to,
    }


def settings(tmp_path, **overrides):
    values = dict(
        data_dir=Path(tmp_path), openai_chat_id=-999,
        min_training_messages=1,
        addressed_cooldown=0, generated_cooldown=0,
        max_generated_per_hour=1000, media_cooldown=0,
        meme_render_cooldown=0, media_template_cooldown=0,
    )
    values.update(overrides)
    return LearningSettings(**values)


def relationship(chat_id=-1, user_id=7):
    return SimpleNamespace(
        chat_id=chat_id, user_id=user_id, affinity=.5, irritation=.1,
        respect=.5, interest=.4, troll_tendency=.3, familiarity=.4,
    )


def test_relationship_is_per_user_and_moves_gradually(tmp_path):
    repository = ChatRepository(tmp_path, -1)
    model = RelationshipModel()
    first = normalize_telegram_event(message(1, "спасибо, отлично", user_id=7))
    before_other = model.current(repository, 8)
    after = model.observe(repository, first, ())
    after_other = model.current(repository, 8)

    assert after.affinity > .5
    assert after.familiarity <= .012
    assert abs(after.affinity - .5) <= .02
    assert before_other == after_other
    assert repository.relationship(7)["user_id"] == 7
    assert repository.relationship(8) is None


def test_moment_detector_finds_required_local_moments():
    detector = MomentDetector()
    contradiction_rows = (
        row(1, "больше энергетики не пью", seconds=0),
        row(2, "третий монстр пошёл", seconds=30),
    )
    kinds = {item.moment_type for item in detector.detect(contradiction_rows, 2)}
    assert {"contradiction", "self_own"} <= kinds

    argument = (
        row(10, "нет, docker релиз сломан", 7, 0),
        row(11, "неправда, docker релиз нормальный", 8, 10),
        row(12, "нет, docker релиз сломан", 7, 20),
        row(13, "чушь, docker релиз нормальный", 8, 30),
    )
    assert "argument_loop" in {
        item.moment_type for item in detector.detect(argument, 13)
    }

    burst = tuple(row(20 + i, f"сообщение {i}", i % 3 + 1, i * 5) for i in range(7))
    assert "message_burst" in {
        item.moment_type for item in detector.detect(burst, 26)
    }


def test_boring_chat_and_burst_prefer_silence():
    selector = ResponseSelector(ZeroRng())
    assert selector.select(
        moment=None, relationship=relationship()
    ).kind == ResponseKind.SILENCE
    burst = Moment("message_burst", .95, (7, 8), (1, 2, 3, 4, 5, 6))
    assert selector.select(
        moment=burst, relationship=relationship()
    ).kind == ResponseKind.SILENCE
    assert all(
        selector.select(moment=None, relationship=relationship()).kind
        == ResponseKind.SILENCE
        for _ in range(100)
    )


def test_evidence_requires_and_preserves_real_source(tmp_path):
    repository = ChatRepository(tmp_path, -1)
    assert repository.add_message(1, 7, "u", "больше кофе не пью", NOW)
    stored = repository.add_evidence(
        user_id=7, source_message_id=1, source_timestamp=NOW.isoformat(),
        source_text="выдуманная цитата", evidence_type="promise",
        normalized_topic="кофе", score=.9,
    )
    assert stored["source_text"] == "больше кофе не пью"
    assert stored["source_message_id"] == 1
    assert repository.add_evidence(
        user_id=7, source_message_id=999, source_timestamp=NOW.isoformat(),
        source_text="fake", evidence_type="statement", score=1,
    ) is None


def test_delayed_callback_is_retrievable_and_reuse_limited(tmp_path):
    repository = ChatRepository(tmp_path, -1)
    engine = EvidenceEngine(reuse_cooldown_days=7)
    detector = MomentDetector()
    old_event = normalize_telegram_event(message(
        1, "больше энергетики не пью", when=NOW - timedelta(days=3)
    ))
    repository.add_message(1, 7, "u", old_event.effective_text, old_event.timestamp)
    stored = engine.capture_message(repository, old_event)
    assert stored and stored[0].message_id == 1

    current_event = normalize_telegram_event(message(20, "третий монстр пошёл"))
    repository.add_message(20, 7, "u", current_event.effective_text, current_event.timestamp)
    moment = detector.primary(repository.recent_messages(40), 20)
    candidates = engine.retrieve(
        repository, current_event.effective_text, moment, 7, 20, current=NOW
    )
    assert candidates and candidates[0].text == "больше энергетики не пью"
    assert engine.callback_text(candidates[0], current_event.effective_text) == (
        "Стул всё видел: «больше энергетики не пью»"
    )
    repository.mark_evidence_used(candidates[0].id, NOW)
    assert engine.retrieve(
        repository, current_event.effective_text, moment, 7, 20, current=NOW
    ) == ()


def test_evidence_storage_is_bounded(tmp_path):
    repository = ChatRepository(tmp_path, -1)
    engine = EvidenceEngine(max_items=50)
    for index in range(60):
        event = normalize_telegram_event(message(
            100 + index, f"никогда не пью кофе номер {index}",
            when=NOW + timedelta(seconds=index),
        ))
        repository.add_message(
            event.message_id, 7, "u", event.effective_text, event.timestamp
        )
        engine.capture_message(repository, event)
    assert len(repository.evidence_candidates(7, limit=100)) == 50


def test_reaction_can_win_and_all_media_forms_remain_selectable():
    selector = ResponseSelector(ZeroRng())
    own = Moment("self_own", .8, (7,), (1,))
    assert selector.select(
        moment=own, relationship=relationship()
    ).kind == ResponseKind.REACTION

    meme = Moment("meme_opportunity", .8, (7,), (2,))
    assert selector.select(
        moment=meme, relationship=relationship(), memory_meme_available=True
    ).kind == ResponseKind.MEME
    assert selector.select(
        moment=meme, relationship=relationship(), media_enabled=True
    ).kind == ResponseKind.GIF
    assert selector.select(
        moment=meme, relationship=relationship(), media_enabled=True,
        recent_media_usage=({"action": "gif"},),
    ).kind == ResponseKind.STICKER


def test_memory_meme_uses_old_chat_image_and_current_real_quote(tmp_path):
    service = LearningService(
        settings(tmp_path), llm_provider=UnavailableProvider(), rng=ZeroRng()
    )
    repository = service.repository(-1)
    repository.add_chat_image(
        1, 7, "file-id", "unique-id", "photo", caption="старое фото",
        width=800, height=600, created_at=NOW - timedelta(days=5),
    )
    repository.add_message(2, 7, "u", "третий монстр пошёл", NOW)
    decision = service.media.memory_meme(
        repository, "третий монстр пошёл", 7, 2
    )
    assert decision.action == "meme"
    assert decision.background_file_id == "file-id"
    assert decision.background_message_id == 1
    assert decision.source_message_id == 2
    assert decision.caption_text == "третий монстр пошёл"


def test_direct_required_question_bypasses_social_silence(tmp_path):
    service = LearningService(
        settings(tmp_path), llm_provider=UnavailableProvider(), rng=ZeroRng()
    )
    service.set_media_enabled(-1, False)
    incoming = message(1, "стул как выбрать монитор?")
    with patch.object(service.response_selector, "select") as selector:
        response = service.maybe_direct_reply(incoming, explicit_address=True)
    assert response
    assert "монитор" in response.casefold()
    selector.assert_not_called()


def test_social_evidence_plan_is_delayed_grounded_and_one_output(tmp_path):
    service = LearningService(
        settings(tmp_path), llm_provider=UnavailableProvider(), rng=ZeroRng()
    )
    service.set_media_enabled(-1, False)
    old = message(1, "больше энергетики не пью", when=NOW - timedelta(days=3))
    with service.telegram_user_event(old):
        service.ingest(old)
    current = message(20, "третий монстр пошёл", when=NOW)
    forced = ConversationDecision(
        "reply", 1, .7, 30, "direct_mocking", 20, 7,
        "forced_social", 1, 0,
    )
    with (
        patch.object(service.conversation_policy, "decide", return_value=forced),
        patch.object(service.triggers, "allowed", return_value=True),
        service.telegram_user_event(current) as event,
    ):
        service.ingest(current)
        service.context_snapshot(current)
        with service.response_planning():
            plan = service.maybe_reply(current)
    assert plan.producer.value == "evidence"
    assert plan.delivery_type == DeliveryType.TEXT
    assert plan.payload.text == "Стул всё видел: «больше энергетики не пью»"
    assert event.permit.call_count == 0
    assert service.commit_response(
        plan, DeliveryReceipt(plan.event_id, True, plan.delivery_type, 501)
    )
    evidence = service.repository(-1).evidence_candidates(7)[0]
    assert evidence["use_count"] == 1


def test_social_reaction_uses_response_plan_lifecycle(tmp_path):
    service = LearningService(
        settings(tmp_path), llm_provider=UnavailableProvider(), rng=ZeroRng()
    )
    service.set_media_enabled(-1, False)
    current = message(30, "опять сломал релиз", when=NOW)
    forced = ConversationDecision(
        "reply", 1, .7, 30, "direct_mocking", 30, 7,
        "forced_social", 1, 0,
    )
    with (
        patch.object(service.conversation_policy, "decide", return_value=forced),
        patch.object(service.triggers, "allowed", return_value=True),
        service.telegram_user_event(current),
    ):
        service.ingest(current)
        service.context_snapshot(current)
        with service.response_planning():
            plan = service.maybe_reply(current)
    assert plan.delivery_type == DeliveryType.REACTION
    assert plan.payload.emoji == "🤡"
    assert service.commit_response(
        plan, DeliveryReceipt(plan.event_id, True, plan.delivery_type)
    )
    assert service.repository(-1).latest_generated()["kind"] == "social_reaction"


def test_local_responses_have_structural_diversity_and_no_canned_skeleton(tmp_path):
    service = LearningService(
        settings(tmp_path), llm_provider=UnavailableProvider(), rng=random.Random(11)
    )
    repository = service.repository(-1)
    outputs = []
    signatures = set()
    for index in range(100):
        result, _ = service.local_responder.respond(
            -1, f"я снова сломал проект номер {index}", "social", repository,
            recent_generated=outputs[-40:], user_id=7, username="саня",
        )
        outputs.append(result)
        signatures.add(result.construction_signature)
    joined = " ".join(outputs).casefold()
    assert len(set(outputs)) >= 95
    assert len(signatures) >= 8
    assert "сильный заход" not in joined
    assert "доказательная база вышла покурить" not in joined
    assert "уверенность есть" not in joined


def test_schema_v6_and_physical_forget_remove_social_state(tmp_path):
    repository = ChatRepository(tmp_path, -1)
    assert repository.current_schema_version() == CURRENT_SCHEMA_VERSION == 6
    repository.add_message(1, 7, "u", "никогда не опаздываю", NOW)
    repository.add_evidence(
        user_id=7, source_message_id=1, source_timestamp=NOW.isoformat(),
        source_text="ignored", evidence_type="statement", score=.8,
    )
    RelationshipModel().observe(
        repository, normalize_telegram_event(message(2, "спасибо")), ()
    )
    repository.record_response_structure("quote_plus_reaction", "quote")
    repository.clear()
    report = repository.persistence_diagnostics()["rows_by_table"]
    assert report["relationships"] == 0
    assert report["evidence"] == 0
    assert report["response_structures"] == 0
