import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from learning import (
    CURRENT_SCHEMA_VERSION,
    DeliveryReceipt,
    LearningService,
    LearningSettings,
    Producer,
    SummaryJob,
)
from learning.normalized_event import normalize_telegram_event
from learning.repository import ChatRepository


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


class Provider:
    available = False
    provider_key = "none"

    def summarize(self, request):
        raise AssertionError


def message(message_id, text="стул тест", chat_id=-1):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id), message_id=message_id,
        text=text, caption=None, content_type="text", date=1_776_000_000,
        from_user=SimpleNamespace(
            id=7, username="u", first_name="U", is_bot=False
        ),
        reply_to_message=None,
    )


def populate_every_data_table(repository):
    stamp = NOW.isoformat()
    with repository._lock, repository._connect() as db, db:
        db.execute("INSERT INTO messages(chat_id,message_id,text,created_at) VALUES(-1,1,'private',?)", (stamp,))
        db.execute("INSERT INTO settings VALUES('talk','0')")
        db.execute("INSERT INTO generated(text,kind,created_at) VALUES('bot','reply',?)", (stamp,))
        db.execute("INSERT INTO gifs(chat_id,message_id,file_id,file_unique_id,created_at) VALUES(-1,1,'f','gu',?)", (stamp,))
        db.execute("INSERT INTO stickers(chat_id,message_id,file_id,file_unique_id,created_at) VALUES(-1,2,'s','su',?)", (stamp,))
        db.execute("INSERT INTO chat_images(chat_id,message_id,file_id,file_unique_id,media_type,created_at) VALUES(-1,3,'i','iu','photo',?)", (stamp,))
        db.execute("INSERT INTO chat_image_usage(file_unique_id,caption_hash,created_at) VALUES('iu','h',?)", (stamp,))
        db.execute("INSERT INTO daily_summaries VALUES('2026-08-18','{}',?)", (stamp,))
        db.execute("INSERT INTO long_memories(memory,updated_at) VALUES('fact',?)", (stamp,))
        db.execute("INSERT INTO memory_candidates VALUES('candidate','Candidate',1,?,?,NULL)", (stamp, stamp))
        db.execute("INSERT INTO chat_stats VALUES('total_messages','1')")
        db.execute(
            """INSERT INTO scheduled_events(
                event_id,event_key,event_kind,scheduled_at,payload,state,
                created_at,updated_at
            ) VALUES('sched_test','event','test',?,'payload','PENDING',?,?)""",
            (stamp, stamp, stamp),
        )
        db.execute("INSERT INTO media_metadata VALUES('gif','gu','[]')")
        db.execute("INSERT INTO media_usage(action,asset_key,cooldown_group,archetype,created_at) VALUES('gif','gu','g','a',?)", (stamp,))
        db.execute("INSERT INTO llm_calls(chat_id,provider,model,call_type,created_at) VALUES(-1,'p','m','reply',?)", (stamp,))
        db.execute("INSERT INTO llm_daily_aggregates(day,provider,model,call_type,calls) VALUES('2026-01-01','p','m','reply',1)")
        db.execute("INSERT INTO routing_events(event_type,created_at) VALUES('route',?)", (stamp,))
        db.execute("INSERT INTO pending_conversations(user_id,chat_id,original_question,clarification_question,intent,created_at) VALUES(7,-1,'q','c','x',?)", (stamp,))
        db.execute("INSERT INTO persistence_meta VALUES('last_retention_at',?)", (stamp,))
        db.execute("CREATE TABLE custom_private(value TEXT)")
        db.execute("INSERT INTO custom_private VALUES('private extension')")


def test_forget_physically_replaces_database_and_all_chat_state(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    populate_every_data_table(repository)
    old_inode = repository.path.stat().st_ino

    assert repository.clear() is True
    assert repository.quick_check() == ("ok",)
    assert repository.path.stat().st_ino != old_inode
    assert repository.current_schema_version() == CURRENT_SCHEMA_VERSION
    report = repository.persistence_diagnostics()["rows_by_table"]
    for table, rows in report.items():
        expected = CURRENT_SCHEMA_VERSION if table == "schema_migrations" else (
            1 if table == "summary_state" else 0
        )
        assert rows == expected, table
    with repository._connect() as db:
        assert db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='custom_private'"
        ).fetchone() is None
    assert repository.setting("talk") is None
    assert repository.pending_conversation(7, 1200, NOW) is None
    assert repository.llm_usage_report("2000-01-01T00:00:00+00:00")["total"]["calls"] == 0
    assert repository.add_message(2, 7, "u", "new clean state", NOW)


def test_old_summary_job_cannot_resurrect_after_forget(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    assert repository.add_message(1, 7, "u", "old secret", NOW)
    claim, status = repository.claim_summary_range(0, 1, "2026-08-18", NOW, 300)
    assert status == "claimed"
    job = SummaryJob(
        event_id=claim["event_id"], chat_id=-1, logical_day="2026-08-18",
        start_cursor=0, end_message_row_id=1, prior_summary_json=None,
        messages=(), created_at=claim["created_at"],
        claim_expires_at=claim["claim_expires_at"],
        attempt_sequence=claim["attempt_sequence"],
    )
    repository.clear()
    result = repository.finalize_summary_job(
        job, {"summary": "resurrected"}, (), NOW
    )
    assert result.status == "stale"
    assert repository.summary_for_day("2026-08-18") is None
    assert repository.stable_memories(10) == []


def test_same_chat_gate_orders_response_commit_before_forget(tmp_path):
    service = LearningService(
        LearningSettings(data_dir=Path(tmp_path)), llm_provider=Provider()
    )
    response_event = normalize_telegram_event(message(10))
    forget_event = normalize_telegram_event(message(11, "/forget_chat confirm"))
    inside = threading.Event()
    release = threading.Event()
    forgotten = threading.Event()

    def response_worker():
        with service.chat_event_slot(response_event):
            plan = service.prepare_text_response(
                response_event, "delivered before forget", producer=Producer.LOCAL,
            )
            inside.set()
            release.wait(timeout=2)
            receipt = DeliveryReceipt(
                plan.event_id, True, plan.delivery_type, 900, None
            )
            assert service.finalize_response(plan, receipt)

    def forget_worker():
        with service.chat_event_slot(forget_event):
            service.forget_chat(-1)
            forgotten.set()

    first = threading.Thread(target=response_worker)
    second = threading.Thread(target=forget_worker)
    first.start()
    assert inside.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    assert not forgotten.is_set()
    release.set()
    first.join(timeout=3)
    second.join(timeout=3)
    assert forgotten.is_set()
    assert service.repository(-1).recent_generated(10) == []
