import json
from datetime import datetime, timedelta, timezone

from learning.repository import ChatRepository


NOW = datetime(2026, 8, 18, 12, tzinfo=timezone.utc)


def bulk_operational_rows(repository, old_count=2000, recent_count=20):
    old = (NOW - timedelta(days=100)).isoformat()
    recent = (NOW - timedelta(days=2)).isoformat()
    with repository._lock, repository._connect() as db, db:
        db.executemany(
            """INSERT INTO llm_calls(
                chat_id,provider,model,call_type,input_tokens,
                cached_input_tokens,output_tokens,reasoning_tokens,
                cost_usd_ticks,event_id,created_at
            ) VALUES(-1,'p','m','reply',2,1,3,0,5,NULL,?)""",
            [(old,)] * old_count + [(recent,)] * recent_count,
        )
        db.executemany(
            "INSERT INTO routing_events(event_type,created_at) VALUES('x',?)",
            [(old,)] * 50 + [(recent,)] * 5,
        )
        db.executemany(
            """INSERT INTO scheduled_events(
                event_id,event_key,event_kind,scheduled_at,payload,state,
                created_at,updated_at,delivered_at
            ) VALUES(?,?, 'test',?,'','SENT',?,?,?)""",
            [
                (f"old-id-{i}", f"old-{i}", old, old, old, old)
                for i in range(10)
            ] + [("recent-id", "recent", recent, recent, recent, recent)],
        )
        db.execute(
            "INSERT INTO media_metadata VALUES('gif','orphan','[]')"
        )
        db.execute(
            "INSERT INTO chat_image_usage(file_unique_id,caption_hash,created_at) "
            "VALUES('orphan','h',?)", (old,),
        )


def test_llm_retention_is_idempotent_and_preserves_historical_cost(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    bulk_operational_rows(repository)
    before = repository.llm_usage_report("2000-01-01T00:00:00+00:00")
    assert before["total"]["calls"] == 2020
    assert before["total"]["cost_usd_ticks"] == 10100

    result = repository.run_persistence_maintenance(NOW, force=True)
    assert result == {
        "status": "completed",
        "llm_calls_pruned": 2000,
        "routing_events_pruned": 50,
        "scheduled_events_pruned": 10,
        "media_metadata_pruned": 1,
        "chat_image_usage_pruned": 1,
    }
    after = repository.llm_usage_report("2000-01-01T00:00:00+00:00")
    assert after["total"] == before["total"]
    diagnostics = repository.persistence_diagnostics()
    assert diagnostics["rows_by_table"]["llm_calls"] == 20
    assert diagnostics["rows_by_table"]["llm_daily_aggregates"] == 1
    assert diagnostics["rows_by_table"]["routing_events"] == 5
    assert diagnostics["rows_by_table"]["scheduled_events"] == 1

    repeated = repository.run_persistence_maintenance(NOW, force=True)
    assert repeated["llm_calls_pruned"] == 0
    assert repository.llm_usage_report(
        "2000-01-01T00:00:00+00:00"
    )["total"] == before["total"]


def test_retention_cadence_and_diagnostics_do_not_expose_content(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    bulk_operational_rows(repository, old_count=2, recent_count=1)
    assert repository.run_persistence_maintenance(NOW, force=True)["status"] == "completed"
    assert repository.run_persistence_maintenance(NOW)["status"] == "not_due"
    report = repository.persistence_diagnostics()
    assert report["schema_version"] == report["latest_schema_version"]
    assert report["page_count"] > 0
    assert report["db_size_bytes"] > 0
    rendered = json.dumps(report, ensure_ascii=False)
    assert "legacy text" not in rendered
    assert repository.quick_check() == ("ok",)


def test_operational_retention_never_prunes_unsummarized_messages(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    for index in range(120):
        assert repository.add_message(
            index, 7, "u", f"protected {index}", NOW + timedelta(seconds=index)
        )
    before = repository.messages_after(0)
    repository.run_persistence_maintenance(NOW + timedelta(days=100), force=True)
    after = repository.messages_after(0)
    assert [row["message_id"] for row in after] == [
        row["message_id"] for row in before
    ]


def test_r7_cleanup_leaves_r5_summary_and_candidate_retention_owned_by_r5(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    old = (NOW - timedelta(days=200)).isoformat()
    with repository._connect() as db, db:
        db.execute(
            "INSERT INTO daily_summaries VALUES('2025-01-01','{}',?)", (old,)
        )
        db.execute(
            "INSERT INTO memory_candidates VALUES('old','old fact',1,?,?,NULL)",
            (old, old),
        )
    repository.run_persistence_maintenance(NOW, force=True)
    with repository._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM daily_summaries").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM memory_candidates").fetchone()[0] == 1


def test_100_day_growth_compacts_operational_detail(tmp_path):
    repository = ChatRepository(tmp_path, -1, 50, 500)
    with repository._lock, repository._connect() as db, db:
        calls = []
        routing = []
        for day in range(100):
            stamp = (NOW - timedelta(days=day)).isoformat()
            calls.extend([(stamp,)] * 20)
            routing.extend([(stamp,)] * 10)
        db.executemany(
            """INSERT INTO llm_calls(
                chat_id,provider,model,call_type,input_tokens,output_tokens,
                event_id,created_at
            ) VALUES(-1,'p','m','reply',1,1,NULL,?)""", calls,
        )
        db.executemany(
            "INSERT INTO routing_events(event_type,created_at) VALUES('x',?)",
            routing,
        )
    repository.run_persistence_maintenance(NOW, force=True)
    rows = repository.persistence_diagnostics()["rows_by_table"]
    assert rows["llm_calls"] <= 91 * 20
    assert rows["llm_daily_aggregates"] <= 10
    assert rows["routing_events"] <= 32 * 10
