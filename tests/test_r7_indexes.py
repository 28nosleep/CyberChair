import sqlite3

from learning.db_migrations import _migration_4, migrate_database


def plan(db, sql, params=()):
    return " ".join(
        str(row[3]).casefold()
        for row in db.execute("EXPLAIN QUERY PLAN " + sql, params)
    )


def test_evidence_indexes_replace_growing_table_scans():
    db = sqlite3.connect(":memory:")
    migrate_database(db)
    db.execute("DROP INDEX idx_messages_reply_to")
    db.execute("DROP INDEX idx_chat_image_usage_asset_caption")
    db.execute("DROP INDEX idx_generated_kind_created")
    before = {
        "reply": plan(db, "SELECT * FROM messages WHERE reply_to_message_id=?", (1,)),
        "image": plan(db, "SELECT 1 FROM chat_image_usage WHERE file_unique_id=? AND caption_hash=?", ("f", "h")),
        "generated": plan(db, "SELECT * FROM generated WHERE kind=? ORDER BY created_at DESC LIMIT 1", ("x",)),
    }
    assert all("scan" in value for value in before.values())

    _migration_4(db)
    after = {
        "reply": plan(db, "SELECT * FROM messages WHERE reply_to_message_id=?", (1,)),
        "image": plan(db, "SELECT 1 FROM chat_image_usage WHERE file_unique_id=? AND caption_hash=?", ("f", "h")),
        "generated": plan(db, "SELECT * FROM generated WHERE kind=? ORDER BY created_at DESC LIMIT 1", ("x",)),
    }
    assert "idx_messages_reply_to" in after["reply"]
    assert "idx_chat_image_usage_asset_caption" in after["image"]
    assert "idx_generated_kind_created" in after["generated"]
    indexes = {
        row[1] for row in db.execute("PRAGMA index_list(memory_candidates)")
    }
    assert "idx_memory_candidates_promoted" not in indexes
    db.close()
