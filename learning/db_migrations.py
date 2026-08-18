"""Forward-only SQLite schema evolution for per-chat databases.

Migrations are deliberately small, local and network-free. Existing databases
without metadata are treated as legacy: every migration uses schema inspection
or idempotent DDL, then records its version only after the transaction commits.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


CURRENT_SCHEMA_VERSION = 5


class SchemaMigrationError(RuntimeError):
    """Base error for an unsupported or failed schema transition."""


class FutureSchemaError(SchemaMigrationError):
    """The database was written by a newer CyberChair version."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: object


BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id INTEGER,
    username TEXT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reply_to_message_id INTEGER,
    is_reply INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    UNIQUE(chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_used ON messages(last_used_at);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generated (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generated_created ON generated(created_at);
CREATE TABLE IF NOT EXISTS gifs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id INTEGER,
    file_id TEXT NOT NULL,
    file_unique_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    UNIQUE(chat_id, file_unique_id)
);
CREATE INDEX IF NOT EXISTS idx_gifs_used ON gifs(last_used_at);
CREATE TABLE IF NOT EXISTS stickers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id INTEGER,
    file_id TEXT NOT NULL,
    file_unique_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    UNIQUE(chat_id, file_unique_id)
);
CREATE INDEX IF NOT EXISTS idx_stickers_used ON stickers(last_used_at);
CREATE TABLE IF NOT EXISTS chat_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id INTEGER,
    file_id TEXT NOT NULL,
    file_unique_id TEXT NOT NULL,
    media_type TEXT NOT NULL,
    mime_type TEXT,
    caption TEXT,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL,
    from_bot INTEGER NOT NULL DEFAULT 0,
    used_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    UNIQUE(chat_id, file_unique_id)
);
CREATE INDEX IF NOT EXISTS idx_chat_images_created
    ON chat_images(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_images_used ON chat_images(last_used_at);
CREATE TABLE IF NOT EXISTS chat_image_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_unique_id TEXT NOT NULL,
    user_id INTEGER,
    caption_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_image_usage_created
    ON chat_image_usage(created_at DESC);
CREATE TABLE IF NOT EXISTS daily_summaries (
    day TEXT PRIMARY KEY,
    summary_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS long_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory TEXT NOT NULL UNIQUE,
    score INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_long_memories_score
    ON long_memories(score DESC, updated_at DESC);
CREATE TABLE IF NOT EXISTS summary_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    last_message_row_id INTEGER NOT NULL DEFAULT 0,
    last_summary_at TEXT,
    pending_since TEXT,
    claim_token TEXT,
    claim_start_cursor INTEGER,
    claim_end_row_id INTEGER,
    claim_day TEXT,
    claimed_at TEXT,
    claim_expires_at TEXT,
    attempt_sequence INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT
);
CREATE TABLE IF NOT EXISTS memory_candidates (
    normalized_memory TEXT PRIMARY KEY,
    memory TEXT NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    promoted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_count
    ON memory_candidates(observation_count DESC, last_seen_at DESC);
CREATE TABLE IF NOT EXISTS chat_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scheduled_events (
    event_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS media_metadata (
    media_type TEXT NOT NULL,
    asset_key TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(media_type, asset_key)
);
CREATE TABLE IF NOT EXISTS media_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    asset_key TEXT NOT NULL,
    template_id TEXT,
    cooldown_group TEXT NOT NULL,
    archetype TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_usage_created
    ON media_usage(created_at DESC);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL DEFAULT 0,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    call_type TEXT NOT NULL,
    input_tokens INTEGER,
    cached_input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    cost_usd_ticks INTEGER,
    event_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created
    ON llm_calls(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_type_created
    ON llm_calls(call_type, created_at DESC);
CREATE TABLE IF NOT EXISTS routing_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    event_id TEXT,
    provider_key TEXT,
    call_type TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routing_events_created
    ON routing_events(created_at DESC);
CREATE TABLE IF NOT EXISTS pending_conversations (
    user_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    bot_message_id INTEGER,
    original_message_id INTEGER,
    original_question TEXT NOT NULL,
    clarification_question TEXT NOT NULL,
    intent TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    expected_type TEXT NOT NULL DEFAULT 'short_answer',
    pending_mode TEXT NOT NULL DEFAULT 'hard',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_created
    ON pending_conversations(created_at);
"""


def _execute_statements(db, script):
    statement = ""
    for line in script.splitlines():
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            sql = statement.strip()
            if sql:
                db.execute(sql)
            statement = ""
    if statement.strip():
        raise SchemaMigrationError("incomplete migration SQL")


def _columns(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _add_column(db, table, column, declaration):
    if column not in _columns(db, table):
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _migration_1(db):
    _execute_statements(db, BASE_SCHEMA)


def _migration_2(db):
    _add_column(db, "llm_calls", "chat_id", "INTEGER NOT NULL DEFAULT 0")
    _add_column(db, "llm_calls", "event_id", "TEXT")
    for column in ("event_id", "provider_key", "call_type"):
        _add_column(db, "routing_events", column, "TEXT")
    _add_column(
        db, "pending_conversations", "pending_mode",
        "TEXT NOT NULL DEFAULT 'hard'",
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_routing_events_event "
        "ON routing_events(event_id, event_type)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_calls_event ON llm_calls(event_id)"
    )


def _migration_3(db):
    additions = {
        "claim_token": "TEXT",
        "claim_start_cursor": "INTEGER",
        "claim_end_row_id": "INTEGER",
        "claim_day": "TEXT",
        "claimed_at": "TEXT",
        "claim_expires_at": "TEXT",
        "attempt_sequence": "INTEGER NOT NULL DEFAULT 0",
        "failure_count": "INTEGER NOT NULL DEFAULT 0",
        "next_attempt_at": "TEXT",
    }
    for column, declaration in additions.items():
        _add_column(db, "summary_state", column, declaration)


def _migration_4(db):
    chat_id_info = next(
        row for row in db.execute("PRAGMA table_info(llm_calls)")
        if row[1] == "chat_id"
    )
    if str(chat_id_info[4]) not in {"0", "'0'", '"0"'}:
        db.execute(
            """CREATE TABLE llm_calls_r7 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                call_type TEXT NOT NULL,
                input_tokens INTEGER,
                cached_input_tokens INTEGER,
                output_tokens INTEGER,
                reasoning_tokens INTEGER,
                cost_usd_ticks INTEGER,
                event_id TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        db.execute(
            """INSERT INTO llm_calls_r7(
                id, chat_id, provider, model, call_type, input_tokens,
                cached_input_tokens, output_tokens, reasoning_tokens,
                cost_usd_ticks, event_id, created_at
            ) SELECT id, chat_id, provider, model, call_type, input_tokens,
                cached_input_tokens, output_tokens, reasoning_tokens,
                cost_usd_ticks, event_id, created_at FROM llm_calls"""
        )
        db.execute("DROP TABLE llm_calls")
        db.execute("ALTER TABLE llm_calls_r7 RENAME TO llm_calls")
        db.execute(
            "CREATE INDEX idx_llm_calls_created ON llm_calls(created_at DESC)"
        )
        db.execute(
            "CREATE INDEX idx_llm_calls_type_created "
            "ON llm_calls(call_type, created_at DESC)"
        )
        db.execute("CREATE INDEX idx_llm_calls_event ON llm_calls(event_id)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS persistence_meta ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS llm_daily_aggregates ("
        "day TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, "
        "call_type TEXT NOT NULL, calls INTEGER NOT NULL DEFAULT 0, "
        "input_tokens INTEGER NOT NULL DEFAULT 0, "
        "cached_input_tokens INTEGER NOT NULL DEFAULT 0, "
        "output_tokens INTEGER NOT NULL DEFAULT 0, "
        "reasoning_tokens INTEGER NOT NULL DEFAULT 0, "
        "cost_usd_ticks INTEGER, "
        "PRIMARY KEY(day, provider, model, call_type))"
    )
    # EXPLAIN evidence: these growing-table lookups were scans before R7.
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_reply_to "
        "ON messages(reply_to_message_id)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_image_usage_asset_caption "
        "ON chat_image_usage(file_unique_id, caption_hash)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_generated_kind_created "
        "ON generated(kind, created_at DESC)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_daily_aggregates_day "
        "ON llm_daily_aggregates(day)"
    )


def _migration_5(db):
    """Replace claim markers with the durable P1 scheduled-delivery lifecycle."""
    db.execute("ALTER TABLE scheduled_events RENAME TO scheduled_events_legacy")
    db.execute(
        """CREATE TABLE scheduled_events (
            event_id TEXT PRIMARY KEY,
            event_key TEXT NOT NULL UNIQUE,
            event_kind TEXT NOT NULL,
            scheduled_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            parse_mode TEXT,
            state TEXT NOT NULL CHECK(state IN (
                'PENDING', 'CLAIMED', 'SENDING', 'SENT', 'RETRY_WAIT',
                'UNKNOWN', 'DEAD'
            )),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            claim_token TEXT,
            claimed_at TEXT,
            claim_expires_at TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            safe_retry_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT,
            telegram_message_id INTEGER,
            delivered_at TEXT,
            last_failure_category TEXT,
            last_failure_at TEXT
        )"""
    )
    # A legacy row only proved that the old scheduler had claimed the logical
    # event. Replaying it would create a historical notification storm, so the
    # only safe upgrade semantics is terminal historical DEAD (unconfirmed),
    # never SENT. Empty payloads make accidental delivery impossible.
    db.execute(
        """INSERT INTO scheduled_events(
            event_id, event_key, event_kind, scheduled_at, payload, state,
            created_at, updated_at, last_failure_category, last_failure_at
        )
        SELECT 'legacy_' || event_key, event_key, 'legacy', created_at, '',
               'DEAD', created_at, created_at, 'legacy_unconfirmed', created_at
        FROM scheduled_events_legacy"""
    )
    db.execute("DROP TABLE scheduled_events_legacy")
    db.execute(
        "CREATE INDEX idx_scheduled_events_eligible "
        "ON scheduled_events(state, next_attempt_at, scheduled_at)"
    )
    db.execute(
        "CREATE INDEX idx_scheduled_events_claim_expiry "
        "ON scheduled_events(state, claim_expires_at)"
    )
    db.execute(
        "CREATE INDEX idx_scheduled_events_delivered "
        "ON scheduled_events(state, delivered_at)"
    )


MIGRATIONS = (
    Migration(1, "legacy_core_schema", _migration_1),
    Migration(2, "event_correlation_and_pending", _migration_2),
    Migration(3, "durable_summary_claims", _migration_3),
    Migration(4, "retention_aggregates_and_indexes", _migration_4),
    Migration(5, "reliable_scheduled_delivery", _migration_5),
)


def _metadata_version(db):
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='schema_migrations'"
    ).fetchone()
    if not exists:
        return 0
    row = db.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
    return int(row[0] or 0)


def schema_version(db):
    pragma = int(db.execute("PRAGMA user_version").fetchone()[0] or 0)
    return max(pragma, _metadata_version(db))


def migrate_database(db, before_apply=None):
    """Upgrade one open connection and return the resulting schema version."""
    found = schema_version(db)
    if found > CURRENT_SCHEMA_VERSION:
        raise FutureSchemaError(
            "database schema is newer than this CyberChair version: "
            f"found={found} supported={CURRENT_SCHEMA_VERSION}"
        )
    db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    applied = {
        int(row[0]) for row in db.execute("SELECT version FROM schema_migrations")
    }
    for migration in MIGRATIONS:
        if migration.version in applied:
            continue
        try:
            db.execute("BEGIN IMMEDIATE")
            if before_apply is not None:
                before_apply(migration)
            migration.apply(db)
            stamp = datetime.now(timezone.utc).isoformat()
            db.execute(
                "INSERT INTO schema_migrations(version, name, applied_at) "
                "VALUES(?, ?, ?)",
                (migration.version, migration.name, stamp),
            )
            db.execute(f"PRAGMA user_version={migration.version}")
            db.commit()
        except Exception as error:
            db.rollback()
            if isinstance(error, SchemaMigrationError):
                raise
            raise SchemaMigrationError(
                f"migration {migration.version:04d}_{migration.name} failed"
            ) from error
    final = schema_version(db)
    if final != CURRENT_SCHEMA_VERSION:
        raise SchemaMigrationError(
            f"schema upgrade incomplete: {final}/{CURRENT_SCHEMA_VERSION}"
        )
    return final


def schema_signature(db):
    """Canonical structural signature used by migration diagnostics/tests."""
    tables = {}
    names = [
        row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for name in names:
        quoted = '"' + str(name).replace('"', '""') + '"'
        columns = tuple(sorted(
            (row[1], row[2], row[3], row[4], row[5])
            for row in db.execute(f"PRAGMA table_info({quoted})")
        ))
        indexes = tuple(
            sorted(
                row[1] for row in db.execute(f"PRAGMA index_list({quoted})")
                if not str(row[1]).startswith("sqlite_autoindex")
            )
        )
        tables[name] = (columns, indexes)
    return tables
