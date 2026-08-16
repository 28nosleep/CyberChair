import json
import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path


def normalize_memory(text):
    normalized = str(text).casefold().replace("ё", "е").strip()
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    normalized = re.sub(r"[_-]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def memories_are_similar(left, right):
    left_normalized = normalize_memory(left)
    right_normalized = normalize_memory(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    left_words = set(left_normalized.split())
    right_words = set(right_normalized.split())
    shared_ratio = len(left_words & right_words) / max(1, min(len(left_words), len(right_words)))
    return (
        SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.78
        or shared_ratio >= 0.65
    )


class ChatRepository:
    """SQLite storage isolated in a separate file for every Telegram chat."""

    def __init__(self, data_dir, chat_id, max_messages=50000):
        self.chat_id = int(chat_id)
        self.max_messages = max_messages
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_id = str(self.chat_id).replace("-", "m")
        self.path = data_dir / f"chat_{safe_id}.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self):
        with self._lock, closing(self._connect()) as db, db:
            db.executescript("""
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
                CREATE INDEX IF NOT EXISTS idx_chat_images_used
                    ON chat_images(last_used_at);
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
                    pending_since TEXT
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
                    chat_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    call_type TEXT NOT NULL,
                    input_tokens INTEGER,
                    cached_input_tokens INTEGER,
                    output_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    cost_usd_ticks INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_llm_calls_created
                    ON llm_calls(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_llm_calls_type_created
                    ON llm_calls(call_type, created_at DESC);
                CREATE TABLE IF NOT EXISTS routing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
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
            """)
            llm_call_columns = {
                row[1] for row in db.execute("PRAGMA table_info(llm_calls)")
            }
            if "chat_id" not in llm_call_columns:
                db.execute(
                    "ALTER TABLE llm_calls ADD COLUMN chat_id INTEGER NOT NULL DEFAULT 0"
                )
            pending_columns = {
                row[1] for row in db.execute("PRAGMA table_info(pending_conversations)")
            }
            if "pending_mode" not in pending_columns:
                db.execute(
                    "ALTER TABLE pending_conversations "
                    "ADD COLUMN pending_mode TEXT NOT NULL DEFAULT 'hard'"
                )
            # One-time/boot migration for databases created by versions that
            # retained a large transcript. Preserve the aggregate count, then
            # keep only the newest short-memory window.
            existing_total = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            if existing_total:
                db.execute(
                    "INSERT OR IGNORE INTO chat_stats(key, value) VALUES('total_messages', ?)",
                    (str(existing_total),),
                )
            summary_state_exists = db.execute(
                "SELECT 1 FROM summary_state WHERE singleton = 1"
            ).fetchone()
            if not summary_state_exists:
                latest_summary = db.execute(
                    "SELECT updated_at FROM daily_summaries ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
                migrated_cursor = (
                    db.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM messages WHERE created_at <= ?",
                        (latest_summary[0],),
                    ).fetchone()[0]
                    if latest_summary
                    else 0
                )
                db.execute(
                    "INSERT INTO summary_state(singleton, last_message_row_id, "
                    "last_summary_at, pending_since) VALUES(1, ?, ?, NULL)",
                    (migrated_cursor, latest_summary[0] if latest_summary else None),
                )
            db.execute(
                "DELETE FROM messages WHERE id NOT IN "
                "(SELECT id FROM messages ORDER BY id DESC LIMIT ?)",
                (self.max_messages,),
            )
            db.execute(
                "DELETE FROM generated WHERE id NOT IN "
                "(SELECT id FROM generated ORDER BY id DESC LIMIT ?)",
                (self.max_messages,),
            )

    def add_message(self, message_id, user_id, username, text, created_at=None,
                    reply_to_message_id=None, is_reply=False):
        created_at = created_at or datetime.now(timezone.utc)
        stamp = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO messages
                (chat_id, message_id, user_id, username, text, created_at,
                 reply_to_message_id, is_reply) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.chat_id, message_id, user_id, username, text, stamp,
                 reply_to_message_id, int(is_reply)),
            )
            inserted = cursor.rowcount > 0
            if inserted:
                self._record_message_stats(db, text, stamp)
                overflow = db.execute("SELECT COUNT(*) - ? FROM messages", (self.max_messages,)).fetchone()[0]
                if overflow > 0:
                    db.execute("DELETE FROM messages WHERE id IN (SELECT id FROM messages ORDER BY id LIMIT ?)", (overflow,))
            return inserted

    def _record_message_stats(self, db, text, stamp):
        """Update non-AI aggregates without retaining another copy of the text."""
        total_row = db.execute(
            "SELECT value FROM chat_stats WHERE key = 'total_messages'"
        ).fetchone()
        total = int(total_row[0]) + 1 if total_row else 1
        db.execute(
            "INSERT INTO chat_stats(key, value) VALUES('total_messages', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(total),),
        )
        db.execute(
            "INSERT INTO chat_stats(key, value) VALUES('last_message_at', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (stamp,),
        )
        try:
            moment = datetime.fromisoformat(stamp)
        except ValueError:
            moment = datetime.now(timezone.utc)
        for key, bucket in (
            ("activity_by_hour", f"{moment.hour:02d}"),
            ("activity_by_minute", moment.replace(second=0, microsecond=0).isoformat()),
        ):
            row = db.execute("SELECT value FROM chat_stats WHERE key = ?", (key,)).fetchone()
            activity = json.loads(row[0]) if row else {}
            activity[bucket] = int(activity.get(bucket, 0)) + 1
            if key == "activity_by_minute" and len(activity) > 120:
                activity = dict(sorted(activity.items())[-120:])
            db.execute(
                "INSERT INTO chat_stats(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(activity, ensure_ascii=False)),
            )
        row = db.execute(
            "SELECT value FROM chat_stats WHERE key = 'word_mentions'"
        ).fetchone()
        mentions = json.loads(row[0]) if row else {}
        for word in re.findall(r"[\wёЁ-]{3,}", text.casefold()):
            mentions[word] = int(mentions.get(word, 0)) + 1
        if len(mentions) > 500:
            mentions = dict(sorted(mentions.items(), key=lambda item: item[1], reverse=True)[:500])
        db.execute(
            "INSERT INTO chat_stats(key, value) VALUES('word_mentions', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(mentions, ensure_ascii=False),),
        )

    def count(self):
        with self._lock, closing(self._connect()) as db, db:
            return db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    def recent_messages(self, limit=5000):
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def meme_source_messages(self, limit=5000):
        """Human quotes plus reply engagement, for deterministic meme ranking."""
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                """SELECT m.*, COUNT(r.id) AS reply_count
                FROM messages AS m
                LEFT JOIN messages AS r ON r.reply_to_message_id = m.message_id
                GROUP BY m.id ORDER BY m.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def recent_activity_count(self, since_iso):
        with self._lock, closing(self._connect()) as db, db:
            return db.execute("SELECT COUNT(*) FROM messages WHERE created_at >= ?", (since_iso,)).fetchone()[0]

    def messages_since(self, since_iso, limit=50):
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT * FROM messages WHERE created_at >= ? ORDER BY id DESC LIMIT ?",
                (since_iso, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def messages_after(self, row_id):
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT * FROM messages WHERE id > ? ORDER BY id ASC",
                (int(row_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def short_term_dialogue(self, since_iso, limit=50):
        """Return the newest dialogue rows satisfying both count and age bounds."""
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                """SELECT row_id, message_id, user_id, text, created_at, speaker,
                    username, reply_to_message_id, is_reply FROM (
                    SELECT id AS row_id, message_id, user_id, text, created_at,
                        'user' AS speaker, username, reply_to_message_id, is_reply
                    FROM messages
                    WHERE created_at >= ?
                    UNION ALL
                    SELECT id AS row_id, NULL AS message_id, NULL AS user_id, text,
                        created_at, 'cyberchair' AS speaker, NULL AS username,
                        NULL AS reply_to_message_id, 0 AS is_reply
                    FROM generated WHERE kind NOT IN ('random_media', 'contextual_media')
                        AND created_at >= ?
                ) ORDER BY created_at DESC LIMIT ?""",
                (since_iso, since_iso, int(limit)),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def mark_used(self, texts):
        if not texts:
            return
        with self._lock, closing(self._connect()) as db, db:
            db.executemany("UPDATE messages SET last_used_at = ? WHERE text = ?", [(datetime.now(timezone.utc).isoformat(), text) for text in texts])

    def setting(self, key, default=None):
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key, value):
        with self._lock, closing(self._connect()) as db, db:
            db.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    def save_pending_conversation(
        self, user_id, original_message_id, original_question,
        clarification_question, intent, context="", expected_type="short_answer",
        pending_mode="hard",
        bot_message_id=None, created_at=None,
    ):
        stamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                """INSERT INTO pending_conversations(
                    user_id, chat_id, bot_message_id, original_message_id,
                    original_question, clarification_question, intent, context,
                    expected_type, pending_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    bot_message_id=excluded.bot_message_id,
                    original_message_id=excluded.original_message_id,
                    original_question=excluded.original_question,
                    clarification_question=excluded.clarification_question,
                    intent=excluded.intent, context=excluded.context,
                    expected_type=excluded.expected_type,
                    pending_mode=excluded.pending_mode,
                    created_at=excluded.created_at""",
                (
                    int(user_id), self.chat_id, bot_message_id,
                    original_message_id, str(original_question),
                    str(clarification_question), str(intent), str(context),
                    str(expected_type), str(pending_mode), stamp,
                ),
            )

    def pending_conversation(self, user_id, ttl_seconds, current=None):
        current = current or datetime.now(timezone.utc)
        cutoff = (current - timedelta(seconds=max(1, int(ttl_seconds)))).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            db.execute("DELETE FROM pending_conversations WHERE created_at < ?", (cutoff,))
            row = db.execute(
                "SELECT * FROM pending_conversations WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
        return dict(row) if row else None

    def clear_pending_conversation(self, user_id):
        with self._lock, closing(self._connect()) as db, db:
            db.execute("DELETE FROM pending_conversations WHERE user_id = ?", (int(user_id),))

    def attach_pending_bot_message(self, user_id, bot_message_id):
        if bot_message_id is None:
            return
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "UPDATE pending_conversations SET bot_message_id = ? WHERE user_id = ?",
                (int(bot_message_id), int(user_id)),
            )

    def claim_scheduled_event(self, event_key):
        """Atomically reserve a scheduler slot across restarts and processes."""
        stamp = datetime.now(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO scheduled_events(event_key, created_at) VALUES(?, ?)",
                (str(event_key), stamp),
            )
            db.execute(
                "DELETE FROM scheduled_events WHERE created_at < ?",
                ((datetime.now(timezone.utc) - timedelta(days=14)).isoformat(),),
            )
            return cursor.rowcount > 0

    def purge_matching_text(self, pattern):
        """Remove forbidden phrases from every store that can feed generation."""
        removed = 0
        with self._lock, closing(self._connect()) as db, db:
            for table, column in (
                ("messages", "text"),
                ("generated", "text"),
                ("daily_summaries", "summary_json"),
                ("long_memories", "memory"),
                ("memory_candidates", "memory"),
            ):
                rows = db.execute(
                    f"SELECT rowid AS purge_rowid, {column} FROM {table}"
                ).fetchall()
                rowids = [
                    row["purge_rowid"]
                    for row in rows
                    if pattern.search(row[column] or "")
                ]
                if rowids:
                    db.executemany(
                        f"DELETE FROM {table} WHERE rowid = ?",
                        ((rowid,) for rowid in rowids),
                    )
                    removed += len(rowids)
        return removed

    def record_generated(self, text, kind, created_at=None):
        """Persist a bot action; ``created_at`` keeps scheduler decisions testable."""
        stamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            db.execute("INSERT INTO generated(text, kind, created_at) VALUES (?, ?, ?)", (text, kind, stamp))
            overflow = db.execute(
                "SELECT COUNT(*) - ? FROM generated", (self.max_messages,)
            ).fetchone()[0]
            if overflow > 0:
                db.execute(
                    "DELETE FROM generated WHERE id IN "
                    "(SELECT id FROM generated ORDER BY id LIMIT ?)", (overflow,)
                )
            row = db.execute("SELECT value FROM chat_stats WHERE key = 'bot_reactions'").fetchone()
            reactions = int(row[0]) + 1 if row else 1
            db.execute(
                "INSERT INTO chat_stats(key, value) VALUES('bot_reactions', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(reactions),)
            )
            db.execute(
                "INSERT INTO chat_stats(key, value) VALUES('last_bot_reply_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (stamp,)
            )

    def record_llm_call(self, provider, model, call_type, usage, created_at=None):
        """Persist API metering only; prompts, output text and secrets never enter it."""
        stamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        values = usage or {}

        def integer(name):
            value = values.get(name)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                """INSERT INTO llm_calls(
                    chat_id, provider, model, call_type, input_tokens, cached_input_tokens,
                    output_tokens, reasoning_tokens, cost_usd_ticks, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.chat_id, str(provider), str(model), str(call_type),
                    integer("input_tokens"), integer("cached_input_tokens"),
                    integer("output_tokens"), integer("reasoning_tokens"),
                    integer("cost_usd_ticks"), stamp,
                ),
            )

    def llm_usage_report(self, since_iso):
        """Return exact local usage totals, grouped for developer diagnostics."""
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                """SELECT call_type, COUNT(*) AS calls,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    SUM(cost_usd_ticks) AS cost_usd_ticks
                FROM llm_calls WHERE created_at >= ? GROUP BY call_type""",
                (since_iso,),
            ).fetchall()
            chats = db.execute(
                "SELECT COUNT(DISTINCT substr(created_at, 1, 10)) FROM llm_calls WHERE created_at >= ?",
                (since_iso,),
            ).fetchone()[0]
        groups = {
            key: {
                "calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "reasoning_tokens": 0,
                "cost_usd_ticks": None,
            }
            for key in ("reply", "summary", "autonomous", "meme")
        }
        for row in rows:
            key = row["call_type"] if row["call_type"] in groups else "reply"
            groups[key] = dict(row)
        totals = {
            key: sum(group[key] or 0 for group in groups.values())
            for key in ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
        }
        known_costs = [group["cost_usd_ticks"] for group in groups.values() if group["cost_usd_ticks"] is not None]
        totals["cost_usd_ticks"] = sum(known_costs) if known_costs else None
        totals["avg_cost_usd_ticks"] = (
            totals["cost_usd_ticks"] // totals["calls"]
            if totals["cost_usd_ticks"] is not None and totals["calls"] else None
        )
        # This repository represents one Telegram chat.  Keep the field in the
        # report shape so a future all-chat aggregator can sum it unchanged.
        totals["active_chats"] = 1 if totals["calls"] else 0
        totals["avg_cost_per_chat_day_usd_ticks"] = totals["cost_usd_ticks"]
        return {"groups": groups, "total": totals}

    def record_routing_event(self, event_type, created_at=None):
        """Store routing counters only; message text is deliberately excluded."""
        stamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO routing_events(event_type, created_at) VALUES (?, ?)",
                (str(event_type), stamp),
            )
            db.execute(
                "DELETE FROM routing_events WHERE created_at < ?",
                ((datetime.now(timezone.utc) - timedelta(days=31)).isoformat(),),
            )

    def routing_report(self, since_iso):
        with self._lock, closing(self._connect()) as db:
            rows = db.execute(
                "SELECT event_type, COUNT(*) AS count FROM routing_events "
                "WHERE created_at >= ? GROUP BY event_type",
                (since_iso,),
            ).fetchall()
        return {row["event_type"]: int(row["count"]) for row in rows}

    def generated_since(self, since_iso, kind=None):
        query = "SELECT text, kind, created_at FROM generated WHERE created_at >= ?"
        params = [since_iso]
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        with self._lock, closing(self._connect()) as db, db:
            return [dict(row) for row in db.execute(query, params).fetchall()]

    def latest_generated(self, kinds=None):
        query = "SELECT text, kind, created_at FROM generated"
        params = []
        if kinds:
            marks = ", ".join("?" for _ in kinds)
            query += f" WHERE kind IN ({marks})"
            params.extend(kinds)
        query += " ORDER BY created_at DESC, id DESC LIMIT 1"
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(query, params).fetchone()
        return dict(row) if row else None

    def recent_dialogue(self, limit=20):
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                """SELECT text, created_at, speaker, username FROM (
                    SELECT text, created_at, 'user' AS speaker, username FROM messages
                    UNION ALL
                    SELECT text, created_at, 'cyberchair' AS speaker, NULL AS username
                    FROM generated WHERE kind NOT IN ('random_media', 'contextual_media')
                ) ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def summary_state(self):
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT last_message_row_id, last_summary_at, pending_since "
                "FROM summary_state WHERE singleton = 1"
            ).fetchone()
        return dict(row)

    def mark_summary_pending(self, stamp):
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "UPDATE summary_state SET pending_since = COALESCE(pending_since, ?) "
                "WHERE singleton = 1",
                (stamp,),
            )

    def summary_for_day(self, day):
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT summary_json FROM daily_summaries WHERE day = ?", (day,)
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return None

    def save_daily_summary(self, day, summary):
        stamp = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO daily_summaries(day, summary_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(day) DO UPDATE SET summary_json=excluded.summary_json, "
                "updated_at=excluded.updated_at",
                (day, payload, stamp),
            )

    def recent_summaries(self, limit=2):
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT day, summary_json, updated_at FROM daily_summaries "
                "ORDER BY day DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            try:
                summary = json.loads(row["summary_json"])
                # Compatibility aliases for callers that still consume the old shape.
                if "main_topics" in summary:
                    summary.setdefault("topics", summary["main_topics"])
                    summary.setdefault(
                        "mood", [summary["current_mood"]] if summary.get("current_mood") else []
                    )
                    summary.setdefault("local_memes", summary.get("inside_jokes", []))
                    summary.setdefault(
                        "people", summary.get("frequently_mentioned_people", [])
                    )
                    summary.setdefault("events", summary.get("notable_events", []))
                result.append({"day": row["day"], **summary})
            except (TypeError, ValueError):
                continue
        return list(reversed(result))

    def remember_stable(self, memories, limit=40):
        stamp = datetime.now(timezone.utc).isoformat()
        clean = [str(item).strip()[:300] for item in memories if str(item).strip()]
        with self._lock, closing(self._connect()) as db, db:
            existing = db.execute("SELECT id, memory FROM long_memories").fetchall()
            for memory in clean:
                match = next(
                    (row for row in existing if memories_are_similar(row["memory"], memory)),
                    None,
                )
                if match:
                    db.execute(
                        "UPDATE long_memories SET score=score+1, updated_at=? WHERE id=?",
                        (stamp, match["id"]),
                    )
                else:
                    cursor = db.execute(
                        "INSERT INTO long_memories(memory, score, updated_at) VALUES (?, 1, ?)",
                        (memory, stamp),
                    )
                    existing.append({"id": cursor.lastrowid, "memory": memory})
            overflow = db.execute("SELECT COUNT(*) - ? FROM long_memories", (limit,)).fetchone()[0]
            if overflow > 0:
                db.execute(
                    "DELETE FROM long_memories WHERE id IN "
                    "(SELECT id FROM long_memories ORDER BY score ASC, updated_at ASC LIMIT ?)",
                    (overflow,),
                )

    def stable_memories(self, limit=20):
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT memory FROM long_memories ORDER BY score DESC, updated_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            if any(memories_are_similar(row[0], existing) for existing in result):
                continue
            result.append(row[0])
            if len(result) >= limit:
                break
        return result

    def memory_candidates(self):
        with self._lock, closing(self._connect()) as db, db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM memory_candidates ORDER BY last_seen_at DESC"
                ).fetchall()
            ]

    def finalize_summary(self, day, summary, last_message_row_id, candidates, limit=40):
        """Atomically save a summary, advance its cursor and observe candidates."""
        stamp = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO daily_summaries(day, summary_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(day) DO UPDATE SET summary_json=excluded.summary_json, "
                "updated_at=excluded.updated_at",
                (day, payload, stamp),
            )
            stable_rows = db.execute(
                "SELECT id, memory FROM long_memories"
            ).fetchall()
            candidate_rows = db.execute(
                "SELECT * FROM memory_candidates"
            ).fetchall()
            for memory, observations in candidates:
                memory = str(memory).strip()[:300]
                if not memory:
                    continue
                observations = max(1, int(observations))
                stable_match = next(
                    (row for row in stable_rows if memories_are_similar(row["memory"], memory)),
                    None,
                )
                candidate_match = next(
                    (
                        row
                        for row in candidate_rows
                        if memories_are_similar(row["memory"], memory)
                    ),
                    None,
                )
                normalized = (
                    candidate_match["normalized_memory"]
                    if candidate_match
                    else normalize_memory(memory)
                )
                previous_count = candidate_match["observation_count"] if candidate_match else 0
                new_count = previous_count + observations
                first_seen = candidate_match["first_seen_at"] if candidate_match else stamp
                promoted_at = candidate_match["promoted_at"] if candidate_match else None
                if stable_match:
                    db.execute(
                        "UPDATE long_memories SET score=score+?, updated_at=? WHERE id=?",
                        (observations, stamp, stable_match["id"]),
                    )
                    promoted_at = promoted_at or stamp
                elif new_count >= 2:
                    cursor = db.execute(
                        "INSERT INTO long_memories(memory, score, updated_at) VALUES (?, ?, ?)",
                        (memory, new_count, stamp),
                    )
                    stable_rows = list(stable_rows) + [
                        {"id": cursor.lastrowid, "memory": memory}
                    ]
                    promoted_at = stamp
                db.execute(
                    "INSERT INTO memory_candidates(normalized_memory, memory, observation_count, "
                    "first_seen_at, last_seen_at, promoted_at) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(normalized_memory) DO UPDATE SET "
                    "observation_count=excluded.observation_count, "
                    "last_seen_at=excluded.last_seen_at, "
                    "promoted_at=COALESCE(memory_candidates.promoted_at, excluded.promoted_at)",
                    (normalized, memory, new_count, first_seen, stamp, promoted_at),
                )
                candidate_rows = db.execute(
                    "SELECT * FROM memory_candidates"
                ).fetchall()
            overflow = db.execute(
                "SELECT COUNT(*) - ? FROM long_memories", (limit,)
            ).fetchone()[0]
            if overflow > 0:
                db.execute(
                    "DELETE FROM long_memories WHERE id IN "
                    "(SELECT id FROM long_memories ORDER BY score ASC, updated_at ASC LIMIT ?)",
                    (overflow,),
                )
            db.execute(
                "UPDATE summary_state SET last_message_row_id=?, last_summary_at=?, "
                "pending_since=NULL WHERE singleton=1",
                (int(last_message_row_id), stamp),
            )

    def statistics(self):
        with self._lock, closing(self._connect()) as db, db:
            values = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM chat_stats")}
        mentions = json.loads(values.get("word_mentions", "{}"))
        activity_by_hour = json.loads(values.get("activity_by_hour", "{}"))
        activity_by_minute = json.loads(values.get("activity_by_minute", "{}"))
        now = datetime.now(timezone.utc)
        recent_cutoff = now.timestamp() - 300
        recent_messages = 0
        for stamp, count in activity_by_minute.items():
            try:
                if datetime.fromisoformat(stamp).timestamp() >= recent_cutoff:
                    recent_messages += int(count)
            except ValueError:
                continue
        silence_seconds = None
        if values.get("last_message_at"):
            try:
                silence_seconds = max(0, int((now - datetime.fromisoformat(values["last_message_at"])).total_seconds()))
            except ValueError:
                pass
        return {
            "total_messages": int(values.get("total_messages", "0")),
            "last_message_at": values.get("last_message_at"),
            "word_mentions": mentions,
            "activity_by_hour": activity_by_hour,
            "messages_per_minute_5m": recent_messages / 5,
            "silence_seconds": silence_seconds,
            "bot_reactions": int(values.get("bot_reactions", "0")),
            "last_bot_reply_at": values.get("last_bot_reply_at"),
        }

    def add_gif(self, message_id, user_id, file_id, file_unique_id, created_at=None,
                max_gifs=1000):
        created_at = created_at or datetime.now(timezone.utc)
        stamp = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                """INSERT INTO gifs
                (chat_id, message_id, user_id, file_id, file_unique_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, file_unique_id) DO UPDATE SET file_id=excluded.file_id""",
                (self.chat_id, message_id, user_id, file_id, file_unique_id, stamp),
            )
            overflow = db.execute("SELECT COUNT(*) - ? FROM gifs", (max_gifs,)).fetchone()[0]
            if overflow > 0:
                db.execute("DELETE FROM gifs WHERE id IN (SELECT id FROM gifs ORDER BY id LIMIT ?)", (overflow,))
            return cursor.rowcount > 0

    def gif_count(self):
        with self._lock, closing(self._connect()) as db, db:
            return db.execute("SELECT COUNT(*) FROM gifs").fetchone()[0]

    def random_gif(self):
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM gifs ORDER BY RANDOM() LIMIT 1").fetchone()
        return dict(row) if row else None

    def mark_gif_used(self, gif_id):
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "UPDATE gifs SET last_used_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), gif_id),
            )

    def add_sticker(self, message_id, user_id, file_id, file_unique_id,
                    created_at=None, max_stickers=1000):
        created_at = created_at or datetime.now(timezone.utc)
        stamp = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                """INSERT INTO stickers
                (chat_id, message_id, user_id, file_id, file_unique_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, file_unique_id) DO UPDATE SET file_id=excluded.file_id""",
                (self.chat_id, message_id, user_id, file_id, file_unique_id, stamp),
            )
            overflow = db.execute("SELECT COUNT(*) - ? FROM stickers", (max_stickers,)).fetchone()[0]
            if overflow > 0:
                db.execute("DELETE FROM stickers WHERE id IN (SELECT id FROM stickers ORDER BY id LIMIT ?)", (overflow,))
            return cursor.rowcount > 0

    def add_chat_image(self, message_id, user_id, file_id, file_unique_id,
                       media_type, mime_type=None, caption=None, file_size=None,
                       width=None, height=None, from_bot=False, created_at=None,
                       max_images=2000):
        """Keep Telegram metadata only; duplicate files refresh their file_id."""
        created_at = created_at or datetime.now(timezone.utc)
        stamp = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        with self._lock, closing(self._connect()) as db, db:
            existed = db.execute(
                "SELECT 1 FROM chat_images WHERE chat_id = ? AND file_unique_id = ?",
                (self.chat_id, str(file_unique_id)),
            ).fetchone()
            db.execute(
                """INSERT INTO chat_images
                (chat_id, message_id, user_id, file_id, file_unique_id,
                 media_type, mime_type, caption, file_size, width, height,
                 created_at, from_bot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, file_unique_id) DO UPDATE SET
                    message_id=excluded.message_id,
                    user_id=excluded.user_id,
                    file_id=excluded.file_id,
                    media_type=excluded.media_type,
                    mime_type=excluded.mime_type,
                    caption=CASE WHEN excluded.caption != '' THEN excluded.caption ELSE chat_images.caption END,
                    file_size=COALESCE(excluded.file_size, chat_images.file_size),
                    width=COALESCE(excluded.width, chat_images.width),
                    height=COALESCE(excluded.height, chat_images.height),
                    from_bot=MAX(chat_images.from_bot, excluded.from_bot)""",
                (
                    self.chat_id, int(message_id), user_id, str(file_id),
                    str(file_unique_id), str(media_type), mime_type,
                    str(caption or "")[:1000], file_size, width, height, stamp,
                    int(bool(from_bot)),
                ),
            )
            overflow = db.execute(
                "SELECT COUNT(*) - ? FROM chat_images", (int(max_images),)
            ).fetchone()[0]
            if overflow > 0:
                db.execute(
                    "DELETE FROM chat_images WHERE id IN "
                    "(SELECT id FROM chat_images ORDER BY id LIMIT ?)",
                    (overflow,),
                )
            return not bool(existed)

    def chat_image_count(self):
        with self._lock, closing(self._connect()) as db, db:
            return db.execute("SELECT COUNT(*) FROM chat_images").fetchone()[0]

    def chat_image_by_unique_id(self, file_unique_id):
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT * FROM chat_images WHERE file_unique_id = ? LIMIT 1",
                (str(file_unique_id),),
            ).fetchone()
        return dict(row) if row else None

    def chat_image_candidates(self, limit=2000):
        """Return human images with lightweight conversation engagement signals."""
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                """SELECT image.*,
                    (SELECT COUNT(*) FROM messages reply
                     WHERE reply.reply_to_message_id = image.message_id) AS reply_count,
                    (SELECT COUNT(*) FROM messages nearby
                     WHERE ABS(
                        (julianday(nearby.created_at) - julianday(image.created_at)) * 1440
                     ) <= 30) AS nearby_message_count
                FROM chat_images image
                WHERE image.from_bot = 0
                ORDER BY image.created_at DESC LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_chat_image_usage(self, limit=12):
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT * FROM chat_image_usage ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def chat_image_caption_used(self, file_unique_id, caption_hash):
        with self._lock, closing(self._connect()) as db, db:
            return bool(db.execute(
                "SELECT 1 FROM chat_image_usage WHERE file_unique_id = ? "
                "AND caption_hash = ? LIMIT 1",
                (str(file_unique_id), str(caption_hash)),
            ).fetchone())

    def mark_chat_image_used(self, file_unique_id, caption_hash, user_id=None):
        stamp = datetime.now(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "UPDATE chat_images SET used_count = used_count + 1, last_used_at = ? "
                "WHERE file_unique_id = ?",
                (stamp, str(file_unique_id)),
            )
            db.execute(
                "INSERT INTO chat_image_usage(file_unique_id, user_id, caption_hash, created_at) "
                "VALUES(?, ?, ?, ?)",
                (str(file_unique_id), user_id, str(caption_hash), stamp),
            )
            db.execute(
                "DELETE FROM chat_image_usage WHERE id NOT IN "
                "(SELECT id FROM chat_image_usage ORDER BY id DESC LIMIT 500)"
            )

    def sticker_count(self):
        with self._lock, closing(self._connect()) as db, db:
            return db.execute("SELECT COUNT(*) FROM stickers").fetchone()[0]

    def random_sticker(self):
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM stickers ORDER BY RANDOM() LIMIT 1").fetchone()
        return dict(row) if row else None

    def mark_sticker_used(self, sticker_id):
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "UPDATE stickers SET last_used_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), sticker_id),
            )

    def mark_media_file_used(self, media_type, file_id):
        table = "gifs" if media_type == "gif" else "stickers"
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                f"UPDATE {table} SET last_used_at = ? WHERE file_id = ?",
                (datetime.now(timezone.utc).isoformat(), file_id),
            )

    def set_media_tags(self, media_type, file_unique_id, tags):
        if media_type not in {"gif", "sticker"}:
            raise ValueError("unsupported media type")
        clean = sorted({normalize_memory(tag) for tag in tags if normalize_memory(tag)})
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO media_metadata(media_type, asset_key, tags_json) VALUES(?, ?, ?) "
                "ON CONFLICT(media_type, asset_key) DO UPDATE SET tags_json=excluded.tags_json",
                (media_type, str(file_unique_id), json.dumps(clean, ensure_ascii=False)),
            )

    def tagged_media(self, media_type):
        if media_type not in {"gif", "sticker"}:
            return []
        table = "gifs" if media_type == "gif" else "stickers"
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                f"SELECT asset.*, metadata.tags_json FROM {table} asset "
                "JOIN media_metadata metadata ON metadata.media_type = ? "
                "AND metadata.asset_key = asset.file_unique_id",
                (media_type,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["tags"] = json.loads(item.pop("tags_json"))
            except (TypeError, ValueError):
                item["tags"] = []
            result.append(item)
        return result

    def record_media_usage(self, action, asset_key, template_id, cooldown_group, archetype):
        stamp = datetime.now(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO media_usage(action, asset_key, template_id, cooldown_group, "
                "archetype, created_at) VALUES(?, ?, ?, ?, ?, ?)",
                (action, asset_key, template_id, cooldown_group, archetype, stamp),
            )
            db.execute(
                "DELETE FROM media_usage WHERE id NOT IN "
                "(SELECT id FROM media_usage ORDER BY id DESC LIMIT 100)")

    def recent_media_usage(self, limit=12):
        with self._lock, closing(self._connect()) as db, db:
            return [
                dict(row) for row in db.execute(
                    "SELECT * FROM media_usage ORDER BY id DESC LIMIT ?", (int(limit),)
                ).fetchall()
            ]

    def has_media_usage_since(self, since_iso, action=None, asset_key=None,
                              cooldown_group=None):
        clauses = ["created_at >= ?"]
        params = [since_iso]
        for column, value in (
            ("action", action), ("asset_key", asset_key),
            ("cooldown_group", cooldown_group),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT 1 FROM media_usage WHERE " + " AND ".join(clauses) + " LIMIT 1",
                params,
            ).fetchone()
        return bool(row)

    def clear(self):
        with self._lock, closing(self._connect()) as db, db:
            db.execute("DELETE FROM messages")
            db.execute("DELETE FROM generated")
            db.execute("DELETE FROM gifs")
            db.execute("DELETE FROM stickers")
            db.execute("DELETE FROM chat_images")
            db.execute("DELETE FROM chat_image_usage")
            db.execute("DELETE FROM daily_summaries")
            db.execute("DELETE FROM long_memories")
            db.execute("DELETE FROM memory_candidates")
            db.execute("DELETE FROM chat_stats")
            db.execute("DELETE FROM settings")
            db.execute("DELETE FROM scheduled_events")
            db.execute("DELETE FROM media_metadata")
            db.execute("DELETE FROM media_usage")
            db.execute("DELETE FROM routing_events")
            db.execute("DELETE FROM pending_conversations")
            db.execute(
                "INSERT OR REPLACE INTO summary_state(singleton, last_message_row_id, "
                "last_summary_at, pending_since) VALUES(1, 0, NULL, NULL)"
            )
