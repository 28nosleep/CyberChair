import json
import re
import sqlite3
import threading
import uuid
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

from .event_context import current_event_id, memory_event_id, scheduled_event_id
from .memory_maintenance import SummaryFinalizeResult
from .db_migrations import (
    CURRENT_SCHEMA_VERSION,
    migrate_database,
    schema_signature,
    schema_version,
)


_repository_profile = ContextVar("cyberchair_repository_profile", default=None)
_sqlite_counter_lock = threading.Lock()
_repository_initialization_lock = threading.RLock()
_sqlite_runtime_counters = {"database_locked": 0, "busy_timeout": 0}


def _record_sqlite_operational_error(error):
    message = str(error).casefold()
    key = "database_locked" if "locked" in message else (
        "busy_timeout" if "busy" in message else None
    )
    if key:
        with _sqlite_counter_lock:
            _sqlite_runtime_counters[key] += 1


class _InstrumentedConnection(sqlite3.Connection):
    """Count lock/busy failures without recording SQL parameters or content."""

    def execute(self, *args, **kwargs):
        try:
            return super().execute(*args, **kwargs)
        except sqlite3.OperationalError as error:
            _record_sqlite_operational_error(error)
            raise

    def executemany(self, *args, **kwargs):
        try:
            return super().executemany(*args, **kwargs)
        except sqlite3.OperationalError as error:
            _record_sqlite_operational_error(error)
            raise


def sqlite_runtime_diagnostics():
    with _sqlite_counter_lock:
        return dict(_sqlite_runtime_counters)


def _profile_statement(profile, statement):
    sql = " ".join(str(statement).strip().split()).casefold()
    if not sql:
        return
    if sql.startswith(("select", "with")):
        profile["reads"] += 1
    elif sql.startswith(("insert", "update", "delete", "replace")):
        profile["writes"] += 1
    if "daily_summaries" in sql and sql.startswith(("select", "with")):
        profile["summary_reads"] += 1
    if (
        "union all" in sql and "from messages" in sql and "from generated" in sql
        and sql.startswith(("select", "with"))
    ):
        profile["dialogue_reads"] += 1
    if "long_memories" in sql and sql.startswith(("select", "with")):
        profile["stable_memory_reads"] += 1
    if (
        "from generated" in sql
        and "union all" not in sql
        and sql.startswith(("select", "with"))
    ):
        profile["generated_history_reads"] += 1
    if any(table in sql for table in (
        "media_usage", "media_metadata", "from gifs", "from stickers",
        "chat_images", "chat_image_usage",
    )) and sql.startswith(("select", "with")):
        profile["media_history_reads"] += 1
    if "from settings" in sql and sql.startswith(("select", "with")):
        profile["settings_reads"] += 1


@contextmanager
def repository_query_profile():
    """Event-local query instrumentation used by R3 diagnostics/tests."""
    profile = {
        "connections": 0,
        "reads": 0,
        "writes": 0,
        "summary_reads": 0,
        "dialogue_reads": 0,
        "stable_memory_reads": 0,
        "generated_history_reads": 0,
        "media_history_reads": 0,
        "settings_reads": 0,
    }
    token = _repository_profile.set(profile)
    try:
        yield profile
    finally:
        _repository_profile.reset(token)


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

    def __init__(
        self, data_dir, chat_id, max_messages=50000,
        max_unsummarized_messages=500,
    ):
        self.chat_id = int(chat_id)
        self.max_messages = max_messages
        self.max_unsummarized_messages = max(
            self.max_messages, int(max_unsummarized_messages)
        )
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_id = str(self.chat_id).replace("-", "m")
        self.path = data_dir / f"chat_{safe_id}.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(
            self.path, timeout=5, factory=_InstrumentedConnection
        )
        connection.row_factory = sqlite3.Row
        profile = _repository_profile.get()
        if profile is not None:
            profile["connections"] += 1
            connection.set_trace_callback(
                lambda statement: _profile_statement(profile, statement)
            )
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=OFF")
        return connection

    def _initialize(self):
        """Migrate once at repository construction, then repair data cursors."""
        with _repository_initialization_lock, self._lock, closing(sqlite3.connect(
            self.path, timeout=5, factory=_InstrumentedConnection
        )) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=5000")
            migrate_database(db)
            # journal_mode is persistent DB state and belongs here, not in every
            # short-lived read/write connection. synchronous remains per-connection.
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA foreign_keys=OFF")
            with db:
                existing_total = db.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()[0]
                if existing_total:
                    db.execute(
                        "INSERT OR IGNORE INTO chat_stats(key, value) "
                        "VALUES('total_messages', ?)",
                        (str(existing_total),),
                    )
                state = db.execute(
                    "SELECT 1 FROM summary_state WHERE singleton=1"
                ).fetchone()
                if not state:
                    latest = db.execute(
                        "SELECT updated_at FROM daily_summaries "
                        "ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    cursor = (
                        db.execute(
                            "SELECT COALESCE(MAX(id), 0) FROM messages "
                            "WHERE created_at <= ?", (latest[0],),
                        ).fetchone()[0]
                        if latest else 0
                    )
                    db.execute(
                        "INSERT INTO summary_state(singleton, last_message_row_id, "
                        "last_summary_at, pending_since) VALUES(1, ?, ?, NULL)",
                        (cursor, latest[0] if latest else None),
                    )
                cursor = db.execute(
                    "SELECT last_message_row_id FROM summary_state "
                    "WHERE singleton=1"
                ).fetchone()[0]
                self._prune_summarized_messages(db, int(cursor or 0))
                db.execute(
                    "DELETE FROM generated WHERE id NOT IN "
                    "(SELECT id FROM generated ORDER BY id DESC LIMIT ?)",
                    (self.max_messages,),
                )

    def add_message(self, message_id, user_id, username, text, created_at=None,
                    reply_to_message_id=None, is_reply=False,
                    return_reason=False):
        created_at = created_at or datetime.now(timezone.utc)
        stamp = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        with self._lock, closing(self._connect()) as db, db:
            duplicate = db.execute(
                "SELECT 1 FROM messages WHERE chat_id=? AND message_id=?",
                (self.chat_id, message_id),
            ).fetchone()
            if duplicate:
                return (False, "duplicate") if return_reason else False
            state = db.execute(
                "SELECT last_message_row_id FROM summary_state WHERE singleton=1"
            ).fetchone()
            cursor_id = int(state[0] if state else 0)
            backlog = db.execute(
                "SELECT COUNT(*) FROM messages WHERE id > ?", (cursor_id,)
            ).fetchone()[0]
            if backlog >= self.max_unsummarized_messages:
                db.execute(
                    "INSERT INTO routing_events(event_type, event_id, call_type, created_at) "
                    "VALUES('summary_backlog_hard_cap', ?, 'summary', ?)",
                    (current_event_id(), stamp),
                )
                return (
                    (False, "summary_backlog_hard_cap")
                    if return_reason else False
                )
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
                self._prune_summarized_messages(db, cursor_id)
            if return_reason:
                return inserted, None if inserted else "duplicate"
            return inserted

    def _prune_summarized_messages(self, db, summarized_cursor):
        """Keep every unsummarized row plus the normal recent chat window."""
        db.execute(
            "DELETE FROM messages WHERE id <= ? AND id NOT IN "
            "(SELECT id FROM messages ORDER BY id DESC LIMIT ?)",
            (int(summarized_cursor), self.max_messages),
        )

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

    def messages_after(self, row_id, limit=None):
        query = "SELECT * FROM messages WHERE id > ? ORDER BY id ASC"
        params = [int(row_id)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(query, params).fetchall()
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

    def load_context_snapshot_inputs(
        self, *, since_iso, dialogue_limit, logical_day,
        generated_limit=50, stable_limit=20, pending_user_id=None,
    ):
        """Load the bounded shared event context using one consistent connection."""
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN")
            dialogue = db.execute(
                """SELECT row_id, message_id, user_id, text, created_at, speaker,
                    username, reply_to_message_id, is_reply FROM (
                    SELECT id AS row_id, message_id, user_id, text, created_at,
                        'user' AS speaker, username, reply_to_message_id, is_reply
                    FROM messages WHERE created_at >= ?
                    UNION ALL
                    SELECT id AS row_id, NULL AS message_id, NULL AS user_id, text,
                        created_at, 'cyberchair' AS speaker, NULL AS username,
                        NULL AS reply_to_message_id, 0 AS is_reply
                    FROM generated WHERE kind NOT IN ('random_media', 'contextual_media')
                        AND created_at >= ?
                ) ORDER BY created_at DESC LIMIT ?""",
                (since_iso, since_iso, int(dialogue_limit)),
            ).fetchall()
            latest_message = db.execute(
                "SELECT * FROM messages ORDER BY id DESC LIMIT 1"
            ).fetchone()
            message_count = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            summary_row = db.execute(
                "SELECT summary_json FROM daily_summaries WHERE day = ?",
                (logical_day,),
            ).fetchone()
            stable_rows = db.execute(
                "SELECT memory FROM long_memories "
                "ORDER BY score DESC, updated_at DESC"
            ).fetchall()
            generated = db.execute(
                "SELECT text, kind, created_at FROM generated "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (max(0, int(generated_limit)),),
            ).fetchall()
            settings = db.execute("SELECT key, value FROM settings").fetchall()
            pending = None
            if pending_user_id is not None:
                pending = db.execute(
                    "SELECT * FROM pending_conversations WHERE user_id = ?",
                    (int(pending_user_id),),
                ).fetchone()
            db.rollback()
        summary = None
        if summary_row:
            try:
                summary = json.loads(summary_row[0])
            except (TypeError, ValueError):
                summary = None
        stable = []
        for row in stable_rows:
            if any(memories_are_similar(row[0], old) for old in stable):
                continue
            stable.append(row[0])
            if len(stable) >= int(stable_limit):
                break
        return {
            "recent_dialogue": [dict(row) for row in reversed(dialogue)],
            "latest_message": dict(latest_message) if latest_message else None,
            "message_count": int(message_count),
            "current_summary": summary,
            "stable_memories": stable,
            "recent_generated": [dict(row) for row in reversed(generated)],
            "settings": {row["key"]: row["value"] for row in settings},
            "pending": dict(pending) if pending else None,
        }

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

    @staticmethod
    def _scheduled_stamp(value=None):
        value = value or datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def ensure_scheduled_event(
        self, event_id, event_key, event_kind, scheduled_at, payload,
        parse_mode=None, current=None,
    ):
        """Persist one immutable logical notification without claiming it."""
        now_stamp = self._scheduled_stamp(current)
        scheduled_stamp = self._scheduled_stamp(scheduled_at)
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO scheduled_events(
                    event_id, event_key, event_kind, scheduled_at, payload,
                    parse_mode, state, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)""",
                (
                    str(event_id), str(event_key), str(event_kind),
                    scheduled_stamp, str(payload),
                    str(parse_mode) if parse_mode is not None else None,
                    now_stamp, now_stamp,
                ),
            )
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_key=?",
                (str(event_key),),
            ).fetchone()
        return (dict(row) if row else None), cursor.rowcount > 0

    def claim_due_scheduled_event(
        self, current, lease_seconds, event_id=None,
    ):
        """Take a short lease; this transaction never represents delivery."""
        now_stamp = self._scheduled_stamp(current)
        expires_stamp = self._scheduled_stamp(
            (current if getattr(current, "tzinfo", None) else current.replace(
                tzinfo=timezone.utc
            )) + timedelta(seconds=max(1, int(lease_seconds)))
        )
        token = uuid.uuid4().hex
        with self._lock, closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            query = (
                "SELECT * FROM scheduled_events WHERE scheduled_at <= ? AND ("
                "state='PENDING' OR "
                "(state='RETRY_WAIT' AND (next_attempt_at IS NULL OR next_attempt_at <= ?)) OR "
                "(state='CLAIMED' AND claim_expires_at IS NOT NULL "
                "AND claim_expires_at <= ?))"
            )
            params = [now_stamp, now_stamp, now_stamp]
            if event_id is not None:
                query += " AND event_id=?"
                params.append(str(event_id))
            query += " ORDER BY scheduled_at, created_at LIMIT 1"
            row = db.execute(query, params).fetchone()
            if row is None:
                return None
            previous_state = row["state"]
            cursor = db.execute(
                """UPDATE scheduled_events SET state='CLAIMED', claim_token=?,
                    claimed_at=?, claim_expires_at=?, updated_at=?
                    WHERE event_id=? AND state=?""",
                (
                    token, now_stamp, expires_stamp, now_stamp,
                    row["event_id"], previous_state,
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?",
                (row["event_id"],),
            ).fetchone()
        result = dict(claimed)
        result["claim_recovered"] = previous_state == "CLAIMED"
        return result

    def mark_scheduled_sending(self, event_id, claim_token, current=None):
        """Durably cross the pre-network boundary for one claimed attempt."""
        stamp = self._scheduled_stamp(current)
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                """UPDATE scheduled_events SET state='SENDING',
                    attempt_count=attempt_count+1, updated_at=?
                    WHERE event_id=? AND state='CLAIMED' AND claim_token=?""",
                (stamp, str(event_id), str(claim_token)),
            )
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?",
                (str(event_id),),
            ).fetchone()
        return dict(row) if cursor.rowcount == 1 and row else None

    def finalize_scheduled_success(
        self, event_id, claim_token, telegram_message_id=None, current=None,
    ):
        """Mark SENT only after a successful Telegram response; idempotent."""
        stamp = self._scheduled_stamp(current)
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT state FROM scheduled_events WHERE event_id=?",
                (str(event_id),),
            ).fetchone()
            if row is None:
                return "stale"
            if row["state"] == "SENT":
                return "sent"
            cursor = db.execute(
                """UPDATE scheduled_events SET state='SENT', delivered_at=?,
                    telegram_message_id=?, updated_at=?, claim_token=NULL,
                    claimed_at=NULL, claim_expires_at=NULL, next_attempt_at=NULL
                    WHERE event_id=? AND state='SENDING' AND claim_token=?""",
                (
                    stamp, telegram_message_id, stamp,
                    str(event_id), str(claim_token),
                ),
            )
        return "sent" if cursor.rowcount == 1 else "stale"

    def finalize_scheduled_failure(
        self, event_id, claim_token, state, category, current=None,
        next_attempt_at=None, safe_retry=False,
    ):
        if state not in {"RETRY_WAIT", "UNKNOWN", "DEAD"}:
            raise ValueError(f"invalid scheduled failure state: {state}")
        stamp = self._scheduled_stamp(current)
        next_stamp = (
            self._scheduled_stamp(next_attempt_at)
            if next_attempt_at is not None else None
        )
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                """UPDATE scheduled_events SET state=?, next_attempt_at=?,
                    safe_retry_count=safe_retry_count+?,
                    last_failure_category=?, last_failure_at=?, updated_at=?,
                    claim_token=NULL, claimed_at=NULL, claim_expires_at=NULL
                    WHERE event_id=? AND state='SENDING' AND claim_token=?""",
                (
                    state, next_stamp, int(bool(safe_retry)), str(category),
                    stamp, stamp, str(event_id), str(claim_token),
                ),
            )
            row = db.execute(
                "SELECT state FROM scheduled_events WHERE event_id=?",
                (str(event_id),),
            ).fetchone()
        return row["state"] if cursor.rowcount == 1 and row else "stale"

    def quarantine_scheduled_after_commit_failure(
        self, event_id, claim_token, category, current=None,
    ):
        """Best-effort no-retry fence after Telegram success/local DB failure."""
        return self.finalize_scheduled_failure(
            event_id, claim_token, "UNKNOWN", category, current=current
        )

    def recover_interrupted_scheduled_sends(self, current=None):
        """A persisted SENDING row means the previous outcome is ambiguous."""
        stamp = self._scheduled_stamp(current)
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT event_id, event_kind FROM scheduled_events "
                "WHERE state='SENDING'"
            ).fetchall()
            if rows:
                db.execute(
                    """UPDATE scheduled_events SET state='UNKNOWN',
                        last_failure_category='interrupted_sending',
                        last_failure_at=?, updated_at=?, claim_token=NULL,
                        claimed_at=NULL, claim_expires_at=NULL,
                        next_attempt_at=NULL WHERE state='SENDING'""",
                    (stamp, stamp),
                )
        return tuple(dict(row) for row in rows)

    def scheduled_event(self, *, event_id=None, event_key=None):
        if (event_id is None) == (event_key is None):
            raise ValueError("provide exactly one scheduled event identity")
        column, value = (
            ("event_id", event_id) if event_id is not None
            else ("event_key", event_key)
        )
        with self._lock, closing(self._connect()) as db:
            row = db.execute(
                f"SELECT * FROM scheduled_events WHERE {column}=?",
                (str(value),),
            ).fetchone()
        return dict(row) if row else None

    def scheduled_events(self):
        with self._lock, closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM scheduled_events ORDER BY scheduled_at, event_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def scheduled_delivery_report(self, current=None, recent_days=14):
        current = current or datetime.now(timezone.utc)
        now_stamp = self._scheduled_stamp(current)
        recent_stamp = self._scheduled_stamp(
            (current if current.tzinfo else current.replace(tzinfo=timezone.utc))
            - timedelta(days=max(1, int(recent_days)))
        )
        with self._lock, closing(self._connect()) as db:
            counts = {
                row["state"].casefold(): int(row["count"])
                for row in db.execute(
                    "SELECT state, COUNT(*) AS count FROM scheduled_events "
                    "GROUP BY state"
                ).fetchall()
            }
            sent_recent = db.execute(
                "SELECT COUNT(*) FROM scheduled_events "
                "WHERE state='SENT' AND delivered_at >= ?", (recent_stamp,)
            ).fetchone()[0]
            safe_retries = db.execute(
                "SELECT COALESCE(SUM(safe_retry_count), 0) "
                "FROM scheduled_events"
            ).fetchone()[0]
            last_success = db.execute(
                "SELECT MAX(delivered_at) FROM scheduled_events "
                "WHERE state='SENT'"
            ).fetchone()[0]
            failure = db.execute(
                "SELECT last_failure_category FROM scheduled_events "
                "WHERE last_failure_at IS NOT NULL "
                "ORDER BY last_failure_at DESC LIMIT 1"
            ).fetchone()
            oldest = db.execute(
                "SELECT MIN(created_at) FROM scheduled_events WHERE state IN "
                "('PENDING','CLAIMED','SENDING','RETRY_WAIT')"
            ).fetchone()[0]
        oldest_age = None
        if oldest:
            try:
                oldest_dt = datetime.fromisoformat(oldest)
                if oldest_dt.tzinfo is None:
                    oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
                now_dt = datetime.fromisoformat(now_stamp)
                oldest_age = max(0, int((now_dt - oldest_dt).total_seconds()))
            except (TypeError, ValueError):
                pass
        return {
            "pending": counts.get("pending", 0),
            "claimed": counts.get("claimed", 0),
            "sending": counts.get("sending", 0),
            "retry_wait": counts.get("retry_wait", 0),
            "unknown": counts.get("unknown", 0),
            "dead": counts.get("dead", 0),
            "sent_recent": int(sent_recent),
            "safe_retries": int(safe_retries),
            "last_success": last_success,
            "last_failure_category": failure[0] if failure else None,
            "oldest_pending_age": oldest_age,
        }

    def claim_scheduled_event(self, event_key):
        """Deprecated compatibility reservation; production P1 never uses it."""
        stamp = datetime.now(timezone.utc)
        _, created = self.ensure_scheduled_event(
            scheduled_event_id(self.chat_id, "legacy_reservation", event_key),
            event_key, "legacy_reservation", stamp, "", current=stamp,
        )
        return created

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

    def record_llm_call(self, provider, model, call_type, usage, created_at=None,
                        event_id=None):
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
                    output_tokens, reasoning_tokens, cost_usd_ticks, event_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.chat_id, str(provider), str(model), str(call_type),
                    integer("input_tokens"), integer("cached_input_tokens"),
                    integer("output_tokens"), integer("reasoning_tokens"),
                    integer("cost_usd_ticks"), event_id or current_event_id(), stamp,
                ),
            )

    def llm_usage_report(self, since_iso):
        """Return exact local usage totals, grouped for developer diagnostics."""
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                """WITH usage AS (
                    SELECT call_type, 1 AS calls,
                        COALESCE(input_tokens, 0) AS input_tokens,
                        COALESCE(cached_input_tokens, 0) AS cached_input_tokens,
                        COALESCE(output_tokens, 0) AS output_tokens,
                        COALESCE(reasoning_tokens, 0) AS reasoning_tokens,
                        cost_usd_ticks
                    FROM llm_calls WHERE created_at >= ?
                    UNION ALL
                    SELECT call_type, calls, input_tokens, cached_input_tokens,
                        output_tokens, reasoning_tokens, cost_usd_ticks
                    FROM llm_daily_aggregates
                    WHERE day >= substr(?, 1, 10)
                )
                SELECT call_type, SUM(calls) AS calls,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    SUM(cost_usd_ticks) AS cost_usd_ticks
                FROM usage GROUP BY call_type""",
                (since_iso, since_iso),
            ).fetchall()
            chats = db.execute(
                """SELECT COUNT(DISTINCT day) FROM (
                    SELECT substr(created_at, 1, 10) AS day FROM llm_calls
                    WHERE created_at >= ?
                    UNION ALL
                    SELECT day FROM llm_daily_aggregates
                    WHERE day >= substr(?, 1, 10)
                )""",
                (since_iso, since_iso),
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

    def record_routing_event(self, event_type, created_at=None, event_id=None,
                             provider_key=None, call_type=None):
        """Store routing counters only; message text is deliberately excluded."""
        stamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO routing_events(event_type, event_id, provider_key, "
                "call_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    str(event_type), event_id or current_event_id(),
                    str(provider_key) if provider_key is not None else None,
                    str(call_type) if call_type is not None else None,
                    stamp,
                ),
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

    def llm_event_invariant_report(self, since_iso):
        """Prove the per-user-event call invariant from attempted-call telemetry."""
        with self._lock, closing(self._connect()) as db:
            user_events = db.execute(
                "SELECT DISTINCT event_id FROM routing_events "
                "WHERE event_type='user_event' AND event_id IS NOT NULL "
                "AND created_at >= ?",
                (since_iso,),
            ).fetchall()
            counts = {
                row["event_id"]: int(row["calls"])
                for row in db.execute(
                    "SELECT event_id, COUNT(*) AS calls FROM routing_events "
                    "WHERE event_type='llm_call_attempt' AND event_id IS NOT NULL "
                    "AND created_at >= ? GROUP BY event_id",
                    (since_iso,),
                ).fetchall()
            }
        values = [counts.get(row["event_id"], 0) for row in user_events]
        report = {
            "user_events": len(values),
            "events_with_0_llm": sum(value == 0 for value in values),
            "events_with_1_llm": sum(value == 1 for value in values),
            "events_with_2plus_llm": sum(value >= 2 for value in values),
            "max_calls_per_user_event": max(values, default=0),
        }
        return report

    def correlated_event_rows(self, event_id):
        """Developer/test diagnostic without message content or credentials."""
        with self._lock, closing(self._connect()) as db:
            routing = [
                dict(row) for row in db.execute(
                    "SELECT event_type, event_id, provider_key, call_type, created_at "
                    "FROM routing_events WHERE event_id = ? ORDER BY id",
                    (str(event_id),),
                ).fetchall()
            ]
            usage = [
                dict(row) for row in db.execute(
                    "SELECT provider, model, call_type, event_id, created_at "
                    "FROM llm_calls WHERE event_id = ? ORDER BY id",
                    (str(event_id),),
                ).fetchall()
            ]
        return {"routing": routing, "usage": usage}

    def memory_lifecycle_report(self, current=None):
        """Content-free R5 operational state and lifecycle counters."""
        current = current or datetime.now(timezone.utc)
        backlog = self.summary_backlog_state(current)
        with self._lock, closing(self._connect()) as db:
            candidate_counts = db.execute(
                "SELECT SUM(CASE WHEN promoted_at IS NULL THEN 1 ELSE 0 END) AS active, "
                "SUM(CASE WHEN promoted_at IS NOT NULL THEN 1 ELSE 0 END) AS promoted "
                "FROM memory_candidates"
            ).fetchone()
            event_counts = {
                row["event_type"]: int(row["count"])
                for row in db.execute(
                    "SELECT event_type, COUNT(*) AS count FROM routing_events "
                    "WHERE event_type LIKE 'summary_%' GROUP BY event_type"
                ).fetchall()
            }
            foreground_calls = db.execute(
                "SELECT COUNT(*) FROM routing_events WHERE event_type='llm_call_attempt' "
                "AND call_type='summary' AND event_id LIKE 'tg_%'"
            ).fetchone()[0]
            maximum = db.execute(
                "SELECT COALESCE(MAX(calls), 0) FROM ("
                "SELECT COUNT(*) AS calls FROM routing_events "
                "WHERE event_type='llm_call_attempt' AND call_type='summary' "
                "AND event_id LIKE 'mem_%' GROUP BY event_id)"
            ).fetchone()[0]
        return {
            "chat_id": self.chat_id,
            "backlog_messages": backlog["backlog_messages"],
            "cursor_lag": backlog["cursor_lag"],
            "oldest_unsummarized_age_seconds": (
                backlog["oldest_unsummarized_age_seconds"]
            ),
            "hard_cap": backlog["hard_cap"],
            "summary_attempts": event_counts.get("summary_attempt", 0),
            "summary_success": event_counts.get("summary_success", 0),
            "summary_failures": event_counts.get("summary_failure", 0),
            "summary_resource_deferrals": event_counts.get(
                "summary_deferred_resource_busy", 0
            ),
            "summary_stale_finalize": event_counts.get(
                "summary_stale_finalize", 0
            ),
            "foreground_summary_calls": int(foreground_calls),
            "max_summary_calls_per_memory_event": int(maximum),
            "active_candidates": int(candidate_counts["active"] or 0),
            "promoted_candidates_retained": int(
                candidate_counts["promoted"] or 0
            ),
            "promoted_pruned": event_counts.get(
                "summary_candidates_promoted_pruned", 0
            ),
            "stale_pruned": event_counts.get(
                "summary_candidates_stale_pruned", 0
            ),
        }

    def generated_since(self, since_iso, kind=None):
        query = "SELECT text, kind, created_at FROM generated WHERE created_at >= ?"
        params = [since_iso]
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        with self._lock, closing(self._connect()) as db, db:
            return [dict(row) for row in db.execute(query, params).fetchall()]

    def recent_generated(self, limit=40):
        """Bounded newest bot texts for local quality scoring."""
        with self._lock, closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT text, kind, created_at FROM generated "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (max(0, int(limit)),),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

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
                "SELECT last_message_row_id, last_summary_at, pending_since, "
                "claim_token, claim_start_cursor, claim_end_row_id, claim_day, "
                "claimed_at, claim_expires_at, attempt_sequence, failure_count, "
                "next_attempt_at "
                "FROM summary_state WHERE singleton = 1"
            ).fetchone()
        return dict(row)

    def mark_summary_pending(self, stamp):
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                "UPDATE summary_state SET pending_since = COALESCE(pending_since, ?) "
                "WHERE singleton = 1 AND pending_since IS NULL",
                (stamp,),
            )
            return cursor.rowcount > 0

    def summary_backlog_state(self, current=None):
        current = current or datetime.now(timezone.utc)
        with self._lock, closing(self._connect()) as db, db:
            state = db.execute(
                "SELECT * FROM summary_state WHERE singleton=1"
            ).fetchone()
            cursor_id = int(state["last_message_row_id"])
            row = db.execute(
                "SELECT COUNT(*) AS count, MIN(created_at) AS oldest, "
                "COALESCE(MAX(id), ?) AS newest FROM messages WHERE id > ?",
                (cursor_id, cursor_id),
            ).fetchone()
        oldest_age = 0
        if row["oldest"]:
            try:
                oldest = datetime.fromisoformat(row["oldest"])
                if oldest.tzinfo is None:
                    oldest = oldest.replace(tzinfo=timezone.utc)
                oldest_age = max(0, int((current - oldest).total_seconds()))
            except (TypeError, ValueError):
                oldest_age = 0
        return {
            **dict(state),
            "backlog_messages": int(row["count"]),
            "cursor_lag": max(0, int(row["newest"]) - cursor_id),
            "oldest_unsummarized_age_seconds": oldest_age,
            "hard_cap": self.max_unsummarized_messages,
        }

    def claim_summary_range(
        self, expected_cursor, end_row_id, logical_day, current,
        lease_seconds,
    ):
        """Conditionally lease one exact cursor range for a bounded job."""
        now = current if isinstance(current, datetime) else datetime.fromisoformat(str(current))
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        stamp = now.isoformat()
        expires = (now + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            state = db.execute(
                "SELECT * FROM summary_state WHERE singleton=1"
            ).fetchone()
            cursor_before = int(state["last_message_row_id"])
            if cursor_before != int(expected_cursor):
                return None, "stale_cursor"
            if state["next_attempt_at"]:
                try:
                    if datetime.fromisoformat(state["next_attempt_at"]) > now:
                        return None, "backoff"
                except (TypeError, ValueError):
                    pass
            if state["claim_token"] and state["claim_expires_at"]:
                try:
                    if datetime.fromisoformat(state["claim_expires_at"]) > now:
                        return None, "claimed"
                except (TypeError, ValueError):
                    return None, "claimed"
            sequence = int(state["attempt_sequence"] or 0) + 1
            event_id = memory_event_id(
                self.chat_id, logical_day, expected_cursor, end_row_id, sequence
            )
            db.execute(
                "UPDATE summary_state SET claim_token=?, claim_start_cursor=?, "
                "claim_end_row_id=?, claim_day=?, claimed_at=?, claim_expires_at=?, "
                "attempt_sequence=? WHERE singleton=1",
                (
                    event_id, int(expected_cursor), int(end_row_id), logical_day,
                    stamp, expires, sequence,
                ),
            )
        return {
            "event_id": event_id,
            "created_at": stamp,
            "claim_expires_at": expires,
            "attempt_sequence": sequence,
        }, "claimed"

    def release_summary_claim(self, event_id):
        with self._lock, closing(self._connect()) as db, db:
            cursor = db.execute(
                "UPDATE summary_state SET claim_token=NULL, claim_start_cursor=NULL, "
                "claim_end_row_id=NULL, claim_day=NULL, claimed_at=NULL, "
                "claim_expires_at=NULL WHERE singleton=1 AND claim_token=?",
                (event_id,),
            )
            return cursor.rowcount > 0

    def fail_summary_claim(
        self, event_id, current, backoff_base_seconds, backoff_cap_seconds,
    ):
        now = current if isinstance(current, datetime) else datetime.fromisoformat(str(current))
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        with self._lock, closing(self._connect()) as db, db:
            state = db.execute(
                "SELECT claim_token, failure_count FROM summary_state WHERE singleton=1"
            ).fetchone()
            if not state or state["claim_token"] != event_id:
                return False
            failures = int(state["failure_count"] or 0) + 1
            delay = min(
                max(1, int(backoff_cap_seconds)),
                max(1, int(backoff_base_seconds)) * (2 ** min(failures - 1, 10)),
            )
            next_attempt = (now + timedelta(seconds=delay)).isoformat()
            db.execute(
                "UPDATE summary_state SET claim_token=NULL, claim_start_cursor=NULL, "
                "claim_end_row_id=NULL, claim_day=NULL, claimed_at=NULL, "
                "claim_expires_at=NULL, failure_count=?, next_attempt_at=? "
                "WHERE singleton=1 AND claim_token=?",
                (failures, next_attempt, event_id),
            )
            return True

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

    def _apply_candidate_observations(self, db, candidates, stamp, limit):
        stable_rows = [
            dict(row) for row in db.execute(
                "SELECT id, memory FROM long_memories"
            ).fetchall()
        ]
        candidate_rows = [
            dict(row) for row in db.execute(
                "SELECT * FROM memory_candidates"
            ).fetchall()
        ]
        promoted = 0
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
                (row for row in candidate_rows if memories_are_similar(row["memory"], memory)),
                None,
            )
            normalized = (
                candidate_match["normalized_memory"]
                if candidate_match else normalize_memory(memory)
            )
            previous_count = (
                int(candidate_match["observation_count"])
                if candidate_match else 0
            )
            new_count = previous_count + observations
            first_seen = (
                candidate_match["first_seen_at"] if candidate_match else stamp
            )
            promoted_at = (
                candidate_match["promoted_at"] if candidate_match else None
            )
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
                stable_rows.append({"id": cursor.lastrowid, "memory": memory})
                promoted_at = stamp
                promoted += 1
            db.execute(
                "INSERT INTO memory_candidates(normalized_memory, memory, observation_count, "
                "first_seen_at, last_seen_at, promoted_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(normalized_memory) DO UPDATE SET "
                "observation_count=excluded.observation_count, "
                "last_seen_at=excluded.last_seen_at, "
                "promoted_at=COALESCE(memory_candidates.promoted_at, excluded.promoted_at)",
                (normalized, memory, new_count, first_seen, stamp, promoted_at),
            )
            updated = {
                "normalized_memory": normalized,
                "memory": memory,
                "observation_count": new_count,
                "first_seen_at": first_seen,
                "last_seen_at": stamp,
                "promoted_at": promoted_at,
            }
            if candidate_match:
                candidate_rows[candidate_rows.index(candidate_match)] = updated
            else:
                candidate_rows.append(updated)
        overflow = db.execute(
            "SELECT COUNT(*) - ? FROM long_memories", (limit,)
        ).fetchone()[0]
        if overflow > 0:
            db.execute(
                "DELETE FROM long_memories WHERE id IN "
                "(SELECT id FROM long_memories ORDER BY score ASC, updated_at ASC LIMIT ?)",
                (overflow,),
            )
        return promoted

    @staticmethod
    def _prune_candidates(
        db, current, max_candidates, stale_days, promoted_retention_days,
    ):
        stale_cutoff = (current - timedelta(days=int(stale_days))).isoformat()
        promoted_cutoff = (
            current - timedelta(days=int(promoted_retention_days))
        ).isoformat()
        stale = db.execute(
            "DELETE FROM memory_candidates WHERE promoted_at IS NULL "
            "AND last_seen_at < ?", (stale_cutoff,)
        ).rowcount
        promoted = db.execute(
            "DELETE FROM memory_candidates WHERE promoted_at IS NOT NULL "
            "AND promoted_at < ?", (promoted_cutoff,)
        ).rowcount
        overflow = db.execute(
            "SELECT COUNT(*) - ? FROM memory_candidates", (int(max_candidates),)
        ).fetchone()[0]
        if overflow > 0:
            # Retained promoted rows are audit-only and are discarded before
            # active evidence. Remaining overflow removes weakest/oldest active.
            removed = db.execute(
                "DELETE FROM memory_candidates WHERE normalized_memory IN ("
                "SELECT normalized_memory FROM memory_candidates "
                "ORDER BY (promoted_at IS NOT NULL) DESC, observation_count ASC, "
                "last_seen_at ASC LIMIT ?)",
                (overflow,),
            ).rowcount
            promoted += removed
        return max(0, int(promoted)), max(0, int(stale))

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
            self._apply_candidate_observations(db, candidates, stamp, limit)
            db.execute(
                "UPDATE summary_state SET last_message_row_id="
                "MAX(last_message_row_id, ?), last_summary_at=?, "
                "pending_since=NULL WHERE singleton=1",
                (int(last_message_row_id), stamp),
            )
            self._prune_summarized_messages(db, int(last_message_row_id))

    def finalize_summary_job(
        self, job, summary, candidates, current, *, stable_limit=40,
        recent_limit=None, max_candidates=200, candidate_stale_days=30,
        promoted_retention_days=7, daily_summary_retention_days=90,
    ):
        """Atomically conditionally finalize one leased immutable range."""
        current = current if isinstance(current, datetime) else datetime.fromisoformat(str(current))
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        stamp = current.isoformat()
        payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        recent_limit = self.max_messages if recent_limit is None else int(recent_limit)
        with self._lock, closing(self._connect()) as db, db:
            state = db.execute(
                "SELECT * FROM summary_state WHERE singleton=1"
            ).fetchone()
            cursor_before = int(state["last_message_row_id"])
            if cursor_before >= int(job.end_message_row_id):
                if state["claim_token"] == job.event_id:
                    db.execute(
                        "UPDATE summary_state SET claim_token=NULL, "
                        "claim_start_cursor=NULL, claim_end_row_id=NULL, "
                        "claim_day=NULL, claimed_at=NULL, claim_expires_at=NULL "
                        "WHERE singleton=1 AND claim_token=?",
                        (job.event_id,),
                    )
                return SummaryFinalizeResult(
                    "already_finalized", cursor_before, cursor_before
                )
            valid = (
                state["claim_token"] == job.event_id
                and int(
                    state["claim_start_cursor"]
                    if state["claim_start_cursor"] is not None else -1
                ) == int(job.start_cursor)
                and int(
                    state["claim_end_row_id"]
                    if state["claim_end_row_id"] is not None else -1
                ) == int(job.end_message_row_id)
                and state["claim_day"] == job.logical_day
                and cursor_before == int(job.start_cursor)
            )
            if not valid:
                return SummaryFinalizeResult("stale", cursor_before, cursor_before)

            db.execute(
                "INSERT INTO daily_summaries(day, summary_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(day) DO UPDATE SET summary_json=excluded.summary_json, "
                "updated_at=excluded.updated_at",
                (job.logical_day, payload, stamp),
            )
            promoted = self._apply_candidate_observations(
                db, candidates, stamp, stable_limit
            )
            promoted_pruned, stale_pruned = self._prune_candidates(
                db, current, max_candidates, candidate_stale_days,
                promoted_retention_days,
            )
            remaining_row = db.execute(
                "SELECT COUNT(*) AS count, MIN(created_at) AS oldest "
                "FROM messages WHERE id > ?",
                (int(job.end_message_row_id),),
            ).fetchone()
            remaining = int(remaining_row["count"])
            pending_since = (
                state["pending_since"] or remaining_row["oldest"]
                if remaining else None
            )
            db.execute(
                "UPDATE summary_state SET last_message_row_id=?, last_summary_at=?, "
                "pending_since=?, claim_token=NULL, claim_start_cursor=NULL, "
                "claim_end_row_id=NULL, claim_day=NULL, claimed_at=NULL, "
                "claim_expires_at=NULL, failure_count=0, next_attempt_at=NULL "
                "WHERE singleton=1 AND claim_token=?",
                (
                    int(job.end_message_row_id), stamp, pending_since,
                    job.event_id,
                ),
            )
            # Temporarily use the requested recent window for this transaction.
            original_recent = self.max_messages
            self.max_messages = max(1, recent_limit)
            try:
                self._prune_summarized_messages(db, int(job.end_message_row_id))
            finally:
                self.max_messages = original_recent
            cutoff_day = (
                datetime.fromisoformat(job.logical_day).date()
                - timedelta(days=int(daily_summary_retention_days))
            ).isoformat()
            db.execute(
                "DELETE FROM daily_summaries WHERE day < ? AND day <> ?",
                (cutoff_day, job.logical_day),
            )
            return SummaryFinalizeResult(
                "committed", cursor_before, int(job.end_message_row_id),
                remaining, promoted, promoted_pruned, stale_pruned,
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

    def load_media_context_inputs(self):
        """Load bounded media decision signals in one read connection."""
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN")
            usage = db.execute(
                "SELECT * FROM media_usage ORDER BY id DESC LIMIT 100"
            ).fetchall()
            tagged = {}
            for media_type, table in (("gif", "gifs"), ("sticker", "stickers")):
                tagged[media_type] = db.execute(
                    f"SELECT asset.*, metadata.tags_json FROM {table} asset "
                    "JOIN media_metadata metadata ON metadata.media_type = ? "
                    "AND metadata.asset_key = asset.file_unique_id",
                    (media_type,),
                ).fetchall()
            db.rollback()
        result = {"recent_usage": [dict(row) for row in usage]}
        for media_type in ("gif", "sticker"):
            items = []
            for row in tagged[media_type]:
                item = dict(row)
                try:
                    item["tags"] = json.loads(item.pop("tags_json"))
                except (TypeError, ValueError):
                    item["tags"] = []
                items.append(item)
            result[f"tagged_{media_type}s"] = items
        return result

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

    def current_schema_version(self):
        with self._lock, closing(self._connect()) as db:
            return schema_version(db)

    def migration_status(self):
        with self._lock, closing(self._connect()) as db:
            rows = db.execute(
                "SELECT version, name, applied_at FROM schema_migrations "
                "ORDER BY version"
            ).fetchall()
        return {
            "current": self.current_schema_version(),
            "latest": CURRENT_SCHEMA_VERSION,
            "migrations": tuple(dict(row) for row in rows),
        }

    def schema_signature(self):
        with self._lock, closing(self._connect()) as db:
            return schema_signature(db)

    def run_persistence_maintenance(
        self, current=None, *, llm_retention_days=90,
        routing_retention_days=31, scheduled_retention_days=14,
        interval_seconds=86400, force=False,
    ):
        """Compact operational rows at most once per configured interval."""
        current = current or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        stamp = current.astimezone(timezone.utc).isoformat()
        llm_cutoff = (
            current - timedelta(days=max(7, int(llm_retention_days)))
        ).astimezone(timezone.utc).isoformat()
        routing_cutoff = (
            current - timedelta(days=max(7, int(routing_retention_days)))
        ).astimezone(timezone.utc).isoformat()
        scheduled_cutoff = (
            current - timedelta(days=max(1, int(scheduled_retention_days)))
        ).astimezone(timezone.utc).isoformat()
        with self._lock, closing(self._connect()) as db, db:
            last_row = db.execute(
                "SELECT value FROM persistence_meta "
                "WHERE key='last_retention_at'"
            ).fetchone()
            if last_row and not force:
                try:
                    last = datetime.fromisoformat(last_row[0])
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    if (current - last).total_seconds() < max(
                        3600, int(interval_seconds)
                    ):
                        return {"status": "not_due", "llm_calls_pruned": 0}
                except (TypeError, ValueError):
                    pass
            raw_before = db.execute(
                "SELECT COUNT(*) FROM llm_calls WHERE created_at < ?",
                (llm_cutoff,),
            ).fetchone()[0]
            db.execute(
                """INSERT INTO llm_daily_aggregates(
                    day, provider, model, call_type, calls, input_tokens,
                    cached_input_tokens, output_tokens, reasoning_tokens,
                    cost_usd_ticks
                )
                SELECT substr(created_at, 1, 10), provider, model, call_type,
                    COUNT(*), COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(cached_input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(reasoning_tokens), 0), SUM(cost_usd_ticks)
                FROM llm_calls WHERE created_at < ?
                GROUP BY substr(created_at, 1, 10), provider, model, call_type
                ON CONFLICT(day, provider, model, call_type) DO UPDATE SET
                    calls=llm_daily_aggregates.calls+excluded.calls,
                    input_tokens=llm_daily_aggregates.input_tokens+excluded.input_tokens,
                    cached_input_tokens=llm_daily_aggregates.cached_input_tokens+excluded.cached_input_tokens,
                    output_tokens=llm_daily_aggregates.output_tokens+excluded.output_tokens,
                    reasoning_tokens=llm_daily_aggregates.reasoning_tokens+excluded.reasoning_tokens,
                    cost_usd_ticks=CASE
                        WHEN llm_daily_aggregates.cost_usd_ticks IS NULL
                         AND excluded.cost_usd_ticks IS NULL THEN NULL
                        ELSE COALESCE(llm_daily_aggregates.cost_usd_ticks, 0)
                           + COALESCE(excluded.cost_usd_ticks, 0) END""",
                (llm_cutoff,),
            )
            db.execute("DELETE FROM llm_calls WHERE created_at < ?", (llm_cutoff,))
            routing_pruned = db.execute(
                "DELETE FROM routing_events WHERE created_at < ?",
                (routing_cutoff,),
            ).rowcount
            scheduled_pruned = db.execute(
                "DELETE FROM scheduled_events WHERE state IN ('SENT','DEAD') "
                "AND COALESCE(delivered_at, updated_at, created_at) < ?",
                (scheduled_cutoff,),
            ).rowcount
            metadata_pruned = db.execute(
                """DELETE FROM media_metadata WHERE
                    (media_type='gif' AND NOT EXISTS(
                        SELECT 1 FROM gifs WHERE file_unique_id=asset_key)) OR
                    (media_type='sticker' AND NOT EXISTS(
                        SELECT 1 FROM stickers WHERE file_unique_id=asset_key))"""
            ).rowcount
            image_usage_pruned = db.execute(
                "DELETE FROM chat_image_usage WHERE NOT EXISTS("
                "SELECT 1 FROM chat_images "
                "WHERE chat_images.file_unique_id=chat_image_usage.file_unique_id)"
            ).rowcount
            db.execute(
                "INSERT INTO persistence_meta(key, value) "
                "VALUES('last_retention_at', ?) ON CONFLICT(key) "
                "DO UPDATE SET value=excluded.value", (stamp,),
            )
        return {
            "status": "completed",
            "llm_calls_pruned": int(raw_before),
            "routing_events_pruned": max(0, int(routing_pruned)),
            "scheduled_events_pruned": max(0, int(scheduled_pruned)),
            "media_metadata_pruned": max(0, int(metadata_pruned)),
            "chat_image_usage_pruned": max(0, int(image_usage_pruned)),
        }

    def persistence_diagnostics(self):
        """Return schema/size/retention facts without user content."""
        with self._lock, closing(self._connect()) as db:
            page_size = int(db.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(db.execute("PRAGMA page_count").fetchone()[0])
            freelist = int(db.execute("PRAGMA freelist_count").fetchone()[0])
            names = [
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            rows = {
                name: int(db.execute(
                    "SELECT COUNT(*) FROM "
                    + '"' + str(name).replace('"', '""') + '"'
                ).fetchone()[0])
                for name in names
            }
            oldest = {}
            for name, column in (
                ("llm_calls", "created_at"),
                ("routing_events", "created_at"),
                ("daily_summaries", "day"),
                ("memory_candidates", "last_seen_at"),
            ):
                oldest[name] = db.execute(
                    f"SELECT MIN({column}) FROM {name}"
                ).fetchone()[0]
            version = schema_version(db)
        wal = self.path.with_name(self.path.name + "-wal")
        return {
            "schema_version": version,
            "latest_schema_version": CURRENT_SCHEMA_VERSION,
            "migration_current": version == CURRENT_SCHEMA_VERSION,
            "db_size_bytes": (self.path.stat().st_size if self.path.exists() else 0)
            + (wal.stat().st_size if wal.exists() else 0),
            "page_count": page_count,
            "page_size": page_size,
            "freelist_count": freelist,
            "rows_by_table": rows,
            "oldest": oldest,
            "sqlite_runtime": sqlite_runtime_diagnostics(),
        }

    def quick_check(self):
        with self._lock, closing(self._connect()) as db:
            rows = db.execute("PRAGMA quick_check").fetchall()
        return tuple(row[0] for row in rows)

    def backup_to(self, destination, *, overwrite=False):
        """Create a consistent online backup with SQLite's backup API."""
        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValueError("backup destination must differ from source")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        if destination.exists():
            destination.unlink()
        with self._lock, closing(self._connect()) as source, closing(
            sqlite3.connect(destination, timeout=5)
        ) as target:
            source.backup(target)
            check = target.execute("PRAGMA quick_check").fetchone()[0]
            if check != "ok":
                raise sqlite3.DatabaseError(f"backup quick_check failed: {check}")
        return destination

    def clear(self):
        """Physically forget the isolated chat DB, including WAL/SHM pages."""
        with self._lock:
            if self.path.exists():
                with closing(self._connect()) as db:
                    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            for path in (
                self.path,
                self.path.with_name(self.path.name + "-wal"),
                self.path.with_name(self.path.name + "-shm"),
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self._initialize()
        return True
