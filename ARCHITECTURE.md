# CyberChair architecture

CyberChair is a single-process modular monolith. Telegram polling, domain
orchestration, provider calls, background maintenance and per-chat SQLite
repositories run in one process, with explicit boundaries between transport,
decision making and persistence.

## Event pipeline

```text
Telegram update
  → NormalizedEvent
  → EventContext
  → per-chat FIFO arbitration
  → ContextSnapshot
  → ForegroundOrchestrator
      → GenerationCoordinator / MediaCoordinator / LocalResponder / Markov
  → ResponsePlan
  → Telegram adapter delivery
  → DeliveryReceipt
  → ResponseLifecycle commit or abort
```

`bot.py` owns Telegram extraction, downloads, chat actions and delivery. Core
components never call Telegram. A `ResponsePlan` contains only the final typed
intent and commit actions; it does not contain locks, provider clients,
repository connections or Telegram objects.

## Component ownership

| Component | Responsibility | Does not own |
|---|---|---|
| `LearningService` | Composition root and supported facade | Domain algorithms or Telegram delivery |
| `ForegroundOrchestrator` | Existing foreground route ordering and final-plan arbitration | Telegram parsing/delivery or persistence schema |
| `GenerationCoordinator` | Provider-neutral LLM, LocalResponder and Markov generation/fallback | Telegram transport or routing precedence |
| `MediaCoordinator` | Media selection, explicit meme preparation and render coordination | Telegram send or renderer implementation |
| `AutonomousCoordinator` | Optional autonomous decision lifecycle | Foreground scheduling or memory maintenance |
| `ResponseLifecycle` | Typed post-delivery commit/abort actions | Telegram delivery |
| `ContextSnapshotBuilder` | One immutable event-local database view | Routing decisions or LLM calls |
| `MemoryFacade` | Memory ingestion and maintenance entry points | Foreground orchestration |
| `MemoryMaintenanceRunner` | Summary claim, background admission and conditional finalize | User responses or `ResponsePlan` |
| `ScheduledDeliveryCoordinator` | Durable utility-event identity, lease, Telegram attempt outcome and diagnostics | Schedule calculation, Telegram internals or foreground state |
| `ShutdownCoordinator` | Process RUNNING/DRAINING/STOPPED state, stop hooks, active-work drain and one grace deadline | Routing, transport/provider logic or SQL business state |
| `ChatRepository` | Short SQLite transactions and domain persistence operations | Provider calls or orchestration |

Dependency direction is one-way:

```text
bot.py
  → LearningService
    → orchestration components
      → specialized domain components
        → repository / provider interfaces
```

Extracted components do not import `LearningService`; dependencies are passed
explicitly by the composition root.

## Concurrency

- Foreground lifecycles for one chat are FIFO-serialized from snapshot through
  delivery commit/abort.
- Different chats remain parallel.
- Global LLM calls and heavy media work use independent bounded admission.
- Foreground LLM work may wait for capacity; optional background work skips or
  defers under pressure.
- Acquisition order is chat gate → resource admission → short repository
  operation. SQLite connections are not held during provider or Telegram calls.

## Runtime lifecycle

```text
STARTING → RUNNING → DRAINING → STOPPED
```

SIGTERM and SIGINT enter the same idempotent drain path. The exact stop order is:
stop Telegram polling, signal the scheduler, close R4 admission/wake waiters,
stop chat-action refreshers, then wait for already active foreground, provider,
media, scheduled and memory work. All waits share one monotonic deadline from
`SHUTDOWN_GRACE_SECONDS`; no component receives its own additional timeout.
After expiry the runtime reports unfinished content-free counters and returns
without forcibly killing Python threads. Existing P1/R5 leases and SQLite WAL
provide restart safety; shutdown does not run checkpoint, retention or backup.

## Memory lifecycle

Foreground ingestion persists a message and marks summary work due. It performs
zero summary LLM calls.

```text
maintenance tick
  → immutable SummaryJob for a bounded row-id range
  → durable claim/lease
  → background LLM admission
  → one provider attempt
  → atomic conditional finalize
```

New messages cannot change an existing job. Failed or stale jobs do not advance
the cursor. Unsummarized rows are protected from normal pruning; the configured
default hard outage envelope is 500 messages. Summary candidates and daily
summaries use the bounded R5 lifecycle.

## Persistence and operations

Each chat has an isolated SQLite database in `LEARNING_DATA_DIR`. Repositories
use WAL, short transactions and forward-only migrations. The current schema is
`CURRENT_SCHEMA_VERSION = 5`; a database created by a newer application version
is rejected without mutation.

Scheduled utility notifications use one durable row per logical event:

```text
PENDING → CLAIMED (expiring lease) → SENDING → SENT
                                      ├→ RETRY_WAIT (definite non-delivery only)
                                      ├→ UNKNOWN (ambiguous outcome; never auto-retried)
                                      └→ DEAD (permanent/exhausted)
```

The payload is fixed when the event is first created. No SQLite transaction or
foreground chat gate is held during Telegram I/O. Only a successful Telegram
response can finalize `SENT`; `telegram_message_id` and `delivered_at` are kept
when available. Existing pre-v5 claim markers migrate as terminal historical
rows and are never replayed.

Detailed LLM telemetry is retained for 90 days by default and compacted into
content-free daily cost aggregates. Routing and scheduled-event detail have
their own bounded retention. Persistence maintenance never prunes protected
unsummarized rows.

Online backup uses `ChatRepository.backup_to()`, which invokes the SQLite Backup
API and `quick_check`. Do not copy an active `.sqlite3` alone while WAL may
contain committed pages. Backups contain sensitive user data and must remain
outside source control.

`/forget_chat confirm` physically replaces the entire isolated per-chat DB,
including WAL/SHM files, and recreates an empty schema-v5 database. Old summary
or scheduled-delivery claims cannot finalize into the replacement database.

## Hard invariants

1. One event performs at most one LLM network call.
2. One user event creates at most one final `ResponsePlan`.
3. Delivered-state persistence happens only after Telegram success.
4. One foreground event builds at most one base `ContextSnapshot`.
5. Same-chat foreground lifecycles are serial.
6. Different chats may execute in parallel.
7. Summary generation never runs inside a foreground Telegram event.
8. Unsummarized message rows are protected from ordinary pruning.
9. SQLite migrations are forward-only and versioned.
10. `/forget_chat` physically replaces the per-chat database.
11. A scheduled claim is never treated as delivery success.
12. `UNKNOWN` scheduled outcomes are never automatically retried.
13. Runtime drain rejects new work and uses one bounded monotonic deadline.

## Accepted boundaries

Graceful process-local SIGTERM/SIGINT draining is implemented. Deployment
automation and production monitoring remain separate production-hardening
work; P3 and later stages are not part of this architecture change.
