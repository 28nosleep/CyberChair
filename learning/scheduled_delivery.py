"""Minimal durable lifecycle for utility notifications sent by the scheduler."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .event_context import scheduled_event_id
from .response_plan import DeliveryReceipt, DeliveryType


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledEventSpec:
    event_key: str
    event_kind: str
    scheduled_at: datetime
    payload: str
    parse_mode: str | None = None


@dataclass(frozen=True)
class ScheduledDeliveryResult:
    event_id: str
    state: str
    attempted: bool = False


@dataclass(frozen=True)
class _Failure:
    outcome: str
    category: str
    retry_after: int | None = None


class ScheduledDeliveryCoordinator:
    """Coordinate short SQLite transactions around one Telegram network call."""

    def __init__(
        self,
        repository,
        *,
        lease_seconds=120,
        max_attempts=5,
        backoff_base_seconds=30,
        backoff_cap_seconds=3600,
        clock=None,
    ):
        self.repository = repository
        self.lease_seconds = max(1, int(lease_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base_seconds = max(1, int(backoff_base_seconds))
        self.backoff_cap_seconds = max(
            self.backoff_base_seconds, int(backoff_cap_seconds)
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._recovered_database_identity = {}

    @staticmethod
    def _utc(value):
        value = value or datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _telemetry(self, repository, event_type, event_id, event_kind, now):
        try:
            repository.record_routing_event(
                event_type,
                created_at=now,
                event_id=event_id,
                call_type=event_kind,
            )
        except Exception as error:
            log.warning(
                "SCHEDULED_TELEMETRY_FAILED event_id=%s event_type=%s error=%s",
                event_id, event_type, type(error).__name__,
            )

    @staticmethod
    def _database_identity(repository):
        try:
            stat = repository.path.stat()
            return stat.st_dev, stat.st_ino
        except OSError:
            return None

    def _recover_after_restart(self, repository, now):
        key = str(repository.path)
        identity = self._database_identity(repository)
        if self._recovered_database_identity.get(key) == identity:
            return
        recovered = repository.recover_interrupted_scheduled_sends(now)
        self._recovered_database_identity[key] = self._database_identity(repository)
        for row in recovered:
            self._telemetry(
                repository, "scheduled_unknown", row["event_id"],
                row["event_kind"], now,
            )

    def ensure_event(self, chat_id, spec, current=None):
        now = self._utc(current or self.clock())
        repository = self.repository(chat_id)
        self._recover_after_restart(repository, now)
        event_id = scheduled_event_id(chat_id, spec.event_kind, spec.event_key)
        row, created = repository.ensure_scheduled_event(
            event_id,
            spec.event_key,
            spec.event_kind,
            self._utc(spec.scheduled_at),
            spec.payload,
            spec.parse_mode,
            now,
        )
        if created:
            self._telemetry(
                repository, "scheduled_created", event_id, spec.event_kind, now
            )
        return row, created

    def deliver_event(self, chat_id, spec, sender, current=None):
        now = self._utc(current or self.clock())
        row, _ = self.ensure_event(chat_id, spec, now)
        if row is None:
            return ScheduledDeliveryResult("", "STALE")
        return self._claim_and_send(
            chat_id, self.repository(chat_id), sender, now, row["event_id"]
        )

    def deliver_pending(self, chat_id, sender, current=None, limit=10):
        now = self._utc(current or self.clock())
        repository = self.repository(chat_id)
        self._recover_after_restart(repository, now)
        results = []
        for _ in range(max(1, int(limit))):
            result = self._claim_and_send(chat_id, repository, sender, now)
            if result.state == "NONE":
                break
            results.append(result)
        return tuple(results)

    def _claim_and_send(
        self, chat_id, repository, sender, now, requested_event_id=None,
    ):
        claim = repository.claim_due_scheduled_event(
            now, self.lease_seconds, requested_event_id
        )
        if claim is None:
            state = "NONE"
            if requested_event_id is not None:
                row = repository.scheduled_event(event_id=requested_event_id)
                state = row["state"] if row else "STALE"
            return ScheduledDeliveryResult(requested_event_id or "", state)
        event_id = claim["event_id"]
        event_kind = claim["event_kind"]
        if claim.get("claim_recovered"):
            self._telemetry(
                repository, "scheduled_claim_recovered", event_id,
                event_kind, now,
            )
        self._telemetry(
            repository, "scheduled_claimed", event_id, event_kind, now
        )
        sending = repository.mark_scheduled_sending(
            event_id, claim["claim_token"], now
        )
        if sending is None:
            return ScheduledDeliveryResult(event_id, "STALE")
        self._telemetry(
            repository, "scheduled_attempt", event_id, event_kind, now
        )

        # No repository connection/transaction is alive while this callback
        # performs Telegram network I/O.
        try:
            transport_result = sender(
                event_id,
                int(chat_id),
                sending["payload"],
                sending["parse_mode"],
            )
            receipt = self._receipt(event_id, transport_result)
        except Exception as error:
            failure = self._classify_exception(error)
            return self._finalize_failure(
                repository, sending, failure, now
            )

        if not receipt.success:
            return self._finalize_failure(
                repository,
                sending,
                self._classify_category(receipt.error_category),
                now,
            )
        try:
            status = repository.finalize_scheduled_success(
                event_id,
                sending["claim_token"],
                receipt.telegram_message_id,
                now,
            )
        except Exception as error:
            log.error(
                "POST_DELIVERY_SCHEDULE_COMMIT_FAILED event_id=%s error=%s",
                event_id, type(error).__name__,
            )
            try:
                repository.quarantine_scheduled_after_commit_failure(
                    event_id,
                    sending["claim_token"],
                    "post_delivery_commit_failed",
                    now,
                )
                self._telemetry(
                    repository, "scheduled_unknown", event_id, event_kind, now
                )
            except Exception as quarantine_error:
                # Persisted SENDING is itself a no-retry state and will become
                # UNKNOWN during restart recovery.
                log.error(
                    "SCHEDULED_UNKNOWN_FENCE_FAILED event_id=%s error=%s",
                    event_id, type(quarantine_error).__name__,
                )
            return ScheduledDeliveryResult(event_id, "UNKNOWN", attempted=True)
        if status == "sent":
            self._telemetry(
                repository, "scheduled_success", event_id, event_kind, now
            )
            return ScheduledDeliveryResult(event_id, "SENT", attempted=True)
        return ScheduledDeliveryResult(event_id, "STALE", attempted=True)

    @staticmethod
    def _receipt(event_id, result):
        if isinstance(result, DeliveryReceipt):
            return result
        if result is None:
            return DeliveryReceipt(
                event_id=event_id,
                success=False,
                delivery_type=DeliveryType.TEXT,
                error_category="empty_telegram_response",
            )
        message_id = getattr(result, "message_id", getattr(result, "id", None))
        return DeliveryReceipt(
            event_id=event_id,
            success=True,
            delivery_type=DeliveryType.TEXT,
            telegram_message_id=message_id,
        )

    @staticmethod
    def _telegram_error_code(error):
        code = getattr(error, "error_code", None)
        if code is None:
            result = getattr(error, "result", None)
            code = getattr(result, "status_code", None)
        try:
            return int(code) if code is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _retry_after(error):
        value = getattr(error, "retry_after", None)
        payload = getattr(error, "result_json", None)
        if value is None and isinstance(payload, dict):
            value = (payload.get("parameters") or {}).get("retry_after")
        try:
            return max(1, int(value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _classify_exception(cls, error):
        code = cls._telegram_error_code(error)
        if code == 429:
            return _Failure("retry", "telegram_rate_limit", cls._retry_after(error))
        if code is not None and code >= 500:
            return _Failure("retry", "telegram_server_error")
        if code is not None:
            return _Failure("dead", f"telegram_http_{code}")
        if isinstance(error, TimeoutError):
            return _Failure("unknown", "telegram_timeout")
        if isinstance(error, (ConnectionError, OSError)):
            return _Failure("unknown", "telegram_transport")
        return _Failure("unknown", "telegram_ambiguous")

    @staticmethod
    def _classify_category(category):
        category = str(category or "telegram_ambiguous")
        lowered = category.casefold()
        if "rate_limit" in lowered:
            return _Failure("retry", category)
        if any(value in lowered for value in ("timeout", "network", "transport", "ambiguous")):
            return _Failure("unknown", category)
        if any(value in lowered for value in ("permanent", "forbidden", "bad_request")):
            return _Failure("dead", category)
        return _Failure("unknown", category)

    def _finalize_failure(self, repository, event, failure, now):
        attempts = int(event["attempt_count"])
        if failure.outcome == "unknown":
            state = "UNKNOWN"
            next_attempt = None
            telemetry = "scheduled_unknown"
            safe_retry = False
        elif failure.outcome == "dead" or attempts >= self.max_attempts:
            state = "DEAD"
            next_attempt = None
            telemetry = "scheduled_dead"
            safe_retry = False
        else:
            backoff = min(
                self.backoff_cap_seconds,
                self.backoff_base_seconds * (2 ** max(0, attempts - 1)),
            )
            if failure.retry_after is not None:
                backoff = max(backoff, failure.retry_after)
            state = "RETRY_WAIT"
            next_attempt = now + timedelta(seconds=backoff)
            telemetry = "scheduled_retry"
            safe_retry = True
        status = repository.finalize_scheduled_failure(
            event["event_id"],
            event["claim_token"],
            state,
            failure.category,
            current=now,
            next_attempt_at=next_attempt,
            safe_retry=safe_retry,
        )
        if status != "stale":
            self._telemetry(
                repository, telemetry, event["event_id"],
                event["event_kind"], now,
            )
        return ScheduledDeliveryResult(
            event["event_id"], status.upper(), attempted=True
        )

    def diagnostics(self, chat_id, current=None):
        return self.repository(chat_id).scheduled_delivery_report(
            self._utc(current or self.clock())
        )

    def format_diagnostics(self, chat_id, current=None):
        report = self.diagnostics(chat_id, current)
        return "\n".join((
            "SCHEDULED DELIVERY",
            f"pending: {report['pending']}",
            f"claimed: {report['claimed']}",
            f"sending: {report['sending']}",
            f"retry_wait: {report['retry_wait']}",
            f"unknown: {report['unknown']}",
            f"dead: {report['dead']}",
            f"sent_recent: {report['sent_recent']}",
            f"safe_retries: {report['safe_retries']}",
            f"last_success: {report['last_success']}",
            f"last_failure_category: {report['last_failure_category']}",
            f"oldest_pending_age: {report['oldest_pending_age']}",
        ))
