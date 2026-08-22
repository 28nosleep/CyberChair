"""Source-grounded evidence capture, retrieval and delayed callbacks."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import re

from .preprocessing import normalize_spaces, significant_words


PROMISE_RE = re.compile(
    r"(?:\bбольше\b.{0,40}\bне\b|\bникогда\s+не\b|\bобещаю\b|\bточно\s+(?:не\s+)?буду\b|\bзавтра\s+сделаю\b)",
    re.I,
)
FUNNY_RE = re.compile(r"(?:ахах|хаха|лол|кек|💀|😂|шиз|крашаут|обосрал|проеб)", re.I)
STRONG_STATEMENT_RE = re.compile(
    r"(?:\bя\s+(?:всегда|никогда|точно|считаю|буду|не\s+буду)\b|\bмой\s+план\b)", re.I,
)


@dataclass(frozen=True)
class Evidence:
    id: int
    chat_id: int
    user_id: int | None
    message_id: int
    timestamp: str
    text: str | None
    image_reference: str | None
    evidence_type: str
    score: float
    use_count: int = 0
    last_used_at: str | None = None

    def debug(self):
        return asdict(self)


class EvidenceEngine:
    """Keep candidates now; surface them only when a later moment fits."""

    def __init__(self, max_items=400, reuse_cooldown_days=7):
        self.max_items = max(50, int(max_items))
        self.reuse_cooldown_days = max(1, int(reuse_cooldown_days))

    @staticmethod
    def _topic(text):
        return " ".join(significant_words(text or "")[:8])

    def capture_message(self, repository, event, moments=()):
        text = normalize_spaces(event.effective_text)
        if not text or event.user_id is None:
            return ()
        candidates = []
        if PROMISE_RE.search(text):
            candidates.append(("promise", .9))
        if STRONG_STATEMENT_RE.search(text):
            candidates.append(("statement", .72))
        if FUNNY_RE.search(text):
            candidates.append(("funny_message", .7))
        moment_types = {moment.moment_type for moment in moments}
        if "repeated_topic" in moment_types:
            candidates.append(("recurring_behavior", .66))
        if "self_own" in moment_types:
            candidates.append(("meme_moment", .74))
        stored = []
        for evidence_type, score in candidates:
            row = repository.add_evidence(
                user_id=event.user_id,
                source_message_id=event.message_id,
                source_timestamp=(
                    event.timestamp.isoformat()
                    if getattr(event.timestamp, "isoformat", None)
                    else str(event.timestamp)
                ),
                source_text=text,
                evidence_type=evidence_type,
                normalized_topic=self._topic(text),
                score=score,
                max_items=self.max_items,
            )
            if row:
                stored.append(self._from_row(row))
        return tuple(stored)

    def capture_image(self, repository, metadata):
        if not metadata or metadata.get("from_bot") or not metadata.get("file_id"):
            return None
        caption = normalize_spaces(metadata.get("caption") or "")
        row = repository.add_evidence(
            user_id=metadata.get("user_id"),
            source_message_id=metadata["message_id"],
            source_timestamp=str(metadata.get("created_at") or datetime.now(timezone.utc).isoformat()),
            source_text=caption or None,
            image_file_id=metadata["file_id"],
            image_file_unique_id=metadata.get("file_unique_id"),
            evidence_type="image",
            normalized_topic=self._topic(caption),
            score=.68 + (.08 if caption else 0.0),
            max_items=self.max_items,
            validate_message=False,
        )
        return self._from_row(row) if row else None

    @staticmethod
    def _from_row(row):
        if not row:
            return None
        image = row.get("image_file_id")
        return Evidence(
            id=int(row["id"]), chat_id=int(row["chat_id"]),
            user_id=row.get("user_id"), message_id=int(row["source_message_id"]),
            timestamp=str(row["source_timestamp"]), text=row.get("source_text"),
            image_reference=image, evidence_type=str(row["evidence_type"]),
            score=float(row["score"]), use_count=int(row.get("use_count") or 0),
            last_used_at=row.get("last_used_at"),
        )

    def retrieve(self, repository, current_text, moment, user_id, current_message_id,
                 limit=5, current=None):
        if moment is None:
            return ()
        current = current or datetime.now(timezone.utc)
        cutoff = (current - timedelta(days=self.reuse_cooldown_days)).isoformat()
        terms = set(significant_words(current_text or ""))
        rows = repository.evidence_candidates(user_id=user_id, limit=120)
        scored = []
        for row in rows:
            if int(row["source_message_id"]) == int(current_message_id):
                continue
            if row.get("last_used_at") and str(row["last_used_at"]) >= cutoff:
                continue
            if int(row.get("use_count") or 0) >= 3:
                continue
            row_terms = set(significant_words(
                f"{row.get('source_text') or ''} {row.get('normalized_topic') or ''}"
            ))
            overlap = len(terms & row_terms)
            kind = row["evidence_type"]
            fit = 0.0
            if moment.moment_type == "contradiction" and kind in {"promise", "statement"}:
                fit += .45
            elif moment.moment_type in {"self_own", "callback_opportunity"} and kind != "image":
                fit += .36
            elif moment.moment_type == "meme_opportunity" and kind in {"image", "funny_message", "meme_moment"}:
                fit += .35
            if kind != "image" and row.get("user_id") != user_id:
                fit -= .35
            score = float(row["score"]) + min(.4, overlap * .12) + fit - int(row.get("use_count") or 0) * .18
            if score >= .75 and (overlap or fit >= .35):
                scored.append((score, int(row["id"]), row))
        return tuple(
            self._from_row(item[2])
            for item in sorted(scored, reverse=True)[: max(0, int(limit))]
        )

    @staticmethod
    def callback_text(evidence, current_text):
        """Compose only from the stored quote; never invent a prior claim."""
        if evidence is None or not normalize_spaces(evidence.text or ""):
            return None
        quote = normalize_spaces(evidence.text)[:180]
        current = normalize_spaces(current_text)[:100]
        if not quote or quote == current:
            return None
        return f"Стул всё видел: «{quote}»"
