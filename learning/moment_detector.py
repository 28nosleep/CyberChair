"""Bounded, deterministic detection of socially interesting chat moments."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re

from .preprocessing import normalize_spaces, significant_words


MOMENT_TYPES = (
    "contradiction", "self_own", "argument_loop", "message_burst",
    "absurd_statement", "unexpected_agreement", "callback_opportunity",
    "meme_opportunity", "pile_on", "repeated_topic",
)
NEGATION_RE = re.compile(r"\b(?:не|никогда|больше\s+не|низа?что)\b", re.I)
SELF_OWN_RE = re.compile(
    r"(?:\bопять\b|\bснова\b|\bтрет(?:ий|ья|ье)\b|\bобосрал\w*\b|"
    r"\bпроеб\w*\b|\bсломал\w*\b|\bзабыл\w*\b|\bне\s+получилось\b)", re.I,
)
ABSURD_RE = re.compile(
    r"(?:рептилоид|плоск\w*\s+земл|вечн\w*\s+двигател|"
    r"телепат|агарт|гипербор|\b1000%\b|\bникогда\s+не\s+сплю\b)", re.I,
)
AGREEMENT_RE = re.compile(r"(?:\bсогласен\b|\bсогласна\b|\bты\s+прав\b|\bда,?\s+точно\b)", re.I)
ARGUMENT_RE = re.compile(r"(?:\bнет\b|\bнеправда\b|\bчушь\b|\bбред\b|\bвр[её]шь\b)", re.I)
FUNNY_RE = re.compile(r"(?:ахах|хаха|лол|кек|💀|😂|мем|шиз|крашаут)", re.I)


def _stamp(value):
    try:
        result = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


@dataclass(frozen=True)
class Moment:
    moment_type: str
    score: float
    participants: tuple[int, ...]
    supporting_message_ids: tuple[int, ...]

    @property
    def confidence(self):
        return self.score

    def debug(self):
        return asdict(self)


class MomentDetector:
    def __init__(self, max_messages=40):
        self.max_messages = max(8, min(80, int(max_messages)))

    @staticmethod
    def _moment(kind, score, rows):
        participants = tuple(dict.fromkeys(
            int(row["user_id"]) for row in rows if row.get("user_id") is not None
        ))
        message_ids = tuple(dict.fromkeys(
            int(row["message_id"]) for row in rows if row.get("message_id") is not None
        ))
        return Moment(kind, round(min(1.0, score), 3), participants, message_ids)

    @staticmethod
    def _topic(row):
        terms = set(significant_words(row.get("text") or ""))
        normalized = normalize_spaces(row.get("text") or "").casefold().replace("ё", "е")
        concepts = (
            ("energy_drink", r"энергетик|монстр|red\s*bull|адреналин"),
            ("alcohol", r"пив|вино|водк|алкогол"),
            ("release", r"релиз|депло|выкат|прод"),
            ("sleep", r"спать|сон|не\s+сплю|бессон"),
        )
        for label, pattern in concepts:
            if re.search(pattern, normalized):
                terms.add(label)
        return terms

    def detect(self, rows, current_message_id=None):
        window = [dict(row) for row in tuple(rows)[-self.max_messages:] if row.get("text")]
        if not window:
            return ()
        current = next(
            (row for row in reversed(window) if row.get("message_id") == current_message_id),
            window[-1],
        )
        current_text = normalize_spaces(current.get("text", ""))
        current_terms = self._topic(current)
        current_user = current.get("user_id")
        prior = [row for row in window if row.get("message_id") != current.get("message_id")]
        moments = []

        for row in reversed(prior[-24:]):
            if row.get("user_id") != current_user:
                continue
            overlap = current_terms & self._topic(row)
            if len(overlap) >= 1 and bool(NEGATION_RE.search(current_text)) != bool(
                NEGATION_RE.search(row.get("text", ""))
            ):
                moments.append(self._moment("contradiction", .78 + min(.14, len(overlap) * .04), (row, current)))
                break

        if SELF_OWN_RE.search(current_text):
            supporting = [current]
            related = next((
                row for row in reversed(prior[-24:])
                if row.get("user_id") == current_user
                and current_terms & self._topic(row)
            ), None)
            if related:
                supporting.insert(0, related)
            moments.append(self._moment("self_own", .76 if related else .68, supporting))

        recent_argument = [row for row in window[-10:] if ARGUMENT_RE.search(row.get("text", ""))]
        argument_users = {row.get("user_id") for row in recent_argument if row.get("user_id") is not None}
        argument_terms = [self._topic(row) for row in recent_argument]
        shared_argument = set.intersection(*argument_terms) if len(argument_terms) >= 2 else set()
        if len(recent_argument) >= 4 and len(argument_users) >= 2 and shared_argument:
            moments.append(self._moment("argument_loop", .82, recent_argument[-6:]))

        current_at = _stamp(current.get("created_at"))
        if current_at:
            burst = [
                row for row in window
                if _stamp(row.get("created_at")) is not None
                and 0 <= (current_at - _stamp(row.get("created_at"))).total_seconds() <= 60
            ]
            if len(burst) >= 6:
                moments.append(self._moment("message_burst", min(.95, .58 + len(burst) * .04), burst))

        if ABSURD_RE.search(current_text):
            moments.append(self._moment("absurd_statement", .75, (current,)))
        prior_conflict = next((
            row for row in reversed(prior[-8:])
            if ARGUMENT_RE.search(row.get("text", ""))
        ), None)
        if AGREEMENT_RE.search(current_text) and prior_conflict:
            moments.append(self._moment(
                "unexpected_agreement", .7, (prior_conflict, current)
            ))

        callback = next((
            row for row in prior[:-5]
            if row.get("user_id") == current_user
            and len(current_terms & self._topic(row)) >= 2
        ), None)
        if callback:
            moments.append(self._moment("callback_opportunity", .67, (callback, current)))
        if FUNNY_RE.search(current_text) or (len(current_text) <= 100 and SELF_OWN_RE.search(current_text)):
            moments.append(self._moment("meme_opportunity", .64, (current,)))

        replies = [row for row in window[-10:] if row.get("reply_to_message_id")]
        targets = {}
        for row in replies:
            targets.setdefault(row["reply_to_message_id"], []).append(row)
        piled = max(targets.values(), key=len, default=[])
        if len({row.get("user_id") for row in piled}) >= 3:
            moments.append(self._moment("pile_on", .8, piled))

        topic_rows = [row for row in window[-12:] if current_terms & self._topic(row)]
        if len(topic_rows) >= 4:
            moments.append(self._moment("repeated_topic", min(.82, .5 + len(topic_rows) * .04), topic_rows[-6:]))
        return tuple(sorted(moments, key=lambda item: item.score, reverse=True))

    def primary(self, rows, current_message_id=None):
        moments = self.detect(rows, current_message_id)
        return moments[0] if moments else None
