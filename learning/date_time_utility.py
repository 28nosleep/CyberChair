"""Deterministic current-date/time answers in the canonical application zone."""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .preprocessing import normalize_spaces


MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
WEEKDAYS = (
    "понедельник", "вторник", "среда", "четверг",
    "пятница", "суббота", "воскресенье",
)


class DateTimeUtility:
    """Narrow intent recognizer and formatter; never delegates facts to an LLM."""

    _date = re.compile(
        r"^(?:(?:стул|стульчик|киберстул)[, ]+)?(?:"
        r"какое сегодня число|какая сегодня дата|сегодня какое число|"
        r"(?:какой|сегодня какой) сегодня день|сегодня какой день|"
        r"какой день недели|дата)\??$", re.I,
    )
    _time = re.compile(
        r"^(?:(?:стул|стульчик|киберстул)[, ]+)?(?:"
        r"который час|сколько (?:сейчас )?времени|какое сейчас время|время)\??$",
        re.I,
    )
    _tomorrow = re.compile(
        r"^(?:(?:стул|стульчик|киберстул)[, ]+)?(?:что завтра за день|"
        r"какой (?:завтра день|день завтра))\??$", re.I,
    )
    _yesterday = re.compile(
        r"^(?:(?:стул|стульчик|киберстул)[, ]+)?(?:какой день был вчера|"
        r"что вчера был за день)\??$", re.I,
    )

    def __init__(self, timezone_name, clock=None):
        self.timezone = ZoneInfo(timezone_name)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def now(self):
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(self.timezone)

    @staticmethod
    def _clean(text):
        return normalize_spaces(text or "").casefold().strip(" .,!?:;—–-")

    @staticmethod
    def _date_text(day):
        return f"{day.day} {MONTHS[day.month - 1]} {day.year}, {WEEKDAYS[day.weekday()]}"

    def answer(self, text):
        clean = self._clean(text)
        # These exclusions are deliberately explicit: semantic/schedule and
        # programming questions containing the same nouns keep their routes.
        if re.search(r"\b(?:python|питон|получить|почему|летит|ид[её]т|"
                     r"заканчива|рабоч|стул заканчивается)\b", clean, re.I):
            return None
        current = self.now()  # sampled at answer construction, never at startup/snapshot
        if self._time.fullmatch(clean):
            return f"сейчас {current:%H:%M}"
        if self._tomorrow.fullmatch(clean):
            return f"завтра {self._date_text((current + timedelta(days=1)).date())}"
        if self._yesterday.fullmatch(clean):
            return f"вчера был {self._date_text((current - timedelta(days=1)).date())}"
        if self._date.fullmatch(clean):
            if "день недели" in clean or clean.endswith("сегодня день") or clean.endswith("сегодня какой день"):
                return f"сегодня {WEEKDAYS[current.weekday()]}"
            return f"сегодня {self._date_text(current.date())}"
        return None
