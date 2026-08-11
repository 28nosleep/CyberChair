import re
from datetime import date, datetime, timedelta
import pytz

# ==========================================
# ПОИСК СЛОВА "СТУЛ"
# ==========================================

STUL_PATTERN = re.compile(
    r"\bстул\w*|\bстуль\w*",
    re.IGNORECASE
)


# ==========================================
# ПРОИЗВОДСТВЕННЫЙ КАЛЕНДАРЬ РФ
# ==========================================

# Переносы 2026 года: постановление Правительства РФ №1466.
NON_WORKING_DAYS = {
    2026: {
        date(2026, 1, day) for day in range(1, 12)
    } | {
        date(2026, 2, day) for day in range(21, 24)
    } | {
        date(2026, 3, day) for day in range(7, 10)
    } | {
        date(2026, 5, day) for day in range(1, 4)
    } | {
        date(2026, 5, day) for day in range(9, 12)
    } | {
        date(2026, 6, day) for day in range(12, 15)
    } | {
        date(2026, 11, 4),
        date(2026, 12, 31),
    },
}

WORKING_WEEKENDS = {
    2026: set(),
}


def is_fixed_public_holiday(current_date):

    month_and_day = (
        current_date.month,
        current_date.day
    )

    return (
        current_date.month == 1
        and 1 <= current_date.day <= 8
    ) or month_and_day in {
        (2, 23),
        (3, 8),
        (5, 1),
        (5, 9),
        (6, 12),
        (11, 4),
    }


def is_stul_message(text):

    if not text:
        return False

    return bool(STUL_PATTERN.search(text))


# ==========================================
# СКЛОНЕНИЯ
# ==========================================

def plural(number, one, two, five):

    number = abs(number) % 100
    last = number % 10

    if 11 <= number <= 19:
        return five

    if last == 1:
        return one

    if 2 <= last <= 4:
        return two

    return five


def format_time(hours, minutes):

    return (
        f"{hours} "
        f"{plural(hours,'час','часа','часов')} "
        f"{minutes} "
        f"{plural(minutes,'минута','минуты','минут')}"
    )


def stul_remaining_variants(hours, minutes):
    remaining = format_time(hours, minutes)
    return [
        f"🪑 Осталось стула: {remaining}",
        f"⏳ Стула осталось: {remaining}",
        f"🪑 До конца стула: {remaining}",
    ]


# ==========================================
# ВРЕМЯ
# ==========================================

def get_timezone(name):

    return pytz.timezone(name)


def get_now(name):

    tz = get_timezone(name)

    return datetime.now(tz)


def get_today_times(
    timezone_name,
    start_hour,
    start_minute,
    end_hour,
    end_minute
):

    tz = get_timezone(timezone_name)

    now = get_now(timezone_name)

    start = tz.localize(
        datetime(
            now.year,
            now.month,
            now.day,
            start_hour,
            start_minute
        )
    )

    end = tz.localize(
        datetime(
            now.year,
            now.month,
            now.day,
            end_hour,
            end_minute
        )
    )

    return start, end


def seconds_to_hm(seconds, round_up=False):

    if round_up:
        total_minutes = (max(seconds, 0) + 59) // 60
    else:
        total_minutes = max(seconds, 0) // 60

    hours, minutes = divmod(total_minutes, 60)

    return hours, minutes


# ==========================================
# ВЫХОДНОЙ?
# ==========================================

def is_weekend(current):

    return current.weekday() >= 5


def is_public_holiday(current):

    current_date = current.date()

    return (
        is_fixed_public_holiday(current_date)
        or current_date in NON_WORKING_DAYS.get(
            current_date.year,
            set()
        )
    )


def is_workday(current):

    current_date = current.date()
    year = current_date.year

    if current_date in WORKING_WEEKENDS.get(year, set()):
        return True

    if (
        is_fixed_public_holiday(current_date)
        or current_date in NON_WORKING_DAYS.get(year, set())
    ):
        return False

    return current.weekday() < 5


def next_workday_start(
    current,
    timezone_name,
    start_hour,
    start_minute
):

    tz = get_timezone(timezone_name)

    day = current

    while True:

        day += timedelta(days=1)

        if is_workday(day):

            return tz.localize(
                datetime(
                    day.year,
                    day.month,
                    day.day,
                    start_hour,
                    start_minute
                )
            )
