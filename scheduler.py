import random
import time
from collections import deque
from datetime import timedelta

from messages import (
    END_MESSAGES,
    MOVIE_QUOTES,
    START_MESSAGES,
    WEEK_SUMMARY_MESSAGES,
    format_movie_quote,
)
from utils import (
    format_time,
    get_now,
    get_today_times,
    is_workday,
    seconds_to_hm,
)


_last_start = None
_last_end = None
_last_week_summary = None
_last_event = None
_last_quote = None
_last_quote_event = None
_next_random_at = None
_recent_quotes = deque(maxlen=7)


def random_without_repeat(items, previous):
    choices = list(items)
    if previous in choices and len(choices) > 1:
        choices.remove(previous)
    return random.choice(choices)


def start_message(bot, chat_id):
    bot.send_message(chat_id, start_message_payload())


def start_message_payload():
    global _last_start
    _last_start = random_without_repeat(START_MESSAGES, _last_start)
    return _last_start


def end_message(bot, chat_id):
    bot.send_message(chat_id, end_message_payload())


def end_message_payload():
    global _last_end
    _last_end = random_without_repeat(END_MESSAGES, _last_end)
    return _last_end


def is_last_workday_of_week(current):
    days_until_sunday = 6 - current.weekday()
    return not any(
        is_workday(current + timedelta(days=offset))
        for offset in range(1, days_until_sunday + 1)
    )


def weekly_summary_message(
    bot,
    chat_id,
    current,
    start_hour,
    start_minute,
    end_hour,
    end_minute,
):
    bot.send_message(
        chat_id,
        weekly_summary_payload(
            current, start_hour, start_minute, end_hour, end_minute
        ),
    )


def weekly_summary_payload(
    current, start_hour, start_minute, end_hour, end_minute,
):
    global _last_week_summary
    monday = current - timedelta(days=current.weekday())
    workdays = sum(
        is_workday(monday + timedelta(days=offset))
        for offset in range(current.weekday() + 1)
    )
    daily_minutes = end_hour * 60 + end_minute - start_hour * 60 - start_minute
    total_hours, total_minutes = seconds_to_hm(workdays * daily_minutes * 60)
    _last_week_summary = random_without_repeat(
        WEEK_SUMMARY_MESSAGES,
        _last_week_summary,
    )
    text = (
        f"{_last_week_summary}\n"
        f"Рабочих дней: {workdays}\n"
        f"По графику отсидели: {format_time(total_hours, total_minutes)}\n"
        "Киберстул временно отпускает людей на восстановление."
    )
    return text


def random_media_message(bot, chat_id, media_callback=None):
    if not media_callback:
        return
    media = media_callback(chat_id)
    if not media:
        return
    media_type, file_id = media
    if media_type == "sticker":
        bot.send_sticker(chat_id, file_id)
    else:
        bot.send_animation(chat_id, file_id)


def schedule_next_random(current, first=False):
    delay_minutes = random.randint(10, 30) if first else random.randint(20, 60)
    return current + timedelta(minutes=delay_minutes)


def daily_quote_minutes(current, count=2):
    """Return two stable minutes in the 11:00-01:00 publication cycle."""
    cycle_date = current.date()
    if current.hour < 1:
        cycle_date -= timedelta(days=1)
    daily_random = random.Random(f"movie-quotes:{cycle_date.isoformat()}")
    return sorted(daily_random.sample(range(11 * 60, 25 * 60), count))


def movie_quote_message(bot, chat_id):
    bot.send_message(chat_id, movie_quote_payload(), parse_mode="HTML")


def movie_quote_payload():
    global _last_quote
    candidates = [quote for quote in MOVIE_QUOTES if quote not in _recent_quotes]
    _last_quote = random.choice(candidates or MOVIE_QUOTES)
    _recent_quotes.append(_last_quote)
    return format_movie_quote(_last_quote)


def daily_freekucher_minute(current):
    """Stable pseudo-random publication minute between 11:00 and 21:59."""
    daily_random = random.Random(f"freekucher:{current.date().isoformat()}")
    return daily_random.randrange(11 * 60, 22 * 60)


def _claim_event(callback, chat_id, event_key):
    return callback is None or callback(chat_id, event_key)


def _pause(stop_event, seconds):
    if stop_event is not None:
        return stop_event.wait(seconds)
    time.sleep(seconds)
    return False


def scheduler(
    bot,
    chat_id,
    timezone,
    start_hour,
    start_minute,
    end_hour,
    end_minute,
    autonomous_callback=None,
    media_callback=None,
    activity_callback=None,
    activity_percent_provider=None,
    event_claim_callback=None,
    autonomous_sender=None,
    daily_freekucher_callback=None,
    memory_maintenance_callback=None,
    scheduled_delivery_callback=None,
    scheduled_retry_callback=None,
    stop_event=None,
):
    global _last_event
    global _last_quote_event
    global _next_random_at

    while stop_event is None or not stop_event.is_set():
        try:
            current = get_now(timezone)
            current_minutes = current.hour * 60 + current.minute

            # R5 maintenance is optional/background work.  The callback owns
            # its short claim, R4 admission and retry/backoff semantics; the
            # scheduler starts at most one summary event per loop iteration.
            if memory_maintenance_callback:
                memory_maintenance_callback(chat_id, current)
            if stop_event is not None and stop_event.is_set():
                break

            # P1 owns only utility delivery. Retry/recovery uses its durable
            # state and does not enter the foreground per-chat gate.
            if scheduled_retry_callback:
                scheduled_retry_callback(chat_id, current)
            if stop_event is not None and stop_event.is_set():
                break

            freekucher_event = f"freekucher:{current.date().isoformat()}"
            freekucher_minute = daily_freekucher_minute(current)
            if daily_freekucher_callback and current_minutes >= freekucher_minute:
                if scheduled_delivery_callback:
                    scheduled_delivery_callback(
                        chat_id,
                        freekucher_event,
                        "freekucher",
                        current.replace(
                            hour=freekucher_minute // 60,
                            minute=freekucher_minute % 60,
                            second=0,
                            microsecond=0,
                        ),
                        "#FREEKUCHER",
                        None,
                    )
                elif _claim_event(
                    event_claim_callback, chat_id, freekucher_event
                ):
                    daily_freekucher_callback(chat_id)
            if stop_event is not None and stop_event.is_set():
                break

            quote_minutes = current_minutes
            quote_cycle_date = current.date()
            if current.hour < 1:
                quote_minutes += 24 * 60
                quote_cycle_date -= timedelta(days=1)
            quote_event = f"quote:{quote_cycle_date}:{quote_minutes}"
            if (
                quote_minutes in daily_quote_minutes(current)
                and _last_quote_event != quote_event
            ):
                if scheduled_delivery_callback:
                    scheduled_delivery_callback(
                        chat_id,
                        quote_event,
                        "movie_quote",
                        current.replace(second=0, microsecond=0),
                        movie_quote_payload(),
                        "HTML",
                    )
                elif _claim_event(event_claim_callback, chat_id, quote_event):
                    movie_quote_message(bot, chat_id)
                _last_quote_event = quote_event
            if stop_event is not None and stop_event.is_set():
                break

            if autonomous_callback:
                autonomous_result = autonomous_callback(
                    chat_id,
                    current,
                    is_workday(current),
                )
                if autonomous_result:
                    if autonomous_sender:
                        autonomous_sender(chat_id, autonomous_result)
                    else:
                        # Compatibility for older text-only autonomous callbacks.
                        bot.send_message(chat_id, autonomous_result)
            if stop_event is not None and stop_event.is_set():
                break

            if _next_random_at is None:
                _next_random_at = schedule_next_random(current, first=True)
            if current >= _next_random_at:
                random_media_message(bot, chat_id, media_callback)
                _next_random_at = schedule_next_random(current)
            if stop_event is not None and stop_event.is_set():
                break

            if not is_workday(current):
                if _pause(stop_event, 30):
                    break
                continue

            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute
            event = f"{current.date()}:{current_minutes}"

            def activity_allowed():
                return activity_callback is None or activity_callback(chat_id)

            if current_minutes == start_minutes and _last_event != event:
                if activity_allowed():
                    if scheduled_delivery_callback:
                        scheduled_delivery_callback(
                            chat_id,
                            f"workday_start:{current.date()}:{start_minutes}",
                            "workday_start",
                            current.replace(second=0, microsecond=0),
                            start_message_payload(),
                            None,
                        )
                    else:
                        start_message(bot, chat_id)
                _last_event = event

            elif current_minutes == end_minutes and _last_event != event:
                allowed = activity_allowed()
                if allowed:
                    if scheduled_delivery_callback:
                        scheduled_delivery_callback(
                            chat_id,
                            f"workday_end:{current.date()}:{end_minutes}",
                            "workday_end",
                            current.replace(second=0, microsecond=0),
                            end_message_payload(),
                            None,
                        )
                    else:
                        end_message(bot, chat_id)
                # End-of-day and weekly-summary delivery are two distinct
                # logical jobs.  A drain requested while the first callback
                # was active must not admit the second one in the same tick.
                if stop_event is not None and stop_event.is_set():
                    _last_event = event
                    break
                if allowed and is_last_workday_of_week(current):
                    if scheduled_delivery_callback:
                        scheduled_delivery_callback(
                            chat_id,
                            f"weekly_summary:{current.date()}:{end_minutes}",
                            "weekly_summary",
                            current.replace(second=0, microsecond=0),
                            weekly_summary_payload(
                                current,
                                start_hour,
                                start_minute,
                                end_hour,
                                end_minute,
                            ),
                            None,
                        )
                    else:
                        weekly_summary_message(
                            bot,
                            chat_id,
                            current,
                            start_hour,
                            start_minute,
                            end_hour,
                            end_minute,
                        )
                _last_event = event

        except Exception as error:
            print(f"[Scheduler] {error}")

        if _pause(stop_event, 30):
            break
