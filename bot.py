import json
import atexit
import logging
import os
import random
import re
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import wraps
from html import escape
from pathlib import Path

from env_loader import load_environment

load_environment(Path(__file__).resolve().parent)

import telebot
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

from utils import (
    format_time,
    get_now,
    get_today_times,
    is_stul_message,
    get_timezone,
    is_workday,
    seconds_to_hm,
)

from scheduler import (
    scheduler,
)

from messages import (
    BOT_VERSION,
    MOVIE_QUOTES,
    REACTIONS,
    format_movie_quote,
)

from learning import (
    ChatActionManager,
    DeliveryReceipt,
    DeliveryType,
    LearningService,
    LearningSettings,
    MEDIA_CHAT_ACTIONS,
    MediaDecision,
    Producer,
    ResponsePlan,
)
from learning.preprocessing import (
    FOREIGN_BOT_COMMAND_RE,
    VOICE_STORY_COMMAND_RE,
    contains_link,
)
from learning.response_plan import GeneratedCommit, SourceUsageCommit
from learning.event_context import current_event_id
from learning.normalized_event import (
    NormalizedEvent,
    normalize_callback_event,
    normalize_telegram_event,
)
from runtime_shutdown import ShutdownCoordinator

# ==========================================
# НАСТРОЙКИ
# ==========================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "0:development"

CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "-1002682965971"))

WORK_START_HOUR = 9
WORK_START_MINUTE = 0

WORK_END_HOUR = 17
WORK_END_MINUTE = 30

TIMEZONE = "Europe/Moscow"

bot = telebot.TeleBot(TOKEN)
learning_settings = LearningSettings()
learning_service = LearningService(learning_settings)
chat_action_manager = ChatActionManager(bot)
learning_service.response_activity = chat_action_manager.activity
_runtime_shutdown_coordinator = None
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@contextmanager
def runtime_work(kind):
    """Track one top-level runtime lifecycle without entering domain code."""
    coordinator = _runtime_shutdown_coordinator
    if coordinator is None:
        yield True
        return
    with coordinator.work(kind) as admitted:
        yield admitted


def _shutdown_diagnostics():
    report = learning_service.concurrency.snapshot()
    report["typing_refreshers"] = chat_action_manager.worker_count()
    return report


def _atexit_shutdown():
    coordinator = _runtime_shutdown_coordinator
    if coordinator is not None:
        coordinator.request_shutdown("atexit")
        coordinator.drain("atexit")


def _stop_telebot_workers(_deadline):
    """Non-blocking final stop for TeleBot's daemon worker pool."""
    pool = getattr(bot, "worker_pool", None)
    for worker in tuple(getattr(pool, "workers", ()) or ()):
        worker.stop()


atexit.register(_atexit_shutdown)


def telegram_user_event_handler(handler):
    """Normalize once, then bind the existing R0 event context and permit."""
    @wraps(handler)
    def correlated(message, *args, **kwargs):
        with runtime_work("foreground") as runtime_admission:
            if not runtime_admission:
                return None
            event = normalize_telegram_event(message)
            with learning_service.chat_event_slot(event) as admission:
                if not admission:
                    return None
                with learning_service.telegram_user_event(event):
                    return handler(message, event, *args, **kwargs)
    return correlated


def telegram_callback_event_handler(handler):
    @wraps(handler)
    def correlated(call, *args, **kwargs):
        with runtime_work("foreground") as runtime_admission:
            if not runtime_admission:
                return None
            event = normalize_callback_event(call)
            with learning_service.chat_event_slot(event) as admission:
                if not admission:
                    return None
                with learning_service.telegram_user_event(event):
                    return handler(call, event, *args, **kwargs)
    return correlated


def log_delivery(chat_id, delivery_type, success, event_id=None):
    logging.getLogger(__name__).info(
        "DELIVERY event_id=%s chat_id=%s delivery_type=%s outcome=%s",
        event_id or current_event_id(), chat_id, delivery_type,
        "success" if success else "failure",
    )


def classify_delivery_error(error):
    """Return a safe transport category without persisting exception text."""
    name = type(error).__name__.casefold()
    status = getattr(error, "error_code", None)
    if "timeout" in name:
        return "telegram_timeout"
    if status == 429 or "too many requests" in name or "ratelimit" in name:
        return "telegram_rate_limit"
    if status == 403 or "forbidden" in name:
        return "telegram_forbidden"
    if status == 400 or "badrequest" in name or "bad_request" in name:
        return "telegram_bad_request"
    if any(part in name for part in ("connection", "network", "request")):
        return "telegram_network"
    return "unknown_transport"


def deliver_response_plan(plan, message=None):
    """The single Telegram transport boundary for one immutable final plan."""
    try:
        learning_service.record_delivery_attempt(plan)
    except Exception:
        logging.getLogger(__name__).exception(
            "DELIVERY_ATTEMPT_TELEMETRY_FAILED event_id=%s", plan.event_id
        )
    rendered = None
    source_path = None
    try:
        reply_to = plan.reply_to_message_id
        if plan.delivery_type == DeliveryType.TEXT:
            sent = (
                bot.reply_to(message, plan.payload.text)
                if message is not None
                else bot.send_message(
                    plan.chat_id,
                    plan.payload.text,
                    reply_to_message_id=reply_to,
                )
            )
        elif plan.delivery_type == DeliveryType.ANIMATION:
            sent = bot.send_animation(
                plan.chat_id,
                plan.payload.decision.asset_id,
                reply_to_message_id=reply_to,
            )
        elif plan.delivery_type == DeliveryType.STICKER:
            sent = bot.send_sticker(
                plan.chat_id,
                plan.payload.decision.asset_id,
                reply_to_message_id=reply_to,
            )
        elif plan.delivery_type == DeliveryType.PHOTO:
            path = plan.payload.prepared_path
            if path is None:
                decision = plan.payload.decision
                if decision.background_file_id:
                    source_path = download_chat_image(decision.background_file_id)
                rendered = learning_service.render_meme(
                    decision, source_path,
                    background=plan.purpose.startswith("autonomous"),
                )
                path = getattr(rendered, "path", None)
            if path is None:
                return DeliveryReceipt(
                    plan.event_id, False, plan.delivery_type,
                    error_category="renderer_failure",
                )
            with Path(path).open("rb") as image:
                sent = bot.send_photo(
                    plan.chat_id, image, reply_to_message_id=reply_to,
                )
        else:
            return DeliveryReceipt(
                plan.event_id, False, plan.delivery_type,
                error_category="unsupported_delivery_type",
            )
        telegram_message_id = getattr(
            sent, "message_id", getattr(sent, "id", None)
        )
        log_delivery(
            plan.chat_id, plan.delivery_type.value, True, plan.event_id
        )
        return DeliveryReceipt(
            plan.event_id, True, plan.delivery_type,
            telegram_message_id=(
                telegram_message_id
                if isinstance(telegram_message_id, int) else None
            ),
        )
    except Exception as error:
        category = classify_delivery_error(error)
        logging.getLogger(__name__).warning(
            "Telegram delivery failed event_id=%s delivery_type=%s category=%s",
            plan.event_id, plan.delivery_type.value, category,
        )
        log_delivery(
            plan.chat_id, plan.delivery_type.value, False, plan.event_id
        )
        return DeliveryReceipt(
            plan.event_id, False, plan.delivery_type,
            error_category=category,
        )
    finally:
        if rendered is not None:
            learning_service.cleanup_rendered_meme(rendered)
        if source_path is not None:
            try:
                Path(source_path).unlink(missing_ok=True)
            except OSError:
                pass


def execute_response_plan(plan, message=None):
    """Deliver once, then commit or abort exactly that transport result."""
    receipt = deliver_response_plan(plan, message)
    learning_service.finalize_response(plan, receipt)
    return receipt

config_path = Path(__file__).with_name("config.txt")
state_path = Path(__file__).with_name("bot_state.json")
restart_gif_path = Path(__file__).with_name("assets") / "cyberstul-restart.gif"
state_lock = threading.RLock()
sglypa_reply_lock = threading.Lock()
trigger_reply_lock = threading.Lock()
freekucher_reply_lock = threading.Lock()
last_sglypa_reply_at = 0
last_trigger_reply_at = {}
last_freekucher_reply_at = {}

SGLYPA_USERNAME = "sglypa_tg_bot"
FREEKUCHER_REPLY_COOLDOWN = 60
CHAIR_REMAINING_COMMAND_RE = re.compile(r"^\s*с\s+стул\s*$", re.IGNORECASE)
CHAIR_MEME_COMMAND_RE = re.compile(
    r"^\s*с\s+м\s+стул(?:\s+(?P<hint>.+?))?\s*$", re.IGNORECASE
)

bot_state = {
    "known_users": {},
}


# ==========================================
# СОСТОЯНИЕ БОТА
# ==========================================

def save_state():

    temporary_path = state_path.with_suffix(".tmp")

    with state_lock:
        serialized_state = {
            "known_users": {
                str(user_id): user
                for user_id, user in bot_state["known_users"].items()
            },
        }

        try:
            temporary_path.write_text(
                json.dumps(
                    serialized_state,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(state_path)
        except OSError as e:
            print(f"[State] Не удалось сохранить состояние: {e}")
            return False

    return True


def load_state():

    if not state_path.exists():
        return

    try:
        saved_state = json.loads(
            state_path.read_text(encoding="utf-8")
        )

        if not isinstance(saved_state, dict):
            raise ValueError("корневой элемент должен быть объектом")

        saved_users = saved_state.get("known_users", {})

        if not isinstance(saved_users, dict):
            raise ValueError("known_users должен быть объектом")

        known_users = {}

        for user_id, user in saved_users.items():
            numeric_user_id = int(user_id)

            if not isinstance(user, dict):
                continue

            known_users[numeric_user_id] = {
                "id": numeric_user_id,
                "username": user.get("username"),
                "name": user.get("name") or "пользователь",
            }

        with state_lock:
            bot_state["known_users"] = known_users

    except (OSError, ValueError, TypeError) as e:
        print(f"[State] Не удалось загрузить состояние: {e}")

# ==========================================
# ЗАГРУЗКА CONFIG
# ==========================================

def load_config():

    global WORK_START_HOUR
    global WORK_START_MINUTE
    global WORK_END_HOUR
    global WORK_END_MINUTE
    global TIMEZONE

    try:
        if not config_path.exists():
            raise FileNotFoundError(
                f"файл {config_path.name} не найден"
            )

        with open(
            config_path,
            encoding="utf-8"
        ) as f:

            lines = f.read().splitlines()

        if len(lines) < 5:
            raise ValueError(
                "ожидалось 5 строк: начало, конец и часовой пояс"
            )

        start_hour = int(lines[0])
        start_minute = int(lines[1])
        end_hour = int(lines[2])
        end_minute = int(lines[3])
        timezone = lines[4].strip()

        if not 0 <= start_hour <= 23:
            raise ValueError("час начала должен быть от 0 до 23")

        if not 0 <= start_minute <= 59:
            raise ValueError("минута начала должна быть от 0 до 59")

        if not 0 <= end_hour <= 23:
            raise ValueError("час окончания должен быть от 0 до 23")

        if not 0 <= end_minute <= 59:
            raise ValueError("минута окончания должна быть от 0 до 59")

        start_minutes = start_hour * 60 + start_minute
        end_minutes = end_hour * 60 + end_minute

        if end_minutes <= start_minutes:
            raise ValueError(
                "время окончания должно быть позже времени начала"
            )

        try:
            get_timezone(timezone)
        except Exception as e:
            raise ValueError(
                f"неизвестный часовой пояс: {timezone}"
            ) from e

        WORK_START_HOUR = start_hour
        WORK_START_MINUTE = start_minute
        WORK_END_HOUR = end_hour
        WORK_END_MINUTE = end_minute
        TIMEZONE = timezone

    except (OSError, ValueError) as e:
        print(
            f"[Config] {e}. Сохраняются текущие значения: "
            f"{WORK_START_HOUR:02d}:{WORK_START_MINUTE:02d}–"
            f"{WORK_END_HOUR:02d}:{WORK_END_MINUTE:02d}, "
            f"{TIMEZONE}."
        )


load_config()
load_state()

# ==========================================
# РЕАКЦИИ
# ==========================================

def reaction_text(message, event=None):

    event = event or normalize_telegram_event(message)
    text = event.effective_text.lower()

    allowed_triggers = {"понедельник", "пятница", "кофе", "домой"}

    for word in allowed_triggers:

        if word not in text:
            continue

        current_time = time.monotonic()

        with trigger_reply_lock:
            last_reply = last_trigger_reply_at.get(event.chat_id)
            if (
                last_reply is not None
                and current_time - last_reply < learning_settings.trigger_reaction_cooldown
            ):
                return False
            if random.random() >= learning_settings.trigger_reaction_chance:
                return False
            # Short-lived collision reservation; a transport failure rolls it back.
            last_trigger_reply_at[event.chat_id] = current_time

        plan = learning_service.prepare_text_response(
            event, random.choice(REACTIONS[word]), "rare_trigger"
        )
        delivered = execute_response_plan(plan, message).success
        if not delivered:
            with trigger_reply_lock:
                if last_trigger_reply_at.get(event.chat_id) == current_time:
                    last_trigger_reply_at.pop(event.chat_id, None)
        return delivered

    return False


def is_freekucher_message(text):

    return bool(
        text
        and re.search(r"(?<!\w)(?:#?freekucher|kucher|кучер|douxross)(?!\w)", text, re.IGNORECASE)
    )


def freekucher_reaction(message, event=None):

    event = event or normalize_telegram_event(message)
    if not is_freekucher_message(event.effective_text):
        return False
    if not learning_service.troll_mode(event.chat_id):
        return False

    current_time = time.monotonic()
    with freekucher_reply_lock:
        last_reply = last_freekucher_reply_at.get(event.chat_id)
        if last_reply is not None and current_time - last_reply < FREEKUCHER_REPLY_COOLDOWN:
            return True
        last_freekucher_reply_at[event.chat_id] = current_time

    plan = learning_service.prepare_text_response(
        event, "#FREEKUCHER", "freekucher", required=True
    )
    delivered = execute_response_plan(plan, message).success
    if not delivered:
        with freekucher_reply_lock:
            if last_freekucher_reply_at.get(event.chat_id) == current_time:
                last_freekucher_reply_at.pop(event.chat_id, None)
    return delivered


def send_daily_freekucher(chat_id):
    """Scheduler-only daily action; it never invokes the LLM."""
    bot.send_message(chat_id, "#FREEKUCHER")


def send_scheduled_text(event_id, chat_id, payload, parse_mode=None):
    """P1 Telegram adapter; no foreground gate or conversation mutation."""
    kwargs = {"parse_mode": parse_mode} if parse_mode else {}
    try:
        result = bot.send_message(chat_id, payload, **kwargs)
    except Exception:
        log_delivery(chat_id, "scheduled_text", False)
        raise
    log_delivery(chat_id, "scheduled_text", True)
    return result


def run_scheduled_notification(
    chat_id, event_key, event_kind, scheduled_at, payload, parse_mode=None,
):
    with runtime_work("scheduled") as admission:
        if not admission:
            return None
        return learning_service.deliver_scheduled_event(
            chat_id,
            event_key,
            event_kind,
            scheduled_at,
            payload,
            send_scheduled_text,
            parse_mode=parse_mode,
            current=get_now(TIMEZONE),
        )


def retry_scheduled_notifications(chat_id, current):
    with runtime_work("scheduled") as admission:
        if not admission:
            return ()
        return learning_service.deliver_pending_scheduled_events(
            chat_id, send_scheduled_text, current=current, limit=10
        )


def chair_remaining_message():
    """Build the workday countdown shown only by the explicit ``с стул`` command."""
    current = get_now(TIMEZONE)
    if not is_workday(current):
        return "🪑 стул сегодня вне смены"

    start, end = get_today_times(
        TIMEZONE,
        WORK_START_HOUR,
        WORK_START_MINUTE,
        WORK_END_HOUR,
        WORK_END_MINUTE,
    )
    if current < start:
        hours, minutes = seconds_to_hm(int((start - current).total_seconds()), round_up=True)
        return f"🪑 до стула: {format_time(hours, minutes)}"
    if current >= end:
        return "🪑 стул на сегодня закончился"

    hours, minutes = seconds_to_hm(int((end - current).total_seconds()), round_up=True)
    return f"🪑 осталось стула: {format_time(hours, minutes)}"


def is_sglypa_message(message_or_event):
    event = (
        message_or_event if isinstance(message_or_event, NormalizedEvent)
        else normalize_telegram_event(message_or_event)
    )
    return bool(
        event.user_is_bot
        and event.username
        and event.username.lower() == SGLYPA_USERNAME
    )


def sglypa_reaction(message, event=None):

    global last_sglypa_reply_at

    current_time = time.monotonic()

    with sglypa_reply_lock:
        if (
            current_time - last_sglypa_reply_at
            < learning_settings.sglypa_reply_cooldown
        ):
            return False

        with learning_service.response_planning():
            learning_service.context_snapshot(
                event or normalize_telegram_event(message)
            )
            reply = learning_service.maybe_sglypa_reply(
                event or normalize_telegram_event(message)
            )
        if not reply:
            return False
        # Temporary collision reservation; rolled back if Telegram rejects it.
        last_sglypa_reply_at = current_time

    delivered = send_contextual_response(
        message, reply, event or normalize_telegram_event(message)
    )
    if not delivered:
        with sglypa_reply_lock:
            if last_sglypa_reply_at == current_time:
                last_sglypa_reply_at = 0
    return delivered


# ==========================================
# "К КТО"
# ==========================================

def remember_user(message_or_event):
    event = (
        message_or_event if isinstance(message_or_event, NormalizedEvent)
        else normalize_telegram_event(message_or_event)
    )
    if event.user_id is None or event.user_is_bot:
        return

    user_data = {
        "id": event.user_id,
        "username": event.username,
        "name": event.first_name or "пользователь",
    }

    with state_lock:
        previous_user_data = bot_state["known_users"].get(event.user_id)

        if previous_user_data == user_data:
            return

        bot_state["known_users"][event.user_id] = user_data
        save_state()


def remove_known_user(user_id):
    with state_lock:
        removed = bot_state["known_users"].pop(user_id, None)
        if removed is not None:
            save_state()
    return removed is not None


def random_known_user(author_id, chat_id):

    with state_lock:
        users = list(bot_state["known_users"].values())

    other_users = [
        user for user in users
        if user["id"] != author_id
    ]

    author_users = [user for user in users if user["id"] == author_id]
    random.shuffle(other_users)
    candidates = other_users + author_users

    for user in candidates:
        try:
            member = bot.get_chat_member(chat_id, user["id"])
        except Exception as error:
            logging.getLogger(__name__).warning(
                "Не удалось проверить участника chat=%s user=%s: %s",
                chat_id,
                user["id"],
                type(error).__name__,
            )
            continue
        if member.status not in {"left", "kicked"}:
            return user
        remove_known_user(user["id"])

    return None


def send_startup_quote():
    terminator_quotes = [
        quote for quote in MOVIE_QUOTES
        if quote[2].startswith("Терминатор")
    ]
    bot.send_message(
        CHAT_ID,
        format_movie_quote(random.choice(terminator_quotes)),
        parse_mode="HTML",
    )


def send_restart_gif():
    if not restart_gif_path.is_file():
        logging.getLogger(__name__).warning(
            "Стартовая GIF не найдена: %s", restart_gif_path
        )
        return False
    try:
        with restart_gif_path.open("rb") as animation:
            bot.send_animation(CHAT_ID, animation)
        return True
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Не удалось отправить стартовую GIF: %s", type(error).__name__
        )
        return False


def send_startup_meme():
    """Send the persistent one-shot meme after the next successful restart."""
    decision = learning_service.startup_meme(CHAT_ID)
    if not decision:
        return False
    rendered = learning_service.render_meme(decision)
    if not rendered:
        return False
    try:
        with rendered.path.open("rb") as image:
            bot.send_photo(CHAT_ID, image)
        learning_service.mark_startup_meme_sent(decision, CHAT_ID)
        return True
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Не удалось отправить стартовый мем: %s", type(error).__name__
        )
        return False
    finally:
        learning_service.cleanup_rendered_meme(rendered)


def download_chat_image(file_id):
    """Download a Telegram image into a bounded, short-lived local file."""
    source_path = None
    try:
        telegram_file = bot.get_file(file_id)
        remote_size = int(getattr(telegram_file, "file_size", 0) or 0)
        if remote_size > learning_settings.max_chat_image_bytes:
            return None
        payload = bot.download_file(telegram_file.file_path)
        if not payload or len(payload) > learning_settings.max_chat_image_bytes:
            return None
        suffix = Path(getattr(telegram_file, "file_path", "image") or "image").suffix[:10]
        fd, raw_path = tempfile.mkstemp(prefix="cyberchair_source_", suffix=suffix)
        os.close(fd)
        source_path = Path(raw_path)
        source_path.write_bytes(payload)
        return source_path
    except Exception as error:
        if source_path is not None:
            source_path.unlink(missing_ok=True)
        logging.getLogger(__name__).warning(
            "Не удалось временно скачать chat image: %s", type(error).__name__
        )
        return None


def send_manual_meme(message, decision=None, hint="", event=None):
    event = event or normalize_telegram_event(message)
    with (
        learning_service.response_planning(),
        chat_action_manager.activity(
            event.chat_id, MEDIA_CHAT_ACTIONS["meme"], "meme"
        ),
    ):
        learning_service.context_snapshot(event)
        if decision is None:
            decision = learning_service.maybe_command_meme(event, hint)
        if not decision:
            return False
        rendered = None
        source_path = None
        used_decision = decision
        try:
            # Admission begins before a potentially large Telegram download and
            # ends after Pillow rendering. Telegram delivery is outside the
            # scarce media slot but remains inside the per-chat lifecycle gate.
            with learning_service.media_work_slot(
                event.chat_id, event.event_id
            ) as admission:
                if not admission:
                    learning_service.discard_command_meme_candidate(decision)
                    return False
                if decision.background_file_id:
                    source_path = download_chat_image(decision.background_file_id)
                    if source_path is not None:
                        rendered = learning_service.render_meme(decision, source_path)
                    if rendered is None:
                        used_decision = learning_service.fallback_command_meme_background(
                            decision, event.chat_id
                        )
                        rendered = (
                            learning_service.render_meme(used_decision)
                            if used_decision else None
                        )
                else:
                    rendered = learning_service.render_meme(decision)
            if not rendered:
                learning_service.discard_command_meme_candidate(used_decision)
                return False
            cleanup_paths = [rendered.path]
            if source_path is not None:
                cleanup_paths.append(source_path)
            plan = learning_service.prepare_manual_meme_response(
                event, used_decision, rendered.path, cleanup_paths
            )
            rendered = None
            source_path = None
            return execute_response_plan(plan).success
        except Exception as error:
            learning_service.discard_command_meme_candidate(used_decision)
            logging.getLogger(__name__).warning(
                "Не удалось отправить мем по команде: %s", type(error).__name__
            )
            log_delivery(event.chat_id, "photo", False)
            return False
        finally:
            try:
                if rendered is not None:
                    learning_service.cleanup_rendered_meme(rendered)
            finally:
                if source_path is not None:
                    try:
                        source_path.unlink(missing_ok=True)
                    except OSError:
                        pass


def user_mention(user):

    if user["username"]:
        return f"@{user['username']}"

    name = escape(user["name"])

    return f'<a href="tg://user?id={user["id"]}">{name}</a>'


WHO_COMMAND_RE = re.compile(
    r"\s*к\s+кто(?:\s+(.+?))?\s*",
    re.IGNORECASE,
)


def handle_who(message, text, event=None):
    event = event or normalize_telegram_event(message)

    match = WHO_COMMAND_RE.fullmatch(text)

    if not match:
        return False

    phrase = match.group(1)

    if not phrase:
        bot.reply_to(message, "А кто что? Киберстулу нужна конкретика.")
        return True

    user = random_known_user(event.user_id, event.chat_id)

    if user is None:
        bot.reply_to(
            message,
            "Киберстул не нашёл актуальных участников этого чата."
        )
        return True

    answer = f"🤖 Киберстул выбрал: {user_mention(user)} {escape(phrase)}"

    bot.reply_to(
        message,
        answer,
        parse_mode="HTML",
    )

    return True

# ==========================================
# ОБРАБОТКА СООБЩЕНИЙ
# ==========================================

_bot_identity = {"id": None, "username": None}


def get_bot_identity():
    if _bot_identity["id"] is None:
        try:
            me = bot.get_me()
            _bot_identity.update(id=me.id, username=me.username)
        except Exception as error:
            logging.getLogger(__name__).warning(
                "Не удалось получить identity бота: %s", type(error).__name__
            )
    return _bot_identity


def is_user_chat_admin(chat_id, user_id):
    if user_id is None:
        return False
    try:
        member = bot.get_chat_member(chat_id, user_id)
        return member.status in {"administrator", "creator"}
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Не удалось проверить права chat=%s: %s",
            chat_id,
            type(error).__name__,
        )
        return False


def is_chat_admin(message):
    user = getattr(message, "from_user", None)
    return bool(user and is_user_chat_admin(message.chat.id, user.id))


def require_admin(message):
    if is_chat_admin(message):
        return True
    bot.reply_to(message, "⛔ Протокол отклонён: нужны права администратора чата.")
    return False


def _on_off(value):
    return "ВКЛ" if value else "ВЫКЛ"


def _provider_title(provider_name, compact=False):
    if provider_name == "grok":
        return "GROK" if compact else learning_settings.xai_model
    return "OPENAI" if compact else learning_settings.openai_model


def chair_main_text(chat_id):
    return (
        "🪑 chairOS // панель управления\n"
        "доступ к внутренностям киберстула\n"
        "если чё-то сломаешь — стул всё запомнит\n\n"
        f"🧠 мозги: {_provider_title(learning_service.llm_provider_name(chat_id))}\n"
        f"😈 тролль: {_on_off(learning_service.troll_mode(chat_id)).casefold()}\n"
        f"🗣 пиздливость: {learning_service.activity_percent(chat_id)}%\n"
        f"🛰 самовольность: {_on_off(learning_service.autonomous_enabled(chat_id)).casefold()}\n"
        f"🖼 мемы: {_on_off(learning_service.media_enabled(chat_id)).casefold()}\n"
        f"🕒 стул: {WORK_START_HOUR:02d}:{WORK_START_MINUTE:02d}–"
        f"{WORK_END_HOUR:02d}:{WORK_END_MINUTE:02d}\n\n"
        "ну чё крутим"
    )


def chair_main_keyboard(chat_id):
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.row(
        InlineKeyboardButton(
            f"😈 тролль: {_on_off(learning_service.troll_mode(chat_id))}",
            callback_data="chair:troll",
        ),
        InlineKeyboardButton(
            f"🧠 мозги: {_provider_title(learning_service.llm_provider_name(chat_id), True)}",
            callback_data="chair:provider",
        ),
    )
    keyboard.row(
        InlineKeyboardButton(
            f"🗣 пиздливость: {learning_service.activity_percent(chat_id)}%",
            callback_data="chair:activity",
        ),
        InlineKeyboardButton(
            f"🛰 самовольность: {_on_off(learning_service.autonomous_enabled(chat_id))}",
            callback_data="chair:auto",
        ),
    )
    keyboard.row(
        InlineKeyboardButton(
            f"🖼 мемы: {_on_off(learning_service.media_enabled(chat_id))}",
            callback_data="chair:media",
        ),
        InlineKeyboardButton("🕒 рабочий стул", callback_data="chair:work"),
    )
    keyboard.row(
        InlineKeyboardButton("📊 статус", callback_data="chair:status"),
        InlineKeyboardButton("❌ закрыть", callback_data="chair:close"),
    )
    return keyboard


def chair_provider_screen(chat_id):
    active = learning_service.llm_provider_name(chat_id)
    keyboard = InlineKeyboardMarkup(row_width=1)
    keyboard.add(
        InlineKeyboardButton(
            f"{'✅ ' if active == 'grok' else ''}GROK 4.5",
            callback_data="chair:provider:grok",
        ),
        InlineKeyboardButton(
            f"{'✅ ' if active == 'openai' else ''}OpenAI",
            callback_data="chair:provider:openai",
        ),
        InlineKeyboardButton("← назад", callback_data="chair:main"),
    )
    return (
        "🧠 chairOS // выбор мозга\n"
        "какой кремний сегодня изображает интеллект",
        keyboard,
    )


def chair_activity_screen(chat_id):
    current = learning_service.activity_percent(chat_id)
    keyboard = InlineKeyboardMarkup(row_width=2)
    buttons = [
        InlineKeyboardButton(
            f"{'✅ ' if current == value else ''}{value}%",
            callback_data=f"chair:activity:{value}",
        )
        for value in (25, 50, 75, 100)
    ]
    keyboard.row(*buttons[:2])
    keyboard.row(*buttons[2:])
    keyboard.row(InlineKeyboardButton("← назад", callback_data="chair:main"))
    return (
        "🗣 chairOS // пиздливость\n"
        "насколько часто этот кусок металла открывает ебало",
        keyboard,
    )


def chair_work_screen():
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("← назад", callback_data="chair:main"))
    return (
        "🕒 chairOS // режим стула\n"
        f"начало: {WORK_START_HOUR:02d}:{WORK_START_MINUTE:02d}\n"
        f"конец: {WORK_END_HOUR:02d}:{WORK_END_MINUTE:02d}\n"
        "дни: пн–пт\n"
        f"timezone: {TIMEZONE}",
        keyboard,
    )


def chair_status_screen(chat_id):
    status = learning_service.status(chat_id)
    repository = learning_service.repository(chat_id)
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("← назад", callback_data="chair:main"))
    provider_state = "работает" if status["provider_available"] else "не завёлся"
    return (
        "📊 chairOS // статус\n"
        f"🧠 {_provider_title(status['provider'])} ({provider_state})\n"
        f"😈 troll mode: {_on_off(status['troll_mode']).casefold()}\n"
        f"🗣 пиздливость: {status['activity_percent']}%\n"
        f"🛰 автономка: {_on_off(status['autonomous_enabled']).casefold()}\n"
        f"🖼 мемы: {_on_off(status['media_enabled']).casefold()}\n"
        f"память: {'работает' if status['learning'] else 'выкл'}\n"
        "summary: работает\n"
        f"media: {'работает' if status['media_enabled'] and status['troll_mode'] else 'выкл'}\n"
        f"сообщений в short-term: {repository.count()}\n"
        f"stable memories: {len(repository.stable_memories())}",
        keyboard,
    )


def edit_chair_screen(call, text, keyboard):
    bot.edit_message_text(
        text,
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        reply_markup=keyboard,
    )


@bot.message_handler(commands=["chair_settings"])
@telegram_user_event_handler
def chair_settings_command(message, event):
    if getattr(message.chat, "type", None) not in {"group", "supergroup"}:
        bot.reply_to(message, "панель работает только внутри конференции")
        return
    if not require_admin(message):
        return
    bot.send_message(
        event.chat_id,
        chair_main_text(event.chat_id),
        reply_markup=chair_main_keyboard(event.chat_id),
    )


@bot.callback_query_handler(func=lambda call: str(getattr(call, "data", "")).startswith("chair:"))
@telegram_callback_event_handler
def chair_settings_callback(call, event):
    chat_id = event.chat_id
    user_id = event.user_id
    if (
        getattr(call.message.chat, "type", None) not in {"group", "supergroup"}
        or not is_user_chat_admin(chat_id, user_id)
    ):
        bot.answer_callback_query(
            call.id, "админка не твоя, руки убрал", show_alert=True
        )
        return

    action = event.data
    if action == "chair:close":
        bot.edit_message_text(
            "панель закрыта\nстул снова делает вид что работает",
            chat_id=chat_id,
            message_id=call.message.message_id,
        )
        bot.answer_callback_query(call.id)
        return
    if action == "chair:troll":
        learning_service.set_troll_mode(chat_id, not learning_service.troll_mode(chat_id))
    elif action == "chair:auto":
        learning_service.set_autonomous_enabled(
            chat_id, not learning_service.autonomous_enabled(chat_id)
        )
    elif action == "chair:media":
        learning_service.set_media_enabled(chat_id, not learning_service.media_enabled(chat_id))
    elif action.startswith("chair:provider:"):
        selected = action.rsplit(":", 1)[-1]
        if not learning_service.set_llm_provider(chat_id, selected):
            bot.answer_callback_query(
                call.id,
                "grok не завёлся. проверь XAI_API_KEY"
                if selected == "grok" else "openai не завёлся. проверь OPENAI_API_KEY",
                show_alert=True,
            )
            return
        text, keyboard = chair_provider_screen(chat_id)
        edit_chair_screen(call, text, keyboard)
        bot.answer_callback_query(call.id)
        return
    elif action.startswith("chair:activity:"):
        learning_service.set_activity_percent(chat_id, int(action.rsplit(":", 1)[-1]))
        text, keyboard = chair_activity_screen(chat_id)
        edit_chair_screen(call, text, keyboard)
        bot.answer_callback_query(call.id)
        return
    elif action == "chair:provider":
        text, keyboard = chair_provider_screen(chat_id)
        edit_chair_screen(call, text, keyboard)
        bot.answer_callback_query(call.id)
        return
    elif action == "chair:activity":
        text, keyboard = chair_activity_screen(chat_id)
        edit_chair_screen(call, text, keyboard)
        bot.answer_callback_query(call.id)
        return
    elif action == "chair:work":
        text, keyboard = chair_work_screen()
        edit_chair_screen(call, text, keyboard)
        bot.answer_callback_query(call.id)
        return
    elif action == "chair:status":
        text, keyboard = chair_status_screen(chat_id)
        edit_chair_screen(call, text, keyboard)
        bot.answer_callback_query(call.id)
        return

    edit_chair_screen(call, chair_main_text(chat_id), chair_main_keyboard(chat_id))
    bot.answer_callback_query(call.id)


def is_creator_message(message_or_event):
    """Whether this message was written by CyberChair's creator.

    A mention of the creator is ordinary chat context, not an invitation for a
    privileged reply.  This keeps mentions from making the bot noticeably more
    talkative than it is with other participants.
    """
    event = (
        message_or_event if isinstance(message_or_event, NormalizedEvent)
        else normalize_telegram_event(message_or_event)
    )
    username = (event.username or "").casefold()
    return username == learning_settings.creator_username


@bot.message_handler(commands=["learn_status"])
@telegram_user_event_handler
def learn_status_command(message, event):
    if not require_admin(message):
        return
    status = learning_service.status(event.chat_id)
    readiness = "готова" if status["ready"] else "ещё обучается"
    bot.reply_to(
        message,
        "🤖 Статус Киберстула\n"
        f"Сообщений: {status['count']} / {learning_settings.min_training_messages}\n"
        f"Локальная модель: {readiness}\n"
        f"Обучение: {'вкл' if status['learning'] else 'выкл'}\n"
        f"Случайные ответы: {'вкл' if status['talk'] else 'выкл'}\n"
        f"TrollMode: {'вкл' if status['troll_mode'] else 'выкл'}\n"
        f"LLM ({status['provider']}): "
        f"{'готов' if status['llm'] else 'нет ключа/выключен'}\n"
        f"Активность: {status['activity_percent']}%\n"
        f"Сохранено GIF: {learning_service.repository(event.chat_id).gif_count()}\n"
        f"Сохранено стикеров: {learning_service.repository(event.chat_id).sticker_count()}",
    )


@bot.message_handler(commands=["activity"])
@telegram_user_event_handler
def activity_command(message, event):
    if not require_admin(message):
        return
    arguments = event.effective_text.split(maxsplit=1)
    if len(arguments) == 1:
        current = learning_service.activity_percent(event.chat_id)
        bot.reply_to(
            message,
            f"🤖 Активность Киберстула: {current}%\n"
            "Доступно: /activity 0, 25, 50, 75 или 100.",
        )
        return
    try:
        percent = int(arguments[1].strip().rstrip("%"))
    except ValueError:
        percent = -1
    if percent not in {0, 25, 50, 75, 100}:
        bot.reply_to(message, "⚠ Укажите один уровень: 0, 25, 50, 75 или 100.")
        return
    learning_service.set_activity_percent(event.chat_id, percent)
    bot.reply_to(message, f"🤖 Активность этого чата установлена на {percent}%.")


@bot.message_handler(commands=["learn_on", "learn_off", "talk_on", "talk_off"])
@telegram_user_event_handler
def toggle_learning_command(message, event):
    if not require_admin(message):
        return
    command = event.effective_text.split()[0].split("@")[0].lower()
    kind = "learning" if command.startswith("/learn_") else "talk"
    enabled = command.endswith("_on")
    learning_service.set_enabled(event.chat_id, kind, enabled)
    bot.reply_to(message, f"Протокол {kind} {'активирован' if enabled else 'остановлен'}.")


@bot.message_handler(commands=["troll_on", "troll_off"])
@telegram_user_event_handler
def toggle_troll_mode_command(message, event):
    if not require_admin(message):
        return
    command = event.effective_text.split()[0].split("@")[0].lower()
    enabled = command.endswith("_on")
    learning_service.set_troll_mode(event.chat_id, enabled)
    bot.reply_to(
        message,
        f"TrollMode {'включён' if enabled else 'выключен'} для этого чата.",
    )


@bot.message_handler(commands=["forget_chat"])
@telegram_user_event_handler
def forget_chat_command(message, event):
    if not require_admin(message):
        return
    arguments = event.effective_text.split(maxsplit=1)
    if len(arguments) != 2 or arguments[1].strip().lower() != "confirm":
        bot.reply_to(message, "⚠ Для удаления базы этого чата: /forget_chat confirm")
        return
    learning_service.forget_chat(event.chat_id)
    bot.reply_to(message, "🤖 Память этого чата уничтожена. Данные других чатов не затронуты.")


@bot.message_handler(commands=["generate"])
@telegram_user_event_handler
def generate_command(message, event):
    if not require_admin(message):
        return
    arguments = event.effective_text.partition(" ")[2].strip()
    provider, separator, remaining = arguments.partition(" ")
    provider = provider.casefold()
    with learning_service.response_planning():
        if provider in {"local", "локально"}:
            generated, source_usage = learning_service.generate_free_response(
                event.chat_id, remaining, return_sources=True
            )
            producer = Producer.LOCAL
        elif provider in {"ai", "openai", "ии"}:
            generated = learning_service.generate_llm(
                event.chat_id, remaining, "reply"
            )
            producer = Producer.LLM
            source_usage = ()
        else:
            generated = learning_service.generate_llm(
                event.chat_id, arguments, "reply"
            )
            producer = Producer.LLM
            source_usage = ()
        actions = [GeneratedCommit(generated, "manual")] if generated else []
        if source_usage:
            actions.append(SourceUsageCommit(tuple(source_usage)))
        plan = (
            learning_service.prepare_text_response(
                event, generated, "manual", producer=producer, required=True,
                actions=actions,
                provider_key=(
                    str(getattr(
                        learning_service.provider_for_chat(event.chat_id),
                        "provider_key", "llm",
                    ))
                    if producer == Producer.LLM else None
                ),
            )
            if generated else None
        )
    if plan:
        execute_response_plan(plan, message)
    else:
        bot.reply_to(message, "🤖 Генерация не удалась: проверьте ключ LLM provider или накопите больше сообщений.")


def send_contextual_response(message, response, event=None):
    """Telegram-only sender for a provider-neutral text/media decision."""
    if isinstance(response, ResponsePlan):
        return execute_response_plan(response, message).success
    event = event or normalize_telegram_event(message)
    if not isinstance(response, MediaDecision):
        try:
            sent = bot.reply_to(message, response)
        except Exception:
            log_delivery(message.chat.id, "text", False)
            raise
        log_delivery(message.chat.id, "text", True)
        learning_service.attach_pending_bot_message(event, sent)
        return True
    if response.action == "gif":
        with chat_action_manager.activity(
            message.chat.id, MEDIA_CHAT_ACTIONS["gif"], "gif"
        ):
            try:
                bot.send_animation(
                    message.chat.id, response.asset_id,
                    reply_to_message_id=message.message_id,
                )
            except Exception:
                log_delivery(message.chat.id, "gif", False)
                raise
            log_delivery(message.chat.id, "gif", True)
        return True
    if response.action == "sticker":
        with chat_action_manager.activity(
            message.chat.id, MEDIA_CHAT_ACTIONS["sticker"], "sticker"
        ):
            try:
                bot.send_sticker(
                    message.chat.id, response.asset_id,
                    reply_to_message_id=message.message_id,
                )
            except Exception:
                log_delivery(message.chat.id, "sticker", False)
                raise
            log_delivery(message.chat.id, "sticker", True)
        return True
    if response.action != "meme":
        return False
    with chat_action_manager.activity(
        message.chat.id, MEDIA_CHAT_ACTIONS["meme"], "meme"
    ):
        rendered = learning_service.render_meme(response)
        if not rendered:
            return False
        try:
            with rendered.path.open("rb") as image:
                bot.send_photo(
                    message.chat.id, image,
                    reply_to_message_id=message.message_id,
                )
            log_delivery(message.chat.id, "photo", True)
            return True
        except OSError as error:
            logging.getLogger(__name__).warning(
                "Не удалось отправить contextual meme chat=%s: %s",
                message.chat.id, type(error).__name__,
            )
            log_delivery(message.chat.id, "photo", False)
            return False
        finally:
            learning_service.cleanup_rendered_meme(rendered)


def send_autonomous_response(chat_id, response):
    """Telegram-only sender for scheduler-originated text or MediaDecision."""
    if isinstance(response, ResponsePlan):
        return execute_response_plan(response).success
    if not isinstance(response, MediaDecision):
        try:
            bot.send_message(chat_id, response)
        except Exception:
            log_delivery(chat_id, "text", False)
            raise
        log_delivery(chat_id, "text", True)
        return True
    if response.action == "gif":
        with chat_action_manager.activity(
            chat_id, MEDIA_CHAT_ACTIONS["gif"], "gif"
        ):
            try:
                bot.send_animation(chat_id, response.asset_id)
            except Exception:
                log_delivery(chat_id, "gif", False)
                raise
            log_delivery(chat_id, "gif", True)
        return True
    if response.action == "sticker":
        with chat_action_manager.activity(
            chat_id, MEDIA_CHAT_ACTIONS["sticker"], "sticker"
        ):
            try:
                bot.send_sticker(chat_id, response.asset_id)
            except Exception:
                log_delivery(chat_id, "sticker", False)
                raise
            log_delivery(chat_id, "sticker", True)
        return True
    if response.action != "meme":
        return False
    with chat_action_manager.activity(
        chat_id, MEDIA_CHAT_ACTIONS["meme"], "meme"
    ):
        rendered = learning_service.render_meme(response)
        if not rendered:
            return False
        try:
            with rendered.path.open("rb") as image:
                bot.send_photo(chat_id, image)
            log_delivery(chat_id, "photo", True)
            return True
        finally:
            learning_service.cleanup_rendered_meme(rendered)


def run_autonomous_response(chat_id, current, workday=True):
    """Optional autonomous lifecycle: skip instead of queueing behind a user."""
    with runtime_work("autonomous") as runtime_admission:
        if not runtime_admission:
            return None
        with learning_service.autonomous_chat_event_slot(
            chat_id, current
        ) as admission:
            if not admission:
                return None
            response = learning_service.prepare_autonomous(chat_id, current, workday)
            if response:
                send_autonomous_response(chat_id, response)
    # The scheduler's legacy sender must not run after this combined lifecycle.
    return None


def run_memory_maintenance(chat_id, current):
    with runtime_work("memory") as admission:
        if not admission:
            return None
        return learning_service.run_memory_maintenance(chat_id, current)


@bot.message_handler(content_types=["animation"])
@telegram_user_event_handler
def remember_animation(message, event):
    learning_service.ingest_gif(event)


@bot.message_handler(content_types=["photo"])
@telegram_user_event_handler
def remember_photo(message, event):
    # Telegram stores photo text in caption. This special command owns the
    # event before collection, direct-address routing, policy or AI text flow.
    meme_match = CHAIR_MEME_COMMAND_RE.fullmatch(event.effective_text)
    if meme_match:
        if not send_manual_meme(
            message, hint=meme_match.group("hint") or "", event=event
        ):
            logging.getLogger(__name__).warning(
                "Не удалось собрать мем по photo caption chat=%s", message.chat.id
            )
        return
    learning_service.ingest_chat_image(event)


@bot.message_handler(
    content_types=["document"],
    func=lambda m: bool(
        getattr(m, "document", None)
        and str(getattr(m.document, "mime_type", "") or "").casefold().startswith("image/")
    ),
)
@telegram_user_event_handler
def remember_image_document(message, event):
    learning_service.ingest_chat_image(event)
    if (event.mime_type or "").casefold() == "image/gif":
        learning_service.ingest_gif(event)


@bot.message_handler(content_types=["sticker"])
@telegram_user_event_handler
def remember_sticker(message, event):
    learning_service.ingest_sticker(event)

@bot.message_handler(func=lambda m: m.text is not None)
@telegram_user_event_handler
def handle_message(message, event):

    text = event.effective_text

    # Kucher mentions have the highest routing priority and are independent of
    # learning, activity level and every other command module.
    if freekucher_reaction(message, event):
        return

    # These are control phrases, not dialogue: never learn them, answer them or
    # pass them to an external model.
    if FOREIGN_BOT_COMMAND_RE.search(text):
        return

    # Do not learn from or react to URLs, including deliberately space-separated
    # ones such as "https www instagram com reel ...".
    if contains_link(text):
        return

    meme_match = CHAIR_MEME_COMMAND_RE.fullmatch(text)
    if meme_match:
        if send_manual_meme(
            message, hint=meme_match.group("hint") or "", event=event
        ):
            return
        logging.getLogger(__name__).warning("Не удалось собрать мем по команде chat=%s", message.chat.id)
        return

    if CHAIR_REMAINING_COMMAND_RE.fullmatch(text):
        plan = learning_service.prepare_text_response(
            event, chair_remaining_message(), "chair_remaining", required=True
        )
        execute_response_plan(plan, message)
        return

    if is_sglypa_message(event):
        if learning_service.activity_allows(event.chat_id):
            sglypa_reaction(message, event)
        return

    remember_user(event)

    if VOICE_STORY_COMMAND_RE.search(text):
        if not learning_service.troll_mode(event.chat_id):
            return
        remaining = learning_service.take_voice_story_cooldown_notice(event.chat_id)
        if remaining > 0:
            minutes, seconds = divmod(remaining, 60)
            plan = learning_service.prepare_text_response(
                event,
                "🎙 Киберстул голос на кулдауне "
                f"осталось {minutes} мин {seconds} сек",
                "voice_cooldown", required=True,
            )
            receipt = execute_response_plan(plan, message)
            if not receipt.success:
                learning_service.release_voice_story_cooldown_notice(event.chat_id)
            return
        with learning_service.response_planning():
            learning_service.context_snapshot(event)
            story = learning_service.maybe_voice_story(event)
        if story:
            send_contextual_response(message, story, event)
        return

    is_who_command = bool(WHO_COMMAND_RE.fullmatch(text))
    identity = get_bot_identity()
    replies_to_chair = event.replies_to_user(identity["id"])
    mentions_chair = bool(
        identity["username"]
        and f"@{identity['username']}".casefold() in text.casefold()
    )
    configured_address = any(
        phrase in text.casefold() for phrase in learning_settings.special_phrases
    )
    explicit_chair = is_stul_message(text)
    direct_candidate = not is_who_command and (
        explicit_chair or replies_to_chair or mentions_chair or configured_address
    )
    pending_candidate = not is_who_command and not direct_candidate and learning_service.is_pending_continuation(
        event, bot_id=identity["id"]
    )
    # A direct turn must not accidentally trigger a summary request and then a
    # second conversational request on the same incoming event.
    if direct_candidate or pending_candidate:
        learning_service.ingest_event(event, refresh_memory=False)
    else:
        learning_service.ingest_event(event)

    # The complete "к кто ..." command has priority over words inside its
    # argument, including "стул" and "стульчик".
    if is_who_command:
        if (
            learning_service.troll_mode(event.chat_id)
            and learning_service.activity_allows(event.chat_id)
        ):
            handle_who(message, text, event)
        return

    # R3 builds one immutable read view after ingest and before response routing.
    learning_service.context_snapshot(event)

    # A pending continuation belongs to the same mandatory-response lane as a
    # direct address, but has higher dialogue priority and one final producer.
    if pending_candidate:
        with learning_service.response_planning():
            generated = learning_service.maybe_pending_continuation(
                event, bot_id=identity["id"]
            )
        if generated:
            send_contextual_response(message, generated, event)
        return

    # Special commands above own their event. All remaining explicit chair
    # addresses and replies enter one mandatory-response arbitration path
    # before activity sampling/cooldowns can discard them.
    if direct_candidate:
        with learning_service.response_planning():
            generated = learning_service.maybe_direct_reply(
                event,
                bot_id=identity["id"],
                bot_username=identity["username"],
                explicit_address=explicit_chair,
            )
        if generated:
            send_contextual_response(message, generated, event)
        return

    if not learning_service.activity_allows(event.chat_id):
        return

    if is_creator_message(event):
        with learning_service.response_planning():
            generated = learning_service.maybe_special_ai(
                event,
                # Share the ordinary random-reply bucket and probability, so
                # the creator never receives more unsolicited reactions than
                # other chat members.
                "random",
                learning_settings.random_reply_chance,
                "creator",
                addressed=False,
            )
        if generated:
            send_contextual_response(message, generated, event)
        return

    if learning_service.troll_mode(event.chat_id) and reaction_text(message, event):
        return

    with learning_service.response_planning():
        generated = learning_service.maybe_reply(
            event,
            bot_id=identity["id"],
            bot_username=identity["username"],
        )
    if generated:
        send_contextual_response(message, generated, event)

# ==========================================
# ЗАПУСК БОТА
# ==========================================

def main():
    global _runtime_shutdown_coordinator

    previous_runtime = _runtime_shutdown_coordinator
    polling_active = threading.Event()
    scheduler_stop = threading.Event()
    coordinator = ShutdownCoordinator(
        learning_settings.shutdown_grace_seconds,
        diagnostics=_shutdown_diagnostics,
    )
    _runtime_shutdown_coordinator = coordinator
    scheduler_thread = threading.Thread(
        target=scheduler,
        args=(
            bot,
            CHAT_ID,
            TIMEZONE,
            WORK_START_HOUR,
            WORK_START_MINUTE,
            WORK_END_HOUR,
            WORK_END_MINUTE,
            run_autonomous_response,
            # Autonomous media is selected only after AutonomousPolicy has
            # accepted a contextual intervention.  Keep the legacy callback
            # available on LearningService, but do not run a second random
            # scheduler stream alongside the policy.
            None,
            learning_service.activity_allows,
            learning_service.activity_percent,
            None,
            None,
            send_daily_freekucher,
            run_memory_maintenance,
            run_scheduled_notification,
            retry_scheduled_notifications,
            scheduler_stop,
        ),
        name="cyberchair-scheduler",
        daemon=True,
    )
    coordinator.register_component(
        "telegram_polling",
        stop=bot.stop_polling,
        stopped=lambda: not polling_active.is_set(),
    )
    coordinator.register_component(
        "scheduler",
        stop=scheduler_stop.set,
        stopped=lambda: not scheduler_thread.is_alive(),
    )
    coordinator.register_component(
        "concurrency_admission",
        stop=learning_service.concurrency.shutdown,
    )
    coordinator.register_component(
        "chat_actions",
        stop=chat_action_manager.shutdown,
        stopped=lambda: chat_action_manager.worker_count() == 0,
    )
    coordinator.register_cleanup("telebot_workers", _stop_telebot_workers)

    previous_signals = {}

    def request_from_signal(signum, _frame):
        coordinator.request_shutdown(signal.Signals(signum).name)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_signals[signum] = signal.getsignal(signum)
            signal.signal(signum, request_from_signal)

        missing = []
        if TOKEN == "0:development":
            missing.append("TELEGRAM_BOT_TOKEN")
        active_provider = learning_service.llm_provider_name(CHAT_ID)
        if not learning_service.provider_available(CHAT_ID, active_provider):
            missing.append(
                "XAI_API_KEY (LLM_PROVIDER=grok)"
                if active_provider == "grok"
                else "OPENAI_API_KEY (LLM_PROVIDER=openai)"
            )
        if missing:
            raise RuntimeError(
                "Не заполнены обязательные настройки: " + ", ".join(missing)
                + ". Укажите их в .env (предпочтительно) или .env.example."
            )

        scheduler_thread.start()

        print("=" * 50)
        print(
            f"🤖 Бот 'Киберстул' "
            f"версии {BOT_VERSION} запущен"
        )
        print(
            f"🕒 Время: "
            f"{WORK_START_HOUR:02d}:{WORK_START_MINUTE:02d}"
            f" - "
            f"{WORK_END_HOUR:02d}:{WORK_END_MINUTE:02d}"
        )
        print(f"🌍 Часовой пояс: {TIMEZONE}")
        print("📅 Режим: Понедельник–Пятница")
        print("=" * 50)
        print("Запускаем бесконечный поллинг...")

        with coordinator.work("startup") as admitted:
            if admitted:
                send_startup_meme()

        while coordinator.is_running:
            try:
                polling_active.set()
                bot.infinity_polling(
                    skip_pending=True,
                    timeout=30,
                    long_polling_timeout=30,
                )
            except Exception as error:
                if coordinator.is_draining:
                    break
                print(f"Ошибка: {error}")
                print("Переподключение через 10 секунд...")
                coordinator.shutdown_event.wait(10)
            finally:
                polling_active.clear()
    finally:
        coordinator.request_shutdown("runtime_exit")
        coordinator.drain("runtime_exit")
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
        # Imported runtimes may be invoked repeatedly by tests/embedders. The
        # real ``python bot.py`` process keeps the stopped guard until exit.
        if __name__ != "__main__":
            _runtime_shutdown_coordinator = previous_runtime


if __name__ == "__main__":
    main()
