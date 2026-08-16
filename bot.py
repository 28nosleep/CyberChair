import json
import logging
import os
import random
import re
import threading
import time
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

from learning import LearningService, LearningSettings, MediaDecision
from learning.preprocessing import FOREIGN_BOT_COMMAND_RE, VOICE_STORY_COMMAND_RE

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
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

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

SGLYPA_REPLY_COOLDOWN = 15
SGLYPA_USERNAME = "sglypa_tg_bot"
FREEKUCHER_REPLY_COOLDOWN = 60
CHAIR_REMAINING_COMMAND_RE = re.compile(r"^\s*с\s+стул\s*$", re.IGNORECASE)
CHAIR_MEME_COMMAND_RE = re.compile(r"^\s*с\s+м\s+стул\s*$", re.IGNORECASE)

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

def reaction_text(message):

    text = message.text.lower()

    allowed_triggers = {"понедельник", "пятница", "кофе", "домой"}

    for word in allowed_triggers:

        if word not in text:
            continue

        current_time = time.monotonic()

        with trigger_reply_lock:
            last_reply = last_trigger_reply_at.get(message.chat.id)
            if (
                last_reply is not None
                and current_time - last_reply < learning_settings.trigger_reaction_cooldown
            ):
                return False
            if random.random() >= learning_settings.trigger_reaction_chance:
                return False
            last_trigger_reply_at[message.chat.id] = current_time

        bot.reply_to(message, random.choice(REACTIONS[word]))

        return True

    return False


def is_freekucher_message(text):

    return bool(
        text
        and re.search(r"(?<!\w)(?:#?freekucher|kucher|кучер|douxross)(?!\w)", text, re.IGNORECASE)
    )


def freekucher_reaction(message):

    if not is_freekucher_message(message.text):
        return False
    if not learning_service.troll_mode(message.chat.id):
        return False

    current_time = time.monotonic()
    with freekucher_reply_lock:
        last_reply = last_freekucher_reply_at.get(message.chat.id)
        if last_reply is not None and current_time - last_reply < FREEKUCHER_REPLY_COOLDOWN:
            return True
        last_freekucher_reply_at[message.chat.id] = current_time

    bot.reply_to(message, "#FREEKUCHER")

    return True


def send_daily_freekucher(chat_id):
    """Scheduler-only daily action; it never invokes the LLM."""
    bot.send_message(chat_id, "#FREEKUCHER")


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


def is_sglypa_message(message):

    user = message.from_user

    return bool(
        user
        and user.is_bot
        and user.username
        and user.username.lower() == SGLYPA_USERNAME
    )


def sglypa_reaction(message):

    global last_sglypa_reply_at

    current_time = time.monotonic()

    with sglypa_reply_lock:
        if (
            current_time - last_sglypa_reply_at
            < SGLYPA_REPLY_COOLDOWN
        ):
            return False

        reply = learning_service.maybe_sglypa_reply(message)
        if not reply:
            return False
        last_sglypa_reply_at = current_time

    bot.reply_to(message, reply)

    return True


# ==========================================
# "К КТО"
# ==========================================

def remember_user(message):

    user = message.from_user

    if not user or user.is_bot:
        return

    user_data = {
        "id": user.id,
        "username": user.username,
        "name": user.first_name or "пользователь",
    }

    with state_lock:
        previous_user_data = bot_state["known_users"].get(user.id)

        if previous_user_data == user_data:
            return

        bot_state["known_users"][user.id] = user_data
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


def send_manual_meme(message, decision):
    rendered = learning_service.render_meme(decision)
    if not rendered:
        return False
    try:
        with rendered.path.open("rb") as image:
            bot.send_photo(
                message.chat.id, image, reply_to_message_id=message.message_id,
            )
        learning_service.mark_command_meme_sent(message.chat.id, decision)
        return True
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Не удалось отправить мем по команде: %s", type(error).__name__
        )
        return False
    finally:
        learning_service.cleanup_rendered_meme(rendered)


def user_mention(user):

    if user["username"]:
        return f"@{user['username']}"

    name = escape(user["name"])

    return f'<a href="tg://user?id={user["id"]}">{name}</a>'


WHO_COMMAND_RE = re.compile(
    r"\s*к\s+кто(?:\s+(.+?))?\s*",
    re.IGNORECASE,
)


def handle_who(message, text):

    match = WHO_COMMAND_RE.fullmatch(text)

    if not match:
        return False

    phrase = match.group(1)

    if not phrase:
        bot.reply_to(message, "А кто что? Киберстулу нужна конкретика.")
        return True

    author_id = message.from_user.id if message.from_user else None
    user = random_known_user(author_id, message.chat.id)

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
def chair_settings_command(message):
    if getattr(message.chat, "type", None) not in {"group", "supergroup"}:
        bot.reply_to(message, "панель работает только внутри конференции")
        return
    if not require_admin(message):
        return
    bot.send_message(
        message.chat.id,
        chair_main_text(message.chat.id),
        reply_markup=chair_main_keyboard(message.chat.id),
    )


@bot.callback_query_handler(func=lambda call: str(getattr(call, "data", "")).startswith("chair:"))
def chair_settings_callback(call):
    chat_id = call.message.chat.id
    user_id = getattr(getattr(call, "from_user", None), "id", None)
    if (
        getattr(call.message.chat, "type", None) not in {"group", "supergroup"}
        or not is_user_chat_admin(chat_id, user_id)
    ):
        bot.answer_callback_query(
            call.id, "админка не твоя, руки убрал", show_alert=True
        )
        return

    action = call.data
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


def is_creator_message(message):
    """Whether this message was written by CyberChair's creator.

    A mention of the creator is ordinary chat context, not an invitation for a
    privileged reply.  This keeps mentions from making the bot noticeably more
    talkative than it is with other participants.
    """
    user = getattr(message, "from_user", None)
    username = (getattr(user, "username", None) or "").casefold()
    return username == learning_settings.creator_username


@bot.message_handler(commands=["learn_status"])
def learn_status_command(message):
    if not require_admin(message):
        return
    status = learning_service.status(message.chat.id)
    readiness = "готова" if status["ready"] else "ещё обучается"
    bot.reply_to(
        message,
        "🤖 Статус Киберстула\n"
        f"Сообщений: {status['count']} / {learning_settings.min_training_messages}\n"
        f"Локальная модель: {readiness}\n"
        f"Обучение: {'вкл' if status['learning'] else 'выкл'}\n"
        f"Случайные ответы: {'вкл' if status['talk'] else 'выкл'}\n"
        f"TrollMode: {'вкл' if status['troll_mode'] else 'выкл'}\n"
        f"OpenAI: {'готов' if status['openai'] else 'нет ключа/выключен'}\n"
        f"Активность: {status['activity_percent']}%\n"
        f"Сохранено GIF: {learning_service.repository(message.chat.id).gif_count()}\n"
        f"Сохранено стикеров: {learning_service.repository(message.chat.id).sticker_count()}",
    )


@bot.message_handler(commands=["activity"])
def activity_command(message):
    if not require_admin(message):
        return
    arguments = message.text.split(maxsplit=1)
    if len(arguments) == 1:
        current = learning_service.activity_percent(message.chat.id)
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
    learning_service.set_activity_percent(message.chat.id, percent)
    bot.reply_to(message, f"🤖 Активность этого чата установлена на {percent}%.")


@bot.message_handler(commands=["learn_on", "learn_off", "talk_on", "talk_off"])
def toggle_learning_command(message):
    if not require_admin(message):
        return
    command = message.text.split()[0].split("@")[0].lower()
    kind = "learning" if command.startswith("/learn_") else "talk"
    enabled = command.endswith("_on")
    learning_service.set_enabled(message.chat.id, kind, enabled)
    bot.reply_to(message, f"Протокол {kind} {'активирован' if enabled else 'остановлен'}.")


@bot.message_handler(commands=["troll_on", "troll_off"])
def toggle_troll_mode_command(message):
    if not require_admin(message):
        return
    command = message.text.split()[0].split("@")[0].lower()
    enabled = command.endswith("_on")
    learning_service.set_troll_mode(message.chat.id, enabled)
    bot.reply_to(
        message,
        f"TrollMode {'включён' if enabled else 'выключен'} для этого чата.",
    )


@bot.message_handler(commands=["forget_chat"])
def forget_chat_command(message):
    if not require_admin(message):
        return
    arguments = message.text.split(maxsplit=1)
    if len(arguments) != 2 or arguments[1].strip().lower() != "confirm":
        bot.reply_to(message, "⚠ Для удаления базы этого чата: /forget_chat confirm")
        return
    learning_service.forget_chat(message.chat.id)
    bot.reply_to(message, "🤖 Память этого чата уничтожена. Данные других чатов не затронуты.")


@bot.message_handler(commands=["generate"])
def generate_command(message):
    if not require_admin(message):
        return
    arguments = message.text.partition(" ")[2].strip()
    provider, separator, remaining = arguments.partition(" ")
    provider = provider.casefold()
    if provider in {"markov", "марков"}:
        generated = learning_service.generate_local(message.chat.id, remaining)
    elif provider in {"ai", "openai", "ии"}:
        generated = learning_service.generate_openai(message.chat.id, remaining, "reply")
    else:
        generated = learning_service.generate_openai(message.chat.id, arguments, "reply")
    if generated:
        learning_service.repository(message.chat.id).record_generated(generated, "manual")
        bot.reply_to(message, generated)
    else:
        bot.reply_to(message, "🤖 Генерация не удалась: проверьте ключ OpenAI или накопите больше сообщений.")


def send_contextual_response(message, response):
    """Telegram-only sender for a provider-neutral text/media decision."""
    if not isinstance(response, MediaDecision):
        bot.reply_to(message, response)
        return True
    if response.action == "gif":
        bot.send_animation(
            message.chat.id, response.asset_id,
            reply_to_message_id=message.message_id,
        )
        return True
    if response.action == "sticker":
        bot.send_sticker(
            message.chat.id, response.asset_id,
            reply_to_message_id=message.message_id,
        )
        return True
    if response.action != "meme":
        return False
    rendered = learning_service.render_meme(response)
    if not rendered:
        return False
    try:
        with rendered.path.open("rb") as image:
            bot.send_photo(
                message.chat.id, image,
                reply_to_message_id=message.message_id,
            )
        return True
    except OSError as error:
        logging.getLogger(__name__).warning(
            "Не удалось отправить contextual meme chat=%s: %s",
            message.chat.id, type(error).__name__,
        )
        return False
    finally:
        learning_service.cleanup_rendered_meme(rendered)


def send_autonomous_response(chat_id, response):
    """Telegram-only sender for scheduler-originated text or MediaDecision."""
    if not isinstance(response, MediaDecision):
        bot.send_message(chat_id, response)
        return True
    if response.action == "gif":
        bot.send_animation(chat_id, response.asset_id)
        return True
    if response.action == "sticker":
        bot.send_sticker(chat_id, response.asset_id)
        return True
    if response.action != "meme":
        return False
    rendered = learning_service.render_meme(response)
    if not rendered:
        return False
    try:
        with rendered.path.open("rb") as image:
            bot.send_photo(chat_id, image)
        return True
    finally:
        learning_service.cleanup_rendered_meme(rendered)


@bot.message_handler(content_types=["animation"])
def remember_animation(message):
    learning_service.ingest_gif(message)


@bot.message_handler(
    content_types=["document"],
    func=lambda m: bool(
        getattr(m, "document", None)
        and getattr(m.document, "mime_type", "") == "image/gif"
    ),
)
def remember_gif_document(message):
    learning_service.ingest_gif(message)


@bot.message_handler(content_types=["sticker"])
def remember_sticker(message):
    learning_service.ingest_sticker(message)

@bot.message_handler(func=lambda m: m.text is not None)
def handle_message(message):

    text = message.text

    # Kucher mentions have the highest routing priority and are independent of
    # learning, activity level and every other command module.
    if freekucher_reaction(message):
        return

    # These are control phrases, not dialogue: never learn them, answer them or
    # pass them to an external model.
    if FOREIGN_BOT_COMMAND_RE.search(text):
        return

    if CHAIR_MEME_COMMAND_RE.fullmatch(text):
        decision = learning_service.maybe_command_meme(message.chat.id)
        if decision and send_manual_meme(message, decision):
            return
        logging.getLogger(__name__).warning("Не удалось собрать мем по команде chat=%s", message.chat.id)
        return

    if CHAIR_REMAINING_COMMAND_RE.fullmatch(text):
        bot.reply_to(message, chair_remaining_message())
        return

    if is_sglypa_message(message):
        if learning_service.activity_allows(message.chat.id):
            sglypa_reaction(message)
        return

    remember_user(message)

    if VOICE_STORY_COMMAND_RE.search(text):
        if not learning_service.troll_mode(message.chat.id):
            return
        remaining = learning_service.take_voice_story_cooldown_notice(message.chat.id)
        if remaining > 0:
            minutes, seconds = divmod(remaining, 60)
            bot.reply_to(
                message,
                "🎙 Киберстул голос на кулдауне "
                f"осталось {minutes} мин {seconds} сек",
            )
            return
        story = learning_service.maybe_voice_story(message)
        if story:
            bot.reply_to(message, story)
        return

    is_who_command = bool(WHO_COMMAND_RE.fullmatch(text))
    identity = get_bot_identity()
    reply_user = getattr(getattr(message, "reply_to_message", None), "from_user", None)
    replies_to_chair = bool(
        reply_user and identity["id"] and reply_user.id == identity["id"]
    )
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
    # A direct turn must not accidentally trigger a summary request and then a
    # second conversational request on the same incoming event.
    if direct_candidate:
        learning_service.ingest(message, refresh_memory=False)
    else:
        learning_service.ingest(message)

    # The complete "к кто ..." command has priority over words inside its
    # argument, including "стул" and "стульчик".
    if is_who_command:
        if (
            learning_service.troll_mode(message.chat.id)
            and learning_service.activity_allows(message.chat.id)
        ):
            handle_who(message, text)
        return

    # Special commands above own their event. All remaining explicit chair
    # addresses and replies enter one mandatory-response arbitration path
    # before activity sampling/cooldowns can discard them.
    if direct_candidate:
        generated = learning_service.maybe_direct_reply(
            message,
            bot_id=identity["id"],
            bot_username=identity["username"],
            explicit_address=explicit_chair,
        )
        if generated:
            send_contextual_response(message, generated)
        return

    if not learning_service.activity_allows(message.chat.id):
        return

    if is_creator_message(message):
        generated = learning_service.maybe_special_ai(
            message,
            # Share the ordinary random-reply bucket and probability, so
            # the creator never receives more unsolicited reactions than
            # other chat members.
            "random",
            learning_settings.random_reply_chance,
            "creator",
            addressed=False,
        )
        if generated:
            bot.reply_to(message, generated)
        return

    if learning_service.troll_mode(message.chat.id) and reaction_text(message):
        return

    generated = learning_service.maybe_reply(
        message,
        bot_id=identity["id"],
        bot_username=identity["username"],
    )
    if generated:
        send_contextual_response(message, generated)

# ==========================================
# ЗАПУСК БОТА
# ==========================================

def main():

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

    threading.Thread(
        target=scheduler,
        args=(
            bot,
            CHAT_ID,
            TIMEZONE,
            WORK_START_HOUR,
            WORK_START_MINUTE,
            WORK_END_HOUR,
            WORK_END_MINUTE,
            learning_service.maybe_autonomous,
            # Autonomous media is selected only after AutonomousPolicy has
            # accepted a contextual intervention.  Keep the legacy callback
            # available on LearningService, but do not run a second random
            # scheduler stream alongside the policy.
            None,
            learning_service.activity_allows,
            learning_service.activity_percent,
            learning_service.claim_scheduled_event,
            send_autonomous_response,
            send_daily_freekucher,
        ),
        daemon=True,
    ).start()

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

    send_startup_meme()

    while True:

        try:

            bot.infinity_polling(
                skip_pending=True,
                timeout=30,
                long_polling_timeout=30,
            )

        except Exception as e:

            print(f"Ошибка: {e}")
            print("Переподключение через 10 секунд...")

            time.sleep(10)


if __name__ == "__main__":
    main()
