import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _data_dir():
    project_dir = Path(__file__).resolve().parent.parent
    configured = Path(os.getenv("LEARNING_DATA_DIR", "data"))
    return configured if configured.is_absolute() else project_dir / configured


def _optional_int(name):
    try:
        value = os.getenv(name)
        return int(value) if value else None
    except ValueError:
        return None


def _timezone_name():
    config_path = Path(__file__).resolve().parent.parent / "config.txt"
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 5 and lines[4].strip():
            return lines[4].strip()
    except OSError:
        pass
    configured = os.getenv("TIMEZONE")
    if configured:
        return configured.strip()
    return "Europe/Moscow"


@dataclass(frozen=True)
class LearningSettings:
    enabled: bool = field(default_factory=lambda: _bool("LEARNING_ENABLED", True))
    default_activity_percent: int = field(default_factory=lambda: _int("DEFAULT_ACTIVITY_PERCENT", 100))
    min_training_messages: int = field(default_factory=lambda: _int("MIN_TRAINING_MESSAGES", 20))
    random_reply_chance: float = field(default_factory=lambda: _float("RANDOM_REPLY_CHANCE", 0.24))
    active_chat_reply_chance: float = field(default_factory=lambda: _float("ACTIVE_CHAT_REPLY_CHANCE", 0.16))
    openai_random_reply_chance: float = field(default_factory=lambda: _float("OPENAI_RANDOM_REPLY_CHANCE", 0.12))
    reply_to_stul_chance: float = field(default_factory=lambda: _float("REPLY_TO_STUL_CHANCE", 0.40))
    stul_markov_reply_chance: float = field(default_factory=lambda: _float("STUL_MARKOV_REPLY_CHANCE", 0.50))
    frequent_stul_markov_chance: float = field(default_factory=lambda: _float("FREQUENT_STUL_MARKOV_CHANCE", 0.80))
    # A bare invocation is much weaker than an addressed question. Keep this
    # separate from the global activity and addressed-reply controls.
    bare_stul_reply_factor: float = field(default_factory=lambda: _float("BARE_STUL_REPLY_FACTOR", 0.35))
    creator_username: str = field(default_factory=lambda: os.getenv("CREATOR_USERNAME", "sssssssssssssss28").strip().lstrip("@").casefold())
    sglypa_reply_chance: float = field(default_factory=lambda: _float("SGLYPA_REPLY_CHANCE", 0.375))
    trigger_reaction_chance: float = field(default_factory=lambda: _float("TRIGGER_REACTION_CHANCE", 0.08))
    trigger_reaction_cooldown: int = field(default_factory=lambda: _int("TRIGGER_REACTION_COOLDOWN", 1800))
    max_generated_words: int = field(default_factory=lambda: _int("MAX_GENERATED_WORDS", 35))
    min_generated_words: int = field(default_factory=lambda: _int("MIN_GENERATED_WORDS", 3))
    generated_cooldown: int = field(default_factory=lambda: _int("GENERATED_MESSAGE_COOLDOWN", 300))
    addressed_cooldown: int = field(default_factory=lambda: _int("ADDRESSED_REPLY_COOLDOWN", 60))
    voice_story_cooldown: int = 600
    max_generated_per_hour: int = field(default_factory=lambda: _int("MAX_GENERATED_PER_HOUR", 10))
    # Raw chat text is deliberately short-lived. Durable context is stored as
    # summaries and stable memories, never as an unbounded transcript.
    max_messages_per_chat: int = field(default_factory=lambda: _int("MAX_MESSAGES_PER_CHAT", 50))
    # This is a hard ceiling for the rare context-heavy request.  Normal chat
    # turns use the smaller per-purpose limits below.
    context_message_limit: int = field(default_factory=lambda: _int("CONTEXT_MESSAGE_LIMIT", 20))
    reply_context_message_limit: int = field(default_factory=lambda: _int("REPLY_CONTEXT_MESSAGE_LIMIT", 8))
    targeted_context_message_limit: int = field(default_factory=lambda: _int("TARGETED_CONTEXT_MESSAGE_LIMIT", 10))
    complex_context_message_limit: int = field(default_factory=lambda: _int("COMPLEX_CONTEXT_MESSAGE_LIMIT", 20))
    autonomous_context_message_limit: int = field(default_factory=lambda: _int("AUTONOMOUS_CONTEXT_MESSAGE_LIMIT", 8))
    short_memory_minutes: int = field(default_factory=lambda: _int("SHORT_MEMORY_MINUTES", 30))
    summary_message_interval: int = field(default_factory=lambda: _int("SUMMARY_MESSAGE_INTERVAL", 50))
    summary_time_interval: int = field(default_factory=lambda: _int("SUMMARY_TIME_INTERVAL", 1200))
    max_long_memories: int = field(default_factory=lambda: _int("MAX_LONG_MEMORIES", 40))
    model_cache_size: int = field(default_factory=lambda: _int("MODEL_CACHE_SIZE", 20))
    markov_exclude_recent_messages: int = field(default_factory=lambda: _int("MARKOV_EXCLUDE_RECENT_MESSAGES", 3))
    markov_min_message_age_seconds: int = field(default_factory=lambda: _int("MARKOV_MIN_MESSAGE_AGE_SECONDS", 120))
    markov_recent_history_size: int = field(default_factory=lambda: _int("MARKOV_RECENT_HISTORY_SIZE", 10))
    max_stored_text_length: int = field(default_factory=lambda: _int("MAX_STORED_TEXT_LENGTH", 2000))
    allow_user_mentions: bool = field(default_factory=lambda: _bool("ALLOW_USER_MENTIONS", False))
    quiet_start_hour: int = field(default_factory=lambda: _int("QUIET_START_HOUR", 23))
    quiet_end_hour: int = field(default_factory=lambda: _int("QUIET_END_HOUR", 8))
    autonomous_on_weekends: bool = field(default_factory=lambda: _bool("AUTONOMOUS_ON_WEEKENDS", False))
    autonomous_min_silence: int = field(default_factory=lambda: _int("AUTONOMOUS_MIN_SILENCE", 300))
    autonomous_max_silence: int = field(default_factory=lambda: _int("AUTONOMOUS_MAX_SILENCE", 21600))
    autonomous_cooldown: int = field(default_factory=lambda: _int("AUTONOMOUS_COOLDOWN", 7200))
    autonomous_bot_pause: int = field(default_factory=lambda: _int("AUTONOMOUS_BOT_PAUSE", 2700))
    autonomous_no_response_cooldown: int = field(default_factory=lambda: _int("AUTONOMOUS_NO_RESPONSE_COOLDOWN", 21600))
    autonomous_daily_limit: int = field(default_factory=lambda: _int("AUTONOMOUS_DAILY_LIMIT", 3))
    autonomous_active_message_count: int = field(default_factory=lambda: _int("AUTONOMOUS_ACTIVE_MESSAGE_COUNT", 6))
    autonomous_probability_cap: float = field(default_factory=lambda: _float("AUTONOMOUS_PROBABILITY_CAP", 0.30))
    autonomous_work_hour_factor: float = field(default_factory=lambda: _float("AUTONOMOUS_WORK_HOUR_FACTOR", 0.85))
    autonomous_evening_factor: float = field(default_factory=lambda: _float("AUTONOMOUS_EVENING_FACTOR", 1.0))
    openai_enabled: bool = field(default_factory=lambda: _bool("OPENAI_ENABLED", True))
    openai_chat_id: int | None = field(default_factory=lambda: _optional_int("OPENAI_CHAT_ID") or _optional_int("TELEGRAM_CHAT_ID"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    openai_daily_min: int = field(default_factory=lambda: _int("OPENAI_DAILY_MIN", 5))
    openai_daily_max: int = field(default_factory=lambda: _int("OPENAI_DAILY_MAX", 7))
    openai_timeout: float = field(default_factory=lambda: _float("OPENAI_TIMEOUT", 20.0))
    # XAI_MODEL remains a backwards-compatible alias for the reply model.
    xai_model: str = field(default_factory=lambda: os.getenv("XAI_MODEL", "grok-4.5"))
    xai_reply_model: str = field(default_factory=lambda: os.getenv(
        "XAI_REPLY_MODEL", os.getenv("XAI_MODEL", "grok-4.5")
    ))
    # A summary is deterministic compression, so its default is deliberately
    # cheaper than the conversational model.  It remains configurable because
    # xAI model availability can differ between accounts/regions.
    xai_summary_model: str = field(default_factory=lambda: os.getenv(
        "XAI_SUMMARY_MODEL", "grok-4.3"
    ))
    xai_reply_reasoning_effort: str = field(default_factory=lambda: os.getenv(
        "XAI_REPLY_REASONING_EFFORT", "low"
    ).strip().casefold())
    xai_summary_reasoning_effort: str = field(default_factory=lambda: os.getenv(
        "XAI_SUMMARY_REASONING_EFFORT", "none"
    ).strip().casefold())
    reply_max_output_tokens: int = field(default_factory=lambda: _int("REPLY_MAX_OUTPUT_TOKENS", 100))
    autonomous_max_output_tokens: int = field(default_factory=lambda: _int("AUTONOMOUS_MAX_OUTPUT_TOKENS", 90))
    meme_max_output_tokens: int = field(default_factory=lambda: _int("MEME_MAX_OUTPUT_TOKENS", 50))
    summary_max_output_tokens: int = field(default_factory=lambda: _int("SUMMARY_MAX_OUTPUT_TOKENS", 240))
    xai_base_url: str = field(default_factory=lambda: os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"))
    xai_timeout: float = field(default_factory=lambda: _float("XAI_TIMEOUT", 60.0))
    # Zero disables the soft budget. It is intentionally priority-aware: P1/P2
    # stay local while useful P3 questions may still use the configured model.
    xai_daily_chat_budget_usd: float = field(default_factory=lambda: _float("XAI_DAILY_CHAT_BUDGET_USD", 0.0))
    direct_social_markov_share: float = field(default_factory=lambda: _float("DIRECT_SOCIAL_MARKOV_SHARE", 0.12))
    gif_enabled: bool = field(default_factory=lambda: _bool("GIF_ENABLED", True))
    gif_post_chance: float = field(default_factory=lambda: _float("GIF_POST_CHANCE", 0.65))
    gif_post_cooldown: int = field(default_factory=lambda: _int("GIF_POST_COOLDOWN", 3600))
    max_gifs_per_chat: int = field(default_factory=lambda: _int("MAX_GIFS_PER_CHAT", 1000))
    sticker_enabled: bool = field(default_factory=lambda: _bool("STICKER_ENABLED", True))
    max_stickers_per_chat: int = field(default_factory=lambda: _int("MAX_STICKERS_PER_CHAT", 1000))
    media_reply_chance: float = field(default_factory=lambda: _float("MEDIA_REPLY_CHANCE", 0.18))
    media_humor_bonus: float = field(default_factory=lambda: _float("MEDIA_HUMOR_BONUS", 0.14))
    media_argument_bonus: float = field(default_factory=lambda: _float("MEDIA_ARGUMENT_BONUS", 0.08))
    media_meme_chance: float = field(default_factory=lambda: _float("MEDIA_MEME_CHANCE", 0.55))
    media_gif_share: float = field(default_factory=lambda: _float("MEDIA_GIF_SHARE", 0.72))
    media_cooldown: int = field(default_factory=lambda: _int("MEDIA_COOLDOWN", 600))
    meme_render_cooldown: int = field(default_factory=lambda: _int("MEME_RENDER_COOLDOWN", 1800))
    media_template_cooldown: int = field(default_factory=lambda: _int("MEDIA_TEMPLATE_COOLDOWN", 3600))
    manual_meme_cooldown: int = field(default_factory=lambda: _int("MANUAL_MEME_COOLDOWN", 120))
    chat_image_background_chance: float = field(default_factory=lambda: _float("CHAT_IMAGE_BACKGROUND_CHANCE", 0.35))
    max_chat_images_per_chat: int = field(default_factory=lambda: _int("MAX_CHAT_IMAGES_PER_CHAT", 2000))
    max_chat_image_bytes: int = field(default_factory=lambda: _int("MAX_CHAT_IMAGE_BYTES", 20 * 1024 * 1024))
    max_chat_image_dimension: int = field(default_factory=lambda: _int("MAX_CHAT_IMAGE_DIMENSION", 12000))
    max_chat_image_pixels: int = field(default_factory=lambda: _int("MAX_CHAT_IMAGE_PIXELS", 48_000_000))
    media_recent_limit: int = field(default_factory=lambda: _int("MEDIA_RECENT_LIMIT", 12))
    meme_quote_max_chars: int = field(default_factory=lambda: _int("MEME_QUOTE_MAX_CHARS", 140))
    meme_quote_hard_limit: int = field(default_factory=lambda: _int("MEME_QUOTE_HARD_LIMIT", 600))
    special_phrases: tuple = field(default_factory=lambda: tuple(
        phrase.strip().casefold()
        for phrase in os.getenv("SPECIAL_PHRASES", "сглыпа стул,киберстул").split(",")
        if phrase.strip()
    ))
    mode_weights: dict = field(default_factory=lambda: {
        "markov": _float("MODE_MARKOV_WEIGHT", 0.42),
        "combine": _float("MODE_COMBINE_WEIGHT", 0.18),
        "quote_mutation": _float("MODE_QUOTE_MUTATION_WEIGHT", 0.16),
        "contextual": _float("MODE_CONTEXTUAL_WEIGHT", 0.20),
        "random_old_phrase": _float("MODE_RANDOM_OLD_PHRASE_WEIGHT", 0.04),
    })
    data_dir: Path = field(default_factory=_data_dir)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "grok"))
    timezone_name: str = field(default_factory=_timezone_name)
    state_low_message_count: int = field(default_factory=lambda: _int("STATE_LOW_MESSAGE_COUNT", 2))
    state_low_silence_seconds: int = field(default_factory=lambda: _int("STATE_LOW_SILENCE_SECONDS", 600))
    state_high_messages_5m: int = field(default_factory=lambda: _int("STATE_HIGH_MESSAGES_5M", 8))
    state_high_messages_1m: int = field(default_factory=lambda: _int("STATE_HIGH_MESSAGES_1M", 4))
    state_burst_messages_1m: int = field(default_factory=lambda: _int("STATE_BURST_MESSAGES_1M", 8))
    state_burst_participants: int = field(default_factory=lambda: _int("STATE_BURST_PARTICIPANTS", 3))
    state_topic_min_occurrences: int = field(default_factory=lambda: _int("STATE_TOPIC_MIN_OCCURRENCES", 2))
    state_topic_min_messages: int = field(default_factory=lambda: _int("STATE_TOPIC_MIN_MESSAGES", 2))
    policy_burst_probability_cap: float = field(default_factory=lambda: _float("POLICY_BURST_PROBABILITY_CAP", 0.48))

    def __post_init__(self):
        # Preserve callers that configured the former single field in Python,
        # while allowing XAI_REPLY_MODEL to take precedence in deployments.
        if (
            "XAI_REPLY_MODEL" not in os.environ
            and self.xai_reply_model == "grok-4.5"
            and self.xai_model != "grok-4.5"
        ):
            object.__setattr__(self, "xai_reply_model", self.xai_model)
