"""Local policy for rare, contextual messages after a conversation goes quiet."""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class AutonomousDecision:
    action: str = "none"  # none | text | media
    probability: float = 0.0
    conversation_decision: object | None = None
    reason: str = ""

    def debug(self):
        data = asdict(self)
        if self.conversation_decision is not None:
            data["conversation_decision"] = self.conversation_decision.debug()
        return data


class AutonomousPolicy:
    """Makes no network calls and never sends a Telegram message."""

    def __init__(self, settings, conversation_policy):
        self.settings = settings
        self.conversation_policy = conversation_policy

    def decide(
        self,
        state,
        *,
        current,
        summary=None,
        prior_activity=0,
        last_bot_at=None,
        last_autonomous_at=None,
        last_human_at=None,
        daily_count=0,
        quiet_hours=False,
        troll_mode=True,
    ):
        if not troll_mode:
            return AutonomousDecision(reason="troll_mode_off")
        if quiet_hours:
            return AutonomousDecision(reason="quiet_hours")
        if daily_count >= self.settings.autonomous_daily_limit:
            return AutonomousDecision(reason="daily_limit")
        silence = state.silence_seconds
        if silence is None or silence < self.settings.autonomous_min_silence:
            return AutonomousDecision(reason="silence_too_short")
        if silence > self.settings.autonomous_max_silence:
            return AutonomousDecision(reason="chat_too_quiet")
        if last_bot_at and (current - last_bot_at).total_seconds() < self.settings.autonomous_bot_pause:
            return AutonomousDecision(reason="bot_pause")
        if last_autonomous_at:
            elapsed = (current - last_autonomous_at).total_seconds()
            if elapsed < self.settings.autonomous_cooldown:
                return AutonomousDecision(reason="autonomous_cooldown")
            # A second unsolicited message without a human reply is especially unwelcome.
            if last_human_at is None or last_human_at <= last_autonomous_at:
                if elapsed < self.settings.autonomous_no_response_cooldown:
                    return AutonomousDecision(reason="awaiting_human_reaction")

        # Reuse ConversationPolicy for the style/intensity profile instead of
        # maintaining a second matrix of conversation-type rules here.
        base = self.conversation_policy.decide(
            state, local_allowed=False, llm_allowed=True, quiet_hours=False
        )
        if base.action == "none":
            return AutonomousDecision(reason="conversation_policy_none")

        if silence <= 25 * 60:
            chance = .20
            window = "recent_pause"
        elif silence <= 60 * 60:
            chance = .12
            window = "cooling_down"
        elif silence <= 3 * 60 * 60:
            chance = .045
            window = "long_pause"
        else:
            chance = .015
            window = "very_long_pause"
        # A workday pause is not an invitation to constantly interrupt: keep
        # office-time nudges rarer than the permitted evening window.
        if 9 <= current.hour < 18:
            chance *= self.settings.autonomous_work_hour_factor
            time_window = "work_hours"
        else:
            chance *= self.settings.autonomous_evening_factor
            time_window = "evening"
        # A lively discussion immediately before the pause provides actual
        # context; an empty, dormant chat must not get revived aggressively.
        if prior_activity >= self.settings.autonomous_active_message_count:
            chance *= 1.35
        elif prior_activity <= 1:
            chance *= .55
        if summary and (summary.get("callback_jokes") or summary.get("inside_jokes")):
            chance *= 1.15
        if state.conversation_type in {"humor", "argument"}:
            chance *= 1.2
        elif state.conversation_type == "serious":
            chance *= .75
        chance = round(min(self.settings.autonomous_probability_cap, chance), 4)
        decision = replace(
            base,
            action="spontaneous",
            reply_probability=chance,
            local_probability=0.0,
            llm_probability=chance,
            reason=f"autonomous+{window}+{time_window}+{base.reason}",
        )
        return AutonomousDecision("text", chance, decision, decision.reason)
