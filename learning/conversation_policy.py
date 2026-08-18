from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConversationDecision:
    action: str
    reply_probability: float
    troll_intensity: float
    max_reply_length: int
    preferred_style: str
    target_message_id: int | None
    target_user_id: int | None
    reason: str
    local_probability: float = 0.0
    llm_probability: float = 0.0

    def debug(self):
        return asdict(self)


class ConversationPolicy:
    def __init__(self, settings):
        self.settings = settings

    def _profile(self, state, addressed):
        activity_factor = {
            "low": 0.65,
            "normal": 1.0,
            "high": 1.15,
            "burst": 1.25,
        }[state.activity_level]
        profiles = {
            "casual": (1.0, 0.6, 35, "chatty"),
            "humor": (1.25, 0.8, 22, "absurd_short"),
            "argument": (1.2, 0.85, 26, "direct_mocking"),
            "serious": (0.9, 0.35, 45, "dry_sarcastic"),
            "work": (1.0 if addressed else 0.75, 0.5, 36, "work_sarcastic"),
        }
        if state.conversation_type == "mixed":
            type_factor = max(
                0.75,
                min(
                    1.2,
                    1.0
                    + state.humor_score * 0.15
                    + state.argument_score * 0.1
                    - state.serious_score * 0.1
                    - state.work_score * 0.15,
                ),
            )
            intensity = max(
                0.2,
                min(
                    0.9,
                    0.6
                    + state.humor_score * 0.2
                    + state.argument_score * 0.2
                    - state.serious_score * 0.2
                    - state.work_score * 0.1,
                ),
            )
            return activity_factor, type_factor, intensity, 32, "adaptive_mixed"
        type_factor, intensity, length, style = profiles[state.conversation_type]
        return activity_factor, type_factor, intensity, length, style

    def decide(
        self,
        state,
        addressed=False,
        local_allowed=True,
        llm_allowed=True,
        quiet_hours=False,
    ):
        activity_factor, type_factor, intensity, length, style = self._profile(
            state, addressed
        )
        reason_parts = [state.activity_level, state.conversation_type]
        if quiet_hours and not addressed:
            return ConversationDecision(
                "none", 0.0, intensity, length, style, None, None,
                "quiet_hours", 0.0, 0.0,
            )
        if addressed:
            # Direct-address routing chooses the producer separately. Policy is
            # no longer allowed to turn an explicit address into IGNORE.
            probability = 1.0
            action = "reply"
            return ConversationDecision(
                action=action,
                reply_probability=round(probability, 4),
                troll_intensity=round(intensity, 3),
                max_reply_length=length,
                preferred_style=style,
                target_message_id=state.target_message_id,
                target_user_id=state.target_user_id,
                reason="+".join(reason_parts + (["addressed"] if addressed else [])),
                local_probability=1.0 if local_allowed or not llm_allowed else 0.0,
                llm_probability=1.0 if llm_allowed else 0.0,
            )
        local_chance = self.settings.random_reply_chance * activity_factor * type_factor
        conditional_llm_chance = (
            self.settings.llm_random_reply_chance * activity_factor * type_factor
        )
        local_probability = min(1.0, local_chance) if local_allowed else 0.0
        llm_probability = (
            (1.0 - local_probability) * min(1.0, conditional_llm_chance)
            if llm_allowed
            else 0.0
        )
        total = local_probability + llm_probability
        if state.activity_level == "burst" and total > self.settings.policy_burst_probability_cap:
            scale = self.settings.policy_burst_probability_cap / total
            local_probability *= scale
            llm_probability *= scale
            total = self.settings.policy_burst_probability_cap
            reason_parts.append("burst_cap")
        action = (
            "reply"
            if total > 0 and state.target_message_id is not None
            else "spontaneous" if total > 0 else "none"
        )
        if action == "none":
            reason_parts.append("hard_limit")
        return ConversationDecision(
            action=action,
            reply_probability=round(total, 4),
            troll_intensity=round(intensity, 3),
            max_reply_length=length,
            preferred_style=style,
            target_message_id=state.target_message_id,
            target_user_id=state.target_user_id,
            reason="+".join(reason_parts),
            local_probability=round(local_probability, 4),
            llm_probability=round(llm_probability, 4),
        )
