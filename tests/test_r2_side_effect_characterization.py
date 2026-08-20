import unittest


# Observed before R2 from live service.py/bot.py. Routing selection telemetry is
# intentionally distinguished from state which claims the payload was sent.
SIDE_EFFECT_MATRIX = (
    ("direct_useful", "text", "route/generated/pending/policy target", "pending bot id attach", "phantom generated/new pending"),
    ("direct_troll_user", "text", "route/generated/policy target", "none", "phantom generated/cooldown history"),
    ("trivial_local", "text", "route/generated/policy target", "pending bot id attach", "phantom generated"),
    ("pending_continuation", "text", "pending delete/route/generated", "none", "pending lost"),
    ("ordinary_ai", "text", "trigger/generated/policy target", "none", "phantom generated/cooldown"),
    ("ordinary_free", "text", "trigger/generated/policy target", "none", "phantom generated/cooldown"),
    ("contextual_gif", "animation", "trigger/media usage/generated", "none", "phantom media usage"),
    ("contextual_sticker", "sticker", "trigger/media usage/generated", "none", "phantom media usage"),
    ("contextual_meme", "photo", "trigger/media usage/generated", "temp cleanup", "usage committed despite failure"),
    ("manual_meme_text", "photo", "transient source only", "usage/generated/source commit", "transient source leak"),
    ("manual_meme_photo", "photo", "transient source/image metadata", "usage/generated/source commit", "transient source leak"),
    ("manual_meme_reply", "photo", "transient source/image metadata", "usage/generated/source commit", "transient source leak"),
    ("autonomous_text", "text", "trigger/generated", "none", "phantom autonomous action"),
    ("autonomous_media", "media", "trigger/media usage/generated", "none", "phantom autonomous media"),
    ("provider_failure_local", "text", "fallback route/generated/pending", "pending bot id attach", "phantom local fallback"),
    ("scheduled_utility", "text", "scheduled claim", "none", "claim remains after failure"),
)


class R2SideEffectCharacterizationTests(unittest.TestCase):
    def test_current_side_effect_matrix_is_complete(self):
        expected = {
            "direct_useful", "direct_troll_user", "trivial_local",
            "pending_continuation", "ordinary_ai", "ordinary_free",
            "contextual_gif", "contextual_sticker", "contextual_meme",
            "manual_meme_text", "manual_meme_photo", "manual_meme_reply",
            "autonomous_text", "autonomous_media", "scheduled_utility",
            "provider_failure_local",
        }
        self.assertEqual({row[0] for row in SIDE_EFFECT_MATRIX}, expected)
        for route, payload, before, after, failure in SIDE_EFFECT_MATRIX:
            with self.subTest(route=route):
                self.assertTrue(payload)
                self.assertTrue(before)
                self.assertTrue(after)
                self.assertTrue(failure)


if __name__ == "__main__":
    unittest.main()
