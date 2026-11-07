import unittest
from copy import deepcopy

from headsup.engine import Hand
from headsup.learning import MIN_RESPONSES, RECENT_HAND_HORIZON, OpponentProfile, PopulationBackedProfile
from headsup.models import Action, GameConfig
from headsup.policies import make_adaptive_policy, make_policy
from tests.support import response_hand


CONTEXT = "preflop:bb:first:medium"


class LearningTests(unittest.TestCase):
    def test_fold_call_raise_are_distinct_observations(self) -> None:
        profile = OpponentProfile()
        for response in ("fold", "call", "raise"):
            profile.observe_hand(response_hand(response).public_history, 0)
        self.assertEqual(profile.estimate(CONTEXT)["counts"], {"fold": 1, "call": 1, "raise": 1})
        self.assertEqual(profile.hands_observed, 3)

    def test_blind_fold_is_not_a_voluntary_bet_response(self) -> None:
        hand = Hand(GameConfig())
        hand.act(Action("fold"))
        profile = OpponentProfile()
        profile.observe_hand(hand.public_history, 1)
        self.assertEqual(profile.summary()["total_responses"], 0)
        self.assertEqual(profile.hands_observed, 1)

    def test_positions_and_individual_readiness_stay_separate(self) -> None:
        profile = OpponentProfile()
        for _ in range(MIN_RESPONSES - 1):
            profile.observe_hand(response_hand().public_history, 0)
        profile.observe_hand(response_hand(opponent=1).public_history, 1)
        self.assertFalse(profile.estimate(CONTEXT)["ready"])
        self.assertEqual(profile.estimate("preflop:button:first:medium")["responses"], 1)
        profile.observe_hand(response_hand().public_history, 0)
        self.assertTrue(profile.estimate(CONTEXT)["ready"])

    def test_recent_memory_limits_evidence_and_expires_old_hands(self) -> None:
        profile = OpponentProfile("recent")
        history = response_hand().public_history
        for _ in range(70):
            profile.observe_hand(history, 0)
        self.assertEqual(profile.estimate(CONTEXT)["responses"], 64)
        for _ in range(RECENT_HAND_HORIZON):
            profile.observe_hand((), 0)
        self.assertEqual(profile.hands_observed, 326)
        self.assertEqual(profile.estimate(CONTEXT)["responses"], 0)
        self.assertFalse(profile.estimate(CONTEXT)["ready"])

    def test_population_support_never_satisfies_individual_gate(self) -> None:
        pool = OpponentProfile()
        history = response_hand().public_history
        for _ in range(100):
            pool.observe_hand(history, 0)
        before = pool.to_dict()
        profile = PopulationBackedProfile(pool, strength=100)
        self.assertFalse(profile.estimate(CONTEXT)["ready"])
        for _ in range(23):
            profile.observe_hand(response_hand("call").public_history, 0)
        estimate = profile.estimate(CONTEXT)
        self.assertEqual(estimate["counts"], {"fold": 0, "call": 23, "raise": 0})
        self.assertEqual(estimate["population_responses"], 100)
        self.assertFalse(estimate["ready"])
        self.assertEqual(pool.to_dict(), before)
        self.assertEqual(profile.to_dict()["population"], before)

    def test_profile_roundtrip_and_export_are_isolated(self) -> None:
        for mode in ("lifetime", "recent"):
            with self.subTest(mode=mode):
                profile = OpponentProfile(mode)
                profile.observe_hand(response_hand().public_history, 0)
                exported = profile.to_dict()
                restored = OpponentProfile.from_dict(exported)
                self.assertEqual(restored.fingerprint, profile.fingerprint)
                restored.observe_hand(response_hand("call").public_history, 0)
                self.assertEqual(profile.hands_observed, 1)
                exported["hands_observed"] = 999
                self.assertEqual(profile.hands_observed, 1)

    def test_malformed_profiles_are_rejected(self) -> None:
        valid = {"schema_version": 1, "hands_observed": 1,
                 "responses": {CONTEXT: {"fold": 1, "call": 0, "raise": 0}}}
        malformed = [dict(valid, hands_observed=True), dict(valid, schema_version=99),
                     dict(valid, responses={"bogus": {"fold": 1, "call": 0, "raise": 0}})]
        negative = deepcopy(valid)
        negative["responses"][CONTEXT]["call"] = -1
        malformed.extend((negative, dict(valid, hands_observed=0)))
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                OpponentProfile.from_dict(value)

    def test_cold_adaptive_policy_preserves_baseline_actions_and_rng(self) -> None:
        for seed in range(8):
            hand = Hand(GameConfig(), seed=seed)
            baseline = make_policy("range", seed, 16)
            adaptive = make_adaptive_policy(seed, 16, OpponentProfile())
            while not hand.finished:
                observation = hand.observation()
                expected = baseline.act(observation)
                self.assertEqual(adaptive.act(observation), expected)
                self.assertEqual(adaptive.rng.getstate(), baseline.rng.getstate())
                hand.act(expected)
