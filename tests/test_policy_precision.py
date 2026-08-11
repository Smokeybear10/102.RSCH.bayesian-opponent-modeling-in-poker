import unittest

from headsup.engine import Hand
from headsup.models import MAX_SAMPLES, Action, GameConfig
from headsup.policies import make_policy
from tests.support import stacked_deck


def facing_flop_bet() -> Hand:
    deck = stacked_deck(("9c", "9d"), ("Jd", "7c"), ("Kd", "Kc", "2s", "3h", "4h"))
    hand = Hand(GameConfig(), deck=deck)
    for action in (Action("raise_to", 5), Action("call"), Action("raise_to", 6)):
        hand.act(action)
    return hand


class PrecisionPolicyTests(unittest.TestCase):
    def test_refinement_can_change_a_borderline_fold_to_a_call(self) -> None:
        observation = facing_flop_bet().observation()
        baseline = make_policy("range", 0, 128)
        candidate = make_policy("precision", 0, 128)
        self.assertEqual(baseline.act(observation), Action("fold"))
        self.assertEqual(candidate.act(observation), Action("call"))
        detail = candidate.last_info["precision"]
        self.assertTrue(detail["refined"])
        self.assertEqual(detail["pilot_equity"], baseline.last_info["equity"])
        self.assertLess(detail["pilot_equity"], candidate.last_info["pot_odds"])
        self.assertGreater(candidate.last_info["equity"], candidate.last_info["pot_odds"])
        self.assertEqual((candidate.last_info["samples"], detail["total_samples"]), (512, 640))
        self.assertEqual(candidate.rng.getstate(), baseline.rng.getstate())

    def test_decisive_equity_keeps_the_pilot_and_action_randomness(self) -> None:
        hand = Hand(GameConfig(), deck=stacked_deck(
            ("Ac", "Ad"), ("Qc", "Qd"), ("Ah", "As", "Ks", "3h", "4h")))
        hand.act(Action("call"))
        hand.act(Action("check"))
        observation = hand.observation()
        baseline = make_policy("range", 4, 128)
        candidate = make_policy("precision", 4, 128)
        self.assertEqual(candidate.act(observation), baseline.act(observation))
        self.assertFalse(candidate.last_info["precision"]["refined"])
        self.assertEqual(candidate.last_info["precision"]["total_samples"], 128)
        self.assertEqual(candidate.last_info["equity"], baseline.last_info["equity"])
        self.assertEqual(candidate.rng.getstate(), baseline.rng.getstate())

    def test_river_retains_exact_enumeration_and_baseline_action(self) -> None:
        hand = facing_flop_bet()
        while len(hand.observation().board) < 5:
            hand.act(Action("call" if hand.observation().to_call else "check"))
        observation = hand.observation()
        baseline = make_policy("range", 3, 128)
        candidate = make_policy("precision", 3, 128)
        self.assertEqual(candidate.act(observation), baseline.act(observation))
        self.assertEqual(candidate.last_info["samples"], 0)
        self.assertEqual(candidate.last_info["equity_standard_error"], 0)
        self.assertEqual(candidate.last_info["equity"], baseline.last_info["equity"])
        self.assertNotIn("precision", candidate.last_info)
        self.assertEqual(candidate.rng.getstate(), baseline.rng.getstate())

    def test_restoring_saved_rng_reproduces_refinement_and_decision(self) -> None:
        observation = facing_flop_bet().observation()
        original = make_policy("precision", 0, 128)
        saved_rng = original.rng.getstate()
        expected = original.act(observation)
        self.assertTrue(original.last_info["precision"]["refined"])
        restored = make_policy("precision", 999, 128)
        restored.rng.setstate(saved_rng)
        self.assertEqual(restored.act(observation), expected)
        self.assertEqual(restored.last_info, original.last_info)
        self.assertEqual(restored.rng.getstate(), original.rng.getstate())

    def test_existing_maximum_budget_is_not_exceeded_by_refinement(self) -> None:
        for samples in (6000, MAX_SAMPLES):
            with self.subTest(samples=samples):
                candidate = make_policy("precision", 0, samples)
                candidate.act(facing_flop_bet().observation())
                detail = candidate.last_info["precision"]
                self.assertFalse(detail["refined"])
                self.assertEqual(detail["total_samples"], samples)

    def test_zero_variance_in_a_tiny_pilot_does_not_skip_refinement(self) -> None:
        candidate = make_policy("precision", 0, 2)
        candidate.act(facing_flop_bet().observation())
        detail = candidate.last_info["precision"]
        self.assertEqual(detail["pilot_standard_error"], 0)
        self.assertTrue(detail["refined"])
        self.assertEqual(detail["total_samples"], 514)

    def test_preflop_all_in_uses_equity_and_rake_instead_of_hand_score(self) -> None:
        hand = Hand(GameConfig(stack_bb=10, rake_bps=500, rake_cap=3), seed=17)
        hand.act(Action("raise_to", 20))
        candidate = make_policy("precision", 9, 64)
        hand.act(candidate.act(hand.observation()))
        self.assertTrue(hand.finished)
        self.assertNotIn("preflop_hand_score", candidate.last_info)
        self.assertTrue(candidate.last_info["rake_in_decision_model"])
        self.assertIn("call_value_chips", candidate.last_info)
