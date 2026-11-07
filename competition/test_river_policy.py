"""Independent payoff oracles and scope/deadline tests for the opt-in policy."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from .equity import Estimate
from .model import Action, LegalActions, Observation, Seat
from .policy import Parameters
from .river_policy import RiverPolicy, terminal_bet_value


def fixture():
    return Observation(
        0, 0, 0, 3, (50, 51), (0, 5, 10, 27, 32),
        (Seat(0, 1000, 100, 0), Seat(1, 1000, 100, 0)),
        LegalActions(0, 20, 1000), 20, (100, 0), 200,
    )


class FakeEquity:
    """An injectable ranker whose independent showdown outcome is known."""
    cards = tuple(range(52))
    research_fixed_samples = True

    def __init__(self, hero_rank=100, other_rank=50):
        self.hero_rank, self.other_rank = hero_rank, other_rank
        self.hands = []
        self.estimates = 0

    def evaluate(self, hand):
        self.hands.append(tuple(hand))
        return self.hero_rank if set(hand[:2]) == {50, 51} else self.other_rank

    def estimate(self, *args, **kwargs):
        self.estimates += 1
        return Estimate(0.5, 100, 0.5, 0, 0, 40, 40)


class AlwaysCall(RiverPolicy):
    def _call_probability(self, *args):
        return 1.0


class AlwaysFold(RiverPolicy):
    def _call_probability(self, *args):
        return 0.0


class RiverPayoffTests(unittest.TestCase):
    def test_independent_fold_call_outcome_oracle(self):
        # Enumerate win/tie/loss and response outcomes without using the EV formula.
        pot, bet = 200, 80
        for share, called_chips in ((1, 280), (0.5, 100), (0, -80)):
            for call_probability in (0, 0.25, 1):
                oracle = call_probability * called_chips + (1 - call_probability) * 200
                self.assertAlmostEqual(
                    terminal_bet_value(share, pot, bet, call_probability), oracle,
                )

    def test_reject_invalid_payoff_inputs(self):
        for args in ((-0.1, 10, 2, 1), (1.1, 10, 2, 1),
                     (1, -1, 2, 1), (1, 10, -1, 1), (1, 10, 2, 1.1)):
            with self.assertRaises(ValueError):
                terminal_bet_value(*args)


class RiverPolicyTests(unittest.TestCase):
    def test_always_call_winning_hand_selects_largest_bet(self):
        policy = AlwaysCall(equity=FakeEquity(), seats=2)
        self.assertEqual(policy.act(fixture()), Action("raise", 200))
        self.assertEqual(policy.last_river.check_ev, 200)
        self.assertEqual(policy.last_river.expected_ev, 400)
        self.assertFalse(policy.last_river.raises_modeled)
        self.assertEqual(policy.last_samples, 40)

    def test_always_call_losing_hand_checks(self):
        policy = AlwaysCall(equity=FakeEquity(hero_rank=0), seats=2)
        self.assertEqual(policy.act(fixture()), Action("check"))
        self.assertEqual(policy.last_river.expected_ev, 0)
        self.assertTrue(all(ev < 0 for _, ev in policy.last_river.action_evs))

    def test_always_fold_losing_hand_uses_smallest_profitable_bet(self):
        policy = AlwaysFold(equity=FakeEquity(hero_rank=0), seats=2)
        self.assertEqual(policy.act(fixture()), Action("raise", 67))
        self.assertEqual(policy.last_river.check_ev, 0)
        self.assertEqual(policy.last_river.expected_ev, 200)

    def test_ties_do_not_manufacture_value_from_called_bets(self):
        policy = AlwaysCall(equity=FakeEquity(hero_rank=50), seats=2)
        self.assertEqual(policy.act(fixture()), Action("check"))
        self.assertEqual(policy.last_river.check_ev, 100)
        self.assertTrue(all(ev == 100 for _, ev in policy.last_river.action_evs))

    def test_bets_cap_risk_at_both_effective_stacks(self):
        for hero_stack, villain_stack in ((1000, 45), (45, 1000)):
            obs = replace(fixture(), seats=(Seat(0, hero_stack, 100, 0),
                                           Seat(1, villain_stack, 100, 0)),
                          legal=LegalActions(0, 20, hero_stack))
            policy = AlwaysCall(equity=FakeEquity(), seats=2)
            self.assertEqual(policy.act(obs), Action("raise", 45))
            self.assertEqual(policy.last_river.expected_ev, 245)

    def test_invalid_or_unmatchable_raise_sizes_are_never_emitted(self):
        for legal in (LegalActions(0), LegalActions(0, 100, 20),
                      LegalActions(0, 0, 100), LegalActions(0, 2000, 3000)):
            obs = replace(fixture(), legal=legal)
            policy = AlwaysCall(equity=FakeEquity(), seats=2)
            self.assertEqual(policy.act(obs), Action("check"))
            self.assertEqual(policy.last_river.reason, "no_effective_legal_bet")

    def test_samples_have_no_collisions_and_respect_range(self):
        backend = FakeEquity()
        policy = AlwaysCall(equity=backend, seats=2)
        obs = fixture()
        policy.opponents.begin_hand(obs.hand)
        policy.opponents.ranges[1] = 0.1
        policy.act(obs)
        known = set(obs.hole + obs.board)
        from .cards import RANGES
        allowed = {frozenset((a, b)) for a, b, _ in RANGES[2]}
        for hand in backend.hands[1:]:
            self.assertEqual(len(set(hand)), 7)
            self.assertFalse(set(hand[:2]) & known)
            self.assertIn(frozenset(hand[:2]), allowed)

    def test_insufficient_samples_check_without_a_second_estimate(self):
        backend = FakeEquity()
        policy = RiverPolicy(equity=backend, seats=2, parameters=Parameters(samples=3))
        self.assertEqual(policy.act(fixture()), Action("check"))
        self.assertEqual(policy.last_river.reason, "insufficient_samples")
        self.assertEqual(backend.estimates, 0)

    def test_expired_external_deadline_wins_over_research_marker(self):
        backend = FakeEquity()
        policy = RiverPolicy(equity=backend, seats=2)
        self.assertEqual(policy.act(fixture(), deadline_ns=0), Action("check"))
        self.assertEqual(policy.last_river.reason, "expired_deadline")
        self.assertEqual(backend.hands, [])

    def test_clock_budget_and_external_deadline_are_enforced(self):
        backend = FakeEquity()
        backend.research_fixed_samples = False
        policy = RiverPolicy(equity=backend, seats=2, parameters=Parameters(equity_budget_us=0))
        self.assertEqual(policy.act(fixture()), Action("check"))
        self.assertEqual(policy.last_river.reason, "insufficient_time")
        backend.research_fixed_samples = True
        with patch("competition.river_policy.perf_counter_ns", side_effect=(0, 40_000)):
            self.assertEqual(policy.act(fixture(), deadline_ns=100_000), Action("check"))
        self.assertEqual(policy.last_river.reason, "insufficient_time")

    def test_ineligible_spots_delegate_to_baseline(self):
        base = fixture()
        cases = (
            replace(base, button=1),  # hero check would not end betting
            replace(base, street=2, board=base.board[:4]),
            replace(base, seats=(base.seats[0], Seat(1, 1000, 90, 0)), pot=190),
            replace(base, legal=LegalActions(20, 40, 1000)),
            replace(base, seats=(base.seats[0], Seat(1, 0, 100, 0))),
            replace(base, seats=base.seats + (Seat(2, 1000, 200, 0, True),), pot=400),
        )
        for obs in cases:
            backend = FakeEquity()
            policy = RiverPolicy(equity=backend, seats=len(obs.seats))
            with patch("competition.policy.TournamentPolicy.act", return_value=Action("check")) as baseline:
                policy.act(obs)
            baseline.assert_called_once()
            self.assertFalse(policy.last_river.used)

    def test_incompatible_backend_delegates(self):
        backend = FakeEquity()
        backend.cards = ()
        policy = RiverPolicy(equity=backend, seats=2)
        policy.act(fixture())
        self.assertEqual(backend.estimates, 1)
        self.assertEqual(policy.last_river.reason, "incompatible_evaluator")

    def test_response_model_is_monotone_and_uses_observed_folds_cautiously(self):
        policy = RiverPolicy(equity=FakeEquity(), seats=2)
        self.assertLess(policy._call_probability(0.2, 100, 200, 1),
                        policy._call_probability(0.8, 100, 200, 1))
        self.assertGreater(policy._call_probability(0.4, 67, 200, 1),
                           policy._call_probability(0.4, 200, 200, 1))
        prior = policy._call_probability(0.5, 100, 200, 1)
        tendency = policy.opponents.seats[1]
        tendency.postflop_faced, tendency.postflop_folds = 100, 90
        self.assertLess(policy._call_probability(0.5, 100, 200, 1), prior)

    def test_fixed_sample_runs_are_reproducible(self):
        first = RiverPolicy(seed=7, equity=FakeEquity(), seats=2)
        second = RiverPolicy(seed=7, equity=FakeEquity(), seats=2)
        self.assertEqual(first.act(fixture()), second.act(fixture()))
        self.assertEqual(first.last_river, second.last_river)
        self.assertEqual(first.equity.hands, second.equity.hands)

    def test_sampled_check_value_matches_exact_river_reference(self):
        # An exact, independent enumeration of legal villain combinations.
        try:
            from .holdem import HoldemEquity
        except ImportError:
            self.skipTest("optional eval7 backend is unavailable")
        from .cards import CARD_INDEX
        obs = replace(fixture(), hole=(CARD_INDEX["As"], CARD_INDEX["Qh"]),
                      board=tuple(CARD_INDEX[c] for c in ("Ah", "Kd", "7s", "4c", "2d")))
        equity = HoldemEquity(17)
        equity.research_fixed_samples = True
        known = set(obs.hole + obs.board)
        board = [equity.cards[c] for c in obs.board]
        own_rank = equity.evaluate([equity.cards[c] for c in obs.hole] + board)
        legal = [c for c in range(52) if c not in known]
        total, count = 0.0, 0
        for i, a in enumerate(legal):
            for b in legal[:i]:
                rank = equity.evaluate([equity.cards[a], equity.cards[b]] + board)
                total += float(own_rank > rank) + 0.5 * (own_rank == rank)
                count += 1
        oracle = total / count * obs.pot
        policy = AlwaysCall(seed=174, equity=equity, seats=2,
                            parameters=Parameters(samples=12000, adapt=False))
        policy.act(obs)
        self.assertAlmostEqual(policy.last_river.check_ev, oracle, delta=3.0)
        self.assertEqual(policy.last_samples, 12000)


if __name__ == "__main__":
    unittest.main()
