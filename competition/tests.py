import unittest
from collections import deque
from dataclasses import replace
from itertools import permutations
from random import Random
from unittest.mock import patch

import eval7
from pokerkit import NoLimitTexasHoldem

from .arena import AUTOMATIONS, CARD_IDS, DECK, Configuration, play
from .cards import CARD_INDEX, starting_strength
from .equity import Estimate
from .holdem import CARDS, HoldemEquity
from .icm import hero_value
from .model import Action, Event, LegalActions, Observation, Seat, pots, settle_pots
from .opponents import Opponents
from .policy import Parameters, TournamentPolicy, _all_in_values
from .sampling import SampledEquity


def cards(names: str) -> tuple[int, ...]:
    return tuple(CARD_INDEX[name] for name in names.split())


def all_in_fixture(players: int = 4) -> Observation:
    seats = (Seat(0, 100, 0, 0), Seat(1, 0, 100, 100)) + tuple(
        Seat(i, 100, 0, 0, True) for i in range(2, players)
    )
    return Observation(
        0,
        0,
        players - 1,
        0,
        cards("As Kh"),
        (),
        seats,
        LegalActions(100),
        20,
        (50, 30, 20) + (0,) * (players - 3),
        100,
    )


class ICMTests(unittest.TestCase):
    def test_against_independent_finish_order_oracle(self) -> None:
        rng = Random(1701)
        checked = 0
        for n in range(2, 7):
            for _ in range(8):
                stacks = tuple(rng.randrange(1, 1000) for _ in range(n))
                payouts = tuple(
                    sorted((rng.randrange(100) for _ in range(n)), reverse=True)
                )
                expected = [0.0] * n
                for order in permutations(range(n)):
                    probability = 1.0
                    remaining = sum(stacks)
                    for i in order:
                        probability *= stacks[i] / remaining
                        remaining -= stacks[i]
                    for rank, i in enumerate(order):
                        expected[i] += probability * payouts[rank]
                for i in range(n):
                    self.assertAlmostEqual(
                        hero_value(stacks, payouts, i), expected[i], places=9
                    )
                    checked += 1
                self.assertAlmostEqual(sum(expected), sum(payouts), places=9)
        self.assertEqual(checked, 160)

    def test_equal_stacks_and_zero_stacks(self) -> None:
        for n in range(2, 9):
            payouts = tuple(range(n, 0, -1))
            for i in range(n):
                self.assertAlmostEqual(
                    hero_value((100,) * n, payouts, i), sum(payouts) / n
                )
        self.assertEqual(hero_value((0, 100, 0), (50, 30, 20), 0), 25)
        self.assertEqual(hero_value((0, 100, 0), (50, 30, 20), 1), 50)

    def test_winner_take_all_is_chip_fraction(self) -> None:
        stacks = (12, 54, 21, 13)
        for i, stack in enumerate(stacks):
            self.assertAlmostEqual(hero_value(stacks, (100, 0, 0, 0), i), stack)

    def test_payout_translation(self) -> None:
        stacks = (12, 54, 21, 13)
        self.assertAlmostEqual(
            hero_value(stacks, (57, 37, 27, 7), 0),
            hero_value(stacks, (50, 30, 20, 0), 0) + 7,
        )

    def test_reject_invalid_input(self) -> None:
        for stacks, payouts in [
            ((1, -1), (2, 1)),
            ((1, 2), (1, 2)),
            ((1, 2), (float("nan"), 0)),
        ]:
            with self.assertRaises(ValueError):
                hero_value(stacks, payouts, 0)


class SettlementTests(unittest.TestCase):
    def test_random_all_in_settlements_against_pokerkit(self) -> None:
        rng = Random(882)
        for n in (2, 3, 8):
            for _ in range(30):
                stacks = tuple(rng.randrange(1, 1000) for _ in range(n))
                state = NoLimitTexasHoldem.create_state(
                    AUTOMATIONS, True, 0, (10, 20), 20, stacks, n
                )
                deck = list(DECK)
                rng.shuffle(deck)
                state.deck_cards = deque(deck)
                while state.can_deal_hole():
                    state.deal_hole()
                holes = tuple(
                    tuple(CARD_IDS[c] for c in hand) for hand in state.hole_cards
                )
                while state.status:
                    if state.actor_index is None:
                        state.deal_board()
                    elif state.can_complete_bet_or_raise_to():
                        state.complete_bet_or_raise_to(
                            state.max_completion_betting_or_raising_to_amount
                        )
                    else:
                        state.check_or_call()
                board = tuple(CARD_IDS[c] for c in state.get_board_cards(0))
                ranks = tuple(
                    eval7.evaluate([CARDS[c] for c in hole + board]) for hole in holes
                )
                expected = settle_pots(pots(stacks, (False,) * n), ranks)
                self.assertEqual(sum(state.stacks), sum(stacks))
                for actual, value in zip(state.stacks, expected):
                    # PokerKit assigns integer odd chips; the estimator splits fractional chips.
                    self.assertLess(abs(actual - value), n)
                if len(set(ranks)) == n:
                    self.assertEqual(tuple(state.stacks), expected)

    def test_main_pot_loss_can_still_win_side_pots(self) -> None:
        layers = pots((100, 60, 20, 60), (False, False, False, True))
        self.assertEqual(settle_pots(layers, (3, 2, 4, -1)), (160, 0, 80, 0))

    def test_folded_money_and_uncalled_excess(self) -> None:
        layers = pots((30, 100, 40), (False, False, True))
        self.assertEqual(settle_pots(layers, (2, 1, -1)), (90, 80, 0))

    def test_rank_ties_split_each_pot(self) -> None:
        layers = pots((100, 60, 20), (False, False, False))
        self.assertEqual(settle_pots(layers, (4, 4, 4)), (100, 60, 20))

    def test_all_in_icm_needs_more_than_chip_break_even(self) -> None:
        fold, win, lose, tie = _all_in_values(all_in_fixture())
        self.assertAlmostEqual(fold, 25)
        self.assertAlmostEqual(win, 38 + 1 / 3)
        self.assertAlmostEqual(lose, 0)
        self.assertAlmostEqual(tie, 25)
        self.assertAlmostEqual((fold - lose) / (win - lose), 0.6521739130434783)


class EquityTests(unittest.TestCase):
    def test_a_provided_ranker_does_not_need_eval7_rank_encoding(self) -> None:
        obs = all_in_fixture()
        native = HoldemEquity(71).estimate(obs, (1,) * 4, (0,) * 4, samples=80)
        supplied = SampledEquity(
            71, cards=CARDS, evaluate=lambda hand: 37 * eval7.evaluate(hand) + 100001
        )
        self.assertEqual(supplied.estimate(obs, (1,) * 4, (0,) * 4, samples=80), native)

    def test_river_sampling_agrees_with_exact_range_enumeration(self) -> None:
        obs = Observation(
            0,
            0,
            1,
            3,
            cards("As Qh"),
            cards("Ah Kd 7s 4c 2d"),
            (Seat(0, 100, 50, 0), Seat(1, 100, 50, 0)),
            LegalActions(0),
            20,
            (100, 0),
            100,
        )
        known = set(obs.hole + obs.board)
        board = [CARDS[c] for c in obs.board]
        own = eval7.evaluate([CARDS[c] for c in obs.hole] + board)
        shares = []
        for a in range(52):
            for b in range(a):
                if a in known or b in known:
                    continue
                opponent = eval7.evaluate([CARDS[a], CARDS[b]] + board)
                shares.append(
                    1.0 if own > opponent else 0.5 if own == opponent else 0.0
                )
        exact = sum(shares) / len(shares)
        estimate = HoldemEquity(113).estimate(obs, (1, 1), (0, 0), samples=10000)
        self.assertAlmostEqual(exact, 0.8888888888888888)
        self.assertLess(abs(estimate.share - exact), 0.02)
        self.assertAlmostEqual(estimate.call_return, estimate.share * 100)

    def test_expired_equity_budget_returns_without_sampling(self) -> None:
        obs = all_in_fixture()
        estimate = HoldemEquity(71).estimate(obs, (1,) * 4, (0,) * 4, deadline=0)
        self.assertEqual(estimate.samples, 0)

    def test_board_royal_splits_main_and_side_pots(self) -> None:
        obs = Observation(
            0,
            0,
            2,
            3,
            cards("2c 3d"),
            cards("As Ks Qs Js Ts"),
            (Seat(0, 100, 40, 0), Seat(1, 0, 10, 0), Seat(2, 100, 40, 0)),
            LegalActions(0),
            20,
            (50, 30, 20),
            90,
        )
        estimate = HoldemEquity(11).estimate(obs, (1, 1, 1), (0, 0, 0), samples=80)
        self.assertAlmostEqual(estimate.share, 1 / 3)
        self.assertAlmostEqual(estimate.call_return, 40)
        self.assertEqual(estimate.win, 0)
        self.assertEqual(estimate.tie, 1)

    def test_seeded_estimate_is_reproducible_without_deadline(self) -> None:
        obs = all_in_fixture()
        first = HoldemEquity(71).estimate(obs, (1,) * 4, (0,) * 4, samples=80)
        second = HoldemEquity(71).estimate(obs, (1,) * 4, (0,) * 4, samples=80)
        self.assertEqual(first, second)

    def test_preflop_strength_has_no_suit_bias(self) -> None:
        self.assertEqual(
            starting_strength(cards("As Ks")), starting_strength(cards("Ac Kc"))
        )
        self.assertEqual(
            starting_strength(cards("As Kh")), starting_strength(cards("Ac Kd"))
        )
        self.assertGreater(
            starting_strength(cards("As Ah")), starting_strength(cards("7s 2h"))
        )


class PolicyTests(unittest.TestCase):
    def test_terminal_chip_call_has_no_future_betting_discount(self) -> None:
        obs = replace(all_in_fixture(), big_blind=2)
        estimate = Estimate(0.6, 120, 0.6, 0, 0.08, 40, 40)
        chip_agent = TournamentPolicy(0, Parameters(use_icm=False), 4)
        with patch.object(chip_agent.equity, "estimate", return_value=estimate):
            self.assertEqual(chip_agent.act(obs), Action("call"))
        icm_agent = TournamentPolicy(0, Parameters(), 4)
        with patch.object(icm_agent.equity, "estimate", return_value=estimate):
            self.assertEqual(icm_agent.act(obs), Action("fold"))

    def test_frequent_callers_get_wider_ranges_and_less_informative_calls(self) -> None:
        model = Opponents(4)
        for hand in range(40):
            model.observe(Event(hand, 1, 0, Action("call"), True, 20))
            model.observe(Event(hand, 1, 1, Action("call"), True, 20))
        self.assertGreater(model.ranges[1], 0.9)
        self.assertLess(model.information()[1], 0.3)

    def test_legal_actions(self) -> None:
        legal = LegalActions(10, 40, 100)
        self.assertEqual(legal.raise_to(20), Action("raise", 40))
        self.assertEqual(legal.raise_to(120), Action("raise", 100))
        self.assertFalse(legal.accepts(Action("check")))
        self.assertFalse(legal.accepts(Action("raise", 39)))
        self.assertEqual(LegalActions(0).passive(), Action("check"))

    def test_stats_persist_only_within_agent_and_ranges_reset(self) -> None:
        model = Opponents(4)
        model.observe(Event(0, 1, 0, Action("raise", 60), True, 20))
        self.assertEqual(model.ranges[1], 0.3)
        model.begin_hand(1)
        self.assertEqual(model.ranges[1], 1)
        self.assertEqual(model.seats[1].raises, 1)
        self.assertEqual(Opponents(4).seats[1].raises, 0)

    def test_betting_does_not_reset_other_players_street_pressure(self) -> None:
        model = Opponents(4)
        model.observe(Event(0, 1, 1, Action("raise", 60), False, 20))
        model.observe(Event(0, 2, 1, Action("call"), True, 20))
        model.begin_street(1)
        self.assertEqual(model.pressure[1:3], [2, 1])
        model.begin_street(2)
        self.assertEqual(model.pressure, [0] * 4)

    def test_preflop_folds_do_not_count_as_postflop_folds(self) -> None:
        model = Opponents(4)
        for hand in range(30):
            model.observe(Event(hand, 1, 0, Action("fold"), True, 20))
        self.assertEqual(model.seats[1].postflop_faced, 0)
        self.assertAlmostEqual(model.seats[1].fold_rate, 5 / 12)

    def test_nut_hand_calls_and_board_tie_calls(self) -> None:
        obs = all_in_fixture()
        agent = TournamentPolicy(6, Parameters(equity_budget_us=100000), 4)
        nuts = replace(
            obs, street=3, hole=cards("As Ks"), board=cards("Qs Js Ts 2c 3d")
        )
        self.assertEqual(agent.act(nuts), Action("call"))
        board_tie = replace(
            obs, street=3, hole=cards("2c 3d"), board=cards("As Ks Qs Js Ts")
        )
        self.assertEqual(agent.act(board_tie), Action("call"))

    def test_arena_chip_and_payout_conservation(self) -> None:
        for n in (2, 3, 8):
            config = Configuration(
                stacks=(200,) * n,
                payouts=tuple(range(n, 0, -1)),
                blinds=(20, 40, 80),
                hands_per_level=5,
                max_hands=20,
            )
            result = play(441 + n, 0, "mixed", Parameters(samples=16), config)
            self.assertEqual(sum(result["final_stacks"]), n * 200)
            self.assertAlmostEqual(sum(result["all_points"]), sum(config.payouts))


if __name__ == "__main__":
    unittest.main()
