import unittest
from dataclasses import replace
from math import sqrt

from .arena import Configuration
from .cards import CARD_INDEX
from .holdem import HoldemEquity
from .league import BoardOpponent, FAMILIES, FixedSampleEquity, opponents, paired_summary, play_harness, policy_factory, validate_manifest
from .model import Action, Event, LegalActions, Observation, Seat
from .policy import Parameters


def cards(names: str) -> tuple[int, ...]:
    return tuple(CARD_INDEX[name] for name in names.split())


def fixture() -> Observation:
    return Observation(
        1, 0, 1, 3, cards("As Ks"), cards("Qs Js Ts 2c 3d"),
        (Seat(0, 900, 100, 0), Seat(1, 900, 100, 0)),
        LegalActions(0, 20, 900), 20, (100, 0), 200,
    )


def result(seed: int, points: float) -> dict:
    return {"seed": seed, "hero_seat": seed % 2, "field": "value", "points": points, "times_ns": [10000, 20000], "overruns": 0, "hands": 10}


class BoardOpponentTests(unittest.TestCase):
    def test_nuts_and_board_tie_use_actual_board(self) -> None:
        obs = fixture()
        for style in FAMILIES[:-1]:
            with self.subTest(style=style):
                agent = BoardOpponent(style, 42)
                self.assertEqual(agent.strength(obs), 1.0)
                royal = replace(obs, hole=cards("2c 3d"), board=cards("As Ks Qs Js Ts"))
                self.assertEqual(agent.strength(royal), 0.5)
                # A weak starting hand becomes the absolute nuts on this board.
                transformed = replace(obs, hole=cards("7c 2d"), board=cards("7d 7h 7s Ac Kd"))
                self.assertEqual(agent.strength(transformed), 1.0)
                self.assertLess(agent.strength(replace(transformed, street=0, board=())), 0.15)

    def test_actions_are_legal_across_streets_sizes_and_raise_rights(self) -> None:
        obs = fixture()
        boards = ((), cards("Qs Js Ts"), cards("Qs Js Ts 2c"), obs.board)
        legal_sets = (LegalActions(0), LegalActions(10), LegalActions(0, 20, 30), LegalActions(10, 40, 900))
        for style in FAMILIES[:-1]:
            agent = BoardOpponent(style, 17)
            for street, board in enumerate(boards):
                for legal in legal_sets:
                    for hole in (obs.hole, cards("4h 8d"), cards("2h 2d")):
                        current = replace(obs, board=board, street=street, legal=legal, hole=hole)
                        for _ in range(5):
                            action = agent.act(current)
                            self.assertTrue(legal.accepts(action), (style, current, action))

    def test_value_family_bets_nuts_and_folds_air_to_large_bet(self) -> None:
        obs = fixture()
        self.assertEqual(BoardOpponent("value", 10).act(obs), Action("raise", 150))
        air = replace(obs, hole=cards("4c 5d"), board=cards("Ah Kh Qs Js 9d"), legal=LegalActions(900))
        self.assertEqual(BoardOpponent("value", 10, samples=100).act(air), Action("fold"))

    def test_no_raising_into_only_all_in_opponent(self) -> None:
        obs = fixture()
        obs = replace(obs, seats=(obs.seats[0], Seat(1, 0, 1000, 0)))
        for style in FAMILIES[:-1]:
            self.assertEqual(BoardOpponent(style).act(obs), Action("check"))

    def test_tricky_observes_public_raise(self) -> None:
        agent = BoardOpponent("tricky", 1)
        agent.observe(Event(1, 1, 3, Action("raise", 200), False, 20))
        self.assertEqual(agent.last_raise, (1, 3, 1))


class LeagueEvaluationTests(unittest.TestCase):
    def test_fixed_samples_removes_expired_deadline_and_is_reproducible(self) -> None:
        obs = fixture()
        first = FixedSampleEquity(HoldemEquity(44)).estimate(obs, (1, 1), (0, 0), samples=19, deadline=0)
        second = FixedSampleEquity(HoldemEquity(44)).estimate(obs, (1, 1), (0, 0), samples=19, deadline=1)
        self.assertEqual(first.samples, 19)
        self.assertEqual(first, second)

    def test_paired_interval_is_on_differences_not_independent_means(self) -> None:
        old = [result(1, 10), result(2, 40), result(3, 80)]
        new = [result(1, 12), result(2, 44), result(3, 86)]
        report = paired_summary(old, new)
        self.assertEqual(report["paired_delta"]["mean"], 4)
        self.assertAlmostEqual(report["paired_delta"]["95pct_interval"][0], 4 - 1.96 * 2 / sqrt(3))
        self.assertAlmostEqual(report["paired_delta"]["95pct_interval"][1], 4 + 1.96 * 2 / sqrt(3))
        self.assertEqual(report["paired_delta"]["positive_tables"], 3)
        with self.assertRaises(ValueError):
            paired_summary(old, list(reversed(new)))

    def test_frozen_manifest_rejects_changes_and_seed_reuse(self) -> None:
        identity = {"candidate": "frozen", "source_sha256": {"policy.py": "abc"}}
        manifest = {"identity": identity, "screening_seed_interval": [10, 29]}
        validate_manifest(manifest, identity, 30, 10)
        with self.assertRaises(ValueError):
            validate_manifest(manifest, identity, 29, 10)
        with self.assertRaises(ValueError):
            validate_manifest(manifest, {"candidate": "changed"}, 30, 10)

    def test_selfplay_is_frozen_original_and_fixed_sample_wrapped(self) -> None:
        roster = opponents("selfplay", 123, 1, 4, True)
        self.assertIsNone(roster[1])
        for seat in (0, 2, 3):
            self.assertEqual(roster[seat].parameters, Parameters())
            self.assertIsInstance(roster[seat].equity, FixedSampleEquity)

    def test_factory_rejects_expressions(self) -> None:
        with self.assertRaises(ValueError):
            policy_factory("competition.policy:TournamentPolicy()", 1, Parameters(), 2, True)

    def test_identical_policies_have_zero_paired_delta_without_clock_noise(self) -> None:
        config = Configuration(stacks=(100, 100), payouts=(100, 0), blinds=(20,), max_hands=4)
        report = play_harness(
            candidate="competition.policy:TournamentPolicy", fields=("value", "pressure", "selfplay"),
            tables=2, seed=2401, config=config, fixed_samples=True,
            baseline_parameters=Parameters(samples=10), candidate_parameters=Parameters(samples=10),
        )
        self.assertFalse(report["configuration"]["enforce_deadline"])
        for field in report["fields"].values():
            self.assertEqual(field["paired_delta"]["mean"], 0)
            self.assertEqual(field["paired_delta"]["95pct_interval"], [0, 0])


if __name__ == "__main__":
    unittest.main()
