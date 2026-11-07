import unittest
from dataclasses import asdict
from math import fsum

from headsup.engine import CARDS, Hand
from headsup.learning import OpponentProfile, PopulationBackedProfile
from headsup.models import Action, GameConfig
from headsup.policies import make_adaptive_policy, make_river_policy
from headsup.protocol import WireAction, WireObservation
from headsup.river import _menu, _posterior, _raise_targets, _terminal
from tests.support import stacked_deck


def river_hand(config: GameConfig, *, tie: bool = False) -> Hand:
    board = ("As", "Ks", "Qs", "Js", "Ts") if tie else ("2h", "3h", "4s", "7d", "8s")
    holes = (("2c", "3d"), ("4c", "5d")) if tie else (("Ac", "Ad"), ("Kc", "Kd"))
    hand = Hand(config, deck=stacked_deck(*holes, board))
    while hand.observation().street != "river":
        hand.act(Action("call" if hand.observation().to_call else "check"))
    return hand


class RiverTests(unittest.TestCase):
    def test_terminal_call_values_match_actual_engine_settlement(self) -> None:
        for hero in (0, 1):
            for tie in (False, True):
                for rake in (0, 2500):
                    with self.subTest(hero=hero, tie=tie, rake=rake):
                        hand = river_hand(GameConfig(rake_bps=rake, rake_cap=3), tie=tie)
                        if hero == 0:
                            hand.act(Action("check"))
                        hand.act(Action("raise_to", 6))
                        observation = hand.observation()
                        self.assertEqual(observation.actor, hero)
                        outcomes, settlement = _terminal(observation, observation.to_call, 0)
                        hand.act(Action("call"))
                        result = hand.result()
                        outcome = 1 if len(result.winners) == 2 else 0 if hero in result.winners else 2
                        self.assertEqual(outcomes[outcome], result.net_stacks[hero] - observation.stacks[hero])
                        self.assertEqual(settlement["contested_pot"], result.contested_pot)
                        self.assertEqual(settlement["rake"], result.rake)

    def test_terminal_refunds_and_odd_chip_match_engine(self) -> None:
        hand = river_hand(GameConfig(rake_bps=2500, rake_cap=1))
        observation = hand.observation()
        outcomes, detail = _terminal(observation, 50, 0)
        hand.act(Action("raise_to", 50))
        hand.act(Action("fold"))
        self.assertEqual(detail["hero_refund"], 50)
        self.assertEqual(detail["rake"], 1)
        self.assertEqual(outcomes[0], hand.result().net_stacks[0] - observation.stacks[0])
        hand = river_hand(GameConfig(rake_bps=2500, rake_cap=1), tie=True)
        bb = hand.observation()
        bb_outcomes, _ = _terminal(bb, 0, 0)
        hand.act(Action("check"))
        sb = hand.observation()
        sb_outcomes, _ = _terminal(sb, 0, 0)
        hand.act(Action("check"))
        self.assertEqual(bb_outcomes[1], sb_outcomes[1] + 1)
        self.assertEqual(hand.result().net_stacks[0] - bb.stacks[0], bb_outcomes[1])
        self.assertEqual(hand.result().net_stacks[1] - sb.stacks[1], sb_outcomes[1])

    def test_exact_river_distribution_excludes_every_visible_card(self) -> None:
        hand = river_hand(GameConfig(), tie=True)
        hand.act(Action("check"))
        observation = hand.observation()
        policy = make_river_policy(5, 16, OpponentProfile())
        policy.act(observation)
        combinations, weights, values, hero_value = policy._river_distribution
        blocked = {CARDS.index(card) for card in observation.hole + observation.board}
        self.assertEqual(len(combinations), 990)
        self.assertTrue(all(not blocked.intersection(pair) for pair in combinations))
        self.assertEqual(len(set(combinations)), 990)
        self.assertTrue(all(value == hero_value for value in values))
        self.assertGreater(fsum(weights), 0)
        self.assertEqual(policy.last_info["equity"], 0.5)
        self.assertEqual(policy.last_info["equity_standard_error"], 0)

    def test_river_candidates_and_modeled_reraises_are_legal(self) -> None:
        for stack_bb in (3, 10, 100):
            hand = river_hand(GameConfig(stack_bb=stack_bb))
            hand.act(Action("check"))
            observation = hand.observation()
            legal = WireObservation.model_validate(asdict(observation))
            for action in _menu(observation, Action("check")):
                with self.subTest(stack_bb=stack_bb, action=action):
                    self.assertTrue(legal.accepts(WireAction.model_validate(asdict(action))))
                    if action.kind != "raise_to":
                        continue
                    targets = _raise_targets(observation, action)
                    self.assertEqual(len(targets), len(set(targets)))
                    for target in targets:
                        replayed = river_hand(GameConfig(stack_bb=stack_bb))
                        replayed.act(Action("check"))
                        replayed.act(action)
                        other = WireObservation.model_validate(asdict(replayed.observation()))
                        self.assertTrue(other.accepts(WireAction(kind="raise_to", amount=target)))

    def test_river_layer_preserves_nonriver_baseline_and_reports_model_scope(self) -> None:
        hand = Hand(GameConfig(), seed=77)
        baseline = make_adaptive_policy(9, 16, OpponentProfile())
        river = make_river_policy(9, 16, OpponentProfile())
        while not hand.finished and hand.observation().street != "river":
            observation = hand.observation()
            self.assertEqual(river.act(observation), baseline.act(observation))
            self.assertEqual(river.rng.getstate(), baseline.rng.getstate())
            self.assertFalse(river.last_info["river_evaluated"])
            hand.act(Action("call" if observation.to_call else "check"))
        hand = river_hand(GameConfig())
        hand.act(Action("check"))
        observation = hand.observation()
        action = river.act(observation)
        self.assertTrue(WireObservation.model_validate(asdict(observation)).accepts(
            WireAction.model_validate(asdict(action))))
        detail = river.last_info["river"]
        self.assertTrue(detail["evaluated"])
        self.assertFalse(detail["default_policy_replaced"])
        for row in detail["candidates"]:
            if row["action"]["kind"] != "raise_to":
                continue
            for scenario in row["scenarios"].values():
                self.assertAlmostEqual(sum(scenario[f"{kind}_probability"]
                                           for kind in ("fold", "call", "raise")), 1, places=10)

    def test_river_posterior_keeps_population_and_individual_support_separate(self) -> None:
        context = "river:bb:first:medium"
        pool = OpponentProfile.from_dict({"schema_version": 1, "hands_observed": 100,
                                         "responses": {context: {"fold": 80, "call": 15, "raise": 5}}})
        profile = PopulationBackedProfile(pool, strength=8)
        before = profile.fingerprint
        posterior = _posterior(profile, context)
        self.assertEqual(posterior["individual_responses"], 0)
        self.assertEqual(posterior["population_responses"], 100)
        self.assertFalse(profile.estimate(context)["ready"])
        self.assertAlmostEqual(sum(posterior["probabilities"].values()), 1)
        self.assertEqual(profile.fingerprint, before)
