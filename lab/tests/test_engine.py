import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from random import Random

from headsup.arena import replay, seed_for
from headsup.engine import CARDS, Hand
from headsup.models import Action, GameConfig
from headsup.protocol import WireObservation
from headsup.training import _play
from tests.support import finish_passively, stacked_deck


class EngineTests(unittest.TestCase):
    def test_blinds_button_and_first_postflop_actor(self) -> None:
        hand = Hand(GameConfig(), seed=7)
        before = hand.observation()
        self.assertEqual((before.actor, before.button, before.bets, before.pot),
                         (1, 1, (2, 1), 3))
        hand.act(Action("call"))
        hand.act(Action("check"))
        after = hand.observation()
        self.assertEqual((after.actor, after.street, after.bets, after.pot),
                         (0, "flop", (0, 0), 4))

    def test_observation_contains_only_own_cards_and_is_immutable(self) -> None:
        deck = stacked_deck(("Ac", "Ad"), ("Kc", "Kd"), ("2c", "3d", "4h", "7s", "8c"))
        hand = Hand(GameConfig(), deck=deck)
        observation = hand.observation()
        self.assertEqual(observation.hole, ("Kc", "Kd"))
        self.assertEqual(observation.board, ())
        self.assertFalse({"deck", "seed", "opponent_hole", "engine"} & asdict(observation).keys())
        with self.assertRaises(FrozenInstanceError):
            observation.actor = 0
        hand.act(Action("call"))
        self.assertEqual(hand.observation().hole, ("Ac", "Ad"))
        self.assertEqual(observation.history, ())

    def test_illegal_actions_leave_state_unchanged(self) -> None:
        hand = Hand(GameConfig(), seed=2)
        before = hand.snapshot()
        for action in (Action("check"), Action("raise_to", 3), Action("raise_to", 201)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                hand.act(action)
            self.assertEqual(hand.snapshot(), before)
        hand.act(Action("call"))
        with self.assertRaises(ValueError):
            hand.act(Action("fold"))

    def test_raise_to_is_total_street_contribution(self) -> None:
        hand = Hand(GameConfig(), seed=5)
        hand.act(Action("raise_to", 6))
        event = hand.public_history[0]
        self.assertEqual((event.paid, event.action.amount), (5, 6))
        self.assertEqual((hand.observation().pot, hand.observation().to_call), (8, 4))
        hand.act(Action("call"))
        self.assertEqual(hand.observation().pot, 12)

    def test_fold_refunds_uncalled_raise_and_no_flop_rake(self) -> None:
        hand = Hand(GameConfig(rake_bps=500, rake_cap=20), seed=3)
        hand.act(Action("raise_to", 100))
        hand.act(Action("fold"))
        result = hand.result()
        self.assertEqual(result.contested_pot, 4)
        self.assertEqual(result.gross_returns, (-2, 2))
        self.assertEqual(result.net_returns, (-2, 2))
        self.assertEqual(result.rake, 0)

    def test_preflop_rake_charges_only_matched_pot(self) -> None:
        hand = Hand(GameConfig(rake_bps=2500, rake_cap=20, no_flop_no_drop=False), seed=3)
        hand.act(Action("raise_to", 100))
        hand.act(Action("fold"))
        self.assertEqual(hand.result().rake, 1)
        self.assertEqual(hand.result().net_returns, (-2, 1))

    def test_known_all_in_winner_and_rake_cap(self) -> None:
        deck = stacked_deck(("Ac", "Ad"), ("Kc", "Kd"), ("2c", "3d", "4h", "7s", "8c"))
        hand = Hand(GameConfig(stack_bb=20, rake_bps=500, rake_cap=3), deck=deck)
        hand.act(Action("raise_to", 40))
        hand.act(Action("call"))
        result = hand.result()
        self.assertEqual((result.winners, result.gross_stacks), ((0,), (80, 0)))
        self.assertEqual((result.rake, result.net_stacks), (3, (77, 0)))
        self.assertEqual(result.board, ("2c", "3d", "4h", "7s", "8c"))

    def test_split_pot_odd_chip_goes_left_of_button(self) -> None:
        deck = stacked_deck(("2c", "3d"), ("4c", "5d"), ("As", "Ks", "Qs", "Js", "Ts"))
        hand = Hand(GameConfig(rake_bps=2500, rake_cap=1), deck=deck)
        finish_passively(hand)
        result = hand.result()
        self.assertEqual(result.winners, (0, 1))
        self.assertEqual(result.gross_returns, (0, 0))
        self.assertEqual(result.net_returns, (0, -1))
        self.assertEqual(result.rake, 1)

    def test_random_legal_games_conserve_chips_and_replay(self) -> None:
        for seed in range(120):
            with self.subTest(seed=seed):
                config = GameConfig(stack_bb=(1, 5, 20, 100)[seed % 4],
                                    rake_bps=(0, 500, 1000)[seed % 3], rake_cap=7,
                                    no_flop_no_drop=seed % 2 == 0)
                hand, rng = Hand(config, seed=seed), Random(seed + 999)
                while not hand.finished:
                    observation = hand.observation()
                    wire = WireObservation.model_validate(asdict(observation))
                    actions = [Action("call" if observation.to_call else "check")]
                    if observation.can_fold:
                        actions.append(Action("fold"))
                    if observation.min_raise_to is not None:
                        actions.extend((Action("raise_to", observation.min_raise_to),
                                        Action("raise_to", observation.max_raise_to)))
                    action = rng.choice(actions)
                    self.assertEqual(sum(observation.stacks) + observation.pot,
                                     2 * config.starting_stack)
                    self.assertEqual(wire.to_domain(), observation)
                    hand.act(action)
                result = hand.result()
                self.assertEqual(sum(result.net_returns) + result.rake, 0)
                self.assertTrue(all(stack >= 0 for stack in result.net_stacks))
                self.assertEqual(replay(hand.record()), result)

    def test_policy_replay_and_corruption_detection(self) -> None:
        for seed in range(4):
            hand = Hand(GameConfig(), seed=seed)
            record = _play(hand, "calling", seed % 2, (31, 43), 16, None, [])
            self.assertEqual(replay(record, verify_policy=True), hand.result())
            changed = deepcopy(record)
            changed["result"]["rake"] += 1
            with self.assertRaisesRegex(ValueError, "settlement"):
                replay(changed)
            changed = deepcopy(record)
            changed["decisions"][0]["actor"] = 1 - changed["decisions"][0]["actor"]
            with self.assertRaisesRegex(ValueError, "actor mismatch"):
                replay(changed)

    def test_invalid_decks_and_noninteger_configuration_are_rejected(self) -> None:
        for deck in (CARDS[:-1], CARDS[:-1] + (CARDS[0],), CARDS[:-1] + ("Joker",)):
            with self.subTest(deck=deck[-2:]), self.assertRaises(ValueError):
                Hand(GameConfig(), deck=deck)
        for values in ({"small_blind": 0}, {"big_blind": True}, {"rake_bps": 10001},
                       {"stack_bb": 1.5}, {"no_flop_no_drop": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GameConfig(**values)

    def test_deck_and_policy_seed_streams_are_distinct_and_repeatable(self) -> None:
        self.assertEqual(seed_for(11, "deck", 3), seed_for(11, "deck", 3))
        self.assertEqual(Hand(GameConfig(), seed=55).snapshot(), Hand(GameConfig(), seed=55).snapshot())
        self.assertEqual(len({seed_for(11, stream, index)
                              for stream in ("deck", "policy") for index in range(100)}), 200)
