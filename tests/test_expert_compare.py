import csv
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from headsup.expert_compare import compare, metrics, parse_record, reconstruct, select_hands
from headsup.models import Action


def record(action: str = "f", cards: str = "AcAd|KcKd", payoff: str = "50|-50",
           players: str = "DeepStack|human.1", number: int = 0, timestamp: str = "1000.1") -> str:
    return f"STATE:{number}:{action}:{cards}:{payoff}:{players} # {timestamp}"


def archive(path: Path, lines: list[str], notes: list[str] | None = None) -> None:
    root = "DeepStack_vs_IFP_pros"
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([f"field{i}" for i in range(18)] + [" Notes"])
    for line, note in zip(lines, notes or [""] * len(lines)):
        hand = parse_record(line)
        row = [""] * 19
        row[5] = "".join(hand.holes[hand.human_seat])
        row[6] = "".join(hand.holes[1 - hand.human_seat])
        row[10] = "bb" if hand.human_seat == 0 else "sb"
        row[18] = note
        writer.writerow(row)
    with ZipFile(path, "w") as zipped:
        zipped.writestr(f"{root}/DeepStack_logs/ACPC/all_hands.human.log", "\n".join(lines))
        zipped.writestr(f"{root}/DeepStack_logs/ACPC/all_hands.log", "\n".join(lines))
        zipped.writestr(f"{root}/AIVAT_analysis/results.human.csv", output.getvalue())


class ExpertReplayTests(unittest.TestCase):
    def test_small_blind_and_big_blind_mapping(self) -> None:
        hand = parse_record(record())
        obs, action = reconstruct(hand)[0]
        self.assertEqual((hand.human_seat, obs.actor, obs.button, obs.to_call), (1, 1, 1, 50))
        self.assertEqual(obs.hole, ("Kc", "Kd"))
        self.assertEqual(action, Action("fold"))
        big = parse_record(record(players="human.2|DeepStack"))
        self.assertEqual(big.human_seat, 0)

    def test_cumulative_raise_amount_is_converted_to_street_total(self) -> None:
        hand = parse_record("STATE:19:r200c/cr400c/cr850f:9hQc|5dQd/4h4c6c/4s:-400|400:DeepStack|human.1 # 1479132064.35509")
        steps = reconstruct(hand)
        raises = [(obs.street, action.amount) for obs, action in steps if action.kind == "raise_to"]
        self.assertEqual(raises, [("preflop", 200), ("flop", 200), ("turn", 450)])
        obs, action = steps[-1]
        self.assertEqual((obs.pot, obs.to_call, action.kind), (1250, 450, "fold"))

    def test_all_in_runout_and_short_raise(self) -> None:
        for action in ("r20000c///", "r14000r20000c///"):
            hand = parse_record(record(action, "AcAd|KcKd/2h3s4d/7c/9h", "20000|-20000"))
            steps = reconstruct(hand)
            self.assertEqual(steps[-1][1], Action("call"))
            self.assertIsNone(steps[-1][0].min_raise_to)

    def test_showdown_and_tie_are_recomputed(self) -> None:
        reconstruct(parse_record(record("cc/cc/cc/cc", "AcAd|KcKd/2h3s4d/7c/9h", "100|-100")))
        reconstruct(parse_record(record("cc/cc/cc/cc", "2c3c|4c5c/AsKsQs/Js/Ts", "0|0")))
        with self.assertRaisesRegex(ValueError, "payoffs"):
            reconstruct(parse_record(record("cc/cc/cc/cc", "AcAd|KcKd/2h3s4d/7c/9h", "-100|100")))

    def test_opponent_cards_do_not_change_human_observations(self) -> None:
        first = reconstruct(parse_record(record("cc/cc/cc/cc", "KcKd|AcAd/2h3s4d/7c/9h", "-100|100")))
        second = reconstruct(parse_record(record("cc/cc/cc/cc", "QcQd|AcAd/2h3s4d/7c/9h", "-100|100")))
        self.assertEqual([obs for obs, _ in first if obs.actor == 1], [obs for obs, _ in second if obs.actor == 1])
        human = [obs for obs, _ in first if obs.actor == 1]
        self.assertEqual([len(obs.board) for obs in human], [0, 3, 4, 5])
        self.assertTrue(all(not hasattr(obs, "deck") and not hasattr(obs, "opponent_hole") for obs in human))

    def test_future_board_does_not_change_earlier_decisions(self) -> None:
        first = reconstruct(parse_record(record("cc/cc/cc/cc", "KcKd|AcAd/2h3s4d/7c/9h", "-100|100")))
        second = reconstruct(parse_record(record("cc/cc/cc/cc", "KcKd|AcAd/2h3s4d/7c/8h", "-100|100")))
        self.assertEqual([obs for obs, _ in first if obs.street != "river"],
                         [obs for obs, _ in second if obs.street != "river"])

    def test_rejects_missing_cards_invalid_tokens_and_incomplete_records(self) -> None:
        invalid = [record(cards="????|KcKd"), record(cards="AcAd|AcKd"), record(action="r0200f"),
                   record(action="k"), record(players="DeepStack|DeepStack"), record(payoff="100|-50"),
                   record(cards="AcAd|KcKd/2h"), record(timestamp="nan")]
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_record(line)
        for line in (record("c"), record("f/"), record("r150f"),
                     record("r200c", "AcAd|KcKd/2h3s4d", "100|-100")):
            with self.subTest(line=line), self.assertRaises(ValueError):
                reconstruct(parse_record(line))


class ExpertComparisonTests(unittest.TestCase):
    def test_selection_uses_canonical_player_chronology_and_excludes_notes_without_replacement(self) -> None:
        lines = [record(number=0, timestamp="1002", players="DeepStack|human.2"),
                 record(number=0, timestamp="1000"), record(number=1, timestamp="1001")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.zip"
            archive(path, lines, ["", "DeepStack disconnected", ""])
            selected, report = select_hands(path, 2, ("human",))
            self.assertEqual(len(selected), 1)
            self.assertEqual((selected[0].human, selected[0].timestamp), ("human", "1001"))
            self.assertEqual(report["available_hands"], 3)
            self.assertEqual(len(report["excluded_annotations"]), 1)
            self.assertEqual(len(report["member_sha256"]), 2)
            with self.assertRaises(ValueError):
                select_hands(path, 2, ("absent",))

    def test_runner_preserves_historical_path_and_uses_only_human_turns(self) -> None:
        line = record("r200c/cr400f", "AcAd|KcKd/2h3s4d", "-200|200")
        observed = []

        def policy(name: str, seed: int, samples: int) -> SimpleNamespace:
            def act(obs: object) -> Action:
                observed.append((name, seed, obs))
                return Action("call" if obs.to_call else "check")
            return SimpleNamespace(act=act, last_info={"test": True})

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.zip"
            archive(source, [line])
            with patch("headsup.expert_compare.make_policy", side_effect=policy):
                report = compare(source, Path(temp) / "output", players=("human",))
            rows = [json.loads(row) for row in (Path(temp) / "output/decisions.jsonl").read_text().splitlines()]
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["validated_completed_hands"], 1)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["human_seat"] == row["observation"]["actor"] == 1 for row in rows))
            self.assertEqual([row["human_action"]["amount"] for row in rows], [200, 200, 200, 200])
            self.assertEqual(observed[0][1], observed[1][1])
            self.assertEqual(observed[2][2].pot, 400)
            self.assertNotIn("winnings", report)

    def test_bad_settlement_invalidates_before_any_policy_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.zip"
            archive(source, [record(payoff="100|-100")])
            with patch("headsup.expert_compare.make_policy") as policy:
                report = compare(source, Path(temp) / "output", players=("human",))
            policy.assert_not_called()
            self.assertEqual((report["status"], report["validated_completed_hands"]), ("invalidated", 0))

    def test_metrics_keep_action_and_conditional_size_agreement_separate(self) -> None:
        obs = {"pot": 200, "to_call": 100}
        rows = [{"observation": obs, "human_action": asdict(human), "bot_action": asdict(bot)}
                for human, bot in ((Action("raise_to", 600), Action("raise_to", 300)),
                                   (Action("call"), Action("fold")), (Action("check"), Action("check")))]
        report = metrics(rows)
        self.assertEqual(report["action_class_agreement"], 2 / 3)
        self.assertEqual(report["both_raise_decisions"], 1)
        self.assertEqual(report["mean_absolute_raise_to_difference_bb"], 3)
        self.assertEqual(report["mean_absolute_raise_to_difference_pot"], 1)
        self.assertEqual(report["exact_action_matches"], 1)


if __name__ == "__main__":
    unittest.main()
