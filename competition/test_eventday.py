import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from .bridge import (
    Bridge,
    ContractError,
    Frame,
    encode_raise,
    normalize_raise,
    validate_frame,
)
from .eventday import build, intake, rehearse
from .policy import Parameters, TournamentPolicy
from .rehearsal.adapter import DemoAdapter
from .rehearsal.sdk import SDKEquity, fixture, legal_output
from .sampling import SampledEquity


def policy() -> TournamentPolicy:
    return TournamentPolicy(equity=SDKEquity())


class BridgeTests(unittest.TestCase):
    def test_a_supplied_sdk_ranker_can_sample_without_native_modules(self) -> None:
        raw = fixture()
        raw["hole_cards"] = ["2c", "3d"]
        raw["board"] = ["As", "Ks", "Qs", "Js", "Ts"]
        obs = DemoAdapter().decode(raw).observation
        # Every seven-card hand ties on this royal-flush board.
        backend = SampledEquity(71, cards=tuple(range(52)), evaluate=lambda hand: 0)
        result = backend.estimate(obs, (1, 1), (0, 0), samples=16)
        self.assertEqual(result.share, 0.5)
        self.assertEqual(result.call_return, 240)

    def test_all_three_raise_conventions_preserve_the_bet(self) -> None:
        for mode, expected in (("to", 140), ("increment", 80), ("additional", 120)):
            frame = DemoAdapter().decode(fixture(mode))
            validate_frame(frame)
            self.assertEqual(encode_raise(140, frame.observation, mode), expected)
            self.assertEqual(normalize_raise(expected, frame.observation, mode), 140)
            result = Bridge(DemoAdapter(), policy).decide(fixture(mode))
            self.assertTrue(legal_output(fixture(mode), result))
            self.assertEqual(result["type"], "raise")

    def test_repeated_snapshots_do_not_duplicate_opponent_actions(self) -> None:
        bridge = Bridge(DemoAdapter(), policy)
        raw = fixture()
        bridge.decide(raw)
        bridge.decide(raw)
        self.assertEqual(bridge.events_seen, 1)
        self.assertEqual(bridge.policy.opponents.seats[1].raises, 1)

    def test_new_tables_with_reused_hand_numbers_reset_learning(self) -> None:
        bridge = Bridge(DemoAdapter(), policy)
        bridge.decide(fixture(table="first"))
        bridge.decide(fixture(table="second"))
        self.assertEqual(bridge.table_resets, 2)
        self.assertEqual(bridge.policy.opponents.seats[1].raises, 1)

    def test_sdk_action_objects_can_use_the_same_bridge(self) -> None:
        class ObjectAdapter(DemoAdapter):
            def encode(self, action, raw, frame):
                return (
                    action.kind,
                    encode_raise(action.amount, frame.observation, "additional")
                    if action.kind == "raise"
                    else 0,
                )

        result = Bridge(ObjectAdapter(), policy).decide(fixture("additional"))
        self.assertEqual(result, ("raise", 328))

    def test_gapped_or_reordered_history_is_rejected(self) -> None:
        raw = fixture()
        raw["history"][0]["sequence"] = 2
        with self.assertRaises(ContractError):
            Bridge(DemoAdapter(), policy).decide(raw)
        raw = fixture()
        raw["history"].append(deepcopy(raw["history"][0]))
        with self.assertRaises(ContractError):
            Bridge(DemoAdapter(), policy).decide(raw)

    def test_hand_regression_requires_a_new_table(self) -> None:
        bridge = Bridge(DemoAdapter(), policy)
        raw = fixture()
        raw["hand_index"] = 2
        raw["history"][0]["hand_index"] = 2
        bridge.decide(raw)
        with self.assertRaises(ContractError):
            bridge.decide(fixture())

    def test_other_variants_cannot_accidentally_use_the_holdem_strategy(self) -> None:
        adapter = DemoAdapter()
        adapter.variant = "omaha"
        with self.assertRaises(ContractError):
            Bridge(adapter, policy)

    def test_missing_history_requires_explicit_stateless_configuration(self) -> None:
        adapter = DemoAdapter()
        adapter.has_history = False
        with self.assertRaises(ContractError):
            Bridge(adapter, policy)
        bridge = Bridge(
            adapter,
            lambda: TournamentPolicy(
                parameters=Parameters(adapt=False), equity=SDKEquity()
            ),
        )
        raw = fixture()
        raw["history"] = []
        self.assertTrue(legal_output(raw, bridge.decide(raw)))

    def test_expired_callback_budget_uses_a_legal_passive_action(self) -> None:
        bridge = Bridge(DemoAdapter(), policy)
        with patch(
            "competition.bridge.perf_counter_ns", side_effect=[0, 1_000_000, 1_000_100]
        ):
            result = bridge.decide(fixture())
        self.assertEqual(result, {"type": "fold"})
        self.assertEqual(bridge.budget_fallbacks, 1)

    def test_capped_calls_and_undersized_allins_are_valid(self) -> None:
        adapter = DemoAdapter()
        raw = fixture()
        raw["players"][0]["remaining_chips"] = 30
        raw["legal"].update(call_cost=30, min_raise=None, max_raise=None)
        validate_frame(adapter.decode(raw))
        raw = fixture()
        raw["players"][0]["remaining_chips"] = 60
        raw["legal"].update(min_raise=80, max_raise=80)
        validate_frame(adapter.decode(raw))

    def test_malformed_card_seat_and_chip_mappings_fail_before_play(self) -> None:
        original = DemoAdapter().decode(fixture())
        obs = original.observation
        invalid = [
            replace(obs, board=(obs.hole[0],) + obs.board[1:]),
            replace(obs, pot=obs.pot + 1),
            replace(obs, seats=(replace(obs.seats[0], id=4), obs.seats[1])),
            replace(obs, legal=replace(obs.legal, call=9999)),
            replace(obs, legal=replace(obs.legal, min_raise_to=100.5)),
            replace(obs, payouts=(0, 100)),
        ]
        for observation in invalid:
            with self.assertRaises(ContractError):
                validate_frame(Frame(original.table, observation, original.events))


class EventToolsTests(unittest.TestCase):
    def test_starter_backend_checks_and_packages_an_independent_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text(
                "from competition.rules import choose\n"
                "class Agent:\n"
                "    def act(self, state):\n"
                "        return choose(state)\n"
            )
            package = root / "competition"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "rules.py").write_text(
                "def choose(state):\n    return state['legal'][0]\n"
            )
            output = root / "out" / "agent.zip"
            command = [
                sys.executable, "-I", "-S", "-c",
                "import sys;sys.path.insert(0,sys.argv[1]);"
                "from agent import Agent;"
                "assert Agent().act({'legal':['draw_three']}) == 'draw_three'",
                "{candidate}",
            ]
            manifest = build(
                root, "agent.py", ("competition",), output, command,
                backend="starter", context="synthetic-draw-variant",
            )
            self.assertFalse(manifest["prepared_holdem_core_included"])
            self.assertEqual(manifest["external_import_roots"], [])
            self.assertEqual(manifest["archive_verification"]["returncode"], 0)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "agent.py", "competition/__init__.py", "competition/rules.py"
                })

    def test_extracted_archive_failure_cannot_replace_the_working_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text("VALUE = 1\n")
            output = root / "working.zip"
            output.write_bytes(b"working candidate")
            command = [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import pathlib,sys;assert pathlib.Path(sys.argv[1]).parent.name != 'extracted'",
                "{entry}",
            ]
            with self.assertRaises(RuntimeError):
                build(root, "agent.py", (), output, command)
            self.assertEqual(output.read_bytes(), b"working candidate")
            report = json.loads(output.with_suffix(".failed.json").read_text())
            self.assertEqual(report["verification"]["phase"], "extracted-archive")

    def test_a_check_that_mutates_candidate_source_cannot_promote_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text("VALUE = 1\n")
            output = root / "working.zip"
            output.write_bytes(b"working candidate")
            command = [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('VALUE = 2')",
                "{entry}",
            ]
            with self.assertRaises(RuntimeError):
                build(root, "agent.py", (), output, command)
            self.assertEqual(output.read_bytes(), b"working candidate")

    def test_starter_inspection_does_not_execute_starter_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            (root / "agent.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\nclass Agent:\n    def act(self, state: dict) -> dict:\n        return state\n"
            )
            report = intake(root, sys.executable)
            self.assertFalse(marker.exists())
            self.assertFalse(report["executed_starter_code"])
            self.assertEqual(report["interfaces"][0]["class"], "Agent")

    def test_checked_build_preserves_previous_artifacts_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent.py"
            entry.write_text("VALUE = 1\n")
            output = root / "out" / "agent.zip"
            command = [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import pathlib,sys; assert 'VALUE = 1' in pathlib.Path(sys.argv[1]).read_text()",
                "{entry}",
            ]
            build(root, "agent.py", (), output, command)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            entry.write_text("VALUE = 2\n")
            with self.assertRaises(RuntimeError):
                build(root, "agent.py", (), output, command)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
            self.assertFalse(
                json.loads(output.with_suffix(".failed.json").read_text())["passed"]
            )
            passed = command.copy()
            passed[-2] = passed[-2].replace("VALUE = 1", "VALUE = 2")
            manifest = build(root, "agent.py", (), output, passed)
            self.assertTrue(Path(manifest["previous_artifact"]).exists())
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertIn("competition/bridge.py", names)
                self.assertNotIn("competition/arena.py", names)
                self.assertNotIn("competition/holdem.py", names)

    def test_a_timed_out_check_cannot_replace_the_working_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text("VALUE = 1\n")
            output = root / "working.zip"
            output.write_bytes(b"previous working candidate")
            command = [sys.executable, "-c", "import time;time.sleep(1)", "{entry}"]
            with self.assertRaises(RuntimeError):
                build(root, "agent.py", (), output, command, timeout=0.01)
            self.assertEqual(output.read_bytes(), b"previous working candidate")

    def test_candidate_paths_and_unbound_checks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text("pass\n")
            with self.assertRaises(ValueError):
                build(
                    root,
                    "../agent.py",
                    (),
                    root / "out.zip",
                    [sys.executable, "{entry}"],
                )
            with self.assertRaises(ValueError):
                build(
                    root,
                    "agent.py",
                    (),
                    root / "out.zip",
                    [sys.executable, "-c", "pass"],
                )

    def test_archive_rehearsal_runs_without_site_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = rehearse(Path(directory), deterministic=True)
            self.assertTrue(report["extracted_archive_verified"])
            self.assertEqual(report["details"]["third_party_runtime_imports"], [])
            self.assertFalse(report["official_sdk"])
            self.assertEqual(report["details"]["deadline_mode"], "deterministic-wiring")


if __name__ == "__main__":
    unittest.main()
