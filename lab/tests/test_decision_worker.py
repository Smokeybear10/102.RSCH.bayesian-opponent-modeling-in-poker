import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from random import Random

from headsup.decision_worker import Confirm, Decide, DecisionWorker, Finish, Ping, WorkerError, run_worker
from headsup.engine import Hand
from headsup.models import Action, GameConfig
from headsup.opponent_memory import OpponentMemory
from headsup.protocol import WireObservation
from headsup.timing import ActionTiming
from tests.support import Clock, identity, stacked_deck, wire_history


class DecisionWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.clock = Clock()
        self.memory = OpponentMemory(self.directory, identity())
        self.addCleanup(self.memory.close)
        self.worker = self.make_worker(self.memory)
        self.hand = Hand(GameConfig(rake_cap=0), deck=stacked_deck(
            ("2c", "3d"), ("As", "Ah"), ("9c", "Td", "Jh", "4s", "5c")))

    def make_worker(self, memory: OpponentMemory) -> DecisionWorker:
        return DecisionWorker(7, samples=16, timing=ActionTiming(0, 0, 1), memory_store=memory,
                              monotonic_clock=lambda: self.clock.monotonic,
                              wall_clock=lambda: self.clock.wall)

    def request(self, hand_id: str = "hand_1", observation: WireObservation | None = None) -> Decide:
        return Decide(kind="decide", hand_id=hand_id,
                      observation=observation or WireObservation.model_validate(asdict(self.hand.observation())),
                      observed_unix_ms=int(self.clock.wall * 1000),
                      deadline_unix_ms=int((self.clock.wall + 20) * 1000))

    def start_raise(self) -> dict:
        response = self.worker.handle(self.request())
        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["action"]["kind"], "raise_to")
        self.hand.act(Action(**response["action"]))
        return response

    def finish_fold(self) -> Finish:
        self.start_raise()
        self.worker.handle(Confirm(kind="confirm", hand_id="hand_1", history=wire_history(self.hand.public_history)))
        self.hand.act(Action("fold"))
        return Finish(kind="finish", hand_id="hand_1", hero_seat=1, history=wire_history(self.hand.public_history))

    def test_duplicate_decision_is_cached_and_rng_commits_only_on_receipt(self) -> None:
        request = self.request()
        first = self.worker.handle(request)
        rng_before = self.worker.policy.rng.getstate()
        self.assertEqual(self.worker.handle(request), first)
        self.assertEqual(self.worker.policy.rng.getstate(), rng_before)
        self.hand.act(Action(**first["action"]))
        receipt = Confirm(kind="confirm", hand_id="hand_1", history=wire_history(self.hand.public_history))
        pending_rng = self.worker.pending.rng_after
        self.assertEqual(self.worker.handle(receipt)["status"], "confirmed")
        self.assertEqual(self.worker.policy.rng.getstate(), pending_rng)
        self.assertEqual(self.worker.handle(receipt)["status"], "confirmed")
        self.assertEqual(self.memory.profile.hands_observed, 0)

    def test_verified_completion_persists_once_and_next_hand_uses_profile(self) -> None:
        completion = self.finish_fold()
        result = self.worker.handle(completion)
        self.assertTrue(result["learning_updated"])
        self.assertEqual(result["learning"]["total_responses"], 1)
        self.worker.close()
        memory = OpponentMemory(self.directory, identity())
        self.addCleanup(memory.close)
        worker = self.make_worker(memory)
        self.assertFalse(worker.handle(completion)["learning_updated"])
        self.assertEqual(worker.handle(Ping(kind="ping"))["learning"]["profile_hands"], 1)
        self.hand = Hand(GameConfig(rake_cap=0), seed=4)
        response = worker.handle(self.request("hand_2"))
        self.assertEqual(response["diagnostics"]["opponent_hands_observed"], 1)
        self.assertEqual(response["diagnostics"]["opponent_profile_sha256"], memory.profile.fingerprint)

    def test_mismatched_confirmation_cannot_learn_or_commit(self) -> None:
        self.start_raise()
        event = wire_history(self.hand.public_history)[0]
        changed = event.model_copy(update={"actor": 0})
        before = self.worker.policy.rng.getstate()
        with self.assertRaisesRegex(WorkerError, "confirmation_action_mismatch"):
            self.worker.handle(Confirm(kind="confirm", hand_id="hand_1", history=(changed,)))
        self.assertEqual(self.worker.policy.rng.getstate(), before)
        self.assertEqual(self.memory.profile.hands_observed, 0)

    def test_unconfirmed_and_unfinished_hands_do_not_enter_memory(self) -> None:
        self.start_raise()
        receipt = wire_history(self.hand.public_history)
        self.hand.act(Action("fold"))
        with self.assertRaisesRegex(WorkerError, "pending_turn_requires_confirmation"):
            self.worker.handle(Finish(kind="finish", hand_id="hand_1", hero_seat=1,
                                      history=wire_history(self.hand.public_history)))
        self.worker.handle(Confirm(kind="confirm", hand_id="hand_1", history=receipt))
        with self.assertRaisesRegex(WorkerError, "hand_has_not_finished_betting"):
            self.worker.handle(Finish(kind="finish", hand_id="hand_1", hero_seat=1, history=receipt))
        self.worker.close()
        memory = OpponentMemory(self.directory, identity())
        self.addCleanup(memory.close)
        self.assertEqual(memory.profile.hands_observed, 0)
        self.assertFalse(memory.contains("hand_1"))

    def test_stale_capture_and_rule_changes_reject_without_action(self) -> None:
        request = self.request()
        stale = request.model_copy(update={"observed_unix_ms": request.observed_unix_ms - 3000})
        with self.assertRaisesRegex(WorkerError, "observation_not_fresh"):
            self.worker.handle(stale)
        for changes in ({"big_blind": 4}, {"rake_bps": 500}, {"rake_cap": 2}):
            with self.subTest(changes=changes), self.assertRaisesRegex(WorkerError, "rules_changed"):
                self.worker.handle(request.model_copy(update={
                    "observation": request.observation.model_copy(update=changes)}))
        self.assertIsNone(self.worker.pending)

    def test_clock_jump_permanently_blocks_worker(self) -> None:
        self.clock.wall += 1
        with self.assertRaisesRegex(WorkerError, "clock_changed"):
            self.worker.handle(Ping(kind="ping"))
        self.clock.wall -= 1
        with self.assertRaisesRegex(WorkerError, "clock_changed"):
            self.worker.handle(self.request())

    def test_blind_fold_before_hero_action_counts_hand_without_false_response(self) -> None:
        hand = Hand(GameConfig(rake_cap=0))
        hand.act(Action("fold"))
        completion = Finish(kind="finish", hand_id="blind_fold", hero_seat=0,
                            history=wire_history(hand.public_history))
        response = self.worker.handle(completion)
        self.assertTrue(response["learning_updated"])
        self.assertEqual(response["learning"]["profile_hands"], 1)
        self.assertEqual(response["learning"]["total_responses"], 0)
        self.assertFalse(self.worker.handle(completion)["learning_updated"])

    def test_json_boundary_rejects_duplicate_keys_and_releases_lock_at_eof(self) -> None:
        source = io.StringIO('{"kind":"ping","kind":"ping"}\n{"kind":"ping","unexpected":true}\n{"kind":"ping"}\n')
        output = io.StringIO()
        self.assertEqual(run_worker(self.worker, source, output), 0)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["status"] for row in rows], ["error", "error", "worker_ready"])
        self.assertEqual(rows[0]["error"], "invalid_request")
        reopened = OpponentMemory(self.directory, identity())
        self.addCleanup(reopened.close)


class TimingTests(unittest.TestCase):
    def test_seeded_timing_varies_within_bounds(self) -> None:
        timing = ActionTiming(0.3, 1.1, 1)
        rng = Random(8)
        delays = [timing.plan(10, rng, 30).target_at - 10 for _ in range(100)]
        self.assertTrue(all(0.3 <= value <= 1.1 for value in delays))
        self.assertGreater(len(set(delays)), 90)
        repeated = Random(8)
        self.assertEqual(delays, [timing.plan(10, repeated, 30).target_at - 10 for _ in range(100)])

    def test_deadline_reserve_shortens_delay_and_blocks_at_boundary(self) -> None:
        schedule = ActionTiming(0.8, 1.1, 1).plan(10, Random(1), 11.2)
        self.assertEqual(schedule.target_at, 10)
        self.assertEqual(schedule.remaining(10), 0)
        self.assertIsNone(schedule.remaining(10.2))
        self.assertIsNone(schedule.remaining(11.2))

    def test_nonfinite_timing_and_clock_regression_are_rejected(self) -> None:
        for values in ((-1, 1, 1), (2, 1, 1), (0, float("nan"), 1), (0, 1, float("inf"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ActionTiming(*values)
        with self.assertRaises(ValueError):
            ActionTiming().plan(10, Random(1), 30).remaining(9)
