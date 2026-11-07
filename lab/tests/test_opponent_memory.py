import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from headsup.learning import OpponentProfile
from headsup.opponent_memory import OpponentMemory, read_opponent_profiles
from tests.support import identity, response_hand, wire_history


class OpponentMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.history = wire_history(response_hand().public_history)

    def open_memory(self, opponent: str = "opponent_a", **kwargs: object) -> OpponentMemory:
        memory = OpponentMemory(self.directory, identity(opponent), **kwargs)
        self.addCleanup(memory.close)
        return memory

    def test_restart_preserves_profile_and_duplicate_receipts(self) -> None:
        memory = self.open_memory()
        self.assertTrue(memory.learn("hand_1", 1, self.history))
        expected = memory.summary()
        memory.close()
        restored = self.open_memory()
        self.assertEqual(restored.summary(), expected)
        self.assertFalse(restored.learn("hand_1", 1, self.history))
        self.assertEqual(restored.summary(), expected)

    def test_same_hand_ids_do_not_mix_different_opponents(self) -> None:
        first, second = self.open_memory("a"), self.open_memory("b")
        first.learn("hand_1", 1, self.history)
        second.learn("hand_1", 1, wire_history(response_hand("call").public_history))
        self.assertNotEqual(first.path, second.path)
        key = "preflop:bb:first:medium"
        self.assertEqual(first.profile.estimate(key)["counts"]["fold"], 1)
        self.assertEqual(second.profile.estimate(key)["counts"]["call"], 1)
        self.assertEqual(second.profile.estimate(key)["counts"]["fold"], 0)

    def test_room_hero_and_blinds_define_separate_scopes(self) -> None:
        original = self.open_memory()
        paths = {original.path}
        for overrides in ({"room_url": "https://www.pokernow.com/games/another_test_room"},
                          {"hero_id": "another_hero"}, {"small_blind": 2}, {"big_blind": 4}):
            memory = OpponentMemory(self.directory, identity(**overrides))
            self.addCleanup(memory.close)
            paths.add(memory.path)
        self.assertEqual(len(paths), 5)

    def test_conflicting_duplicate_is_rejected_without_rewriting_profile(self) -> None:
        memory = self.open_memory()
        memory.learn("hand_1", 1, self.history)
        expected = memory.summary()
        with self.assertRaisesRegex(ValueError, "completed_history_changed"):
            memory.learn("hand_1", 1, wire_history(response_hand("call").public_history))
        with self.assertRaisesRegex(ValueError, "completed_history_changed"):
            memory.learn("hand_1", 0, self.history)
        self.assertEqual(memory.summary(), expected)

    def test_failed_transaction_preserves_profile_and_receipt(self) -> None:
        memory = self.open_memory()
        expected = memory.summary()
        memory.db.execute("CREATE TRIGGER fail_update BEFORE UPDATE ON memory BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            memory.learn("hand_1", 1, self.history)
        self.assertEqual(memory.summary(), expected)
        self.assertFalse(memory.contains("hand_1"))
        self.assertEqual(read_opponent_profiles(self.directory)[0], expected)
        memory.db.execute("DROP TRIGGER fail_update")
        self.assertTrue(memory.learn("hand_1", 1, self.history))

    def test_single_writer_lock_and_read_only_inspection(self) -> None:
        memory = self.open_memory()
        memory.learn("hand_1", 1, self.history)
        before = memory.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already in use"):
            self.open_memory()
        report = read_opponent_profiles(self.directory)
        self.assertEqual(report, [memory.summary()])
        self.assertEqual(memory.path.read_bytes(), before)

    def test_read_only_report_filters_without_creating_files(self) -> None:
        missing = self.directory / "missing"
        self.assertEqual(read_opponent_profiles(missing), [])
        self.assertFalse(missing.exists())
        self.open_memory("a")
        self.open_memory("b")
        selected = read_opponent_profiles(self.directory, opponent_id="b")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["identity"]["opponent_id"], "b")
        self.assertEqual(read_opponent_profiles(self.directory, opponent_id="absent"), [])

    def test_nested_report_mutations_do_not_change_live_memory(self) -> None:
        memory = self.open_memory()
        memory.learn("hand_1", 1, self.history)
        key = "preflop:bb:first:medium"
        report = memory.summary()
        report["identity"]["opponent_id"] = "intruder"
        report["behavior"]["contexts"][key]["counts"]["fold"] = 999
        current = memory.summary()
        self.assertEqual(current["identity"]["opponent_id"], "opponent_a")
        self.assertEqual(current["behavior"]["contexts"][key]["counts"]["fold"], 1)

    def test_corrupt_fingerprint_is_rejected_by_reader_and_restart(self) -> None:
        memory = self.open_memory()
        data = json.loads(memory.db.execute("SELECT data FROM memory").fetchone()[0])
        data["profile_sha256"] = "0" * 64
        with memory.db:
            memory.db.execute("UPDATE memory SET data=?", (json.dumps(data),))
        memory.close()
        with self.assertRaises(ValueError):
            read_opponent_profiles(self.directory)
        with self.assertRaises(ValueError):
            self.open_memory()

    def test_missing_completed_receipt_is_rejected(self) -> None:
        memory = self.open_memory()
        memory.learn("hand_1", 1, self.history)
        with memory.db:
            memory.db.execute("DELETE FROM completed WHERE hand_id='hand_1'")
        memory.close()
        with self.assertRaises(ValueError):
            read_opponent_profiles(self.directory)
        with self.assertRaises(ValueError):
            self.open_memory()

    def test_memory_mode_cannot_silently_change(self) -> None:
        memory = self.open_memory(memory="recent")
        memory.learn("hand_1", 1, self.history)
        memory.close()
        with self.assertRaises(ValueError):
            self.open_memory(memory="lifetime")

    def test_population_prior_reloads_and_never_becomes_personal_evidence(self) -> None:
        pool = OpponentProfile()
        for _ in range(24):
            pool.observe_hand(response_hand().public_history, 0)
        artifact = {"schema_version": 1, "kind": "population-fold-prior", "strength": 8.0,
                    "population": pool.to_dict(), "population_sha256": pool.fingerprint}
        prior = self.directory / "prior.json"
        prior.write_text(json.dumps(artifact))
        memory = self.open_memory(population_prior=prior)
        self.assertEqual(memory.summary()["behavior"]["contexts"], {})
        memory.learn("hand_1", 1, wire_history(response_hand("call").public_history))
        expected = memory.summary()
        memory.close()
        prior.unlink()
        restored = self.open_memory()
        self.assertEqual(restored.summary(), expected)
        context = restored.summary()["behavior"]["contexts"]["preflop:bb:first:medium"]
        self.assertEqual(context["rates"], {"fold": 0.0, "call": 1.0, "raise": 0.0})
        self.assertFalse(context["ready"])
        self.assertEqual(restored.profile.to_dict()["population"], pool.to_dict())
