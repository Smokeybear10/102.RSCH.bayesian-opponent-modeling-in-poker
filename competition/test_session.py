import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

from .session import CONTRACT_FIELDS, main, start, status


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "supplied"
        self.source.mkdir()
        self.marker = self.root / "executed"
        self.agent = (
            "from pathlib import Path\n"
            f"Path({str(self.marker)!r}).write_text('executed')\n"
            "class Agent:\n"
            "    def act(self, observation: dict) -> dict:\n"
            "        return {'action': 'check'}\n"
        )
        (self.source / "agent.py").write_text(self.agent)
        (self.source / "rules.md").write_text("The revealed variant and payout rules will apply.\n")
        self.output = self.root / "day" / "session"

    def test_start_preserves_source_and_baseline_and_never_executes_starter(self) -> None:
        (self.source / ".settings").mkdir()
        (self.source / ".settings" / "sdk.data").write_bytes(b"\x00\xff")
        (self.source / "dist").mkdir()
        (self.source / "dist" / "required.bin").write_bytes(b"required")
        (self.source / ".venv").mkdir()
        (self.source / ".venv" / "unneeded.txt").write_text("local runtime")
        (self.source / ".env.example").write_text("TOKEN=\n")
        original = (self.source / "agent.py").read_bytes()

        report = start(self.source, self.output, sys.executable)

        self.assertTrue(report["baseline_intact"])
        self.assertFalse(report["ready"])
        self.assertFalse(report["contract_complete"])
        self.assertEqual(set(report["unresolved_contract_fields"]), set(CONTRACT_FIELDS))
        self.assertEqual(report["candidate_changes"], {"added": [], "removed": [], "modified": []})
        self.assertIn(".venv/", report["excluded_paths"])
        self.assertFalse(self.marker.exists())
        self.assertEqual((self.source / "agent.py").read_bytes(), original)
        for tree in ("starter", "candidate"):
            self.assertEqual((self.output / tree / "agent.py").read_bytes(), original)
            self.assertEqual((self.output / tree / ".settings" / "sdk.data").read_bytes(), b"\x00\xff")
            self.assertTrue((self.output / tree / "dist" / "required.bin").exists())
            self.assertTrue((self.output / tree / ".env.example").exists())
            self.assertFalse((self.output / tree / ".venv").exists())
        contract = json.loads((self.output / "contract.json").read_text())
        self.assertEqual(contract["fields"]["variant"]["status"], "unresolved")
        self.assertTrue(contract["fields"]["variant"]["candidate_evidence"])
        self.assertEqual(contract["fields"]["interface"]["candidate_evidence"][0]["class"], "Agent")
        intake = json.loads((self.output / "intake.json").read_text())
        self.assertEqual(intake["starter"], str(self.output.resolve() / "starter"))
        self.assertFalse(intake["executed_starter_code"])
        manifest = json.loads((self.output / "session.json").read_text())
        self.assertIn("sha256", manifest["baseline_inventory"]["agent.py"])
        self.assertIn("official simulator", (self.output / "AGENT_HANDOFF.md").read_text())

    def test_status_distinguishes_candidate_changes_from_baseline_tampering(self) -> None:
        start(self.source, self.output)
        (self.output / "candidate" / "agent.py").write_text("# improved\n")
        (self.output / "candidate" / "new.txt").write_text("new")
        (self.output / "candidate" / "rules.md").unlink()
        report = status(self.output)
        self.assertTrue(report["baseline_intact"])
        self.assertEqual(report["candidate_changes"], {
            "added": ["new.txt"], "removed": ["rules.md"], "modified": ["agent.py"],
        })
        (self.output / "starter" / "agent.py").write_text("# changed baseline\n")
        report = status(self.output)
        self.assertFalse(report["baseline_intact"])
        self.assertEqual(report["baseline_changes"]["modified"], ["agent.py"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["status", str(self.output)]), 1)
        self.assertEqual((self.source / "agent.py").read_text(), self.agent)
        self.assertFalse(self.marker.exists())

    def test_contract_completion_is_not_submission_readiness(self) -> None:
        start(self.source, self.output)
        path = self.output / "contract.json"
        contract = json.loads(path.read_text())
        for field in contract["fields"].values():
            field.update(status="resolved", value="explicitly checked", evidence=["rules.md:1"])
        path.write_text(json.dumps(contract))
        report = status(self.output)
        self.assertTrue(report["contract_complete"])
        self.assertFalse(report["ready"])
        contract["fields"]["variant"]["evidence"] = []
        path.write_text(json.dumps(contract))
        self.assertEqual(status(self.output)["unresolved_contract_fields"], ["variant"])

    def test_refuses_existing_output_and_source_nested_output(self) -> None:
        self.output.mkdir(parents=True)
        marker = self.output / "keep.txt"
        marker.write_text("existing")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            start(self.source, self.output)
        self.assertEqual(marker.read_text(), "existing")
        with self.assertRaisesRegex(ValueError, "outside"):
            start(self.source, self.source / "session")
        self.assertFalse((self.source / "session").exists())

    def test_refuses_source_symlinks_secrets_and_special_files(self) -> None:
        unsafe = self.source / "linked.py"
        unsafe.symlink_to(self.source / "agent.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            start(self.source, self.output)
        unsafe.unlink()
        secret = self.source / ".env.local"
        secret.write_text("TOKEN=not-a-real-secret")
        with self.assertRaisesRegex(ValueError, "secret"):
            start(self.source, self.output)
        secret.unlink()
        if hasattr(os, "mkfifo"):
            os.mkfifo(self.source / "pipe")
            with self.assertRaisesRegex(ValueError, "special"):
                start(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".session-*")), [])

    def _archive(self, records) -> Path:
        archive = self.root / "starter.zip"
        with warnings.catch_warnings(), zipfile.ZipFile(archive, "w") as output:
            warnings.simplefilter("ignore", UserWarning)
            for name, data in records:
                output.writestr(name, data)
        return archive

    def test_zip_start_keeps_archive_root_and_does_not_run_code(self) -> None:
        archive = self._archive([
            ("official/", b""), ("official/agent.py", self.agent),
            ("official/data.bin", b"\x00\xff"),
        ])
        before = archive.read_bytes()
        report = start(archive, self.output)
        self.assertTrue(report["baseline_intact"])
        self.assertEqual((self.output / "candidate" / "official" / "agent.py").read_text(), self.agent)
        self.assertEqual(archive.read_bytes(), before)
        self.assertFalse(self.marker.exists())

    def test_zip_rejects_unsafe_duplicate_and_colliding_paths_atomically(self) -> None:
        cases = [
            [("../escape.py", b"bad")],
            [("/absolute.py", b"bad")],
            [("C:/drive.py", b"bad")],
            [("safe\\escape.py", b"bad")],
            [("a/./b.py", b"bad")],
            [("a//b.py", b"bad")],
            [("a.py", b"first"), ("a.py", b"second")],
            [("A.py", b"first"), ("a.py", b"second")],
            [("A/x.py", b"first"), ("a/y.py", b"second")],
            [("a", b"file"), ("a/b.py", b"child")],
            [(".env", b"secret")],
        ]
        for records in cases:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    start(self._archive(records), self.output)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.output.parent.glob(".session-*")), [])
        self.assertFalse((self.root / "escape.py").exists())

    def test_zip_rejects_symlinks_special_files_and_resource_excess(self) -> None:
        for file_type in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR):
            with self.subTest(file_type=file_type):
                member = zipfile.ZipInfo("unsafe")
                member.create_system = 3
                member.external_attr = (file_type | 0o644) << 16
                with self.assertRaisesRegex(ValueError, "special file"):
                    start(self._archive([(member, "target")]), self.output)
        for setting, limit, records in (
            ("MAX_ENTRIES", 1, [("a", b"a"), ("b", b"b")]),
            ("MAX_MEMBER_BYTES", 4, [("a", b"12345")]),
            ("MAX_TOTAL_BYTES", 4, [("a", b"123"), ("b", b"456")]),
        ):
            with self.subTest(setting=setting), patch(f"competition.session.{setting}", limit):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    start(self._archive(records), self.output)
                self.assertFalse(self.output.exists())

    def test_cli_pending_contract_is_success_and_missing_baseline_is_failure(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["start", str(self.source), "--output", str(self.output)]), 0)
            self.assertEqual(main(["status", str(self.output)]), 0)
        (self.output / "starter").rename(self.output / "baseline-moved")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["status", str(self.output)]), 1)
        report = status(self.output)
        self.assertFalse(report["baseline_intact"])
        self.assertTrue(report["baseline_errors"])
        self.assertFalse(self.marker.exists())

    def test_invalid_session_json_reports_a_controlled_cli_error(self) -> None:
        start(self.source, self.output)
        for filename in ("session.json", "contract.json"):
            path = self.output / filename
            original = path.read_text()
            for value in ([], None):
                with self.subTest(filename=filename, value=value):
                    path.write_text(json.dumps(value))
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        self.assertEqual(main(["status", str(self.output)]), 1)
                    self.assertIn("malformed", errors.getvalue())
            path.write_text(original)


if __name__ == "__main__":
    unittest.main()
