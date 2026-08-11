import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from scripts import run_research
from scripts.audit_research_run import verify_policy_assignment
from scripts.verify_research import verify_interval


class ResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source"
        self.source.mkdir()
        (self.source / "policy.py").write_text("frozen policy source\n")
        self.args = argparse.Namespace(seeds=[123], warmup=8, pairs=4, samples=32)
        self.config = {"small_blind": 10, "big_blind": 20, "stack_bb": 100,
                       "rake_bps": 0, "rake_cap": 0, "no_flop_no_drop": True}
        self.environment = {"python": "3.12.0", "platform": "test-platform",
                            "dependencies": {"scipy": "1.0"}}
        self.manifest = {"seeds": [123], "styles": list(run_research.STYLES),
                         "warmup_hands_per_stratum": 8, "matched_evaluation_units_per_stratum": 4,
                         "samples": 32, "config": self.config, "environment": self.environment,
                         "source_sha256": {"policy.py": run_research.digest(self.source / "policy.py")}}
        self.previous = {"status": "incomplete", "strata": [], "completed_hands": 0}

    def test_resume_accepts_matching_frozen_source_settings_and_environment(self) -> None:
        with patch.object(run_research, "environment", return_value=deepcopy(self.environment)):
            run_research.validate_resume(self.manifest, self.previous, self.source, self.args, self.config)

    def test_resume_rejects_each_environment_drift_before_mutation(self) -> None:
        for field in ("python", "platform", "dependencies"):
            changed = deepcopy(self.environment)
            changed[field] = {"scipy": "2.0"} if field == "dependencies" else "changed"
            with self.subTest(field=field), patch.object(run_research, "environment", return_value=changed):
                with self.assertRaisesRegex(ValueError, "original Python, platform and dependency"):
                    run_research.validate_resume(self.manifest, self.previous, self.source, self.args, self.config)
        self.assertEqual(sorted(path.name for path in self.directory.iterdir()), ["source"])

    def test_resume_cli_environment_failure_preserves_original_receipts(self) -> None:
        run_research.write_json(self.directory / "manifest.json", self.manifest)
        run_research.write_json(self.directory / "report.json", self.previous)
        original = {path.name: path.read_bytes() for path in self.directory.glob("*.json")}
        argv = ["run_research.py", "--output", str(self.directory), "--resume", "--seeds", "123",
                "--warmup", "8", "--pairs", "4", "--samples", "32"]
        with patch("sys.argv", argv), patch.object(run_research, "environment", return_value={}), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                run_research.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.directory / "attempts").exists())
        self.assertEqual({path.name: path.read_bytes() for path in self.directory.glob("*.json")}, original)

    def test_export_collision_fails_before_creating_run(self) -> None:
        output = self.directory / "new-run"
        argv = ["run_research.py", "--output", str(output), "--export", str(self.source)]
        with patch("sys.argv", argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                run_research.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())

    def test_policy_assignment_checks_style_arm_and_seat(self) -> None:
        for arm, hero in (("baseline", "range"), ("adaptive", "adaptive")):
            for seat in (0, 1):
                names = [hero, "calling"] if seat == 0 else ["calling", hero]
                record = {"arm": arm, "hero_seat": seat, "policies": names}
                with self.subTest(arm=arm, seat=seat):
                    verify_policy_assignment(record, "calling")
                    with self.assertRaisesRegex(ValueError, "declared arm, style and seat"):
                        verify_policy_assignment(record, "tight")
                    with self.assertRaises(ValueError):
                        verify_policy_assignment({**record, "policies": names[::-1]}, "calling")
                    with self.assertRaises(ValueError):
                        verify_policy_assignment({**record, "arm": "adaptive" if arm == "baseline" else "baseline"}, "calling")

    def test_single_replicate_values_are_verified_even_without_interval(self) -> None:
        estimate = {"independent_replicates": 1, "adjustment_endpoints": 1,
                    "mean_bb100": 1.0, "ci95": None, "standard_error": None,
                    "replicate_values_bb100": [1.0]}
        verify_interval(estimate, [1.0], 1)
        with self.assertRaisesRegex(ValueError, "replicate values disagree"):
            verify_interval({**estimate, "replicate_values_bb100": [999.0]}, [1.0], 1)


if __name__ == "__main__":
    unittest.main()
