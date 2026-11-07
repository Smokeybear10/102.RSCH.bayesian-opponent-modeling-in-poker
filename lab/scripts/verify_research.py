import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from headsup.learning import OpponentProfile


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10)


def verify_interval(actual: dict, values: list[float], endpoints: int) -> None:
    if actual["independent_replicates"] != len(values) or actual["adjustment_endpoints"] != endpoints:
        raise ValueError("replicate count or endpoint correction disagrees")
    if not close(actual["mean_bb100"], statistics.mean(values)):
        raise ValueError("reported mean disagrees with paired results")
    if len(actual["replicate_values_bb100"]) != len(values) or not all(
            close(a, b) for a, b in zip(actual["replicate_values_bb100"], values, strict=True)):
        raise ValueError("reported replicate values disagree")
    if len(values) == 1:
        if actual["ci95"] is not None or actual["standard_error"] is not None:
            raise ValueError("one replicate cannot supply a replicate-level interval")
        return
    se = statistics.stdev(values) / math.sqrt(len(values))
    margin = t.ppf(1 - .05 / (2 * endpoints), len(values) - 1) * se
    expected = (statistics.mean(values) - margin, statistics.mean(values) + margin)
    if not close(actual["standard_error"], se) or not all(close(a, b) for a, b in zip(actual["ci95"], expected, strict=True)):
        raise ValueError("reported uncertainty disagrees with independent replicate calculation")


def verify_bundle(directory: Path) -> dict:
    expected_files = {"manifest.json", "summary.json", "paired-results.csv", "profiles.json"}
    hashes = json.loads((directory / "checksums.json").read_text())
    if set(hashes) != expected_files:
        raise ValueError("bundle checksum manifest has unexpected files")
    for name, expected in hashes.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {name}")
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "summary.json").read_text())
    profiles = json.loads((directory / "profiles.json").read_text())
    if report["status"] != "completed" or not all(report[key] for key in (
            "all_restarts_verified", "all_evaluation_profiles_unchanged", "frozen_source_unchanged")):
        raise ValueError("incomplete or invalidated experiment")
    seeds, styles = manifest["seeds"], manifest["styles"]
    pairs, warmup = manifest["matched_evaluation_units_per_stratum"], manifest["warmup_hands_per_stratum"]
    if len(set(seeds)) != len(seeds) or len(set(styles)) != len(styles):
        raise ValueError("replicate seeds and styles must be distinct")
    expected_keys = {(seed, style, unit) for seed in seeds for style in styles for unit in range(pairs)}
    seen = set()
    groups = defaultdict(list)
    arms = defaultdict(lambda: {"baseline": [], "adaptive": []})
    with (directory / "paired-results.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["seed"]), row["style"], int(row["unit"]))
            if key in seen or key not in expected_keys:
                raise ValueError("duplicate or unexpected matched evaluation unit")
            seen.add(key)
            baseline, adaptive, delta = (float(row[name]) for name in ("baseline_bb100", "adaptive_bb100", "delta_bb100"))
            if not all(math.isfinite(value) for value in (baseline, adaptive, delta)) or not close(adaptive - baseline, delta):
                raise ValueError("invalid paired return")
            groups[key[:2]].append(delta)
            arms[key[:2]]["baseline"].append(baseline)
            arms[key[:2]]["adaptive"].append(adaptive)
    if seen != expected_keys:
        raise ValueError("missing paired evaluation units")
    if (report["matched_deck_units"] != len(seen) or report["evaluation_hands"] != 4 * len(seen)
            or report["learning_hands"] != len(seeds) * len(styles) * warmup
            or report["completed_hands"] != manifest["planned_hands"]
            or report["completed_hands"] != report["learning_hands"] + report["evaluation_hands"]):
        raise ValueError("reported hand accounting disagrees with design")
    strata = {(row["seed"], row["style"]): row for row in report["strata"]}
    if len(strata) != len(report["strata"]) or set(strata) != set(groups):
        raise ValueError("reported stratum identities disagree")
    for key, values in groups.items():
        row = strata[key]
        if (not close(row["delta_mean_bb100"], statistics.mean(values))
                or not close(row["baseline_mean_bb100"], statistics.mean(arms[key]["baseline"]))
                or not close(row["adaptive_mean_bb100"], statistics.mean(arms[key]["adaptive"]))):
            raise ValueError("stratum outcomes disagree with raw units")
    primary = [statistics.mean(statistics.mean(groups[(seed, style)]) for style in styles) for seed in seeds]
    verify_interval(report["primary_equal_style_pool"], primary, 1)
    if set(report["secondary_by_style"]) != set(styles):
        raise ValueError("secondary endpoints disagree")
    for style in styles:
        values = [statistics.mean(groups[(seed, style)]) for seed in seeds]
        verify_interval(report["secondary_by_style"][style], values, len(styles))
    if set(profiles) != {str(seed) for seed in seeds}:
        raise ValueError("frozen profile replicates disagree")
    for seed in seeds:
        if set(profiles[str(seed)]) != set(styles):
            raise ValueError("frozen profile style identities disagree")
        for style in styles:
            if OpponentProfile.from_dict(profiles[str(seed)][style]).hands_observed != warmup:
                raise ValueError("frozen profile hand count disagrees")
    return {"status": "passed", "replicates": len(seeds), "styles": len(styles),
            "paired_units_verified": len(seen), "frozen_profiles_verified": len(seeds) * len(styles),
            "scope": "Portable artifact integrity and recomputed statistics. Full action/policy replay requires local raw journals from the reproduction command."}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify portable research artifacts and recompute their statistics.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = verify_bundle(args.directory)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
