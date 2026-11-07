import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import shutil
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from scipy.stats import t

ROOT = Path(__file__).resolve().parent.parent
STYLES = ("calling", "tight", "aggressive", "heldout_fit_fold", "heldout_pressure", "heldout_loose")
DEFAULT_SEEDS = (20261107101, 20261107102, 20261107103)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": {name: version(name) for name in ("pokerkit", "eval7", "scipy", "pydantic")}}


def validate_resume(manifest: dict, previous: dict, source: Path, args: argparse.Namespace, config: dict) -> None:
    if previous["status"] not in ("incomplete", "failed"):
        raise ValueError("only an incomplete or failed campaign may resume")
    if (list(args.seeds) != manifest["seeds"] or list(STYLES) != manifest["styles"]
            or args.warmup != manifest["warmup_hands_per_stratum"]
            or args.pairs != manifest["matched_evaluation_units_per_stratum"]
            or args.samples != manifest["samples"] or config != manifest["config"]):
        raise ValueError("resume requires identical seeds, styles, learning/evaluation sizes, samples and game settings")
    if environment() != manifest["environment"]:
        raise ValueError("resume requires the original Python, platform and dependency versions")
    if any(digest(source / name) != value for name, value in manifest["source_sha256"].items()):
        raise ValueError("frozen source differs from the recorded campaign")


def interval(values: list[float], endpoints: int = 1) -> dict:
    n = len(values)
    mean = statistics.mean(values) if n else None
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    margin = float(t.ppf(1 - 0.05 / (2 * endpoints), n - 1)) * se if se is not None else None
    return {"independent_replicates": n, "mean_bb100": mean, "standard_error": se,
            "ci95": [mean - margin, mean + margin] if margin is not None else None,
            "adjustment_endpoints": endpoints, "replicate_values_bb100": values}


def run_stratum(task: dict) -> dict:
    sys.path.insert(0, task["source_directory"])
    from headsup.arena import seed_for
    from headsup.engine import Hand
    from headsup.learning import OpponentProfile
    from headsup.models import GameConfig
    from headsup.opponent_memory import OpponentIdentity, OpponentMemory
    from headsup.protocol import WireEvent
    from headsup.training import _play

    seed, style = task["seed"], task["style"]
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=False)
    config = GameConfig(**task["config"])
    identity = OpponentIdentity(room_url=f"https://www.pokernow.com/games/synthetic_research_{seed}",
                                hero_id="research_hero", opponent_id=style,
                                small_blind=config.small_blind, big_blind=config.big_blind)
    memory = OpponentMemory(output / "memory", identity)
    changed = {"learning": 0, "evaluation": 0}
    completed = {"learning": 0, "evaluation": 0}
    units = []
    phase, index, arm, leg = "learning", 0, "adaptive", 0
    hand, decisions, frozen = None, [], None
    try:
        with gzip.open(output / "hands.jsonl.gz", "xt", compresslevel=3) as journal:
            for index in range(task["warmup"]):
                leg = index % 2
                hand = Hand(config, seed_for(seed, "learning", style, index, "deck"))
                seeds = tuple(seed_for(seed, "learning", style, index, seat, "policy") for seat in (0, 1))
                decisions = []
                record = _play(hand, style, leg, seeds, task["samples"], memory.profile, decisions)
                journal.write(json.dumps({**record, "phase": phase, "index": index, "arm": arm}) + "\n")
                journal.flush()
                public = tuple(WireEvent.model_validate(asdict(event)) for event in hand.public_history)
                memory.learn(f"learning-{index}", leg, public)
                completed[phase] += 1
                changed[phase] += sum(bool(item["info"].get("adaptation_applied")) for item in decisions
                                      if item["actor"] == leg)
            frozen = memory.profile.to_dict()
            frozen_hash = memory.profile.fingerprint
            memory.close()
            memory = OpponentMemory(output / "memory", identity)
            if memory.profile.to_dict() != frozen or memory.profile.fingerprint != frozen_hash:
                raise ValueError("saved opponent profile changed after restart")
            write_json(output / "frozen-profile.json", frozen)
            phase = "evaluation"
            for index in range(task["pairs"]):
                returns = {"baseline": [], "adaptive": []}
                for arm in ("baseline", "adaptive"):
                    for leg in (0, 1):
                        hand = Hand(config, seed_for(seed, phase, style, index, "deck"))
                        seeds = tuple(seed_for(seed, phase, style, index, leg, seat, "policy") for seat in (0, 1))
                        profile = OpponentProfile.from_dict(frozen) if arm == "adaptive" else None
                        decisions = []
                        record = _play(hand, style, leg, seeds, task["samples"], profile, decisions)
                        journal.write(json.dumps({**record, "phase": phase, "index": index, "arm": arm}) + "\n")
                        journal.flush()
                        completed[phase] += 1
                        changed[phase] += sum(bool(item["info"].get("adaptation_applied")) for item in decisions
                                              if item["actor"] == leg)
                        returns[arm].append(hand.result().net_returns[leg])
                scale = 100 / (2 * config.big_blind)
                baseline, adaptive = sum(returns["baseline"]) * scale, sum(returns["adaptive"]) * scale
                units.append({"seed": seed, "style": style, "unit": index,
                              "baseline_bb100": baseline, "adaptive_bb100": adaptive,
                              "delta_bb100": adaptive - baseline})
            if memory.profile.to_dict() != frozen:
                raise ValueError("evaluation mutated the frozen learning profile")
        summary = memory.profile.summary()
        result = {"status": "completed", "seed": seed, "style": style, "completed_hands": completed,
                  "restart_verified": True, "evaluation_profile_unchanged": True,
                  "adapted_decisions": changed, "profile": summary, "units": units,
                  "baseline_mean_bb100": statistics.mean(unit["baseline_bb100"] for unit in units),
                  "adaptive_mean_bb100": statistics.mean(unit["adaptive_bb100"] for unit in units),
                  "delta_mean_bb100": statistics.mean(unit["delta_bb100"] for unit in units)}
    except (Exception, KeyboardInterrupt) as exc:
        result = {"status": "failed", "seed": seed, "style": style, "completed_hands": completed,
                  "error_type": type(exc).__name__, "error": str(exc), "phase": phase,
                  "index": index, "arm": arm, "leg": leg, "units": units}
        write_json(output / "failure.json", {**result, "hand": hand.snapshot() if hand else None,
                                             "decisions": decisions})
    finally:
        memory.close()
    write_json(output / "report.json", result)
    return result


def aggregate(results: list[dict], manifest: dict) -> dict:
    completed = all(item["status"] == "completed" for item in results) and len(results) == len(manifest["seeds"]) * len(STYLES)
    report = {"status": "completed" if completed else "incomplete", "strata": [],
              "completed_hands": sum(sum(item["completed_hands"].values()) for item in results),
              "learning_hands": sum(item["completed_hands"]["learning"] for item in results),
              "evaluation_hands": sum(item["completed_hands"]["evaluation"] for item in results),
              "matched_deck_units": sum(len(item["units"]) for item in results),
              "all_restarts_verified": completed and all(item["restart_verified"] for item in results),
              "all_evaluation_profiles_unchanged": completed and all(item["evaluation_profile_unchanged"] for item in results)}
    for item in sorted(results, key=lambda x: (x["seed"], STYLES.index(x["style"]))):
        report["strata"].append({key: value for key, value in item.items() if key not in ("units", "profile")})
    if not completed:
        return report
    grouped = {(item["seed"], item["style"]): item for item in results}
    values = [statistics.mean(grouped[(seed, style)]["delta_mean_bb100"] for style in STYLES)
              for seed in manifest["seeds"]]
    report["primary_equal_style_pool"] = interval(values)
    report["secondary_by_style"] = {
        style: interval([grouped[(seed, style)]["delta_mean_bb100"] for seed in manifest["seeds"]], len(STYLES))
        for style in STYLES
    }
    report["statistical_unit"] = "Independent training-seed replicate; each replicate averages equal-weight style strata and all matched evaluation units."
    report["primary_endpoint"] = "One prespecified primary endpoint, equal-style mean adaptive-minus-baseline net bb/100, unadjusted 95% Student-t interval across independent training seeds."
    report["secondary_endpoints"] = "Six style-specific differences; Bonferroni-adjusted 95% intervals across independent training seeds."
    report["limitations"] = [
        "Three default training replicates give only two degrees of freedom; normality is unverified and intervals are fragile.",
        "Synthetic opponents only. No human-player, live-site, or monetary-profit inference.",
        "Each unfamiliar style has its own warmup; this is adaptation, not zero-shot transfer.",
        "Fixed 100BB fresh stacks and one configured rake schedule per campaign do not cover tournament or stake dynamics.",
        "No population prior; individual lifetime learning and the adaptive policy match current defaults.",
        "The profile records calls and raises separately, but adaptive betting adjustments primarily use fold probabilities.",
    ]
    return report


def export_bundle(output: Path, destination: Path, manifest: dict, report: dict, results: list[dict]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "summary.json", report)
    with (destination / "paired-results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "style", "unit", "baseline_bb100", "adaptive_bb100", "delta_bb100"))
        writer.writeheader()
        for item in sorted(results, key=lambda value: (value["seed"], STYLES.index(value["style"]))):
            writer.writerows(item["units"])
    write_json(destination / "profiles.json", {
        str(item["seed"]): {other["style"]: json.loads((output / f"seed-{item['seed']}" / other["style"] / "frozen-profile.json").read_text())
                           for other in results if other["seed"] == item["seed"]}
        for item in results
    })
    write_json(destination / "checksums.json", {path.name: digest(path) for path in sorted(destination.iterdir()) if path.is_file()})


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Replicate frozen-profile opponent learning on matched held-out deals.")
    parser.add_argument("--output", type=Path, required=True, help="new local run directory, including raw replay records")
    parser.add_argument("--export", type=Path, help="new portable result directory; no real player memory or raw hands")
    parser.add_argument("--resume", action="store_true", help="retain complete strata and repeat only failed strata with identical source, seeds and settings")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--warmup", type=positive, default=600)
    parser.add_argument("--pairs", type=positive, default=128)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--jobs", type=positive, default=4)
    parser.add_argument("--rake-bps", type=int, default=0)
    parser.add_argument("--rake-cap", type=int, default=0)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("training seeds must be distinct")
    if args.samples > 8192 or not 0 <= args.rake_bps <= 10000 or args.rake_cap < 0:
        parser.error("invalid sample count or rake settings")
    output = args.output.resolve()
    if args.export:
        destination = args.export.resolve()
        if destination.exists():
            parser.error("export destination already exists; choose a new directory")
        if destination == output or destination in output.parents or output in destination.parents:
            parser.error("export destination must be separate from the raw run directory")
    source = output / "source"
    config = {"small_blind": 10, "big_blind": 20, "stack_bb": 100,
              "rake_bps": args.rake_bps, "rake_cap": args.rake_cap, "no_flop_no_drop": True}
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        previous = json.loads((output / "report.json").read_text())
        try:
            validate_resume(manifest, previous, source, args, config)
        except ValueError as exc:
            parser.error(str(exc))
        attempt_number = len(manifest.get("recovery_attempts", [])) + 1
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        shutil.copy2(output / "report.json", attempts / f"incomplete-report-{attempt_number}.json")
        shutil.copy2(output / "manifest.json", attempts / f"manifest-before-recovery-{attempt_number}.json")
        runner_name = f"run_research-recovery-{attempt_number}.py"
        shutil.copy2(Path(__file__), source / runner_name)
        manifest["source_sha256"][runner_name] = digest(source / runner_name)
        manifest.setdefault("recovery_attempts", []).append({
            "attempt": attempt_number, "created_utc": datetime.now(timezone.utc).isoformat(),
            "runner_file": runner_name, "runner_sha256": digest(source / runner_name),
            "previous_completed_hands": previous["completed_hands"],
            "protocol": "Completed strata retained; failed strata archived and repeated from their first learning hand with the same frozen policy source and seeds. No outcome-based parameter changes.",
        })
    else:
        output.mkdir(parents=True, exist_ok=False)
        source.mkdir()
        shutil.copytree(ROOT / "headsup", source / "headsup", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copy2(ROOT / name, source / name)
        shutil.copy2(Path(__file__), source / "run_research.py")
        manifest = {
        "schema_version": 1, "experiment": "independent_training_seed_replication",
        "created_utc": datetime.now(timezone.utc).isoformat(), "seeds": args.seeds,
        "styles": STYLES, "warmup_hands_per_stratum": args.warmup,
        "matched_evaluation_units_per_stratum": args.pairs, "hands_per_evaluation_unit": 4,
        "samples": args.samples, "jobs": args.jobs, "config": config,
        "planned_hands": len(args.seeds) * len(STYLES) * (args.warmup + 4 * args.pairs),
        "protocol": "Per style and seed: learn only completed public responses, persist each hand, restart and verify, freeze profile, compare range and adaptive on identical new decks and policy seeds in both seats. Evaluation never updates memory.",
        "selection_policy": "All seeds, styles, sample sizes and one primary endpoint are fixed before play. No default promotion, outcome-driven extension or selective export.",
        "source_sha256": {str(path.relative_to(source)): digest(path) for path in sorted(source.rglob("*")) if path.is_file()},
        "environment": environment(),
        }
    write_json(output / "manifest.json", manifest)
    tasks = [{"seed": seed, "style": style, "warmup": args.warmup, "pairs": args.pairs,
              "samples": args.samples, "config": config, "source_directory": str(source),
              "output": str(output / f"seed-{seed}" / style)} for seed in args.seeds for style in STYLES]
    results = []
    pending = []
    previous_by_identity = {(row["seed"], row["style"]): row for row in previous["strata"]} if args.resume else {}
    for task in tasks:
        path = Path(task["output"])
        if not args.resume:
            pending.append(task)
            continue
        previous_stratum = previous_by_identity.get((task["seed"], task["style"]))
        if previous_stratum and previous_stratum["status"] == "completed":
            saved = json.loads((path / "report.json").read_text())
            if saved["status"] != "completed" or len(saved["units"]) != args.pairs:
                raise ValueError("completed stratum no longer matches its original receipt")
            results.append(saved)
            continue
        if path.exists():
            path.rename(attempts / f"seed-{task['seed']}-{task['style']}-attempt-{attempt_number}")
        pending.append(task)
    if args.resume:
        manifest["recovery_attempts"][-1]["retained_completed_strata"] = len(results)
        manifest["recovery_attempts"][-1]["repeated_strata"] = [{"seed": task["seed"], "style": task["style"]} for task in pending]
        write_json(output / "manifest.json", manifest)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_stratum, task): task for task in pending}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                task = futures[future]
                result = {"status": "failed", "seed": task["seed"], "style": task["style"],
                          "completed_hands": {"learning": 0, "evaluation": 0}, "units": [],
                          "error_type": type(exc).__name__, "error": str(exc),
                          "scope": "Worker failed outside its hand journal; inspect retained raw files."}
            results.append(result)
            print(f"{result['status']}: seed {result['seed']} {result['style']} ({len(results)}/{len(tasks)})", flush=True)
    report = aggregate(results, manifest)
    report["frozen_source_unchanged"] = all(digest(source / name) == value for name, value in manifest["source_sha256"].items())
    if not report["frozen_source_unchanged"]:
        report["status"] = "invalidated"
    write_json(output / "report.json", report)
    if args.export and report["status"] == "completed":
        export_bundle(output, args.export, manifest, report, results)
    print(json.dumps({key: value for key, value in report.items() if key in ("status", "completed_hands", "primary_equal_style_pool")}, indent=2))
    return int(report["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
