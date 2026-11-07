import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

from scipy.stats import t

from .arena import environment, seed_for, summarize, timing_summary, write_json
from .engine import Hand
from .learning import OpponentProfile
from .models import MAX_SAMPLES, GameConfig
from .opponents import HELDOUT_NAMES, RIVER_SIZING_NAMES, make_opponent
from .policies import make_adaptive_policy, make_policy

TRAINING_OPPONENTS = ("calling", "tight", "aggressive")
MODEL_VERSION = "public-fold-response-v2"


def save_checkpoint(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    write_json(temporary, data)
    temporary.replace(path)


def load_checkpoint(path: Path, config: GameConfig) -> dict:
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or data.get("model_version") != MODEL_VERSION):
        raise ValueError("unsupported learning checkpoint schema or model version")
    if data.get("status") not in ("training", "completed", "incomplete", "failed"):
        raise ValueError("checkpoint is invalidated or has an unsupported status")
    if data.get("source_environment") != environment():
        raise ValueError("checkpoint source/environment differs; automatic resume requires the same implementation")
    if data.get("config") != asdict(config):
        raise ValueError("checkpoint game/rake configuration does not match this run")
    if not isinstance(data.get("profiles"), dict) or set(data["profiles"]) != set(TRAINING_OPPONENTS):
        raise ValueError("checkpoint must contain the expected separate training opponents")
    seeds = data.get("training_seeds")
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("checkpoint training seeds must be unique integers")
    for profile in data["profiles"].values():
        OpponentProfile.from_dict(profile)
    return data


def _opponent(name: str, seed: int, samples: int):
    return (make_opponent(name, seed, samples) if name in (*HELDOUT_NAMES, *RIVER_SIZING_NAMES)
            else make_policy(name, seed, samples))


def _play(hand: Hand, opponent: str, hero_seat: int, seeds: tuple[int, int], samples: int,
          profile: OpponentProfile | None, decisions: list[dict]) -> dict:
    snapshot = profile.to_dict() if profile is not None else None
    hero = (make_adaptive_policy(seeds[hero_seat], samples, profile) if profile is not None
            else make_policy("range", seeds[hero_seat], samples))
    other = _opponent(opponent, seeds[1 - hero_seat], samples)
    policies = (hero, other) if hero_seat == 0 else (other, hero)
    names = ("adaptive" if profile is not None else "range", opponent)
    if hero_seat == 1:
        names = names[::-1]
    while not hand.finished:
        observation = hand.observation()
        policy = policies[observation.actor]
        decision = {"actor": observation.actor, "street": observation.street, "accepted": False}
        decisions.append(decision)
        started = perf_counter_ns()
        try:
            action = policy.act(observation)
        finally:
            decision["latency_ns"] = perf_counter_ns() - started
        decision.update(action=asdict(action), info=policy.last_info)
        hand.act(action)
        decision["accepted"] = True
    profiles = [None, None]
    profiles[hero_seat] = snapshot
    return {**hand.record(), "policies": names, "policy_seeds": seeds, "samples": samples,
            "policy_profiles": profiles, "decisions": decisions, "hero_seat": hero_seat}


def _comparison(values: list[float], endpoints: int) -> dict:
    result = summarize(values)
    interval = None
    if len(values) >= 2:
        margin = float(t.ppf(1 - 0.05 / (2 * endpoints), len(values) - 1)) * result["standard_error"]
        interval = [result["mean_bb100"] - margin, result["mean_bb100"] + margin]
    return {**result, "adjusted_ci95": interval,
            "positive_adjusted_lower": interval is not None and interval[0] > 0,
            "negative_adjusted_upper": interval is not None and interval[1] < 0}


def train(output: Path, config: GameConfig, hands: int = 600, warmup: int = 600,
          pairs: int = 128, seed: int = 20261106, samples: int = 256,
          resume: Path | None = None, progress: Callable[[str], None] | None = None,
          memory: str = "lifetime") -> dict:
    for name, value in (("hands", hands), ("warmup", warmup), ("pairs", pairs), ("samples", samples)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if samples > MAX_SAMPLES or type(seed) is not int:
        raise ValueError("samples exceeds the limit or seed is not an integer")
    if memory not in ("lifetime", "recent"):
        raise ValueError("memory must be lifetime or recent")
    parent = load_checkpoint(resume, config) if resume else None
    used_seeds = parent["training_seeds"] if parent else []
    if seed in used_seeds:
        raise ValueError("resume requires a new --seed; reusing training deals would double-count evidence")
    profiles = {name: OpponentProfile.from_dict(parent["profiles"][name]) if parent else OpponentProfile(memory)
                for name in TRAINING_OPPONENTS}
    if any(profile.memory != memory for profile in profiles.values()):
        raise ValueError("--memory must match the resumed checkpoint; old counts cannot reconstruct recent history")
    profiles.update({name: OpponentProfile(memory) for name in HELDOUT_NAMES})
    opponents = (*TRAINING_OPPONENTS, *HELDOUT_NAMES)
    endpoints = len(opponents) + 3
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "mode": "automatic_opponent_learning", "model_version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "training_opponents": TRAINING_OPPONENTS, "heldout_styles": HELDOUT_NAMES, "memory": memory,
        "training_hands_per_opponent": hands, "heldout_warmup_hands_per_opponent": warmup,
        "evaluation_pairs_per_opponent": pairs, "seed": seed, "samples": samples,
        "environment": environment(),
        "parent_checkpoint": {"path": str(resume.resolve()), "sha256": hashlib.sha256(resume.read_bytes()).hexdigest()} if resume else None,
        "comparison_endpoints": endpoints,
        "evaluation_protocol": "Freeze profiles after training/warmup; four games per deck: both seats for range and adaptive. Identical deck and policy seeds across arms. No evaluation updates.",
        "statistical_unit": "Difference of two seat-averaged net returns on the same deck; conditional on the frozen profile. Aggregate blocks average the same pair index across independent opponent strata.",
        "selection_policy": "Save every candidate; never automatically replace the default or select on repeated test results.",
    }
    write_json(output / "manifest.json", manifest)
    checkpoint = {
        "schema_version": 1, "model_version": MODEL_VERSION, "config": asdict(config),
        "status": "training",
        "training_seeds": [*used_seeds, seed], "profiles": {},
        "source_environment": manifest["environment"], "parent_checkpoint": manifest["parent_checkpoint"],
    }

    def persist() -> None:
        checkpoint["profiles"] = {name: profiles[name].to_dict() for name in TRAINING_OPPONENTS}
        save_checkpoint(output / "checkpoint.json", checkpoint)
        save_checkpoint(output / "heldout-warmup.json", {name: profiles[name].to_dict() for name in HELDOUT_NAMES})

    persist()
    baseline = {name: [] for name in opponents}
    candidate = {name: [] for name in opponents}
    differences = {name: [] for name in opponents}
    profile_before = {name: profile.to_dict() for name, profile in profiles.items()}
    frozen = None
    completed = {"training": 0, "heldout_warmup": 0, "evaluation": 0}
    changed_actions = {"training": 0, "heldout_warmup": 0, "evaluation": 0}
    hero_times = []
    phase, opponent, index, leg, arm = "initializing", None, None, None, None
    hand, decisions, failure = None, [], None

    def save_hand(journal, record: dict) -> None:
        record.update(phase=phase, opponent=opponent, index=index, leg=leg, arm=arm)
        journal.write(json.dumps(record, allow_nan=False) + "\n")
        journal.flush()
        completed[phase] += 1
        hero_decisions = [item for item in record["decisions"] if item["actor"] == record["hero_seat"]]
        hero_times.extend(item["latency_ns"] for item in hero_decisions)
        changed_actions[phase] += sum(bool(item["info"].get("adaptation_applied")) for item in hero_decisions)

    try:
        with (output / "hands.jsonl").open("x") as journal:
            for phase, names, count in (("training", TRAINING_OPPONENTS, hands),
                                        ("heldout_warmup", HELDOUT_NAMES, warmup)):
                for opponent in names:
                    for index in range(count):
                        hand, decisions = None, []
                        leg, arm = index % 2, "adaptive"
                        hand = Hand(config, seed_for(seed, phase, opponent, index, "deck"))
                        seeds = tuple(seed_for(seed, phase, opponent, index, seat, "policy") for seat in (0, 1))
                        record = _play(hand, opponent, leg, seeds, samples, profiles[opponent], decisions)
                        save_hand(journal, record)
                        profiles[opponent].observe_hand(hand.public_history, 1 - leg)
                        persist()
                        if progress and ((index + 1) % 100 == 0 or index + 1 == count):
                            progress(f"{phase}: {opponent} {index + 1}/{count}")
            frozen = {name: profile.to_dict() for name, profile in profiles.items()}
            write_json(output / "frozen-profiles.json", frozen)
            phase = "evaluation"
            for opponent in opponents:
                for index in range(pairs):
                    returns = {"baseline": [], "adaptive": []}
                    for arm in ("baseline", "adaptive"):
                        for leg in (0, 1):
                            hand, decisions = None, []
                            hand = Hand(config, seed_for(seed, phase, opponent, index, "deck"))
                            seeds = tuple(seed_for(seed, phase, opponent, index, leg, seat, "policy") for seat in (0, 1))
                            snapshot = OpponentProfile.from_dict(frozen[opponent]) if arm == "adaptive" else None
                            record = _play(hand, opponent, leg, seeds, samples, snapshot, decisions)
                            save_hand(journal, record)
                            returns[arm].append(hand.result().net_returns[leg])
                    scale = 100 / (2 * config.big_blind)
                    b, a = sum(returns["baseline"]) * scale, sum(returns["adaptive"]) * scale
                    baseline[opponent].append(b)
                    candidate[opponent].append(a)
                    differences[opponent].append(a - b)
                if progress:
                    progress(f"evaluation: {opponent} {pairs}/{pairs} matched comparisons")
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc), "phase": phase,
                   "opponent": opponent, "index": index, "leg": leg, "arm": arm,
                   "hand": hand.snapshot() if hand else None, "decisions": decisions}
        write_json(output / "failure.json", failure)
    final_environment = environment()
    source_unchanged = final_environment == manifest["environment"]
    profiles_unchanged = frozen is None or frozen == {name: p.to_dict() for name, p in profiles.items()}
    if not source_unchanged or not profiles_unchanged:
        write_json(output / "provenance-failure.json", {"environment_before": manifest["environment"],
                   "environment_after": final_environment, "evaluation_profiles_unchanged": profiles_unchanged})
    by_opponent = {name: {
        "group": "training_opponent" if name in TRAINING_OPPONENTS else "heldout_style_after_separate_warmup",
        "baseline_net": summarize(baseline[name]), "adaptive_net": summarize(candidate[name]),
        "adaptive_minus_baseline": _comparison(differences[name], endpoints),
        "paired_delta_bb100": differences[name],
        "baseline_pair_bb100": baseline[name], "adaptive_pair_bb100": candidate[name],
        "profile": profiles[name].summary(),
    } for name in opponents}
    aggregate = {}
    for group, names in (("all", opponents), ("training_opponents", TRAINING_OPPONENTS), ("heldout_after_warmup", HELDOUT_NAMES)):
        count = min(len(differences[name]) for name in names)
        blocks = [sum(differences[name][i] for name in names) / len(names) for i in range(count)]
        aggregate[group] = {**_comparison(blocks, endpoints), "paired_delta_blocks_bb100": blocks}
    status = ("invalidated" if not source_unchanged or not profiles_unchanged
              else "incomplete" if failure and failure["error_type"] == "KeyboardInterrupt"
              else "failed" if failure else "completed")
    report = {
        "status": status, "completed_hands": completed,
        "complete_evaluation_units": sum(len(v) for v in differences.values()),
        "incomplete_evaluation_hands_excluded": completed["evaluation"] - 4 * sum(len(v) for v in differences.values()),
        "adapted_decisions": changed_actions, "hero_timing": timing_summary(hero_times),
        "by_opponent": by_opponent, "aggregate_lift": aggregate,
        "profiles_before": profile_before, "profiles_after": {name: p.to_dict() for name, p in profiles.items()},
        "source_environment_unchanged": source_unchanged, "evaluation_profiles_unchanged": profiles_unchanged,
        "environment_after": final_environment, "run_failures": int(failure is not None or status == "invalidated"),
        "failure": failure,
        "positive_lift_evidence_on_tested_pool": status == "completed" and aggregate["all"]["positive_adjusted_lower"],
        "default_policy_replaced": False, "real_money_readiness_established": False,
        "interval_method": f"Approximate Student t over frozen-profile matched differences; Bonferroni across {endpoints} lift endpoints. Absolute net-return intervals are descriptive and unadjusted.",
        "limitations": [
            "Learning means public opponent fold-response estimates, not full-game neural self-play or equilibrium solving.",
            "Uncertainty is conditional on this checkpoint; it excludes variability across independently trained replicas.",
            "Held-out styles receive separate warmup; this measures adaptation to new styles, not zero-shot transfer.",
            "Synthetic opponents and generic rake do not establish profitability against real players.",
            "More training is not guaranteed to improve performance; each run preserves negative and inconclusive results.",
            "Resuming adds training against known opponents; it is not resuming an interrupted evaluation or independent replication.",
        ],
    }
    checkpoint["status"] = status
    save_checkpoint(output / "checkpoint.json", checkpoint)
    write_json(output / "report.json", report)
    if progress:
        progress(f"{status}: saved checkpoint and report to {output}")
    return report
