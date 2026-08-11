import hashlib
import json
import math
import platform
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

from scipy.stats import t

from .engine import Hand
from .models import MAX_SAMPLES, Action, GameConfig, HandResult
from .policies import POLICY_NAMES, make_policy


def seed_for(master: int, *labels: object) -> int:
    data = json.dumps([master, *labels], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def environment() -> dict:
    root = Path(__file__).resolve().parent.parent
    files = [*sorted((root / "headsup").glob("*.py")), root / "pyproject.toml", root / "uv.lock"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in (
            "pokerkit", "eval7", "scipy", "fastapi", "pydantic", "uvicorn", "httpx",
        )},
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if path.is_file()
        },
    }


def play_hand(hand: Hand, names: tuple[str, str], seeds: tuple[int, int], samples: int,
              decisions: list[dict] | None = None) -> dict:
    policies = [make_policy(name, seed, samples) for name, seed in zip(names, seeds)]
    decisions = decisions if decisions is not None else []
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
    return {**hand.record(), "policies": names, "policy_seeds": seeds,
            "samples": samples, "decisions": decisions}


def summarize(values: list[float]) -> dict:
    n = len(values)
    if not n:
        return {"pairs": 0, "mean_bb100": None, "standard_error": None, "ci95": None}
    mean = statistics.mean(values)
    if n < 2:
        return {"pairs": n, "mean_bb100": mean, "standard_error": None, "ci95": None}
    se = statistics.stdev(values) / math.sqrt(n)
    margin = float(t.ppf(0.975, n - 1)) * se
    return {"pairs": n, "mean_bb100": mean, "standard_error": se,
            "ci95": [mean - margin, mean + margin]}


def combined_summary(groups: dict[str, list[float]], weights: dict[str, float]) -> dict:
    stats = {name: summarize(values) for name, values in groups.items()}
    if any(item["pairs"] == 0 for item in stats.values()):
        return {"mean_bb100": None, "standard_error": None, "ci95": None}
    mean = sum(weights[name] * item["mean_bb100"] for name, item in stats.items())
    if any(item["standard_error"] is None for item in stats.values()):
        return {"mean_bb100": mean, "standard_error": None, "ci95": None}
    terms = {name: (weights[name] * item["standard_error"]) ** 2 for name, item in stats.items()}
    variance = sum(terms.values())
    denominator = sum(terms[name] ** 2 / (item["pairs"] - 1) for name, item in stats.items())
    df = variance ** 2 / denominator if denominator else None
    margin = float(t.ppf(0.975, df)) * math.sqrt(variance) if df else 0.0
    return {"mean_bb100": mean, "standard_error": math.sqrt(variance),
            "ci95": [mean - margin, mean + margin], "welch_degrees_of_freedom": df}


def timing_summary(values: list[int]) -> dict:
    if not values:
        return {"decisions": 0, "max_ms": None, "p50_ms": None, "p95_ms": None, "over_500ms": 0}
    ordered = sorted(values)
    return {"decisions": len(values), "max_ms": ordered[-1] / 1e6,
            "p50_ms": statistics.median(ordered) / 1e6,
            "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1] / 1e6,
            "over_500ms": sum(value > 500_000_000 for value in values),
            "boundary": "Policy.act only; excludes engine transitions, imports and initialization."}


def benchmark(output: Path, config: GameConfig, hero: str, opponents: tuple[str, ...],
              pairs: int, seed: int, samples: int) -> dict:
    if hero not in POLICY_NAMES or not opponents or any(name not in POLICY_NAMES for name in opponents):
        raise ValueError("choose supported hero and opponent policy names")
    if len(set(opponents)) != len(opponents):
        raise ValueError("opponents must be unique; weights are equal per named opponent")
    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be an integer from 1 to {MAX_SAMPLES}")
    output.mkdir(parents=True, exist_ok=False)
    weights = {name: 1 / len(opponents) for name in opponents}
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "local_simulation", "config": asdict(config), "hero": hero,
                "opponents": opponents, "opponent_weights": weights, "pairs_per_opponent": pairs,
                "seed": seed, "samples": samples, "environment": environment(),
                "scope": "Fixed synthetic opponents and seed campaign; no human-profitability inference."}
    write_json(output / "manifest.json", manifest)
    gross = {name: [] for name in opponents}
    net = {name: [] for name in opponents}
    timings = []
    all_timings = []
    total_rake = 0
    completed_hands = 0
    failure = None
    hand = None
    opponent = None
    pair = None
    leg = None
    active_decisions = []
    active_timings_accounted = False
    try:
        with (output / "hands.jsonl").open("x") as journal:
            for opponent in opponents:
                for pair in range(pairs):
                    pair_gross, pair_net = [], []
                    deck_seed = seed_for(seed, "deck", opponent, pair)
                    for leg in (0, 1):
                        active_decisions = []
                        active_timings_accounted = False
                        hero_seat = leg
                        hand = None
                        hand = Hand(config, deck_seed)
                        names = (hero, opponent) if leg == 0 else (opponent, hero)
                        policy_seeds = tuple(seed_for(seed, "policy", opponent, pair, leg, seat) for seat in (0, 1))
                        record = play_hand(hand, names, policy_seeds, samples, active_decisions)
                        record.update(opponent=opponent, pair=pair, leg=leg, hero_seat=hero_seat)
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        completed_hands += 1
                        result = hand.result()
                        total_rake += result.rake
                        pair_gross.append(result.gross_returns[hero_seat])
                        pair_net.append(result.net_returns[hero_seat])
                        all_timings.extend(item["latency_ns"] for item in record["decisions"])
                        timings.extend(item["latency_ns"] for item in record["decisions"] if item["actor"] == hero_seat)
                        active_timings_accounted = True
                    gross[opponent].append(sum(pair_gross) * 100 / (2 * config.big_blind))
                    net[opponent].append(sum(pair_net) * 100 / (2 * config.big_blind))
    except (Exception, KeyboardInterrupt) as exc:
        if not active_timings_accounted:
            all_timings.extend(item["latency_ns"] for item in active_decisions if "latency_ns" in item)
            timings.extend(item["latency_ns"] for item in active_decisions
                           if "latency_ns" in item and item["actor"] == leg)
        failure = {"error_type": type(exc).__name__, "error": str(exc), "opponent": opponent,
                   "pair": pair, "leg": leg, "hand": hand.snapshot() if hand else None,
                   "decisions": active_decisions}
        write_json(output / "failure.json", failure)
    final_environment = environment()
    source_unchanged = final_environment == manifest["environment"]
    if not source_unchanged:
        write_json(output / "provenance-failure.json", {
            "before": manifest["environment"], "after": final_environment,
            "reason": "source or environment identity changed during the campaign",
        })
    report = {
        "status": ("invalidated" if not source_unchanged
                   else "incomplete" if failure and failure["error_type"] == "KeyboardInterrupt"
                   else "failed" if failure else "completed"),
        "completed_hands": completed_hands,
        "complete_pairs": sum(len(values) for values in net.values()),
        "unpaired_hands_excluded": completed_hands - 2 * sum(len(values) for values in net.values()),
        "rake_chips_all_completed_hands": total_rake,
        "by_opponent": {name: {"gross": summarize(gross[name]), "net": summarize(net[name])} for name in opponents},
        "aggregate_gross": combined_summary(gross, weights), "aggregate_net": combined_summary(net, weights),
        "hero_timing": timing_summary(timings), "all_policy_timing": timing_summary(all_timings),
        "action_failures": int(bool(failure and failure["error_type"] != "KeyboardInterrupt"
                                    and active_decisions and not active_decisions[-1]["accepted"])),
        "run_failures": int(failure is not None or not source_unchanged),
        "source_environment_unchanged": source_unchanged,
        "environment_after": final_environment,
        "interval_method": "Student t over duplicate-deal pair means; equal-weight strata use Welch approximation.",
        "interpretation": "Finite local sample against named synthetic policies. Intervals are approximate, exclude tuning uncertainty, and do not establish human or real-money profitability.",
        "failure": failure,
    }
    write_json(output / "report.json", report)
    return report


def replay(record: dict, *, verify_policy: bool = False) -> HandResult:
    if not isinstance(record, dict):
        raise ValueError("hand record must be a JSON object")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("unsupported hand schema_version")
    config = GameConfig(**record["config"])
    hand = Hand(config, deck=tuple(record["deck"]))
    decisions = record.get("decisions")
    if decisions is not None and (not isinstance(decisions, list) or len(decisions) != len(record["actions"])):
        raise ValueError("decision trail length does not match canonical actions")
    policies = None
    if verify_policy:
        names, seeds = record["policies"], record["policy_seeds"]
        if len(names) != 2 or len(seeds) != 2:
            raise ValueError("policy replay requires two policy identities and seeds")
        from .learning import OpponentProfile
        from .opponents import HELDOUT_NAMES, RIVER_SIZING_NAMES, make_opponent
        from .policies import LEARNING_POLICY_NAMES, make_adaptive_policy, make_river_policy

        profiles = record.get("policy_profiles", [None, None])
        if not isinstance(profiles, list) or len(profiles) != 2:
            raise ValueError("policy replay requires two profile slots")
        policies = []
        for seat, (name, seed) in enumerate(zip(names, seeds)):
            if name in LEARNING_POLICY_NAMES:
                profile = OpponentProfile.from_dict(profiles[seat])
                factory = make_river_policy if name == "river" else make_adaptive_policy
                policies.append(factory(seed, record["samples"], profile))
            elif profiles[seat] is not None:
                raise ValueError("only a learning policy may have an opponent profile")
            elif name in (*HELDOUT_NAMES, *RIVER_SIZING_NAMES):
                policies.append(make_opponent(name, seed, record["samples"]))
            else:
                policies.append(make_policy(name, seed, record["samples"]))
    for index, value in enumerate(record["actions"]):
        if hand.finished:
            raise ValueError(f"extra action after hand completion at index {index}")
        action = Action(**value)
        observation = hand.observation()
        if decisions is not None:
            decision = decisions[index]
            if not isinstance(decision, dict):
                raise ValueError(f"decision {index} must be a JSON object")
            if type(decision.get("actor")) is not int or decision["actor"] != observation.actor:
                raise ValueError(f"decision actor mismatch at action {index}")
            if Action(**decision["action"]) != action:
                raise ValueError(f"decision trail action mismatch at action {index}")
            if "street" in decision and decision["street"] != observation.street:
                raise ValueError(f"decision street mismatch at action {index}")
            if "accepted" in decision and decision["accepted"] is not True:
                raise ValueError(f"recorded action was not accepted at index {index}")
        if policies:
            policy = policies[observation.actor]
            actual = policy.act(observation)
            if actual != action:
                raise ValueError(f"policy decision mismatch at action {index}: {actual} != {action}")
            if names[observation.actor] in LEARNING_POLICY_NAMES and decisions is not None:
                recorded_hash = decisions[index].get("info", {}).get("opponent_profile_sha256")
                if recorded_hash != policy.last_info["opponent_profile_sha256"]:
                    raise ValueError(f"opponent profile identity mismatch at action {index}")
        hand.act(action)
    if not hand.finished:
        raise ValueError("record stops before the hand completes")
    expected = json.dumps(record["result"], sort_keys=True, allow_nan=False)
    actual = json.dumps(asdict(hand.result()), sort_keys=True, allow_nan=False)
    if actual != expected:
        raise ValueError("recorded settlement does not match replay")
    return hand.result()
