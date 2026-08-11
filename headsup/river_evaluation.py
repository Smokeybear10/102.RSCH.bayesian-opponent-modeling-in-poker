import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from math import fsum
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

from scipy.stats import t

from .arena import environment, seed_for, summarize, timing_summary, write_json
from .connection import MAX_PRIOR_BYTES, load_population_prior
from .engine import Hand
from .learning import OpponentProfile, PopulationBackedProfile
from .models import MAX_SAMPLES, GameConfig
from .training import _opponent

ARMS = ("adaptive", "river")
SCENARIOS = {
    "calling": ("calling", "calling"),
    "tight": ("tight", "tight"),
    "aggressive": ("aggressive", "aggressive"),
    "heldout_fit_fold": ("heldout_fit_fold", "heldout_fit_fold"),
    "heldout_pressure": ("heldout_pressure", "heldout_pressure"),
    "heldout_loose": ("heldout_loose", "heldout_loose"),
    "pressure_to_fit_fold": ("heldout_pressure", "heldout_fit_fold"),
}
SECONDARY_ENDPOINTS = len(SCENARIOS) + 1
SCENARIO_SUITES = {
    "original": SCENARIOS,
    "raise_sizes_v2": {
        **SCENARIOS,
        "river_small_merged": ("river_small_merged", "river_small_merged"),
        "river_large_polarized": ("river_large_polarized", "river_large_polarized"),
        "river_bluff_pressure": ("river_bluff_pressure", "river_bluff_pressure"),
        "polarized_to_bluff_pressure": ("river_large_polarized", "river_bluff_pressure"),
    },
}


def _prior_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(MAX_PRIOR_BYTES + 1)
    if len(raw) > MAX_PRIOR_BYTES:
        raise ValueError("population prior exceeds the 1 MiB size limit")
    return raw


def _summary(values: list[float], endpoints: int = 1) -> dict:
    result = summarize(values)
    result["sessions"] = result.pop("pairs")
    interval = None
    if len(values) >= 2:
        margin = float(t.ppf(1 - 0.05 / (2 * endpoints), len(values) - 1)) * result["standard_error"]
        interval = [result["mean_bb100"] - margin, result["mean_bb100"] + margin]
    return {**result, "comparison_ci95": interval, "comparison_endpoints": endpoints,
            "positive_lower": interval is not None and interval[0] > 0,
            "negative_upper": interval is not None and interval[1] < 0,
            "session_values_bb100": values}


def _play(hand: Hand, opponent: str, hero_seat: int, seeds: tuple[int, int], samples: int,
          arm: str, profile: OpponentProfile, decisions: list[dict]) -> dict:
    from .policies import make_adaptive_policy, make_river_policy

    snapshot = profile.to_dict()
    factory = make_river_policy if arm == "river" else make_adaptive_policy
    hero = factory(seeds[hero_seat], samples, profile)
    other = _opponent(opponent, seeds[1 - hero_seat], samples)
    policies = (hero, other) if hero_seat == 0 else (other, hero)
    names = (arm, opponent) if hero_seat == 0 else (opponent, arm)
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
    if profile.to_dict() != snapshot:
        raise RuntimeError("policy changed its learning profile during a hand")
    profiles = [None, None]
    profiles[hero_seat] = snapshot
    return {**hand.record(), "policies": names, "policy_seeds": seeds, "samples": samples,
            "policy_profiles": profiles, "decisions": decisions, "hero_seat": hero_seat}


def evaluate_river(output: Path, config: GameConfig, prior_path: Path, sessions: int = 12,
                   hands: int = 64, seed: int = 2026110607, samples: int = 64,
                   progress: Callable[[str], None] | None = None, suite: str = "original") -> dict:
    for name, value in (("sessions", sessions), ("hands", hands), ("samples", samples)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if hands < 2:
        raise ValueError("hands must leave observations on both sides of the style switch")
    if samples > MAX_SAMPLES or type(seed) is not int:
        raise ValueError("samples exceeds the limit or seed is not an integer")
    if suite not in SCENARIO_SUITES:
        raise ValueError(f"unknown river evaluation suite: {suite}")
    scenarios = SCENARIO_SUITES[suite]
    switch_names = tuple(name for name, styles in scenarios.items() if styles[0] != styles[1])
    secondary_endpoints = len(scenarios) + len(switch_names)
    prior_bytes = _prior_bytes(prior_path)
    prior = load_population_prior(prior_path)
    if _prior_bytes(prior_path) != prior_bytes:
        raise ValueError("population prior changed while loading")
    pool = OpponentProfile.from_dict(prior["population"])
    prior_hash = hashlib.sha256(prior_bytes).hexdigest()
    switch_at = hands // 2
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1 if suite == "original" else 2, "mode": "river_policy_session_comparison",
        "suite": suite,
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "sessions_per_scenario": sessions, "hands_per_learning_stream": hands,
        "seat_swapped_learning_streams_per_arm": 2, "switch_at": switch_at,
        "arms": ARMS, "scenarios": scenarios, "seed": seed, "samples": samples,
        "prior_path": str(prior_path.resolve()), "prior_sha256": prior_hash,
        "population_sha256": pool.fingerprint, "strength": prior["strength"], "memory": "recent",
        "environment": environment(),
        "primary_endpoint": f"Equal-weight mean of {len(scenarios)} full-session river-minus-adaptive net bb/100 strata",
        "secondary_endpoints": [*scenarios, *(f"{name}:post_switch" for name in switch_names)],
        "protocol": "Each scenario/session starts four independent copies of the same frozen population prior with empty recent individual memory: two policy arms times two seat-swapped learning streams. Each stream alternates hero seats. Decks and seat-indexed policy seeds match across arms and legs. Each profile observes only its own stream once after each complete hand. The style switch is not revealed to the hero.",
        "statistical_unit": "Complete paired session, averaging both seat-swapped streams. Primary blocks average independently seeded opponent strata with the same session index. Hands and the two streams within a session are correlated and are never treated as independent samples.",
        "selection_policy": "Fixed candidate and campaign; retain all outcomes, do not tune on scored payoffs, and do not replace any default policy automatically.",
        "guardrail": f"Positive primary 95% lower bound, no clear losses on {secondary_endpoints} Bonferroni-adjusted secondary comparisons, unchanged source/prior, and no action or run failures. Passing is local evidence only, not a human-profitability claim.",
    }
    write_json(output / "manifest.json", manifest)
    manifest_hash = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    net = {scenario: {arm: [] for arm in ARMS} for scenario in scenarios}
    differences = {scenario: [] for scenario in scenarios}
    switch_differences = {name: [] for name in switch_names}
    completed_hands = {arm: 0 for arm in ARMS}
    coverage = {arm: {"hero_decisions": 0, "river_decisions": 0, "river_evaluated": 0,
                      "river_action_changed": 0, "fold_adaptation_applied": 0} for arm in ARMS}
    timings = {arm: [] for arm in ARMS}
    river_timings = {arm: [] for arm in ARMS}
    complete_sessions = 0
    scenario, session, hand_index, leg, arm, opponent = None, None, None, None, None, None
    hand, decisions, profiles, pre_hand_profile = None, [], {}, None
    failure = None
    try:
        with (output / "hands.jsonl").open("x") as journal, (output / "sessions.jsonl").open("x") as session_journal:
            for scenario, styles in scenarios.items():
                for session in range(sessions):
                    hand, decisions, pre_hand_profile = None, [], None
                    hand_index, leg, arm, opponent = None, None, None, None
                    profiles = {
                        name: [PopulationBackedProfile(pool, prior["strength"], "recent") for _ in (0, 1)]
                        for name in ARMS
                    }
                    returns = {name: [[], []] for name in ARMS}
                    for hand_index in range(hands):
                        opponent = styles[int(hand_index >= switch_at)]
                        deck_seed = seed_for(seed, "river-session-deck", scenario, session, hand_index)
                        seeds = tuple(seed_for(seed, "river-session-policy", scenario, session, hand_index, seat)
                                      for seat in (0, 1))
                        for leg in (0, 1):
                            hero_seat = (hand_index + leg) % 2
                            for arm in ARMS:
                                hand, decisions = None, []
                                profile = profiles[arm][leg]
                                pre_hand_profile = profile.to_dict()
                                before_hash = profile.fingerprint
                                hand = Hand(config, deck_seed)
                                record = _play(hand, opponent, hero_seat, seeds, samples, arm, profile, decisions)
                                profile.observe_hand(hand.public_history, 1 - hero_seat)
                                after = profile.to_dict()
                                if (profile.hands_observed != hand_index + 1
                                        or after["population"] != prior["population"]):
                                    raise RuntimeError("completed-hand learning altered the population or duplicated a hand")
                                record.update(
                                    scenario=scenario, session=session, hand_index=hand_index, leg=leg,
                                    arm=arm, opponent=opponent, post_switch=hand_index >= switch_at,
                                    manifest_sha256=manifest_hash, profile_before_sha256=before_hash,
                                    profile_after_sha256=profile.fingerprint,
                                )
                                journal.write(json.dumps(record, allow_nan=False) + "\n")
                                journal.flush()
                                completed_hands[arm] += 1
                                returns[arm][leg].append(hand.result().net_returns[hero_seat])
                                for item in decisions:
                                    if item["actor"] != hero_seat:
                                        continue
                                    timings[arm].append(item["latency_ns"])
                                    coverage[arm]["hero_decisions"] += 1
                                    if item["street"] == "river":
                                        river_timings[arm].append(item["latency_ns"])
                                        coverage[arm]["river_decisions"] += 1
                                    for key in ("river_evaluated", "river_action_changed"):
                                        coverage[arm][key] += bool(item["info"].get(key))
                                    coverage[arm]["fold_adaptation_applied"] += bool(item["info"].get("adaptation_applied"))
                    scores = {name: fsum(value for stream in streams for value in stream) * 100
                              / (2 * hands * config.big_blind) for name, streams in returns.items()}
                    post_switch = {name: fsum(value for stream in streams for value in stream[switch_at:]) * 100
                                   / (2 * (hands - switch_at) * config.big_blind)
                                   for name, streams in returns.items()}
                    session_record = {
                        "scenario": scenario, "session": session, "manifest_sha256": manifest_hash,
                        "net_chips_by_arm_and_leg": returns, "net_bb100": scores,
                        "post_switch_net_bb100": post_switch,
                        "final_profiles": {name: [profile.to_dict() for profile in streams]
                                           for name, streams in profiles.items()},
                    }
                    session_journal.write(json.dumps(session_record, allow_nan=False) + "\n")
                    session_journal.flush()
                    for name in ARMS:
                        net[scenario][name].append(scores[name])
                    differences[scenario].append(scores["river"] - scores["adaptive"])
                    if scenario in switch_differences:
                        switch_differences[scenario].append(post_switch["river"] - post_switch["adaptive"])
                    complete_sessions += 1
                    if progress:
                        progress(f"river evaluation: {scenario} session {session + 1}/{sessions}")
    except (Exception, KeyboardInterrupt) as exc:
        failure = {
            "error_type": type(exc).__name__, "error": str(exc), "scenario": scenario,
            "session": session, "hand_index": hand_index, "leg": leg, "arm": arm,
            "opponent": opponent, "hand": hand.snapshot() if hand else None,
            "decisions": decisions, "pre_hand_profile": pre_hand_profile,
            "profiles_after_completed_updates": {name: [profile.to_dict() for profile in streams]
                                                  for name, streams in profiles.items()},
        }
        write_json(output / "failure.json", failure)
    after_environment = environment()
    source_unchanged = after_environment == manifest["environment"]
    try:
        prior_unchanged = _prior_bytes(prior_path) == prior_bytes
    except (OSError, ValueError):
        prior_unchanged = False
    manifest_unchanged = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest() == manifest_hash
    unchanged = source_unchanged and prior_unchanged and manifest_unchanged and pool.fingerprint == prior["population_sha256"]
    status = ("invalidated" if not unchanged
              else "incomplete" if failure and failure["error_type"] == "KeyboardInterrupt"
              else "failed" if failure else "completed")
    by_opponent = {
        name: {"net": {arm: _summary(values) for arm, values in net[name].items()},
               "river_minus_adaptive": _summary(differences[name], secondary_endpoints)}
        for name in scenarios
    }
    complete_blocks = min(map(len, differences.values()))
    blocks = [fsum(differences[name][index] for name in scenarios) / len(scenarios)
              for index in range(complete_blocks)]
    aggregate = _summary(blocks)
    aggregate_net = {
        arm: _summary([fsum(net[name][arm][index] for name in scenarios) / len(scenarios)
                       for index in range(complete_blocks)])
        for arm in ARMS
    }
    post_switch_results = {name: _summary(values, secondary_endpoints)
                           for name, values in switch_differences.items()}
    clear_losses = [name for name, result in by_opponent.items()
                    if result["river_minus_adaptive"]["negative_upper"]]
    clear_losses.extend(f"{name}:post_switch" for name, result in post_switch_results.items()
                        if result["negative_upper"])
    action_failures = int(bool(failure and failure["error_type"] != "KeyboardInterrupt"
                               and decisions and not decisions[-1]["accepted"]))
    report = {
        "status": status, "suite": suite, "completed_hands": sum(completed_hands.values()),
        "completed_hands_by_arm": completed_hands, "complete_matched_sessions": complete_sessions,
        "complete_primary_session_blocks": complete_blocks,
        "partial_session_hands_excluded": sum(completed_hands.values()) - complete_sessions * hands * 4,
        "by_opponent": by_opponent, "aggregate_net": aggregate_net,
        "aggregate_river_minus_adaptive": aggregate,
        "style_change_post_switch_river_minus_adaptive": post_switch_results["pressure_to_fit_fold"],
        "style_changes_post_switch_river_minus_adaptive": post_switch_results,
        "coverage": coverage, "hero_timing": {arm: timing_summary(values) for arm, values in timings.items()},
        "river_timing": {arm: timing_summary(values) for arm, values in river_timings.items()},
        "timing_and_coverage_scope": "All completed journaled hands, including retained partial sessions; only complete four-stream sessions enter return estimates.",
        "source_environment_unchanged": source_unchanged, "prior_unchanged": prior_unchanged,
        "manifest_unchanged": manifest_unchanged, "population_unchanged": pool.fingerprint == prior["population_sha256"],
        "environment_after": after_environment, "action_failures": action_failures,
        "run_failures": int(failure is not None or not unchanged), "failure": failure,
        "promotion_evidence": {
            "positive_primary_lower": aggregate["positive_lower"], "clear_secondary_losses": clear_losses,
            "passes_local_gate": status == "completed" and not action_failures
                                 and aggregate["positive_lower"] and not clear_losses,
            "default_policy_replaced": False,
            "interpretation": "An advisory gate for this fixed local campaign; absence of a clear loss does not prove noninferiority or human profitability.",
        },
        "default_policy_replaced": False, "real_money_readiness_established": False,
        "interval_method": f"Approximate Student t over independent complete paired sessions. The primary {len(scenarios)}-stratum mean uses session-index blocks and a 95% interval; secondary guardrails use Bonferroni across {secondary_endpoints} prespecified endpoints. Absolute net intervals are descriptive.",
        "limitations": [
            "Synthetic opponents and this prescribed style switch do not establish performance against humans.",
            "Learning creates dependent hands; inference uses complete sessions, not individual decisions or hands.",
            "Paired decks reduce chance variation but divergent actions produce different public learning evidence.",
            "Short sessions may not accumulate enough evidence in rare contexts; coverage is reported separately.",
            "This measures a fixed candidate with the supplied prior; response and range models remain approximations.",
            "No payoff labels are training inputs, no automatic promotion occurs, and unfavorable or inconclusive results are retained.",
        ],
    }
    write_json(output / "report.json", report)
    if progress:
        progress(f"{status}: saved river comparison to {output}")
    return report
