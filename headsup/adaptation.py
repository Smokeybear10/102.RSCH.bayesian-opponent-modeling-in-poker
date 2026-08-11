import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .arena import environment, seed_for, summarize, timing_summary, write_json
from .engine import Hand
from .learning import OpponentProfile
from .models import MAX_SAMPLES, GameConfig
from .training import _comparison, _play

ARMS = ("range", "lifetime", "recent")
SCENARIOS = {
    "stable_fit_fold": ("heldout_fit_fold", "heldout_fit_fold"),
    "stable_calling": ("calling", "calling"),
    "fit_fold_to_calling": ("heldout_fit_fold", "calling"),
    "calling_to_fit_fold": ("calling", "heldout_fit_fold"),
}
WINDOWS = ("full_session", "post_switch")
CONTRASTS = {"recent_minus_lifetime": "lifetime", "recent_minus_range": "range"}
ENDPOINTS = (len(SCENARIOS) + 1) * len(WINDOWS) * len(CONTRASTS)


def _summary(values: list[float], adjusted: bool = False) -> dict:
    result = _comparison(values, ENDPOINTS) if adjusted else summarize(values)
    result["sessions"] = result.pop("pairs")
    return result


def evaluate_adaptation(output: Path, config: GameConfig, sessions: int = 12,
                        hands: int = 384, switch_at: int = 192,
                        seed: int = 202611051101, samples: int = 128,
                        progress: Callable[[str], None] | None = None) -> dict:
    for name, value in (("sessions", sessions), ("hands", hands), ("switch_at", switch_at), ("samples", samples)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if switch_at >= hands:
        raise ValueError("switch_at must leave at least one hand after the switch")
    if samples > MAX_SAMPLES or type(seed) is not int:
        raise ValueError("samples exceeds the limit or seed is not an integer")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "mode": "online_adaptation_sessions",
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "sessions_per_scenario": sessions, "hands_per_session_per_arm": hands,
        "switch_at": switch_at, "seed": seed, "samples": samples,
        "arms": ARMS, "scenarios": SCENARIOS, "windows": WINDOWS,
        "contrasts": CONTRASTS, "comparison_endpoints": ENDPOINTS,
        "environment": environment(),
        "protocol": "Each independently seeded session starts fresh profiles. All arms receive matching fresh decks and policy seeds; hero seats alternate. Each arm updates its own memory once after each complete hand. One opponent identity persists through a style switch, whose label and timing are never passed to the hero.",
        "statistical_unit": "Whole independent session, paired across arms. Session-index blocks average independent scenario strata for fixed equal-weight aggregates. Hands within a learning session are not independent observations.",
        "post_switch_definition": "Zero-based hand indexes switch_at through hands_per_session_per_arm - 1; the same window is used for stationary controls.",
        "selection_policy": "One fixed campaign, all results retained, no automatic default replacement. Previous public actions are online learning inputs; scored outcomes are never learning inputs.",
    }
    write_json(output / "manifest.json", manifest)
    net = {scenario: {window: {arm: [] for arm in ARMS} for window in WINDOWS}
           for scenario in SCENARIOS}
    deltas = {scenario: {window: {contrast: [] for contrast in CONTRASTS} for window in WINDOWS}
              for scenario in SCENARIOS}
    completed_hands = {arm: 0 for arm in ARMS}
    adaptations = {arm: {window: 0 for window in WINDOWS} for arm in ARMS}
    hero_times, all_times = [], []
    complete_sessions = 0
    scenario, session, hand_index, arm, opponent = None, None, None, None, None
    hand, decisions, profiles, pre_hand_profile = None, [], {}, None
    failure = None
    try:
        with (output / "hands.jsonl").open("x") as journal, (output / "sessions.jsonl").open("x") as session_journal:
            for scenario, styles in SCENARIOS.items():
                for session in range(sessions):
                    hand, decisions, profiles, pre_hand_profile = None, [], {}, None
                    hand_index, arm, opponent = None, None, None
                    profiles = {"range": None, "lifetime": OpponentProfile(memory="lifetime"),
                                "recent": OpponentProfile(memory="recent")}
                    returns = {name: [] for name in ARMS}
                    for hand_index in range(hands):
                        opponent = styles[int(hand_index >= switch_at)]
                        hero_seat = (session + hand_index) % 2
                        deck_seed = seed_for(seed, "adaptation-deck", scenario, session, hand_index)
                        seeds = tuple(seed_for(seed, "adaptation-policy", scenario, session, hand_index, seat)
                                      for seat in (0, 1))
                        for arm in ARMS:
                            hand, decisions = None, []
                            profile = profiles[arm]
                            pre_hand_profile = profile.to_dict() if profile is not None else None
                            hand = Hand(config, deck_seed)
                            record = _play(hand, opponent, hero_seat, seeds, samples, profile, decisions)
                            record.update(scenario=scenario, session=session, hand_index=hand_index,
                                          arm=arm, opponent=opponent, post_switch=hand_index >= switch_at)
                            journal.write(json.dumps(record, allow_nan=False) + "\n")
                            journal.flush()
                            completed_hands[arm] += 1
                            returns[arm].append(hand.result().net_returns[hero_seat])
                            all_times.extend(item["latency_ns"] for item in decisions)
                            hero_decisions = [item for item in decisions if item["actor"] == hero_seat]
                            hero_times.extend(item["latency_ns"] for item in hero_decisions)
                            changes = sum(bool(item["info"].get("adaptation_applied")) for item in hero_decisions)
                            adaptations[arm]["full_session"] += changes
                            if hand_index >= switch_at:
                                adaptations[arm]["post_switch"] += changes
                            if profile is not None:
                                profile.observe_hand(hand.public_history, 1 - hero_seat)
                    scores = {window: {name: sum(values[start:]) * 100 / ((hands - start) * config.big_blind)
                                       for name, values in returns.items()}
                              for window, start in (("full_session", 0), ("post_switch", switch_at))}
                    session_record = {
                        "scenario": scenario, "session": session, "net_chips_by_arm": returns,
                        "net_bb100": scores,
                        "final_profiles": {name: profile.to_dict() if profile is not None else None
                                           for name, profile in profiles.items()},
                    }
                    session_journal.write(json.dumps(session_record, allow_nan=False) + "\n")
                    session_journal.flush()
                    for window in WINDOWS:
                        for name in ARMS:
                            net[scenario][window][name].append(scores[window][name])
                        for contrast, baseline in CONTRASTS.items():
                            deltas[scenario][window][contrast].append(scores[window]["recent"] - scores[window][baseline])
                    complete_sessions += 1
                    if progress:
                        progress(f"adaptation: {scenario} session {session + 1}/{sessions}")
    except (Exception, KeyboardInterrupt) as exc:
        failure = {
            "error_type": type(exc).__name__, "error": str(exc), "scenario": scenario,
            "session": session, "hand_index": hand_index, "arm": arm, "opponent": opponent,
            "hand": hand.snapshot() if hand else None, "decisions": decisions,
            "pre_hand_profile": pre_hand_profile,
            "profiles_after_completed_updates": {name: profile.to_dict() if profile is not None else None
                                                  for name, profile in profiles.items()},
        }
        write_json(output / "failure.json", failure)
    after = environment()
    unchanged = after == manifest["environment"]
    if not unchanged:
        write_json(output / "provenance-failure.json", {"before": manifest["environment"], "after": after})
    by_scenario = {
        name: {window: {
            "net": {arm: {**_summary(values), "session_net_bb100": values}
                    for arm, values in net[name][window].items()},
            "comparisons": {contrast: {**_summary(values, adjusted=True), "session_delta_bb100": values}
                            for contrast, values in deltas[name][window].items()},
        } for window in WINDOWS} for name in SCENARIOS
    }
    aggregate = {}
    for window in WINDOWS:
        aggregate[window] = {}
        for contrast in CONTRASTS:
            count = min(len(deltas[name][window][contrast]) for name in SCENARIOS)
            blocks = [sum(deltas[name][window][contrast][i] for name in SCENARIOS) / len(SCENARIOS)
                      for i in range(count)]
            aggregate[window][contrast] = {**_summary(blocks, adjusted=True), "session_delta_blocks_bb100": blocks}
    status = ("invalidated" if not unchanged
              else "incomplete" if failure and failure["error_type"] == "KeyboardInterrupt"
              else "failed" if failure else "completed")
    report = {
        "status": status, "completed_hands": sum(completed_hands.values()),
        "completed_hands_by_arm": completed_hands, "complete_matched_sessions": complete_sessions,
        "partial_session_hands_excluded": sum(completed_hands.values()) - complete_sessions * hands * len(ARMS),
        "by_scenario": by_scenario, "aggregate": aggregate, "adapted_decisions": adaptations,
        "timing_and_adaptation_scope": "All completed hands, including any retained partial session; only complete three-arm sessions enter return estimates.",
        "hero_timing": timing_summary(hero_times), "all_policy_timing": timing_summary(all_times),
        "source_environment_unchanged": unchanged, "environment_after": after,
        "run_failures": int(failure is not None or not unchanged), "failure": failure,
        "positive_aggregate_lift_over_lifetime": {
            window: status == "completed" and aggregate[window]["recent_minus_lifetime"]["positive_adjusted_lower"]
            for window in WINDOWS
        },
        "default_policy_replaced": False, "real_money_readiness_established": False,
        "interval_method": f"Approximate Student t over independent complete-session differences; Bonferroni across {ENDPOINTS} prespecified comparisons. Absolute net-return intervals are descriptive and unadjusted.",
        "limitations": [
            "Whole sessions include learning variability; individual hands and within-session phases are correlated.",
            "Only fixed synthetic styles and prespecified abrupt switches are tested; no human-profitability inference.",
            "Recent memory can discard useful stationary evidence; negative and inconclusive comparisons remain visible.",
            "All arms start empty profiles. This campaign does not evaluate importing historical checkpoints or longer-term retention.",
            "Matched deck streams reduce variance but divergent actions create different public evidence for each arm.",
            "There is no automatic promotion, repeated testing until favorable, or training on evaluation payoffs.",
        ],
    }
    write_json(output / "report.json", report)
    if progress:
        progress(f"{status}: saved session comparison to {output}")
    return report
