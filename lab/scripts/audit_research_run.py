import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path


def verify_policy_assignment(record: dict, style: str) -> None:
    hero = "adaptive" if record["arm"] == "adaptive" else "range"
    expected = (hero, style) if record["hero_seat"] == 0 else (style, hero)
    if tuple(record["policies"]) != expected:
        raise ValueError("recorded policies do not match the declared arm, style and seat")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay local research journals and independently verify pairing, learning and settlement.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--policy-hands", type=int, default=2, help="first N learning and evaluation hands per stratum, plus first changed decision per stratum")
    args = parser.parse_args()
    if args.policy_hands < 0:
        parser.error("policy-hands cannot be negative")
    directory = args.directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    source = directory / "source"
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen source changed: {name}")
    sys.path.insert(0, str(source))
    from headsup.arena import replay, seed_for
    from headsup.engine import Hand
    from headsup.learning import OpponentProfile
    from headsup.models import Action, GameConfig

    total_hands, total_actions, policy_replays = 0, 0, 0
    strata = []
    for seed in manifest["seeds"]:
        for style in manifest["styles"]:
            run = directory / f"seed-{seed}" / style
            report = json.loads((run / "report.json").read_text())
            frozen = json.loads((run / "frozen-profile.json").read_text())
            profile = OpponentProfile()
            warmup_decks, evaluation_decks, identities, pairs = set(), set(), set(), {}
            counts = {"learning": 0, "evaluation": 0}
            changed_verified = False
            journal = run / "hands.jsonl.gz"
            stream = gzip.open(journal, "rt") if journal.exists() else (run / "hands.jsonl").open()
            with stream:
                for line in stream:
                    record = json.loads(line)
                    phase, index, arm, seat = record["phase"], record["index"], record["arm"], record["hero_seat"]
                    key = (phase, index, arm, seat)
                    if key in identities:
                        raise ValueError("duplicate hand identity")
                    identities.add(key)
                    if record["config"] != manifest["config"] or record["samples"] != manifest["samples"]:
                        raise ValueError("hand configuration changed")
                    verify_policy_assignment(record, style)
                    expected_seed = seed_for(seed, phase, style, index, "deck")
                    seeded = Hand(GameConfig(**manifest["config"]), expected_seed)
                    if tuple(record["deck"]) != seeded.snapshot()["deck"]:
                        raise ValueError("deck seed does not match prespecified stream")
                    policy_seed = tuple(seed_for(seed, phase, style, index, *( (seat, actor) if phase == "evaluation" else (actor,) ), "policy") for actor in (0, 1))
                    if tuple(record["policy_seeds"]) != policy_seed:
                        raise ValueError("policy seed does not match paired stream")
                    if not all(item["accepted"] for item in record["decisions"]):
                        raise ValueError("unaccepted action in completed hand")
                    expected_profile = (profile.to_dict() if phase == "learning" else frozen) if arm == "adaptive" else None
                    if record["policy_profiles"][seat] != expected_profile or record["policy_profiles"][1 - seat] is not None:
                        raise ValueError("wrong or leaked opponent profile")
                    adapted = any(item["info"].get("adaptation_applied") for item in record["decisions"] if item["actor"] == seat)
                    check_policy = counts[phase] < args.policy_hands or (adapted and not changed_verified)
                    replay(record, verify_policy=check_policy)
                    policy_replays += int(check_policy)
                    changed_verified = changed_verified or (adapted and check_policy)
                    counts[phase] += 1
                    total_hands += 1
                    total_actions += len(record["decisions"])
                    if phase == "learning":
                        if arm != "adaptive" or seat != index % 2 or index != counts[phase] - 1:
                            raise ValueError("learning chronology changed")
                        warmup_decks.add(expected_seed)
                        for action in record["actions"]:
                            seeded.act(Action(**action))
                        profile.observe_hand(seeded.public_history, 1 - seat)
                    else:
                        evaluation_decks.add(expected_seed)
                        pairs.setdefault(index, []).append(record)
            if profile.to_dict() != frozen or profile.hands_observed != manifest["warmup_hands_per_stratum"]:
                raise ValueError("public learning reconstruction disagrees with saved profile")
            if warmup_decks & evaluation_decks:
                raise ValueError("learning and evaluation decks overlap")
            if counts != report["completed_hands"] or len(pairs) != manifest["matched_evaluation_units_per_stratum"]:
                raise ValueError("hand accounting disagrees")
            for index, records in pairs.items():
                if len(records) != 4 or {(r["arm"], r["hero_seat"]) for r in records} != {(a, s) for a in ("adaptive", "baseline") for s in (0, 1)}:
                    raise ValueError("incomplete matched unit")
                if any(record["deck"] != records[0]["deck"] for record in records):
                    raise ValueError("unmatched evaluation decks")
                arms = {arm: sum(record["result"]["net_returns"][record["hero_seat"]] for record in records if record["arm"] == arm) * 100 / (2 * manifest["config"]["big_blind"]) for arm in ("baseline", "adaptive")}
                unit = report["units"][index]
                for key, value in (("baseline_bb100", arms["baseline"]), ("adaptive_bb100", arms["adaptive"]), ("delta_bb100", arms["adaptive"] - arms["baseline"])):
                    if not math.isclose(unit[key], value, abs_tol=1e-10):
                        raise ValueError("paired returns disagree with settlements")
            strata.append({"seed": seed, "style": style, "status": "passed", "hands": sum(counts.values()), "profile_sha256": profile.fingerprint})
            print(f"audited: seed {seed} {style}", flush=True)
    result = {"status": "passed", "replayed_hands": total_hands, "replayed_actions": total_actions,
              "policy_verified_hands": policy_replays, "strata": strata,
              "scope": "Every hand replays legal actions and exact settlement; selected hands also reproduce policy choices and profile fingerprints. All learning profiles reconstructed from public action journals; all matched evaluation units and returns verified."}
    (directory / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "strata"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
