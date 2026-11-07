import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from itertools import groupby
from pathlib import Path

from .arena import environment, write_json
from .connection import MAX_PRIOR_BYTES, _finite_json_number, _unique_json_fields, load_population_prior
from .history_import import ImportedHand, import_histories
from .human_learning import responses, score
from .learning import OpponentProfile, PopulationBackedProfile

STRENGTHS = (8.0, 24.0, 64.0)
MODELS = ("fixed_prior", "population", "player", "population_backed")
MEMORY = "recent"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _files(paths: list[Path]) -> dict:
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _bounded_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{path.name} exceeds the {limit} byte size limit")
    return raw


def _hand_hash(hand: ImportedHand) -> str:
    value = asdict(hand)
    del value["source"], value["row"]
    return _hash(value)


def peer_population(population: OpponentProfile, contribution: OpponentProfile | None) -> OpponentProfile:
    data = population.to_dict()
    if data["schema_version"] != 1:
        raise ValueError("population requires lifetime counts")
    if contribution is None:
        return OpponentProfile.from_dict(data)
    own = contribution.to_dict()
    if own["schema_version"] != 1:
        raise ValueError("population exclusion requires all lifetime contributions")
    data["hands_observed"] -= own["hands_observed"]
    for context, counts in own["responses"].items():
        if context not in data["responses"]:
            raise ValueError("individual contribution is missing from the population")
        for action, count in counts.items():
            data["responses"][context][action] -= count
        if not any(data["responses"][context].values()):
            del data["responses"][context]
    return OpponentProfile.from_dict(data)


def _train(hands: list[ImportedHand]) -> tuple[OpponentProfile, dict, dict]:
    population = OpponentProfile()
    individuals, contributions = {}, {}
    for hand in hands:
        for seat, player in enumerate(hand.players):
            individuals.setdefault(player, OpponentProfile(MEMORY)).observe_hand(hand.history, seat)
            contributions.setdefault(player, OpponentProfile()).observe_hand(hand.history, seat)
            population.observe_hand(hand.history, seat)
    return population, individuals, contributions


def _predict(hands: list[ImportedHand], population: OpponentProfile, seeds: dict,
             contributions: dict, strengths: tuple[float, ...], online: bool) -> tuple[list[dict], dict]:
    players, peers, backed = {}, {}, {}
    rows = []
    for _, group in groupby(hands, key=lambda hand: hand.timestamp):
        simultaneous = list(group)
        for hand in simultaneous:
            for player in hand.players:
                if player not in players:
                    players[player] = (OpponentProfile.from_dict(seeds[player].to_dict()) if player in seeds
                                       else OpponentProfile(MEMORY))
                    peers[player] = peer_population(population, contributions.get(player))
                    backed[player] = {strength: PopulationBackedProfile(peers[player], strength, MEMORY, players[player])
                                      for strength in strengths}
            for row in responses(hand):
                player, context = row["player"], row["context"]
                own = players[player].estimate(context)
                pooled = peers[player].estimate(context)
                estimates = {strength: model.estimate(context) for strength, model in backed[player].items()}
                row.update(
                    probabilities={"fixed_prior": 0.25, "population": pooled["fold_mean"], "player": own["fold_mean"],
                                   **{f"k{strength:g}": value["fold_mean"] for strength, value in estimates.items()}},
                    seen_at_fit=player in seeds,
                    individual_responses=own["responses"],
                    individual_hands=players[player].hands_observed,
                    ready=own["ready"],
                    peer_population_responses=pooled["responses"],
                    weights={f"k{strength:g}": value["individual_weight"] for strength, value in estimates.items()},
                )
                rows.append(row)
        if online:
            for hand in simultaneous:
                for seat, player in enumerate(hand.players):
                    players[player].observe_hand(hand.history, seat)
                    for model in backed[player].values():
                        model.observe_hand(hand.history, seat)
    return rows, {player: profile.to_dict() for player, profile in sorted(players.items())}


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def fit_population(paths: list[Path], output: Path) -> dict:
    source = environment()
    inputs = _files(paths)
    hands, imported = import_histories(paths)
    if len(hands) < 2:
        raise ValueError("at least two eligible historical hands are required")
    cutoff = hands[min(len(hands) - 1, max(1, int(len(hands) * 0.7)))].timestamp
    training = [hand for hand in hands if hand.timestamp < cutoff]
    calibration = [hand for hand in hands if hand.timestamp >= cutoff]
    if not training or not calibration:
        raise ValueError("development timestamps cannot produce both training and calibration sets")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {
        "strength_candidates": STRENGTHS, "memory": MEMORY,
        "selection_metric": "mean fold Brier loss across all development responses",
        "tie_break": "larger prior strength", "development_split": "70% whole-hand chronological cutoff; ties together",
        "population_exclusion": "all lifetime target contributions excluded, including expired individual evidence",
        "update_order": "predict each whole timestamp group, then observe completed hands",
        "fresh_evaluation": "required separately; new hands strictly after the development corpus end",
        "source_environment": source, "input_sha256": inputs,
    })
    pool, individuals, contributions = _train(training)
    rows, _ = _predict(calibration, pool, individuals, contributions, STRENGTHS, True)
    if not rows:
        raise ValueError("development calibration has no eligible responses")
    losses = {strength: math.fsum((row["probabilities"][f"k{strength:g}"] - (row["action"] == "fold")) ** 2
                                  for row in rows) / len(rows) for strength in STRENGTHS}
    selected = min(STRENGTHS, key=lambda strength: (losses[strength], -strength))
    _write_rows(output / "calibration-predictions.jsonl", rows)
    pool, individuals, contributions = _train(hands)
    prior = {
        "schema_version": 1, "kind": "population-fold-prior", "population": pool.to_dict(),
        "population_sha256": pool.fingerprint, "strength": selected,
        "training": {"hands": len(hands), "players": len(individuals),
                     "last_timestamp_utc": hands[-1].timestamp, "input_sha256": inputs,
                     "population_observation_unit": "player-hand exposure; two per physical hand"},
        "calibration": {"memory": MEMORY, "candidates": STRENGTHS, "brier": {str(k): v for k, v in losses.items()},
                        "selected": selected, "responses": len(rows), "cutoff_utc": cutoff,
                        "scope": "development selection only; forward evaluation must use new records"},
    }
    write_json(output / "population-prior.json", prior)
    load_population_prior(output / "population-prior.json")
    checkpoint = {
        "schema_version": 1, "kind": "population-learning-checkpoint", "memory": MEMORY,
        "prior_sha256": hashlib.sha256((output / "population-prior.json").read_bytes()).hexdigest(),
        "individuals": {player: profile.to_dict() for player, profile in sorted(individuals.items())},
        "contributions": {player: profile.to_dict() for player, profile in sorted(contributions.items())},
        "hand_fingerprints": {hand.key: _hand_hash(hand) for hand in hands},
        "last_timestamp": hands[-1].timestamp,
    }
    write_json(output / "checkpoint.json", checkpoint)
    unchanged = source == environment() and inputs == _files(paths)
    report = {"status": "completed" if unchanged else "invalidated", "import": imported,
              "development_training_hands": len(training), "development_calibration_hands": len(calibration),
              "refit_hands": len(hands), "refit_players": len(individuals),
              "strength": selected, "memory": MEMORY, "candidate_brier": {str(k): v for k, v in losses.items()},
              "inputs_and_source_unchanged": unchanged, "fresh_evaluation_performed": False,
              "prior_sha256": checkpoint["prior_sha256"], "source_environment": source}
    write_json(output / "report.json", report)
    if not unchanged:
        raise ValueError("fit invalidated by changed inputs or implementation; do not use its prior")
    return report


def _checkpoint(path: Path) -> tuple[dict, dict, OpponentProfile, dict, dict]:
    raw = _bounded_bytes(path, 32 * 1024 * 1024)
    try:
        data = json.loads(raw, object_pairs_hook=_unique_json_fields,
                          parse_float=_finite_json_number, parse_constant=_finite_json_number)
    except RecursionError:
        raise ValueError("population learning checkpoint is too deeply nested") from None
    expected = {"schema_version", "kind", "memory", "prior_sha256", "individuals",
                "contributions", "hand_fingerprints", "last_timestamp"}
    if (not isinstance(data, dict) or set(data) != expected or type(data["schema_version"]) is not int
            or data["schema_version"] != 1 or data["kind"] != "population-learning-checkpoint" or data["memory"] != MEMORY):
        raise ValueError("unsupported population learning checkpoint")
    prior_path = path.parent / "population-prior.json"
    prior_bytes = _bounded_bytes(prior_path, MAX_PRIOR_BYTES)
    if hashlib.sha256(prior_bytes).hexdigest() != data["prior_sha256"]:
        raise ValueError("population prior differs from the fitted checkpoint")
    prior = load_population_prior(prior_path)
    training = json.loads(prior_bytes).get("training")
    if not isinstance(training, dict) or training.get("last_timestamp_utc") != data["last_timestamp"]:
        raise ValueError("checkpoint timestamp differs from the fitted population prior")
    pool = OpponentProfile.from_dict(prior["population"])
    if not isinstance(data["individuals"], dict) or not isinstance(data["contributions"], dict):
        raise ValueError("checkpoint player profiles must be mappings")
    if any(not isinstance(value, dict) or value.get("schema_version") != 2
           for value in data["individuals"].values()):
        raise ValueError("checkpoint individual profiles must use recent schema 2")
    individuals = {key: OpponentProfile.from_dict(value) for key, value in data["individuals"].items()}
    contributions = {key: OpponentProfile.from_dict(value) for key, value in data["contributions"].items()}
    if set(individuals) != set(contributions):
        raise ValueError("individual and contribution identities differ")
    combined = defaultdict(lambda: {"fold": 0, "call": 0, "raise": 0})
    for key, individual in individuals.items():
        contribution = contributions[key]
        if (not re.fullmatch(r"[0-9a-f]{64}", key) or individual.memory != MEMORY
                or contribution.to_dict()["schema_version"] != 1
                or individual.hands_observed != contribution.hands_observed):
            raise ValueError("invalid individual or lifetime contribution metadata")
        lifetime = contribution.to_dict()["responses"]
        for context, estimate in individual.summary()["contexts"].items():
            if any(count > lifetime.get(context, {}).get(action, 0) for action, count in estimate["counts"].items()):
                raise ValueError("recent observations exceed lifetime contributions")
        for context, counts in lifetime.items():
            for action, count in counts.items():
                combined[context][action] += count
    fingerprints = data["hand_fingerprints"]
    if (not isinstance(fingerprints, dict) or not fingerprints
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in (*fingerprints, *fingerprints.values()))):
        raise ValueError("invalid training hand fingerprints")
    exposures = sum(value.hands_observed for value in contributions.values())
    if (pool.to_dict()["responses"] != dict(combined) or pool.hands_observed != exposures
            or exposures != 2 * len(fingerprints)):
        raise ValueError("population counts disagree with individual contributions or hand count")
    timestamp = data["last_timestamp"]
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", timestamp):
        raise ValueError("checkpoint requires a canonical UTC timestamp")
    datetime.fromisoformat(timestamp)
    return data, prior, pool, individuals, contributions


def evaluate_population(checkpoint_path: Path, paths: list[Path], output: Path) -> dict:
    source = environment()
    inputs = _files(paths)
    checkpoint_bytes = _bounded_bytes(checkpoint_path, 32 * 1024 * 1024)
    data, prior, pool, individuals, contributions = _checkpoint(checkpoint_path)
    hands, imported = import_histories(paths)
    future = []
    duplicate = earlier = 0
    for hand in hands:
        if hand.key in data["hand_fingerprints"]:
            if _hand_hash(hand) != data["hand_fingerprints"][hand.key]:
                raise ValueError("new input conflicts with a training hand")
            duplicate += 1
        elif hand.timestamp <= data["last_timestamp"]:
            earlier += 1
        else:
            future.append(hand)
    if not future:
        raise ValueError("no new eligible hands strictly after the training corpus")
    output.mkdir(parents=True, exist_ok=False)
    strength = prior["strength"]
    prior_hash = pool.fingerprint
    model_hash = _hash({"prior": prior, "individuals": data["individuals"], "contributions": data["contributions"]})
    write_json(output / "frozen-evaluation-model.json", {
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "model_sha256": model_hash, "strength": strength, "memory": MEMORY,
        "population_sha256": prior_hash, "input_sha256": inputs, "source_environment": source,
        "update_protocol": "Predict all hands sharing a start timestamp before observing their completed public actions",
    })
    results = {}
    with (output / "hands.jsonl").open("x") as stream:
        for hand in future:
            stream.write(json.dumps(asdict(hand), allow_nan=False) + "\n")
    for mode in ("frozen", "online"):
        rows, final_profiles = _predict(future, pool, individuals, contributions, (strength,), mode == "online")
        for row in rows:
            row["probabilities"]["population_backed"] = row["probabilities"].pop(f"k{strength:g}")
            row["individual_weight"] = row.pop("weights")[f"k{strength:g}"]
        _write_rows(output / f"{mode}-predictions.jsonl", rows)
        if mode == "online":
            write_json(output / "online-individuals.json", final_profiles)
        results[mode] = {
            "all": score(rows, MODELS),
            "new_players": score([row for row in rows if not row["seen_at_fit"]], MODELS),
            "known_players": score([row for row in rows if row["seen_at_fit"]], MODELS),
            "supported_contexts": score([row for row in rows if row["ready"]], MODELS),
            "mean_individual_weight": math.fsum(row["individual_weight"] for row in rows) / len(rows) if rows else None,
        }
    models_unchanged = (_hash({"prior": prior, "individuals": {k: v.to_dict() for k, v in individuals.items()},
                               "contributions": {k: v.to_dict() for k, v in contributions.items()}}) == model_hash
                        and pool.fingerprint == prior_hash
                        and _bounded_bytes(checkpoint_path, 32 * 1024 * 1024) == checkpoint_bytes
                        and hashlib.sha256(_bounded_bytes(checkpoint_path.parent / "population-prior.json", MAX_PRIOR_BYTES)).hexdigest() == data["prior_sha256"])
    unchanged = models_unchanged and inputs == _files(paths) and source == environment()
    report = {
        "status": "completed" if unchanged and results["online"]["all"]["responses"] else "invalidated",
        "import": imported, "excluded_training_duplicates": duplicate, "excluded_earlier_or_equal_hands": earlier,
        "future_hands": len(future), "future_players": len({player for hand in future for player in hand.players}),
        "training_last_timestamp": data["last_timestamp"], "future_first_timestamp": future[0].timestamp,
        "future_last_timestamp": future[-1].timestamp, "strength": strength, "memory": MEMORY,
        "evaluation": results, "inputs_source_and_frozen_model_unchanged": unchanged,
        "default_policy_replaced": False, "real_money_readiness_established": False,
        "limitations": [
            "Only public fold versus continue responses are predicted; no hidden-card or outcome learning.",
            "Population excludes every historical contribution from the target, including expired individual evidence.",
            "Online evaluation observes each completed timestamp group only after predicting every response in that group.",
            "Start timestamps lack completion times; this is historical sequential evaluation, not strict real-time availability validation.",
            "Archive identities are local to the dataset and cannot identify people on another platform.",
            "Metrics are descriptive; hands and players are correlated, and no significance or profitability claim is made.",
            "Policy uses individual support and additional confidence gates; prediction improvement does not establish stronger play.",
        ],
    }
    write_json(output / "report.json", report)
    return report
