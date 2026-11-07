import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .arena import environment, write_json
from .history_import import ImportedHand, import_histories
from .learning import OpponentProfile, response_context

MODEL_NAMES = ("fixed_prior", "global_population", "population", "player")


def responses(hand: ImportedHand) -> list[dict]:
    raises = defaultdict(int)
    aggressor = {}
    rows = []
    for index, event in enumerate(hand.history):
        if event.to_call_before > 0 and aggressor.get(event.street) == 1 - event.actor:
            action = "raise" if event.action.kind == "raise_to" else event.action.kind
            if action not in ("fold", "call", "raise"):
                raise ValueError("priced human response must fold, call or raise")
            rows.append({
                "hand": hand.key, "timestamp": hand.timestamp, "event": index,
                "player": hand.players[event.actor], "action": action,
                "context": response_context(event.street, event.actor, raises[event.street],
                                            event.pot_before, event.to_call_before),
            })
        if event.action.kind == "raise_to":
            raises[event.street] += 1
            aggressor[event.street] = event.actor
    return rows


def loss(probability: float, folded: bool) -> tuple[float, float]:
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("fold probability must be finite and strictly between zero and one")
    return (probability - int(folded)) ** 2, -math.log(probability if folded else 1 - probability)


def score(rows: list[dict], model_names: tuple[str, ...] = MODEL_NAMES) -> dict:
    result = {"responses": len(rows), "players": len({row["player"] for row in rows}),
              "hands": len({row["hand"] for row in rows}), "models": {}}
    for model in model_names:
        values = [loss(row["probabilities"][model], row["action"] == "fold") for row in rows]
        by_player = defaultdict(list)
        for row, value in zip(rows, values):
            by_player[row["player"]].append(value)
        means = [(math.fsum(v[0] for v in group) / len(group),
                  math.fsum(v[1] for v in group) / len(group)) for group in by_player.values()]
        result["models"][model] = {
            "brier": math.fsum(v[0] for v in values) / len(values) if values else None,
            "log_loss": math.fsum(v[1] for v in values) / len(values) if values else None,
            "player_macro_brier": math.fsum(v[0] for v in means) / len(means) if means else None,
            "player_macro_log_loss": math.fsum(v[1] for v in means) / len(means) if means else None,
        }
    result["paired_deltas"] = {}
    for candidate, baseline in (("global_population", "fixed_prior"), ("population", "fixed_prior"),
                                ("population", "global_population"), ("player", "fixed_prior"),
                                ("player", "global_population"), ("player", "population"),
                                ("population_backed", "population"), ("population_backed", "player")):
        if candidate not in model_names or baseline not in model_names:
            continue
        result["paired_deltas"][f"{candidate}_minus_{baseline}"] = {
            metric: (result["models"][candidate][metric] - result["models"][baseline][metric]
                     if rows else None)
            for metric in ("brier", "log_loss", "player_macro_brier", "player_macro_log_loss")
        }
    return result


def learn_hands(paths: list[Path], output: Path, train_fraction: float = 0.7,
                min_stack_bb: float = 80, max_stack_bb: float = 120) -> dict:
    if not math.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train fraction must be between zero and one")
    source_environment = environment()
    input_hashes_before = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    hands, import_report = import_histories(paths, min_stack_bb, max_stack_bb)
    hands = sorted(hands, key=lambda hand: (hand.timestamp, hand.key))
    if len(hands) < 2:
        raise ValueError("at least two eligible completed hands are required")
    cutoff = hands[min(len(hands) - 1, max(1, int(len(hands) * train_fraction)))].timestamp
    training = [hand for hand in hands if hand.timestamp < cutoff]
    evaluation = [hand for hand in hands if hand.timestamp >= cutoff]
    if not training or not evaluation:
        raise ValueError("timestamps cannot produce nonempty chronological training and evaluation sets")
    output.mkdir(parents=True, exist_ok=False)
    population = OpponentProfile()
    profiles: dict[str, OpponentProfile] = {}
    response_counts = defaultdict(lambda: defaultdict(lambda: {"fold": 0, "call": 0, "raise": 0}))
    training_responses = 0
    with (output / "hands.jsonl").open("x") as journal:
        for hand in hands:
            journal.write(json.dumps({**asdict(hand), "split": "train" if hand.timestamp < cutoff else "test"},
                                     allow_nan=False) + "\n")
    for hand in training:
        for seat, player in enumerate(hand.players):
            profiles.setdefault(player, OpponentProfile()).observe_hand(hand.history, seat)
            population.observe_hand(hand.history, seat)
        for row in responses(hand):
            response_counts[row["player"]][row["context"]][row["action"]] += 1
            training_responses += 1
    for player, profile in profiles.items():
        if profile.to_dict()["responses"] != dict(response_counts[player]):
            raise RuntimeError("evaluation response extraction differs from the existing learner")
    population_counts = population.to_dict()["responses"].values()
    total_folds = sum(counts["fold"] for counts in population_counts)
    if sum(sum(counts.values()) for counts in population_counts) != training_responses:
        raise RuntimeError("pooled training counts disagree with public response extraction")
    global_rate = (2 + total_folds) / (8 + training_responses)
    frozen = {
        "schema_version": 1, "model_version": "historical-public-fold-response-v1",
        "purpose": "Offline human response prediction; not a live-play checkpoint",
        "training_cutoff_utc": cutoff, "source_environment": source_environment,
        "input_sha256": input_hashes_before,
        "import": import_report, "population": population.to_dict(),
        "population_observation_unit": "player-hand exposure; two exposures per unique training hand",
        "global_population": {"folds": total_folds, "responses": training_responses, "fold_probability": global_rate},
        "players": {key: value.to_dict() for key, value in sorted(profiles.items())},
    }
    write_json(output / "profiles.json", frozen)
    frozen_bytes = (output / "profiles.json").read_bytes()
    saved = json.loads(frozen_bytes)
    population = OpponentProfile.from_dict(saved["population"])
    profiles = {key: OpponentProfile.from_dict(value) for key, value in saved["players"].items()}
    empty = OpponentProfile()
    rows = []
    with (output / "predictions.jsonl").open("x") as journal:
        for hand in evaluation:
            for row in responses(hand):
                individual = profiles.get(row["player"], empty).estimate(row["context"])
                pooled = population.estimate(row["context"])
                row.update(
                    probabilities={"fixed_prior": 0.25,
                                   "global_population": saved["global_population"]["fold_probability"],
                                   "population": pooled["fold_mean"],
                                   "player": individual["fold_mean"]},
                    seen_player=row["player"] in profiles,
                    player_training_responses=individual["responses"],
                    population_training_responses=pooled["responses"],
                    player_context_supported=individual["ready"],
                )
                rows.append(row)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
    source_unchanged = environment() == source_environment
    profiles_unchanged = (
        population.to_dict() == saved["population"]
        and {key: value.to_dict() for key, value in profiles.items()} == saved["players"]
        and (output / "profiles.json").read_bytes() == frozen_bytes
    )
    input_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    inputs_unchanged = input_hashes == input_hashes_before
    train_ids, test_ids = {hand.key for hand in training}, {hand.key for hand in evaluation}
    disjoint = train_ids.isdisjoint(test_ids)
    subsets = {
        "all": rows,
        "seen_players": [row for row in rows if row["seen_player"]],
        "unseen_players": [row for row in rows if not row["seen_player"]],
        "seen_player_contexts": [row for row in rows if row["player_training_responses"] > 0],
        "supported_player_contexts": [row for row in rows if row["player_context_supported"]],
    }
    by_player = defaultdict(list)
    for row in rows:
        by_player[row["player"]].append(row)
    report = {
        "schema_version": 1,
        "status": "completed" if source_unchanged and profiles_unchanged and inputs_unchanged and disjoint and rows else "invalidated",
        "scope": "Predict public fold versus continue responses from historical records; no policy-value or profit evaluation",
        "import": import_report,
        "split": {"method": "Global chronological whole-hand split; tied start timestamps stay together",
                  "requested_train_fraction": train_fraction, "cutoff_utc": cutoff,
                  "training_hands": len(training), "test_hands": len(evaluation),
                  "training_responses": training_responses, "training_players": len(profiles),
                  "test_players": len({player for hand in evaluation for player in hand.players}),
                  "hands_disjoint": disjoint,
                  "training_last_timestamp": training[-1].timestamp,
                  "test_first_timestamp": evaluation[0].timestamp},
        "models": {"fixed_prior": "Constant fold probability 0.25 from the existing Beta(2,6) prior",
                   "global_population": "One training-only fold posterior across all players and contexts",
                   "population": "Training-only fold posterior pooled across players within each context",
                   "player": "Training-only fold posterior for each player/context; unseen context uses 0.25"},
        "metrics": {key: score(value) for key, value in subsets.items()},
        "per_player": {key: score(value) for key, value in sorted(by_player.items())},
        "player_context_minimum_responses": empty.summary()["minimum_responses"],
        "frozen_profiles_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "source_environment_unchanged": source_unchanged,
        "evaluation_profiles_unchanged": profiles_unchanged,
        "input_files_unchanged": inputs_unchanged,
        "input_sha256_after": input_hashes,
        "default_policy_replaced": False,
        "real_money_readiness_established": False,
        "limitations": [
            "Imported records are attributed to their dataset publisher; legal consistency is not independent proof of origin.",
            "Historical player identities are dataset-local and cannot identify a new ClubGG opponent.",
            "Only public fold/call/raise responses are learned; hidden cards, future outcomes and showdown strength are excluded.",
            "All-row posterior predictions include sparse contexts; the existing action policy requires 24 responses plus further confidence gates.",
            "Per-hand start times do not prove completion-before-cutoff availability; this is frozen offline holdout evaluation.",
            "Metrics are descriptive; repeated hands, players and opponent pairs are correlated. No independent-sample significance claim.",
            "Replaying recorded actions cannot estimate winnings under different actions our bot might take.",
            "Historical play, differing stack depths and missing rake metadata do not establish present-day playing strength or profitability.",
        ],
    }
    write_json(output / "report.json", report)
    return report
