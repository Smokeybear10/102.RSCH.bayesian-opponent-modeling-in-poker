from bisect import bisect_left, bisect_right
from dataclasses import asdict, replace
from functools import lru_cache
from itertools import combinations as card_combinations
from math import exp, fsum, isfinite, log1p
from typing import Callable

import eval7
from scipy.stats import beta

from .learning import OpponentProfile, response_context
from .models import Action, Observation

_CODES = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
_INDEX = {card: index for index, card in enumerate(_CODES)}
_CARDS = tuple(eval7.Card(card) for card in _CODES)
_SCENARIOS = ("central", "less_folding_more_raising", "more_folding_less_raising")
_FRACTIONS = (0.33, 0.65, 1.0)
_RAISE_FRACTIONS = (0.33, 0.75, 1.5)
_SCENARIO_PRIORS = {scenario: 1 / len(_SCENARIOS) for scenario in _SCENARIOS}
_MARGIN_BB = 0.25


def _normalize(weights: tuple[float, ...]) -> tuple[float, ...]:
    if not weights or any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("river range requires finite nonnegative weights")
    total = fsum(weights)
    if not isfinite(total) or total <= 0:
        raise ValueError("river range must have positive finite mass")
    return tuple(weight / total for weight in weights)


@lru_cache(maxsize=128)
def _board_range(board: tuple[str, ...]) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    known = {_INDEX[card] for card in board}
    available = tuple(card for card in range(52) if card not in known)
    public = [_CARDS[_INDEX[card]] for card in board]
    combinations = tuple(card_combinations(available, 2))
    values = tuple(eval7.evaluate([_CARDS[a], _CARDS[b], *public]) for a, b in combinations)
    return combinations, values


def _response_odds(board: tuple[str, ...], values: tuple[int, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    public = sorted(_board_range(board)[1])
    quality = tuple((bisect_left(public, value) + bisect_right(public, value)) / (2 * len(public))
                    for value in values)
    return (tuple(exp(-6 * (value - 0.5)) for value in quality),
            tuple(exp(8 * (value - 0.5)) for value in quality))


def _tilt(target: float, weights: tuple[float, ...], odds: tuple[float, ...],
          target_odds: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    total = fsum(weights)
    if not 0 <= target <= 1 or total < 0:
        raise ValueError("response calibration requires a probability and nonnegative mass")
    if target in (0, 1) or total == 0:
        value = target if total else 0.0
        return (value,) * len(target_odds), (value,) * len(odds)
    low, high = -80.0, 80.0
    for _ in range(36):
        scale = exp((low + high) / 2)
        mean = fsum(weight * (scale * value / (1 + scale * value))
                    for weight, value in zip(weights, odds)) / total
        if mean < target:
            low = (low + high) / 2
        else:
            high = (low + high) / 2
    scale = exp((low + high) / 2)
    return (tuple(scale * value / (1 + scale * value) for value in target_odds),
            tuple(scale * value / (1 + scale * value) for value in odds))


def _posterior(profile: OpponentProfile, context: str) -> dict:
    estimate = profile.estimate(context)
    counts = estimate["counts"]
    prior = (2.0, 5.0, 1.0)
    data = profile.to_dict()
    if data["schema_version"] == 3:
        population = data["population"]["responses"].get(context, {"fold": 0, "call": 0, "raise": 0})
        total = 8 + sum(population.values())
        strength = min(data["strength"], total)
        prior = tuple(strength * (base + population[action]) / total
                      for base, action in zip(prior, ("fold", "call", "raise")))
    alpha = tuple(base + counts[action] for base, action in zip(prior, ("fold", "call", "raise")))
    raise_lower, raise_upper = beta.ppf((0.05, 0.95), alpha[2], alpha[1])
    return {"context": context, "individual_responses": estimate["responses"],
            "population_responses": estimate.get("population_responses", 0),
            "probabilities": dict(zip(("fold", "call", "raise"), (value / sum(alpha) for value in alpha))),
            "fold_lower": estimate["fold_lower"], "fold_upper": estimate["fold_upper"],
            "raise_given_continue": alpha[2] / (alpha[1] + alpha[2]),
            "raise_given_continue_lower": float(raise_lower),
            "raise_given_continue_upper": float(raise_upper),
            "prior": dict(zip(("fold", "call", "raise"), prior))}


def _terminal(obs: Observation, hero_paid: int, other_paid: int) -> tuple[tuple[float, float, float], dict]:
    hero, other = obs.actor, 1 - obs.actor
    difference = obs.bets[hero] + hero_paid - obs.bets[other] - other_paid
    contested = obs.pot + hero_paid + other_paid - abs(difference)
    refund = max(difference, 0)
    rake = min(obs.rake_cap, contested * obs.rake_bps // 10000)
    payout = contested - rake
    tie_award = payout // 2 + (payout % 2 if hero == 0 else 0)
    outcomes = (payout + refund - hero_paid, tie_award + refund - hero_paid, refund - hero_paid)
    return outcomes, {"contested_pot": contested, "rake": rake, "hero_refund": refund,
                      "hero_paid": hero_paid, "opponent_paid": other_paid}


def _showdown(outcomes: tuple[float, float, float], values: tuple[int, ...], hero_value: int) -> tuple[float, ...]:
    win, tie, lose = outcomes
    return tuple(win if hero_value > value else tie if hero_value == value else lose for value in values)


def _raise_targets(obs: Observation, action: Action) -> tuple[int, ...]:
    hero, other = obs.actor, 1 - obs.actor
    paid = action.amount - obs.bets[hero]
    call = min(obs.stacks[other], action.amount - obs.bets[other])
    if paid >= obs.stacks[hero] or call >= obs.stacks[other]:
        return ()
    minimum = action.amount + max(obs.big_blind, action.amount - max(obs.bets))
    maximum = obs.bets[other] + obs.stacks[other]
    effective = min(maximum, obs.bets[hero] + obs.stacks[hero])
    pot = obs.pot + paid + call
    requested = [action.amount + max(1, round(fraction * pot)) for fraction in _RAISE_FRACTIONS]
    requested.append(effective)
    return tuple(sorted({min(maximum, max(minimum, min(effective, target))) for target in requested
                         if min(maximum, max(minimum, min(effective, target))) > action.amount}))


def _size_coordinates(obs: Observation, action: Action, targets: tuple[int, ...]) -> tuple[float, ...]:
    hero, other = obs.actor, 1 - obs.actor
    paid = action.amount - obs.bets[hero]
    call = min(obs.stacks[other], action.amount - obs.bets[other])
    pot = obs.pot + paid + call
    return tuple(log1p((target - action.amount) / pot) for target in targets)


def _size_probabilities(obs: Observation, action: Action, targets: tuple[int, ...],
                        values: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    if not targets:
        return ()
    public = sorted(_board_range(obs.board)[1])
    quality = tuple((bisect_left(public, value) + bisect_right(public, value)) / (2 * len(public))
                    for value in values)
    coordinates = _size_coordinates(obs, action, targets)
    rows = []
    for value in quality:
        logits = tuple(2 * (value - 0.5) * coordinate for coordinate in coordinates)
        maximum = max(logits)
        scores = tuple(exp(logit - maximum) for logit in logits)
        total = fsum(scores)
        rows.append(tuple(score / total for score in scores))
    return tuple(tuple(row[index] for row in rows) for index in range(len(targets)))


def _observed_size_bin(obs: Observation, action: Action, targets: tuple[int, ...], actual: int) -> int:
    coordinates = _size_coordinates(obs, action, targets)
    observed = _size_coordinates(obs, action, (actual,))[0]
    return min(range(len(targets)), key=lambda index: abs(coordinates[index] - observed))


def _conditioned_call(weights: tuple[float, ...], raises: dict[str, tuple[float, ...]],
                      size_likelihood: tuple[float, ...], payoffs: tuple[float, ...],
                      fold_value: float) -> dict:
    scenarios = {}
    for scenario in _SCENARIOS:
        joint = tuple(weight * raised * size for weight, raised, size
                      in zip(weights, raises[scenario], size_likelihood))
        mass = fsum(joint)
        gain = fsum(weight * (payoff - fold_value) for weight, payoff in zip(joint, payoffs)) / mass if mass else None
        scenarios[scenario] = {"raise_range_mass": mass,
                               "call_value_chips": gain + fold_value if gain is not None else None,
                               "call_gain_chips": gain, "fold_value_chips": fold_value}
    evidence = fsum(_SCENARIO_PRIORS[name] * row["raise_range_mass"] for name, row in scenarios.items())
    posterior = {name: _SCENARIO_PRIORS[name] * row["raise_range_mass"] / evidence if evidence else 0.0
                 for name, row in scenarios.items()}
    mixture_gain = (fsum(posterior[name] * (row["call_gain_chips"] or 0.0)
                        for name, row in scenarios.items()) if evidence else None)
    call = mixture_gain is not None and mixture_gain >= 0
    return {"scenarios": scenarios, "model_prior_weights": dict(_SCENARIO_PRIORS),
            "model_posterior_weights": posterior, "raise_size_evidence": evidence,
            "mixture_call_value_chips": mixture_gain + fold_value if mixture_gain is not None else None,
            "fold_value_chips": fold_value, "mixture_call_gain_chips": mixture_gain,
            "continuation_action": "call" if call else "fold"}


def _responses(obs: Observation, action: Action, weights: tuple[float, ...],
               odds: tuple[tuple[float, ...], tuple[float, ...]], posterior: dict,
               scenario: str, can_raise: bool,
               public_weights: tuple[float, ...],
               public_odds: tuple[tuple[float, ...], tuple[float, ...]]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    fold = posterior["probabilities"]["fold"]
    raised = posterior["raise_given_continue"]
    if scenario == "less_folding_more_raising":
        fold, raised = posterior["fold_lower"], posterior["raise_given_continue_upper"]
    elif scenario == "more_folding_less_raising":
        fold, raised = posterior["fold_upper"], posterior["raise_given_continue_lower"]
    folds, public_folds = _tilt(fold, public_weights, public_odds[0], odds[0])
    continuing = tuple(weight * (1 - probability) for weight, probability in zip(public_weights, public_folds))
    conditional_raises = (_tilt(raised, continuing, public_odds[1], odds[1])[0]
                          if can_raise else (0.0,) * len(weights))
    raises = tuple((1 - folded) * probability for folded, probability in zip(folds, conditional_raises))
    calls = tuple((1 - folded) * (1 - raised) for folded, raised in zip(folds, conditional_raises))
    return folds, calls, raises


def _context(obs: Observation, action: Action) -> str:
    hero, other = obs.actor, 1 - obs.actor
    paid = action.amount - obs.bets[hero]
    call = min(obs.stacks[other], action.amount - obs.bets[other])
    raises = 1 + sum(event.street == "river" and event.action.kind == "raise_to" for event in obs.history)
    return response_context("river", other, raises, obs.pot + paid, call)


def _score_raise(obs: Observation, action: Action, profile: OpponentProfile,
                 weights: tuple[float, ...], odds: tuple[tuple[float, ...], tuple[float, ...]],
                 values: tuple[int, ...], hero_value: int, posteriors: dict,
                 public_weights: tuple[float, ...],
                 public_odds: tuple[tuple[float, ...], tuple[float, ...]]) -> dict:
    hero, other = obs.actor, 1 - obs.actor
    paid = action.amount - obs.bets[hero]
    call = min(obs.stacks[other], action.amount - obs.bets[other])
    context = _context(obs, action)
    if context not in posteriors:
        posteriors[context] = _posterior(profile, context)
    posterior = posteriors[context]
    fold_values, fold_chips = _terminal(obs, paid, 0)
    call_values, call_chips = _terminal(obs, paid, call)
    call_payoffs = _showdown(call_values, values, hero_value)
    targets = _raise_targets(obs, action)
    sizes = _size_probabilities(obs, action, targets, values)
    scenarios, raising_ranges = {}, {}
    for scenario in _SCENARIOS:
        folds, calls, raises = _responses(obs, action, weights, odds, posterior, scenario, bool(targets),
                                         public_weights, public_odds)
        raising_ranges[scenario] = raises
        fold_mass, call_mass, raise_mass = (fsum(weight * probability for weight, probability in zip(weights, branch))
                                          for branch in (folds, calls, raises))
        called_value = fsum(weight * probability * payoff for weight, probability, payoff in zip(weights, calls, call_payoffs))
        call_equity = (fsum(weight * probability * (float(hero_value > value) + 0.5 * (hero_value == value))
                           for weight, probability, value in zip(weights, calls, values)) / call_mass
                       if call_mass else None)
        scenarios[scenario] = {
            "fold_probability": fold_mass, "call_probability": call_mass, "raise_probability": raise_mass,
            "call_range_equity": call_equity, "fold_value_chips": fold_values[0],
            "called_value_chips": called_value / call_mass if call_mass else 0.0,
            "expected_chips": fold_mass * fold_values[0] + called_value,
        }
    branches = []
    for target, likelihood in zip(targets, sizes):
        hero_total = min(obs.stacks[hero], target - obs.bets[hero])
        other_total = target - obs.bets[other]
        outcomes, settlement = _terminal(obs, hero_total, other_total)
        branch = _conditioned_call(weights, raising_ranges, likelihood,
                                   _showdown(outcomes, values, hero_value), -paid)
        branch.update(raise_to=target, settlement=settlement)
        for scenario, row in branch["scenarios"].items():
            value = (row["call_value_chips"] or 0.0) if branch["continuation_action"] == "call" else -paid
            scenarios[scenario]["expected_chips"] += row["raise_range_mass"] * value
        branches.append(branch)
    return {"action": asdict(action), "context": context, "response_posterior": posterior,
            "scenarios": scenarios, "fold_settlement": fold_chips, "call_settlement": call_chips,
            "modeled_raise_targets": targets, "raise_branches": branches}


def _passive_row(obs: Observation, action: Action, weights: tuple[float, ...],
                 values: tuple[int, ...], hero_value: int) -> dict:
    if action.kind == "fold":
        value, settlement = 0.0, None
    else:
        outcomes, settlement = _terminal(obs, obs.to_call, 0)
        value = fsum(weight * payoff for weight, payoff in zip(weights, _showdown(outcomes, values, hero_value)))
    return {"action": asdict(action), "scenarios": {scenario: {"expected_chips": value} for scenario in _SCENARIOS},
            "settlement": settlement}


def _menu(obs: Observation, baseline: Action) -> list[Action]:
    actions = [Action("call" if obs.to_call else "check")]
    if obs.can_fold:
        actions.append(Action("fold"))
    if obs.min_raise_to is not None and obs.max_raise_to is not None:
        hero, other = obs.actor, 1 - obs.actor
        unmatched = max(0, obs.bets[other] - obs.bets[hero] - obs.to_call)
        pot = obs.pot - unmatched + obs.to_call
        effective = min(obs.max_raise_to, obs.bets[other] + obs.stacks[other])
        for fraction in _FRACTIONS:
            requested = obs.bets[hero] + obs.to_call + max(1, round(fraction * pot))
            target = min(obs.max_raise_to, max(obs.min_raise_to, min(effective, requested)))
            actions.append(Action("raise_to", target))
    if baseline not in actions:
        actions.append(baseline)
    return list(dict.fromkeys(actions))


def _continuation(obs: Observation, baseline: Action, profile: OpponentProfile,
                  combinations: tuple[tuple[int, int], ...], values: tuple[int, ...], hero_value: int,
                  odds: tuple[tuple[float, ...], tuple[float, ...]],
                  range_weights: Callable[[Observation], tuple[float, ...]] | None,
                  public_range_weights: Callable[[Observation, tuple[tuple[int, int], ...]], tuple[float, ...]] | None) -> tuple[Action, dict]:
    if range_weights is None or len(obs.history) < 2:
        return baseline, {"applied": False, "reason": "prior raise requires a reconstructed range"}
    own, other = obs.history[-2:]
    if (own.street != "river" or own.actor != obs.actor or own.action.kind != "raise_to"
            or other.street != "river" or other.actor == obs.actor or other.action.kind != "raise_to"):
        return baseline, {"applied": False, "reason": "unsupported prior-raise history"}
    paid = tuple(sum(event.paid for event in (own, other) if event.actor == seat) for seat in (0, 1))
    previous = replace(obs, stacks=tuple(stack + amount for stack, amount in zip(obs.stacks, paid)),
                       bets=tuple(bet - amount for bet, amount in zip(obs.bets, paid)),
                       pot=own.pot_before, to_call=own.to_call_before, history=obs.history[:-2],
                       min_raise_to=None, max_raise_to=None, can_fold=own.to_call_before > 0)
    weights = _normalize(tuple(range_weights(previous)))
    if len(weights) != len(combinations):
        raise ValueError("reconstructed river range does not match combinations")
    posterior = _posterior(profile, _context(previous, own.action))
    public_combinations, public_values = _board_range(obs.board)
    public_weights = _normalize(tuple(public_range_weights(previous, public_combinations)) if public_range_weights
                                else (1.0,) * len(public_combinations))
    if len(public_weights) != len(public_combinations):
        raise ValueError("public river range does not match board combinations")
    public_odds = _response_odds(obs.board, public_values)
    outcomes, settlement = _terminal(obs, obs.to_call, 0)
    payoffs = _showdown(outcomes, values, hero_value)
    targets = _raise_targets(previous, own.action)
    if not targets:
        return baseline, {"applied": False, "reason": "reconstructed raise has no legal response sizes"}
    sizes = _size_probabilities(previous, own.action, targets, values)
    mapped_index = _observed_size_bin(previous, own.action, targets, other.action.amount)
    raising_ranges = {
        scenario: _responses(previous, own.action, weights, odds, posterior, scenario, True,
                             public_weights, public_odds)[2]
        for scenario in _SCENARIOS
    }
    detail = _conditioned_call(weights, raising_ranges, sizes[mapped_index], payoffs, 0.0)
    action = Action(detail["continuation_action"])
    return action, {"evaluated": True, "applied": action != baseline, "bounded_continuation": True,
                    "reason": "aggregate call/fold against the size-conditioned model mixture",
                    **detail, "settlement": settlement,
                    "response_posterior": posterior, "previous_hero_raise_to": own.action.amount,
                    "modeled_opponent_raise_targets": targets,
                    "mapped_opponent_raise_to": targets[mapped_index],
                    "actual_opponent_raise_to": other.action.amount,
                    "off_menu_raise": other.action.amount not in targets,
                    "continuation_gate": "Call when the raise-size-conditioned mixture call value is nonnegative; otherwise fold.",
                    "raise_size_assumption": "Observed raises map to the nearest legal size bin in log1p(pot fraction), with lower-bin ties; actual call price is exact. Public-board hand strength sets the normalized size probabilities. This is a fixed heuristic, not a learned size tendency or point-size likelihood."}


def choose_river_action(obs: Observation, baseline: Action, profile: OpponentProfile,
                        combinations: tuple[tuple[int, int], ...], weights: tuple[float, ...],
                        values: tuple[int, ...], hero_value: int,
                        range_weights: Callable[[Observation], tuple[float, ...]] | None = None,
                        public_range_weights: Callable[[Observation, tuple[tuple[int, int], ...]], tuple[float, ...]] | None = None) -> tuple[Action, dict]:
    scope = {"evaluated": False, "model": "bounded river action values under heuristic conditional ranges", "model_version": 2,
             "range_assumptions": "Public-board strength ranks control responses and relative raise-size probabilities; aggregate historical rates do not identify true calling/raising ranges or size tendencies.",
             "sensitivity_gate": "Initial action changes require at least 0.25BB improvement over baseline in every response scenario. Raise continuations use an equal-prior heuristic model mixture conditioned on the observed size bin. Neither is calibrated Bayesian uncertainty, a confidence interval or a safety guarantee.",
             "response_calibration": "Rates are calibrated against public-board-legal combinations and public history, without hero-card blockers; hero blockers only restrict payoff integration.",
             "tree": "One hero bet/raise, opponent fold/call or legal 33%/75%/150%-pot/effective-all-in raise bins, then one shared hero call/fold decision per size bin. Collapsed legal sizes are deduplicated before probabilities are normalized.",
             "default_policy_replaced": False}
    if obs.street != "river" or len(obs.board) != 5:
        return baseline, {**scope, "applied": False, "reason": "river only"}
    if obs.to_call == 0 and obs.actor != obs.button:
        return baseline, {**scope, "applied": False, "reason": "out-of-position checking requires another betting subtree"}
    if not (len(combinations) == len(weights) == len(values)) or not combinations:
        raise ValueError("river combinations, weights and values must have matching nonzero lengths")
    weights = _normalize(weights)
    odds = _response_odds(obs.board, values)
    if any(event.street == "river" and event.actor == obs.actor and event.action.kind == "raise_to"
           for event in obs.history):
        action, info = _continuation(obs, baseline, profile, combinations, values, hero_value, odds, range_weights,
                                     public_range_weights)
        return action, {**scope, **info}
    public_combinations, public_values = _board_range(obs.board)
    public_weights = _normalize(tuple(public_range_weights(obs, public_combinations)) if public_range_weights
                                else (1.0,) * len(public_combinations))
    if len(public_weights) != len(public_combinations):
        raise ValueError("public river range does not match board combinations")
    public_odds = _response_odds(obs.board, public_values)
    posteriors = {}
    rows = []
    for action in _menu(obs, baseline):
        row = (_score_raise(obs, action, profile, weights, odds, values, hero_value, posteriors, public_weights, public_odds)
               if action.kind == "raise_to" else _passive_row(obs, action, weights, values, hero_value))
        rows.append(row)
    reference = next(row for row in rows if row["action"] == asdict(baseline))
    margin = _MARGIN_BB * obs.big_blind
    for row in rows:
        row["worst_case_gain_chips"] = min(row["scenarios"][scenario]["expected_chips"]
                                           - reference["scenarios"][scenario]["expected_chips"] for scenario in _SCENARIOS)
    eligible = [row for row in rows if row["worst_case_gain_chips"] >= margin]
    chosen = max(eligible, key=lambda row: row["worst_case_gain_chips"]) if eligible else reference
    action = Action(**chosen["action"])
    return action, {**scope, "evaluated": True, "applied": action != baseline, "candidates": rows,
                    "margin_chips": margin, "scenario_names": _SCENARIOS,
                    "baseline_action": asdict(baseline), "selected_action": asdict(action),
                    "worst_case_gain_chips": chosen["worst_case_gain_chips"],
                    "reason": "improves on baseline across response scenarios" if action != baseline else "no robust improvement over baseline"}
