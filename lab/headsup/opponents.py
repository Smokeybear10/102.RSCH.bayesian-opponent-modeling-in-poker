from collections import Counter
from random import Random

import eval7

from .models import MAX_SAMPLES, Action, Observation

HELDOUT_NAMES = ("heldout_fit_fold", "heldout_pressure", "heldout_loose")
RIVER_SIZING_NAMES = ("river_small_merged", "river_large_polarized", "river_bluff_pressure")
_RANKS = "23456789TJQKA"
_CATEGORY = {
    "High Card": 0.10, "Pair": 0.44, "Two Pair": 0.70, "Trips": 0.78,
    "Straight": 0.86, "Flush": 0.90, "Full House": 0.96,
    "Quads": 0.99, "Straight Flush": 1.0,
}


def _card_features(obs: Observation) -> tuple[float, bool]:
    ranks = [_RANKS.index(card[0]) + 2 for card in obs.hole]
    high, low = max(ranks), min(ranks)
    if not obs.board:
        if high == low:
            return 0.55 + (high - 2) / 30, False
        score = (high + low - 4) / 32
        score += 0.09 if obs.hole[0][1] == obs.hole[1][1] else 0
        score += 0.07 if high - low == 1 else 0.03 if high - low == 2 else 0
        return min(0.95, score), False
    cards = obs.hole + obs.board
    category = eval7.handtype(eval7.evaluate([eval7.Card(card) for card in cards]))
    score = _CATEGORY[category]
    board_ranks = [_RANKS.index(card[0]) + 2 for card in obs.board]
    if category == "Pair":
        counts = Counter(ranks + board_ranks)
        paired = max(rank for rank, count in counts.items() if count == 2)
        if ranks.count(paired) == 0:
            score = 0.17 + (high - 2) / 100
        elif paired >= max(board_ranks):
            score += 0.17
        else:
            score += 0.06
    elif category == "High Card":
        score += (high - 2) / 100
    draw = False
    if len(obs.board) < 5:
        suits = Counter(card[1] for card in cards)
        flush = any(count == 4 and any(card[1] == suit for card in obs.hole)
                    for suit, count in suits.items())
        distinct = set(ranks + board_ranks)
        if 14 in distinct:
            distinct.add(1)
        own = set(ranks) | ({1} if 14 in ranks else set())
        straight = any(len(distinct.intersection(range(start, start + 5))) == 4
                       and own.intersection(range(start, start + 5))
                       for start in range(1, 11))
        draw = flush or straight
        score += 0.12 if flush else 0.06 if straight else 0
    return min(score, 1.0), draw


class SyntheticOpponent:
    def __init__(self, name: str, seed: int, samples: int = 256) -> None:
        if name not in HELDOUT_NAMES:
            raise ValueError(f"unknown held-out opponent: {name}")
        if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
            raise ValueError(f"samples must be an integer from 1 to {MAX_SAMPLES}")
        self.name = name
        self.rng = Random(seed)
        self.last_info: dict = {}

    def act(self, obs: Observation) -> Action:
        score, draw = _card_features(obs)
        price = obs.to_call / max(1, obs.pot + obs.to_call)
        pressure = self.name == "heldout_pressure"
        loose = self.name == "heldout_loose"
        can_raise = obs.min_raise_to is not None and obs.max_raise_to is not None
        passive = Action("call" if obs.to_call else "check")
        action, reason = passive, "continue passively"
        fraction = 0.5
        raise_requested = False
        if not obs.board:
            voluntary_raise = any(event.action.kind == "raise_to" for event in obs.history)
            opening = obs.actor == obs.button and not obs.history
            threshold = (0.10 if loose else 0.20 if pressure else 0.36)
            if voluntary_raise:
                threshold += (0.18 if loose else 0.28) * price
                threshold += 0.20 * obs.to_call / max(1, obs.stacks[obs.actor])
            if obs.to_call and score < threshold and obs.can_fold:
                action, reason = Action("fold"), "preflop card filter"
            elif opening:
                raise_requested = not loose or score > 0.64
                reason = "button open" if raise_requested else "button limp"
            elif can_raise and (score > (0.80 if loose else 0.72)
                                or pressure and not voluntary_raise and self.rng.random() < 0.28):
                raise_requested, reason = True, "preflop reraise or isolation"
            fraction = 0.75 if pressure else 0.5
        else:
            threshold = ((0.14 + 0.30 * price) if loose else
                         (0.24 + 0.38 * price) if pressure else (0.41 + 0.42 * price))
            if draw and price < 0.22:
                threshold -= 0.10
            if obs.to_call and score < threshold and obs.can_fold:
                action, reason = Action("fold"), "decline price with weak card features"
            else:
                value_threshold = 0.74 if loose else 0.63 if pressure else 0.68
                value = score > value_threshold
                bluff_chance = 0.24 if pressure else 0.015 if loose else 0.045
                bluff = not obs.to_call and self.rng.random() < bluff_chance
                raise_requested = can_raise and ((value and self.rng.random() < 0.75) or bluff)
                if raise_requested:
                    reason = "feature-based value bet" if value else "randomized bluff"
                fraction = self.rng.choice((0.33, 0.75, 1.25)) if pressure else 0.55 if loose else 0.70
        if raise_requested and can_raise:
            target = obs.bets[obs.actor] + obs.to_call + max(1, round((obs.pot + obs.to_call) * fraction))
            if not obs.board and not obs.history:
                target = round((3.0 if pressure else 2.5) * obs.big_blind)
            action = Action("raise_to", min(obs.max_raise_to, max(obs.min_raise_to, target)))
        self.last_info = {
            "policy": self.name, "model": "independent synthetic card-feature heuristic; not calibrated equity",
            "card_feature_score": score, "draw": draw, "reason": reason,
            "action": {"kind": action.kind, "amount": action.amount},
        }
        return action


class RiverSizingOpponent:
    def __init__(self, name: str, seed: int, samples: int = 256) -> None:
        if name not in RIVER_SIZING_NAMES:
            raise ValueError(f"unknown river sizing opponent: {name}")
        self.name = name
        self.base = SyntheticOpponent("heldout_loose", seed, samples)
        self.rng = self.base.rng
        self.last_info: dict = {}

    def act(self, obs: Observation) -> Action:
        if obs.street != "river":
            action = self.base.act(obs)
            self.last_info = {**self.base.last_info, "policy": self.name}
            return action
        score, _ = _card_features(obs)
        price = obs.to_call / max(1, obs.pot + obs.to_call)
        can_raise = obs.min_raise_to is not None and obs.max_raise_to is not None
        own_raises = sum(event.street == "river" and event.actor == obs.actor
                         and event.action.kind == "raise_to" for event in obs.history)
        if self.name == "river_small_merged":
            value_threshold, value_frequency, bluff_frequency = 0.61, 0.65, 0.04
            fractions = (0.18, 0.38, 0.55)
            call_threshold = 0.23 + 0.46 * price
        elif self.name == "river_large_polarized":
            value_threshold, value_frequency, bluff_frequency = 0.86, 0.85, 0.16
            fractions = (1.40, 2.10, 3.00)
            call_threshold = 0.28 + 0.50 * price
        else:
            value_threshold, value_frequency, bluff_frequency = 0.70, 0.75, 0.36
            fractions = (0.24, 0.90, 1.80)
            call_threshold = 0.19 + 0.43 * price
        value = score >= value_threshold
        bluff_eligible = score < 0.35 if self.name == "river_large_polarized" else score < 0.61
        frequency = value_frequency if value else bluff_frequency if bluff_eligible else 0.0
        if own_raises and not value:
            frequency *= 0.35
        raise_requested = can_raise and self.rng.random() < frequency
        fraction = None
        if raise_requested:
            fraction = self.rng.choice(fractions)
            target = obs.bets[obs.actor] + obs.to_call + max(1, round((obs.pot + obs.to_call) * fraction))
            action = Action("raise_to", min(obs.max_raise_to, max(obs.min_raise_to, target)))
            reason = "river feature-based value raise" if value else "river randomized bluff raise"
        elif obs.to_call and obs.can_fold and score < call_threshold:
            action, reason = Action("fold"), "river feature-based price filter"
        else:
            action = Action("call" if obs.to_call else "check")
            reason = "river passive continuation"
        self.last_info = {
            "policy": self.name, "model": "independent synthetic card-feature strategy; not estimated equity",
            "card_feature_score": score, "reason": reason, "river_raise_fraction": fraction,
            "action": {"kind": action.kind, "amount": action.amount},
        }
        return action


def make_opponent(name: str, seed: int, samples: int = 256) -> SyntheticOpponent | RiverSizingOpponent:
    if name in RIVER_SIZING_NAMES:
        return RiverSizingOpponent(name, seed, samples)
    return SyntheticOpponent(name, seed, samples)
