"""Seeded heuristic policies for the local arena; none is a solved HUNL strategy."""

from collections import Counter
from hashlib import sha256
from itertools import combinations
from math import fsum, sqrt
from random import Random

import eval7

from .models import MAX_SAMPLES, Action, Event, Observation
from .learning import OpponentProfile, response_context

POLICY_NAMES = ("range", "equity", "calling", "tight", "aggressive", "precision")
LEARNING_POLICY_NAMES = ("adaptive", "river")
_RANKS = "23456789TJQKA"
_CODES = tuple(rank + suit for rank in _RANKS for suit in "cdhs")
_INDEX = {card: index for index, card in enumerate(_CODES)}
_CARDS = tuple(eval7.Card(card) for card in _CODES)
_HOLES = tuple(combinations(range(52), 2))
_MADE_STRENGTH = {
    "High Card": 0.20, "Pair": 0.52, "Two Pair": 0.70, "Trips": 0.78,
    "Straight": 0.86, "Flush": 0.90, "Full House": 0.95,
    "Quads": 0.98, "Straight Flush": 1.0,
}


def _preflop(hole: tuple[int, int]) -> float:
    high, low = sorted((card // 4 + 2 for card in hole), reverse=True)
    if high == low:
        return 0.48 + 0.50 * (high - 2) / 12
    suited = hole[0] % 4 == hole[1] % 4
    gap = high - low
    connected = 0.06 if gap == 1 else 0.03 if gap == 2 else 0
    wheel = 0.03 if high == 14 and low <= 5 else 0
    return min(0.97, 0.48 * (high - 2) / 12 + 0.23 * (low - 2) / 12
               + 0.07 * suited + connected + wheel)


def _strength(hole: tuple[int, int], board: tuple[int, ...]) -> float:
    if not board:
        return _preflop(hole)
    cards = hole + board
    value = eval7.evaluate([_CARDS[card] for card in cards])
    category = eval7.handtype(value)
    strength = _MADE_STRENGTH[category]
    ranks = Counter(card // 4 + 2 for card in cards)
    board_ranks = Counter(card // 4 + 2 for card in board)
    hole_ranks = {card // 4 + 2 for card in hole}
    if category == "High Card":
        strength += 0.12 * _preflop(hole)
    elif category == "Pair":
        pair = max(rank for rank, count in ranks.items() if count >= 2)
        strength += 0.12 * (pair - 2) / 12
        if board_ranks[pair] >= 2 and pair not in hole_ranks:
            strength -= 0.18
    if len(board) < 5:
        suits = Counter(card % 4 for card in cards)
        flush_draw = any(count == 4 and any(card % 4 == suit for card in hole)
                         for suit, count in suits.items())
        distinct = set(ranks)
        if 14 in distinct:
            distinct.add(1)
        own = hole_ranks | ({1} if 14 in hole_ranks else set())
        straight_draw = any(len(distinct & set(range(start, start + 5))) == 4
                            and bool(own & set(range(start, start + 5)))
                            for start in range(1, 11))
        strength += 0.10 * flush_draw + 0.05 * straight_draw
    return min(1.0, max(0.0, strength))


def _likelihood(hole: tuple[int, int], event: Event) -> float:
    board = tuple(_INDEX[card] for card in event.board)
    quality = _strength(hole, board)
    if event.action.kind == "raise_to":
        size = min(3.0, max(0, event.paid - event.to_call_before)
                   / max(1, event.pot_before + event.to_call_before))
        return 0.04 + (3.0 + size) * quality ** (2.0 + size)
    if event.action.kind == "call":
        return 0.12 + quality * (1.8 - 0.8 * quality)
    return 0.35 + 0.65 * (1.0 - quality)


def _range_weights(obs: Observation, pool: tuple[tuple[int, int], ...]) -> tuple[float, ...]:
    evidence = tuple(event for event in obs.history
                     if event.actor != obs.actor and event.action.kind in ("check", "call", "raise_to"))[-8:]
    weights = []
    for hand in pool:
        weight = 1.0
        for event in evidence:
            weight *= _likelihood(hand, event)
        weights.append(weight)
    return tuple(weights)


class Policy:
    def __init__(self, name: str, seed: int, samples: int = 256) -> None:
        if name not in POLICY_NAMES:
            raise ValueError(f"unknown policy: {name}")
        if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
            raise ValueError(f"samples must be an integer from 1 to {MAX_SAMPLES}")
        self.name = name
        self.samples = samples
        self.rng = Random(seed)
        self.last_info: dict = {}

    def _finish(self, action: Action, reason: str) -> Action:
        self.last_info.update(action={"kind": action.kind, "amount": action.amount}, reason=reason)
        return action

    def _passive(self, obs: Observation) -> Action:
        return Action("call" if obs.to_call else "check")

    def _fold(self, obs: Observation) -> Action:
        return Action("fold") if obs.to_call and obs.can_fold else self._passive(obs)

    def _raise(self, obs: Observation, target: int) -> Action:
        if obs.min_raise_to is None or obs.max_raise_to is None:
            return self._passive(obs)
        return Action("raise_to", min(obs.max_raise_to, max(obs.min_raise_to, target)))

    def _sized_raise(self, obs: Observation, fraction: float) -> Action:
        after_call = self.last_info["contestable_pot"] + obs.to_call
        target = obs.bets[obs.actor] + obs.to_call + max(1, round(fraction * after_call))
        return self._raise(obs, target)

    def _equity(self, obs: Observation, hole: tuple[int, int], board: tuple[int, ...]) -> float:
        known = set(hole + board)
        available = tuple(card for card in range(52) if card not in known)
        evidence = tuple(event for event in obs.history
                         if event.actor != obs.actor and event.action.kind in ("check", "call", "raise_to"))[-8:]
        if self.name not in ("range", "precision"):
            evidence = ()
        pool = tuple(hand for hand in _HOLES if not known.intersection(hand))
        weights = []
        if evidence:
            weights = list(_range_weights(obs, pool))
        own_cards = [_CARDS[card] for card in hole]
        board_cards = [_CARDS[card] for card in board]
        effective = fsum(weights) ** 2 / fsum(weight * weight for weight in weights) if weights else len(pool)
        self.last_info.update(evidence_events=len(evidence), range_combinations=len(pool),
                              effective_range_combinations=effective)
        if len(board) == 5:
            own_value = eval7.evaluate(own_cards + board_cards)
            values = [eval7.evaluate([_CARDS[card] for card in hand] + board_cards) for hand in pool]
            masses = weights or [1.0] * len(pool)
            self._river_distribution = (pool, tuple(masses), tuple(values), own_value)
            total_weight = fsum(masses)
            win = fsum(weight for value, weight in zip(values, masses) if own_value > value) / total_weight
            tie = fsum(weight for value, weight in zip(values, masses) if own_value == value) / total_weight
            equity = win + tie / 2
            self.last_info.update(equity=equity, win_probability=win, tie_probability=tie,
                                  equity_standard_error=0.0, samples=0, evaluated_combinations=len(pool),
                                  equity_method="exact enumeration conditional on the heuristic range")
            return equity
        opponents = (self.rng.choices(pool, weights=weights, k=self.samples) if weights
                     else [self.rng.sample(available, 2) for _ in range(self.samples)])
        wins = ties = 0
        for other in opponents:
            remaining = [card for card in available if card not in other]
            runout = board_cards + [_CARDS[card] for card in self.rng.sample(remaining, 5 - len(board))]
            own_value = eval7.evaluate(own_cards + runout)
            other_value = eval7.evaluate([_CARDS[card] for card in other] + runout)
            wins += own_value > other_value
            ties += own_value == other_value
        equity = (wins + ties / 2) / self.samples
        square = wins + ties / 4
        standard_error = (sqrt(max(0.0, square / self.samples - equity * equity) / (self.samples - 1))
                          if self.samples > 1 else None)
        self.last_info.update(equity=equity, equity_standard_error=standard_error, samples=self.samples,
                              win_probability=wins / self.samples, tie_probability=ties / self.samples,
                              equity_method="fixed-sample conditional Monte Carlo")
        if self.name == "precision":
            return self._refine_equity(obs, hole, board, equity)
        return equity

    def _refine_equity(self, obs: Observation, hole: tuple[int, int], board: tuple[int, ...],
                       equity: float) -> float:
        thresholds = []
        if obs.to_call:
            terminal = obs.stacks[1 - obs.actor] == 0 or obs.to_call == obs.stacks[obs.actor]
            if terminal:
                final_pot = self.last_info["contestable_pot"] + obs.to_call
                payout = final_pot - min(obs.rake_cap, final_pot * obs.rake_bps // 10000)
                if payout:
                    tie_award = payout // 2 + (payout % 2 if obs.actor == (obs.button + 1) % 2 else 0)
                    thresholds.append((obs.to_call - self.last_info["tie_probability"]
                                       * (tie_award - payout / 2)) / payout)
            else:
                thresholds.append(self.last_info["pot_odds"])
        if obs.min_raise_to is not None and obs.max_raise_to is not None:
            thresholds.extend((0.76,) if obs.to_call else (0.38, 0.62))
        standard_error = self.last_info["equity_standard_error"]
        radius = max(0.02, 1 / sqrt(self.samples), 2 * (standard_error or 0))
        budget = min(MAX_SAMPLES - self.samples, max(512, 4 * self.samples))
        detail = {"refined": False, "pilot_equity": equity, "pilot_samples": self.samples,
                  "pilot_standard_error": standard_error, "thresholds": thresholds,
                  "refinement_samples": 0, "total_samples": self.samples,
                  "uncertainty_scope": "sampling only; heuristic range error is not measured"}
        self.last_info["precision"] = detail
        if budget <= self.samples or not any(abs(equity - threshold) <= radius for threshold in thresholds):
            return equity
        # A separate fixed batch avoids reusing the pilot selected for being near a threshold.
        # Its RNG must not move the action-mixing stream or require extra reconnect state.
        state = repr(self.rng.getstate()).encode()
        seed = int.from_bytes(sha256(b"precision-equity-v1:" + state).digest()[:8], "big")
        refined = Policy("range", seed, budget)
        equity = refined._equity(obs, hole, board)
        self.last_info.update(refined.last_info,
                              equity_method="independent fixed refinement batch conditional on the heuristic range")
        detail.update(refined=True, refinement_samples=budget, total_samples=self.samples + budget)
        return equity

    def _preflop_action(self, obs: Observation, score: float) -> Action:
        raises = sum(event.actor != obs.actor and event.action.kind == "raise_to"
                     for event in obs.history[-16:])
        position = obs.actor == obs.button
        opener = not any(event.action.kind in ("call", "raise_to") for event in obs.history)
        aggressive = self.name == "aggressive"
        tight = self.name == "tight"
        if opener and position:
            threshold = 0.56 if tight else 0.24 if aggressive else 0.33
            if score < threshold:
                return self._finish(self._fold(obs), "heuristic small-blind opening filter")
            return self._finish(self._raise(obs, round(2.5 * obs.big_blind)), "heuristic small-blind open")
        if not obs.to_call:
            threshold = 0.73 if tight else 0.49 if aggressive else 0.64
            if score > threshold:
                return self._finish(self._raise(obs, 3 * obs.big_blind), "heuristic value raise versus limp")
            return self._finish(Action("check"), "take the free preflop option")
        price = self.last_info["pot_odds"]
        commitment = obs.to_call / max(1, obs.stacks[obs.actor])
        threshold = 0.26 + 0.30 * price + 0.18 * commitment + 0.07 * min(raises, 3)
        threshold += 0.13 if tight else -0.07 if aggressive else 0
        if score < threshold:
            return self._finish(self._fold(obs), "hand score below heuristic calling threshold")
        raise_threshold = 0.82 if tight else 0.66 if aggressive else 0.76
        if score > raise_threshold and raises < 3 and self.rng.random() < 0.8:
            return self._finish(self._sized_raise(obs, 0.75), "heuristic preflop value reraise")
        return self._finish(self._passive(obs), "hand score supports a heuristic preflop call")

    def _reference_postflop(self, obs: Observation, quality: float) -> Action:
        tight = self.name == "tight"
        price = self.last_info["pot_odds"]
        if obs.to_call and quality < (0.43 if tight else 0.22) + 0.45 * price:
            return self._finish(self._fold(obs), "reference style declines the price")
        value = quality >= (0.75 if tight else 0.60)
        bluff = not obs.to_call and not tight and self.rng.random() < 0.20
        raisable = obs.min_raise_to is not None and obs.max_raise_to is not None
        if raisable and ((value and self.rng.random() < (0.65 if tight else 0.85)) or bluff):
            return self._finish(self._sized_raise(obs, 0.65), "reference value bet" if value else "reference randomized bluff")
        return self._finish(self._passive(obs), "reference style continues passively")

    def act(self, obs: Observation) -> Action:
        other = 1 - obs.actor
        unmatched = max(0, obs.bets[other] - obs.bets[obs.actor] - obs.to_call)
        contestable = obs.pot - unmatched
        self.last_info = {
            "policy": self.name, "model": "heuristic; not GTO or calibrated opponent probabilities",
            "equity": None, "samples": 0, "evidence_events": 0,
            "contestable_pot": contestable,
            "pot_odds": obs.to_call / (contestable + obs.to_call) if obs.to_call else 0.0,
            "rake_in_decision_model": False,
        }
        if self.name == "calling":
            return self._finish(self._passive(obs), "calling reference takes every call and free check")
        hole = tuple(_INDEX[card] for card in obs.hole)
        board = tuple(_INDEX[card] for card in obs.board)
        terminal_call = (self.name in ("range", "equity", "precision") and obs.to_call > 0
                         and (obs.stacks[other] == 0 or obs.to_call == obs.stacks[obs.actor]))
        if not board and not terminal_call:
            score = _preflop(hole)
            self.last_info["preflop_hand_score"] = score
            return self._preflop_action(obs, score)
        if self.name in ("tight", "aggressive"):
            quality = _strength(hole, board)
            self.last_info["made_hand_draw_score"] = quality
            return self._reference_postflop(obs, quality)
        equity = self._equity(obs, hole, board)
        if obs.to_call and (terminal_call or len(board) == 5):
            final_pot = contestable + obs.to_call
            # A terminal call runs out the full board, including preflop all-ins.
            rake = min(obs.rake_cap, final_pot * obs.rake_bps // 10000)
            payout = final_pot - rake
            tie_award = payout // 2 + (payout % 2 if obs.actor == (obs.button + 1) % 2 else 0)
            call_value = (self.last_info["win_probability"] * payout
                          + self.last_info["tie_probability"] * tie_award - obs.to_call)
            self.last_info.update(rake_in_decision_model=True, call_rake_chips=rake,
                                  call_value_chips=call_value, call_tie_award=tie_award,
                                  rake_adjustment_scope="terminal call versus fold only")
            if call_value < 0:
                return self._finish(self._fold(obs), "conditional terminal call value is negative after rake")
        elif obs.to_call and equity < self.last_info["pot_odds"]:
            return self._finish(self._fold(obs), "sampled equity below the immediate call price")
        raisable = obs.min_raise_to is not None and obs.max_raise_to is not None
        if raisable and equity >= (0.76 if obs.to_call else 0.62) and self.rng.random() < 0.80:
            return self._finish(self._sized_raise(obs, 0.65), "mixed heuristic value bet")
        if raisable and not obs.to_call and equity < 0.38 and self.rng.random() < 0.12:
            return self._finish(self._sized_raise(obs, 0.65), "mixed heuristic bluff; fold response is not calibrated")
        return self._finish(self._passive(obs), "continue at the current price" if obs.to_call else "check and preserve showdown value")


def make_policy(name: str, seed: int, samples: int = 256) -> Policy:
    return Policy(name, seed, samples)


class AdaptivePolicy(Policy):
    def __init__(self, seed: int, samples: int, profile: OpponentProfile) -> None:
        super().__init__("range", seed, samples)
        self._profile = OpponentProfile.from_dict(profile.to_dict())
        self._profile_hash = self._profile.fingerprint
        self._estimates: dict[str, dict] = {}

    def _response_estimate(self, obs: Observation, action: Action) -> dict:
        opponent = 1 - obs.actor
        paid = action.amount - obs.bets[obs.actor]
        call = min(obs.stacks[opponent], action.amount - obs.bets[opponent])
        raises = 1 + sum(event.street == obs.street and event.action.kind == "raise_to"
                         for event in obs.history)
        context = response_context(obs.street, opponent, raises, obs.pot + paid, call)
        if context not in self._estimates:
            self._estimates[context] = self._profile.estimate(context)
        return self._estimates[context]

    def act(self, obs: Observation) -> Action:
        baseline = super().act(obs)
        self.last_info.update(policy="adaptive", base_policy="range", adaptation_applied=False,
                              opponent_profile_sha256=self._profile_hash,
                              opponent_hands_observed=self._profile.hands_observed,
                              baseline_action={"kind": baseline.kind, "amount": baseline.amount})
        if obs.min_raise_to is None or obs.max_raise_to is None:
            return baseline
        opening_fold = (not obs.board and obs.actor == obs.button and baseline.kind == "fold"
                        and not any(event.action.kind in ("call", "raise_to") for event in obs.history))
        weak_free_action = (bool(obs.board) and not obs.to_call
                            and self.last_info["equity"] is not None and self.last_info["equity"] < 0.38)
        baseline_bluff = weak_free_action and baseline.kind == "raise_to"
        if not opening_fold and not weak_free_action:
            return baseline
        proposed = (self._raise(obs, round(2.5 * obs.big_blind)) if opening_fold
                    else baseline if baseline_bluff else self._sized_raise(obs, 0.65))
        if proposed.kind != "raise_to":
            return baseline
        estimate = self._response_estimate(obs, proposed)
        self.last_info["opponent_response"] = estimate
        if not estimate["ready"]:
            return baseline
        risk = proposed.amount - obs.bets[obs.actor]
        fold_pot = self.last_info["contestable_pot"] + obs.to_call
        rake = (0 if obs.no_flop_no_drop and not obs.board else
                min(obs.rake_cap, fold_pot * obs.rake_bps // 10000))
        fold_gain = self.last_info["contestable_pot"] - rake
        break_even = risk / (risk + fold_gain)
        self.last_info.update(bluff_fold_break_even=break_even,
                              adaptation_scope="heuristic fold-response adjustments; not calibrated full action value")
        if baseline_bluff and estimate["fold_upper"] < break_even:
            self.last_info["adaptation_applied"] = True
            return self._finish(Action("check"), "learned low fold rate suppresses the baseline bluff")
        if baseline.kind not in ("fold", "check") or estimate["fold_lower"] <= break_even + 0.05:
            return baseline
        # Cap both additional exposure and mixing; repeated opponents can change style.
        if risk > 0.25 * obs.stacks[obs.actor] or (opening_fold and risk > 4 * obs.big_blind):
            return baseline
        frequency = min(0.60, estimate["responses"] / (estimate["responses"] + 80))
        self.last_info["adaptive_bet_frequency"] = frequency
        if self.rng.random() >= frequency:
            return baseline
        self.last_info["adaptation_applied"] = True
        return self._finish(proposed, "learned high fold rate supports a bounded additional bet")


def make_adaptive_policy(seed: int, samples: int, profile: OpponentProfile) -> AdaptivePolicy:
    return AdaptivePolicy(seed, samples, profile)


class RiverPolicy(AdaptivePolicy):
    def act(self, obs: Observation) -> Action:
        baseline = super().act(obs)
        self.last_info.update(policy="river", river_evaluated=False, river_action_changed=False)
        if obs.street != "river" or len(obs.board) != 5:
            return baseline
        from .river import choose_river_action

        combinations, weights, values, hero_value = self._river_distribution
        action, detail = choose_river_action(
            obs, baseline, self._profile, combinations, weights, values, hero_value,
            range_weights=lambda earlier: _range_weights(earlier, combinations),
            public_range_weights=_range_weights,
        )
        self.last_info.update(river=detail, river_evaluated=detail.get("evaluated", False),
                              river_action_changed=action != baseline,
                              river_baseline_action={"kind": baseline.kind, "amount": baseline.amount})
        if self.last_info["river_evaluated"]:
            self.last_info.update(
                baseline_rake_in_decision_model=self.last_info["rake_in_decision_model"],
                baseline_rake_adjustment_scope=self.last_info.get("rake_adjustment_scope"),
                rake_in_decision_model=True,
                rake_adjustment_scope="bounded river candidate values and call/fold continuation",
            )
        if action == baseline:
            return baseline
        return self._finish(action, detail.get("reason", "experimental river action-value comparison"))


def make_river_policy(seed: int, samples: int, profile: OpponentProfile) -> RiverPolicy:
    return RiverPolicy(seed, samples, profile)
