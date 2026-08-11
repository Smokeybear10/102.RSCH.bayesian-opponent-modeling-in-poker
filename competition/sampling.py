from collections.abc import Callable
from math import sqrt
from random import Random
from time import perf_counter_ns

from .cards import COMBINATIONS as _ORDERED
from .cards import RANGES
from .equity import Estimate
from .model import Observation, pots

STRAIGHTS = tuple(31 << i for i in range(9)) + ((1 << 12) | 15,)


def _likelihood(
    hole: tuple[int, int],
    board: tuple[int, ...],
    board_cards: list[object],
    board_mask: int,
    top: int,
    suit_counts: list[int],
    board_value: int,
    pressure: int,
    cards: tuple[object, ...],
    evaluate: Callable[[list[object]], int],
    classify: Callable[[int], int],
) -> float:
    if not pressure:
        return 1.0
    rank = evaluate([cards[hole[0]], cards[hole[1]]] + board_cards)
    category = classify(rank)
    first, second = hole[0] // 4, hole[1] // 4
    paired = (bool(board_mask & (1 << first)), bool(board_mask & (1 << second)))
    if category >= 2:
        quality = 0.95
        if len(board) == 5 and rank == board_value:
            quality = 0.3
    elif first == second:
        quality = 0.75 if first >= top else 0.4
    elif any(paired):
        quality = (
            0.72
            if (paired[0] and first == top) or (paired[1] and second == top)
            else 0.4
        )
    else:
        quality = 0.12
    if len(board) < 5:
        suits = list(suit_counts)
        suits[hole[0] % 4] += 1
        suits[hole[1] % 4] += 1
        ranks = board_mask | (1 << first) | (1 << second)
        draw = max(suits) >= 4 or any(
            (ranks & straight).bit_count() >= 4 for straight in STRAIGHTS
        )
        if draw:
            quality = max(quality, 0.55)
    return (
        0.35 + 0.65 * quality
        if pressure == 1
        else 0.1 + 0.9 * quality ** min(pressure, 3)
    )


class SampledEquity:
    def __init__(
        self,
        seed: int,
        *,
        cards: tuple[object, ...],
        evaluate: Callable[[list[object]], int],
        classify: Callable[[int], int] | None = None,
    ) -> None:
        if len(cards) != 52:
            raise ValueError(
                "provide one SDK card value for each canonical card in CARD_NAMES order"
            )
        self.cards = cards
        self.evaluate = evaluate
        self.classify = classify
        self.rng = Random(seed)
        self.known_mask = -1
        self.ranges = {}

    def estimate(
        self,
        obs: Observation,
        ranges: tuple[float, ...],
        pressure: tuple[int, ...],
        samples: int = 40,
        deadline: int | None = None,
        information: tuple[float, ...] | None = None,
    ) -> Estimate:
        cards, evaluate, classify = self.cards, self.evaluate, self.classify
        known = sum(1 << c for c in obs.hole + obs.board)
        if known != self.known_mask:
            self.known_mask = known
            self.ranges.clear()
        opponents = obs.opponents
        choices = []
        for seat in opponents:
            fraction = max(0.1, min(1.0, ranges[seat.id]))
            bucket = min(20, max(2, round(fraction * 20)))
            if bucket not in self.ranges:
                allowed = RANGES[bucket]
                self.ranges[bucket] = tuple(c for c in allowed if not c[2] & known)
            choices.append(self.ranges[bucket])
        committed = tuple(
            s.committed + (obs.legal.call if s.id == obs.hero else 0) for s in obs.seats
        )
        layers = tuple(
            (p.amount, sum(1 << i for i in p.eligible))
            for p in pots(committed, tuple(s.folded for s in obs.seats))
            if obs.hero in p.eligible
        )
        hero_cards = [cards[c] for c in obs.hole]
        board_cards = [cards[c] for c in obs.board]
        board_mask = 0
        suit_counts = [0] * 4
        for card in obs.board:
            board_mask |= 1 << (card // 4)
            suit_counts[card % 4] += 1
        top = max((c // 4 for c in obs.board), default=0)
        board_value = (
            evaluate(board_cards)
            if len(board_cards) == 5 and classify is not None
            else 0
        )
        random = self.rng
        total_weight = squared_weight = weighted_share = weighted_square = (
            weighted_return
        ) = wins = ties = 0.0
        completed = 0
        for trial in range(samples):
            if deadline is not None and perf_counter_ns() >= deadline:
                break
            used = known
            holes = []
            weight = 1.0
            for seat, available in zip(opponents, choices):
                selected = None
                for _ in range(12):
                    candidate = available[random.randrange(len(available))]
                    if not candidate[2] & used:
                        selected = candidate
                        break
                if selected is None:
                    legal = [c for c in available if not c[2] & used]
                    if not legal:
                        # Sequential narrow ranges can conflict; widen instead of dealing duplicate cards.
                        legal = [c for c in _ORDERED if not c[2] & used]
                    selected = legal[random.randrange(len(legal))]
                a, b, mask = selected
                used |= mask
                holes.append((a, b))
                if pressure[seat.id] and classify is not None:
                    likelihood = _likelihood(
                        (a, b),
                        obs.board,
                        board_cards,
                        board_mask,
                        top,
                        suit_counts,
                        board_value,
                        pressure[seat.id],
                        cards,
                        evaluate,
                        classify,
                    )
                    confidence = (
                        information[seat.id] if information is not None else 1.0
                    )
                    weight *= 1 + confidence * (likelihood - 1)
            if len(obs.board) < 5:
                runout = board_cards.copy()
                for _ in range(5 - len(obs.board)):
                    for _ in range(16):
                        card = random.randrange(52)
                        if not used & (1 << card):
                            break
                    else:
                        remaining = [i for i in range(52) if not used & (1 << i)]
                        card = remaining[random.randrange(len(remaining))]
                    used |= 1 << card
                    runout.append(cards[card])
            else:
                runout = board_cards
            own_rank = evaluate(hero_cards + runout)
            better = 0
            tied = 1 << obs.hero
            for seat, hole in zip(opponents, holes):
                rank = evaluate([cards[hole[0]], cards[hole[1]]] + runout)
                if rank > own_rank:
                    better |= 1 << seat.id
                elif rank == own_rank:
                    tied |= 1 << seat.id
            share = 0.0 if better else 1 / tied.bit_count()
            gross = 0.0
            for amount, eligible in layers:
                if not better & eligible:
                    gross += amount / (tied & eligible).bit_count()
            total_weight += weight
            squared_weight += weight * weight
            weighted_share += weight * share
            weighted_square += weight * share * share
            weighted_return += weight * gross
            wins += weight * (share == 1)
            ties += weight * (0 < share < 1)
            completed += 1
        if completed == 0:
            return Estimate(0, 0, 0, 0, 1, 0, 0)
        mean = weighted_share / total_weight
        effective = total_weight * total_weight / squared_weight
        variance = max(0, weighted_square / total_weight - mean * mean)
        return Estimate(
            mean,
            weighted_return / total_weight,
            wins / total_weight,
            ties / total_weight,
            sqrt(variance / max(1, effective - 1)),
            completed,
            effective,
        )
