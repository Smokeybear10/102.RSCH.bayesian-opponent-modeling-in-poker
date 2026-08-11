from functools import lru_cache
from itertools import pairwise
from math import isfinite


@lru_cache(maxsize=8)
def _subsets(n: int) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    return tuple(
        (mask.bit_count(), tuple((i, 1 << i) for i in range(n) if not mask & (1 << i)))
        for mask in range(1 << n)
    )


def hero_value(
    stacks: tuple[float, ...], payouts: tuple[float, ...], hero: int
) -> float:
    if not 0 <= hero < len(stacks) or len(payouts) != len(stacks):
        raise ValueError("one payout per stack and a valid hero index are required")
    if any(not isfinite(s) or s < 0 for s in stacks):
        raise ValueError("stacks must be finite and nonnegative")
    if any(not isfinite(p) or p < 0 for p in payouts) or any(
        a < b for a, b in pairwise(payouts)
    ):
        raise ValueError("payouts must be finite, nonnegative, and descending")
    positive = tuple(i for i, s in enumerate(stacks) if s > 0)
    if not positive:
        return sum(payouts) / len(payouts)
    if stacks[hero] == 0:
        return sum(payouts[len(positive) :]) / (len(stacks) - len(positive))
    if len(positive) == 1:
        return payouts[0]
    own = stacks[hero]
    other = tuple(stacks[i] for i in positive if i != hero)
    total = own + sum(other)
    size = 1 << len(other)
    removed = [0.0] * size
    reach = [0.0] * size
    reach[0] = 1.0
    baseline = payouts[len(positive) - 1]
    value = baseline
    for mask, (place, remaining) in enumerate(_subsets(len(other))):
        reward = payouts[place] - baseline
        if reward == 0:
            continue
        if mask:
            bit = mask & -mask
            removed[mask] = removed[mask ^ bit] + other[bit.bit_length() - 1]
        probability = reach[mask]
        denominator = total - removed[mask]
        value += probability * own / denominator * reward
        if not remaining or payouts[place + 1] == baseline:
            continue
        for i, bit in remaining:
            reach[mask | bit] += probability * other[i] / denominator
    return value
