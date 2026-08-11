CARD_NAMES = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
CARD_INDEX = {name: i for i, name in enumerate(CARD_NAMES)}


def _chen(a: int, b: int) -> float:
    high, low = sorted((a // 4 + 2, b // 4 + 2), reverse=True)
    base = {14: 10, 13: 8, 12: 7, 11: 6}.get(high, high / 2)
    if high == low:
        return max(5, base * 2)
    gap = high - low - 1
    score = base + (2 if a % 4 == b % 4 else 0)
    score -= (0, 1, 2, 4, 5)[min(gap, 4)]
    return score + (1 if gap <= 1 and high < 12 else 0)


COMBINATIONS = sorted(
    ((a, b, (1 << a) | (1 << b)) for a in range(52) for b in range(a)),
    key=lambda c: _chen(c[0], c[1]),
    reverse=True,
)
PERCENTILES = [[0.0] * 52 for _ in range(52)]
_groups = {}
for _a, _b, _ in COMBINATIONS:
    _groups.setdefault(_chen(_a, _b), []).append((_a, _b))
_offset = 0
for _group in _groups.values():
    _percentile = 1 - (_offset + len(_group) / 2) / len(COMBINATIONS)
    for _a, _b in _group:
        PERCENTILES[_a][_b] = PERCENTILES[_b][_a] = _percentile
    _offset += len(_group)
del _groups
RANGES = {
    bucket: tuple(c for c in COMBINATIONS if PERCENTILES[c[0]][c[1]] >= 1 - bucket / 20)
    for bucket in range(2, 21)
}


def starting_strength(hole: tuple[int, int]) -> float:
    return PERCENTILES[hole[0]][hole[1]]
