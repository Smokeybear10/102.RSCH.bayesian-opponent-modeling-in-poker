import eval7

from .cards import CARD_NAMES
from .sampling import SampledEquity

CARDS = tuple(eval7.Card(name) for name in CARD_NAMES)


def _classify(rank: int) -> int:
    return rank >> 24


class HoldemEquity(SampledEquity):
    def __init__(self, seed: int) -> None:
        super().__init__(seed, cards=CARDS, evaluate=eval7.evaluate, classify=_classify)
