from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class Action:
    kind: Literal["fold", "check", "call", "raise"]
    amount: int = 0


@dataclass(frozen=True, slots=True)
class Seat:
    id: int
    stack: int
    committed: int
    bet: int
    folded: bool = False


@dataclass(frozen=True, slots=True)
class LegalActions:
    call: int
    min_raise_to: int | None = None
    max_raise_to: int | None = None

    def passive(self) -> Action:
        return Action("fold" if self.call else "check")

    def raise_to(self, amount: int) -> Action:
        if self.min_raise_to is None or self.max_raise_to is None:
            return Action("call" if self.call else "check")
        return Action("raise", max(self.min_raise_to, min(amount, self.max_raise_to)))

    def accepts(self, action: Action) -> bool:
        if action.kind == "raise":
            return (
                self.min_raise_to is not None
                and self.max_raise_to is not None
                and self.min_raise_to <= action.amount <= self.max_raise_to
            )
        if action.amount:
            return False
        if action.kind == "check":
            return self.call == 0
        return action.kind in ("fold", "call") and self.call > 0


@dataclass(frozen=True, slots=True)
class Observation:
    hand: int
    hero: int
    button: int
    street: int
    hole: tuple[int, int]
    board: tuple[int, ...]
    seats: tuple[Seat, ...]
    legal: LegalActions
    big_blind: int
    payouts: tuple[float, ...]
    pot: int

    @property
    def player(self) -> Seat:
        return self.seats[self.hero]

    @property
    def opponents(self) -> tuple[Seat, ...]:
        return tuple(s for s in self.seats if s.id != self.hero and not s.folded)


@dataclass(frozen=True, slots=True)
class Event:
    hand: int
    seat: int
    street: int
    action: Action
    facing_bet: bool
    big_blind: int


class Player(Protocol):
    def act(self, obs: Observation) -> Action: ...

    def observe(self, event: Event) -> None: ...


@dataclass(frozen=True, slots=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]


def pots(committed: tuple[int, ...], folded: tuple[bool, ...]) -> tuple[Pot, ...]:
    levels = sorted({c for c in committed if c > 0})
    result = []
    previous = 0
    for level in levels:
        contributors = [i for i, amount in enumerate(committed) if amount >= level]
        eligible = tuple(i for i in contributors if not folded[i])
        if not eligible:
            raise ValueError("a pot must have an eligible player")
        result.append(Pot((level - previous) * len(contributors), eligible))
        previous = level
    return tuple(result)


def settle_pots(layers: tuple[Pot, ...], ranks: tuple[int, ...]) -> tuple[float, ...]:
    winnings = [0.0] * len(ranks)
    for pot in layers:
        best = max(ranks[i] for i in pot.eligible)
        winners = [i for i in pot.eligible if ranks[i] == best]
        for i in winners:
            winnings[i] += pot.amount / len(winners)
    return tuple(winnings)
