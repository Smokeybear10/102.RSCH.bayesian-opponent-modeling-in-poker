from dataclasses import dataclass
from typing import Literal

MAX_SAMPLES = 8192


@dataclass(frozen=True, slots=True)
class GameConfig:
    small_blind: int = 1
    big_blind: int = 2
    stack_bb: int = 100
    rake_bps: int = 0
    rake_cap: int = 6
    no_flop_no_drop: bool = True

    def __post_init__(self) -> None:
        for field in ("small_blind", "big_blind", "stack_bb", "rake_bps", "rake_cap"):
            if type(getattr(self, field)) is not int:
                raise ValueError(f"{field} must be an integer")
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("blinds must satisfy 0 < small blind <= big blind")
        if self.stack_bb < 1:
            raise ValueError("stack_bb must be at least 1")
        if not 0 <= self.rake_bps <= 10000 or self.rake_cap < 0:
            raise ValueError("rake_bps must be 0..10000 and rake_cap must be nonnegative")
        if type(self.no_flop_no_drop) is not bool:
            raise ValueError("no_flop_no_drop must be boolean")

    @property
    def starting_stack(self) -> int:
        return self.stack_bb * self.big_blind


@dataclass(frozen=True, slots=True)
class Action:
    kind: Literal["fold", "check", "call", "raise_to"]
    amount: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("fold", "check", "call", "raise_to"):
            raise ValueError(f"unknown action: {self.kind}")
        if type(self.amount) is not int or self.amount < 0:
            raise ValueError("action amount must be a nonnegative integer")
        if self.kind == "raise_to" and self.amount == 0:
            raise ValueError("raise_to requires a positive street-total amount")
        if self.kind != "raise_to" and self.amount != 0:
            raise ValueError("only raise_to accepts an amount")


@dataclass(frozen=True, slots=True)
class Event:
    actor: int
    street: str
    action: Action
    pot_before: int
    to_call_before: int
    paid: int
    board: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    actor: int
    hole: tuple[str, str]
    board: tuple[str, ...]
    street: str
    button: int
    stacks: tuple[int, int]
    bets: tuple[int, int]
    pot: int
    to_call: int
    min_raise_to: int | None
    max_raise_to: int | None
    can_fold: bool
    history: tuple[Event, ...]
    big_blind: int
    rake_bps: int = 0
    rake_cap: int = 0
    no_flop_no_drop: bool = True


@dataclass(frozen=True, slots=True)
class HandResult:
    gross_stacks: tuple[int, int]
    net_stacks: tuple[int, int]
    gross_returns: tuple[int, int]
    net_returns: tuple[int, int]
    rake: int
    board: tuple[str, ...]
    winners: tuple[int, ...]
    contested_pot: int
