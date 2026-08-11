from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns
from typing import Literal, Protocol

from .model import Action, Event, Observation
from .policy import TournamentPolicy

RaiseMode = Literal["to", "increment", "additional"]


class ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PublicEvent:
    sequence: int
    event: Event


@dataclass(frozen=True, slots=True)
class Frame:
    table: str
    observation: Observation
    events: tuple[PublicEvent, ...] = ()


class Adapter(Protocol):
    variant: str
    has_history: bool

    def decode(self, raw: object) -> Frame: ...

    def encode(self, action: Action, raw: object, frame: Frame) -> object: ...


def normalize_raise(amount: int, obs: Observation, mode: RaiseMode) -> int:
    if mode == "to":
        return amount
    if mode == "increment":
        return amount + max(s.bet for s in obs.seats)
    if mode == "additional":
        return amount + obs.player.bet
    raise ContractError(f"unknown raise convention: {mode}")


def encode_raise(amount: int, obs: Observation, mode: RaiseMode) -> int:
    if mode == "to":
        return amount
    if mode == "increment":
        return amount - max(s.bet for s in obs.seats)
    if mode == "additional":
        return amount - obs.player.bet
    raise ContractError(f"unknown raise convention: {mode}")


def validate_frame(frame: Frame) -> None:
    obs = frame.observation
    n = len(obs.seats)
    if (
        not isinstance(frame.table, str)
        or not frame.table
        or not 2 <= n <= 8
        or type(obs.hero) is not int
        or type(obs.button) is not int
        or not 0 <= obs.hero < n
        or not 0 <= obs.button < n
    ):
        raise ContractError("table ID, hero, button, or roster is invalid")
    if (
        type(obs.hand) is not int
        or obs.hand < 0
        or type(obs.street) is not int
        or obs.street not in range(4)
        or len(obs.board) != (0, 3, 4, 5)[obs.street]
    ):
        raise ContractError(
            "hand number, street, or board length is invalid for Hold'em"
        )
    cards = obs.hole + obs.board
    if (
        len(obs.hole) != 2
        or len(set(cards)) != len(cards)
        or any(type(c) is not int or not 0 <= c < 52 for c in cards)
    ):
        raise ContractError("cards must be distinct canonical integers in 0..51")
    if tuple(s.id for s in obs.seats) != tuple(range(n)):
        raise ContractError(
            "keep a stable, dense roster; map SDK seat IDs in the adapter"
        )
    if any(
        type(v) is not int or v < 0
        for s in obs.seats
        for v in (s.stack, s.committed, s.bet)
    ) or any(s.bet > s.committed for s in obs.seats):
        raise ContractError("stack and contribution accounting is invalid")
    if (
        type(obs.pot) is not int
        or obs.pot != sum(s.committed for s in obs.seats)
        or type(obs.big_blind) is not int
        or obs.big_blind <= 0
    ):
        raise ContractError(
            "pot must include all current-hand contributions and blinds"
        )
    if (
        obs.player.folded
        or obs.player.stack <= 0
        or obs.seats[obs.button].stack + obs.seats[obs.button].committed <= 0
    ):
        raise ContractError(
            "actor or button is absent; normalize the SDK's seat conventions"
        )
    if (
        len(obs.payouts) != n
        or any(not isfinite(p) or p < 0 for p in obs.payouts)
        or sum(obs.payouts) <= 0
        or any(a < b for a, b in zip(obs.payouts, obs.payouts[1:]))
    ):
        raise ContractError(
            "supply actual descending payouts, including zero-paying places"
        )
    legal = obs.legal
    expected_call = min(
        obs.player.stack, max(s.bet for s in obs.seats) - obs.player.bet
    )
    if type(legal.call) is not int or legal.call != expected_call:
        raise ContractError(
            "call must be chips added, capped by the actor's remaining stack"
        )
    if (legal.min_raise_to is None) != (legal.max_raise_to is None):
        raise ContractError("both raise bounds must be present or absent")
    if legal.min_raise_to is not None and (
        type(legal.min_raise_to) is not int or type(legal.max_raise_to) is not int
    ):
        raise ContractError("raise amounts must be whole chips")
    if (
        legal.min_raise_to is not None
        and not max(s.bet for s in obs.seats)
        < legal.min_raise_to
        <= legal.max_raise_to
        <= obs.player.bet + obs.player.stack
    ):
        raise ContractError(
            "raise bounds must be current-street totals, including short all-ins"
        )
    last = -1
    for public in frame.events:
        event = public.event
        if (
            type(public.sequence) is not int
            or public.sequence <= last
            or event.hand != obs.hand
            or not 0 <= event.seat < n
            or not 0 <= event.street <= obs.street
        ):
            raise ContractError(
                "history must contain ordered current-hand public events"
            )
        if (
            event.action.kind not in {"fold", "check", "call", "raise"}
            or event.action.amount < 0
        ):
            raise ContractError("history contains an invalid canonical action")
        last = public.sequence


class Bridge:
    def __init__(
        self,
        adapter: Adapter,
        factory: Callable[[], TournamentPolicy],
        budget_us: int = 800,
        reserve_us: int = 150,
    ) -> None:
        if adapter.variant != "nlhe-standard-52":
            raise ContractError(
                "this strategy requires standard 52-card no-limit Hold'em; adapt the strategy for the revealed variant"
            )
        if not 0 < reserve_us < budget_us <= 1000:
            raise ContractError(
                "leave part of the 1ms limit for SDK and serialization overhead"
            )
        self.adapter = adapter
        self.factory = factory
        self.policy = self._new_policy()
        self.budget_ns = budget_us * 1000
        self.reserve_ns = reserve_us * 1000
        self.table = None
        self.hand = -1
        self.cursor = -1
        self.decisions = self.events_seen = self.table_resets = (
            self.budget_fallbacks
        ) = 0
        self.last_elapsed_ns = 0

    def _new_policy(self) -> TournamentPolicy:
        policy = self.factory()
        if not self.adapter.has_history and policy.parameters.adapt:
            raise ContractError(
                "history is unavailable; set Parameters(adapt=False) explicitly"
            )
        return policy

    def decide(self, raw: object) -> object:
        started = perf_counter_ns()
        deadline = started + self.budget_ns - self.reserve_ns
        frame = self.adapter.decode(raw)
        validate_frame(frame)
        obs = frame.observation
        if frame.table != self.table:
            if self.table is not None:
                self.policy = self._new_policy()
            self.table = frame.table
            self.hand = -1
            self.cursor = -1
            self.table_resets += 1
        if obs.hand < self.hand:
            raise ContractError("hand IDs went backwards without a new table ID")
        if obs.hand != self.hand:
            self.hand, self.cursor = obs.hand, -1
        if not self.adapter.has_history and frame.events:
            raise ContractError("adapter claims no history but supplied events")
        for public in frame.events:
            if public.sequence <= self.cursor:
                continue
            if public.sequence != self.cursor + 1:
                raise ContractError("public history has a gap; fix the cursor mapping")
            self.policy.observe(public.event)
            self.cursor = public.sequence
            self.events_seen += 1
        if perf_counter_ns() >= deadline:
            self.budget_fallbacks += 1
            action = obs.legal.passive()
        else:
            action = self.policy.act(obs, deadline_ns=deadline)
        if not obs.legal.accepts(action):
            raise ContractError(f"policy emitted an illegal action: {action}")
        result = self.adapter.encode(action, raw, frame)
        self.decisions += 1
        self.last_elapsed_ns = perf_counter_ns() - started
        return result
